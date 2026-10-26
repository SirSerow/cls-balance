"""Phase 4 checks extend the frozen suite; generated fixtures are not results."""

import json

import pytest
import torch
from torch import nn

from cls_balance.artifacts import file_hash, object_hash, seed_everything
from cls_balance.checkpoints import load_checkpoint, save_checkpoint, state_hash
from cls_balance.data import AircraftData, AircraftDataset
from cls_balance.infer import Ancestors
from cls_balance.models.cascade import LocalDataset, NodeSpec, node_plan
from cls_balance.models.flat import build_flat
from cls_balance.report_flat import read_json
from cls_balance.residency import OnDemandCNN, cascade_with_policy
from cls_balance.train_independent import IndependentConfig, audit_node, compatibility, load_cascade, read_config, train_seed
from cls_balance.transforms import Preprocessing


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    seed_everything(42)


def test_node_plans_and_singleton_exclusion(tiny_data):
    two = node_plan(tiny_data.taxonomy, 2)
    three = node_plan(tiny_data.taxonomy, 3)
    assert [s.key for s in two] == ["manufacturer", "variant-manufacturer-1"]
    assert [s.key for s in three] == ["manufacturer", "variant-family-0"]
    assert two[1].global_ids == (1, 2)
    assert NodeSpec.from_dict(json.loads(json.dumps(two[1].to_dict()))) == two[1]
    with pytest.raises(ValueError):
        node_plan(tiny_data.taxonomy, 4)


@pytest.mark.parametrize("depth", [2, 3])
def test_local_targets_are_explicit_and_branch_complete(tiny_data, depth):
    base = AircraftDataset(tiny_data, "train", Preprocessing(), training=True)
    for spec in node_plan(tiny_data.taxonomy, depth):
        local = LocalDataset(base, spec, split="train")
        assert set(local.targets) == set(range(len(spec.global_ids)))
        assert len(local) == len(local.image_ids)
        for i, source_index in enumerate(local.indices):
            v, f, m = tiny_data.taxonomy.ids(base.rows[source_index])
            assert spec.global_ids[local[i]["variant_id"]] == {"variant": v, "family": f, "manufacturer": m}[spec.stage]
            if spec.parent_stage:
                assert {"family": f, "manufacturer": m}[spec.parent_stage] == spec.parent_id
    with pytest.raises(ValueError, match="train/val"):
        LocalDataset(base, spec, split="test")


@pytest.mark.parametrize("kwargs", [{"depths": ()}, {"depths": (2, 2)}, {"depths": (4,)}, {"depths": (True,)}])
def test_invalid_depths(kwargs):
    with pytest.raises(ValueError):
        IndependentConfig("data", "prepared", "out", **kwargs)


def test_config_relative_paths(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data_root: data\nprepared_run: prepared\noutput_root: out\ndepths: [2, 3]\n")
    cfg = read_config(path)
    assert cfg.depths == (2, 3) and cfg.data_root == str(tmp_path / "data")


def test_real_full_cnn_node_checkpoint_and_no_weight_sharing(tmp_path, tiny_data):
    hashes = compatibility(tiny_data, Preprocessing())
    spec = node_plan(tiny_data.taxonomy, 2)[0]
    model = build_flat(pretrained=False, classes=len(spec.global_ids)).eval()
    other = build_flat(pretrained=False, classes=len(spec.global_ids))
    assert next(model.parameters()).data_ptr() != next(other.parameters()).data_ptr()
    path = tmp_path / "node.pt"
    save_checkpoint(path, model, {**hashes, "node": spec.to_dict()}, kind="IND-node")
    loaded, metadata = load_checkpoint(path, **hashes, kind="IND-node")
    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        assert torch.equal(model(image), loaded(image))
    assert state_hash(model) == state_hash(loaded)
    assert metadata["node"]["global_ids"] == list(spec.global_ids)
    for policy in ("cold-disk", "preloaded-cpu"):
        lazy = OnDemandCNN(path, hashes, policy=policy, checkpoint_sha256=file_hash(path))
        with torch.inference_mode():
            assert torch.equal(model(image), lazy(image))
            assert torch.equal(model(image), lazy(image))
        assert lazy.calls == 2 and lazy.loads == (2 if policy == "cold-disk" else 0)
    with pytest.raises(ValueError, match="checksum"):
        OnDemandCNN(path, hashes, policy="cold-disk", checkpoint_sha256="0" * 64)


class TinyCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3, 4, 1))
        self.classifier = nn.Linear(4, classes)

    def forward(self, value):
        return self.classifier(self.features(value).flatten(1))


def test_training_bundle_routing_residency_and_retry(tmp_path, full_metadata, monkeypatch):
    """Actual optimizer/export/evaluator path, generated pixels; no claimed accuracy."""
    import cls_balance.checkpoints as checkpointing
    import cls_balance.evaluate as evaluation
    import cls_balance.train_independent as training

    splits, taxonomy, _ = full_metadata
    data = AircraftData(tmp_path, splits, taxonomy, {})

    class GeneratedDataset:
        def __init__(self, data, split, preprocessing, **kwargs):
            assert split in ("train", "val"), "Test pixels must never be accessed"
            self.rows, self.taxonomy = data.splits[split], data.taxonomy
            self.image = torch.zeros(3, 224, 224)

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            row = self.rows[i]
            v, f, m = self.taxonomy.ids(row)
            return {"image": self.image, "image_id": row.image_id, "variant_id": v,
                    "family_id": f, "manufacturer_id": m}

    monkeypatch.setattr(training, "AircraftDataset", GeneratedDataset)
    monkeypatch.setattr(evaluation, "AircraftDataset", GeneratedDataset)
    # Fix only initial features across node widths, as published ImageNet does.
    def factory(*, classes=100, **kwargs):
        seed_everything(42)
        return TinyCNN(classes)
    monkeypatch.setattr(training, "build_flat", factory)
    monkeypatch.setattr(checkpointing, "build_flat", factory)
    config = IndependentConfig(str(tmp_path), str(tmp_path), str(tmp_path / "runs"), seeds=(42,),
        depths=(2, 3), epochs=1, workers=0, batch_size=256, cpu_threads=2)
    run = train_seed(config, data, Preprocessing(), {"run_id": "generated-test"}, 42)
    summary = read_json(run / "summary.json")
    from cls_balance.report_independent import audit_training_run, publish
    assert audit_training_run(run) == summary
    # Exercise publication allowlist and traceability without a plotting dependency.
    monkeypatch.setattr("cls_balance.report_independent.draw_figures", lambda *args: None)
    published = publish([run], tmp_path / "publication")
    assert read_json(published / "aggregate.json")["test_evaluated"] is False
    assert not list(published.rglob("*.pt"))
    assert not list(published.rglob("contract.json"))
    assert not any("training_image_ids" in p.read_text(encoding="utf-8") for p in published.rglob("*.json"))
    assert summary["test_evaluated"] is False
    assert set(summary["evaluations"]) == {f"IND-{d}/{r}" for d in (2, 3) for r in ("predicted", "oracle")}
    image = torch.zeros(3, 224, 224)
    for depth in (2, 3):
        reference = load_cascade(run, depth).predict(image)
        assert reference.backbone_calls == reference.classifier_calls
        for policy in ("preloaded-cpu", "cold-disk"):
            adapter = cascade_with_policy(run, depth, policy)
            assert adapter.predict(image) == reference
            # The oracle never executes upstream CNNs; singleton variants add none.
            oracle = adapter.predict(image, routing="oracle", truth=Ancestors(0, 0))
            assert oracle.backbone_calls == oracle.classifier_calls
            assert oracle.manufacturer_id == 0
    first = next((run / "nodes").glob("*/*/summary.json")).parent
    node = audit_node(first)
    assert node["exact_reload_logits"]
    expected = read_json(first / "contract.json")
    with pytest.raises(ValueError, match="contract"):
        audit_node(first, {**expected, "seed": 44})
    # A complete run cannot be accidentally overwritten/retrained.
    with pytest.raises(ValueError, match="completed"):
        train_seed(config, data, Preprocessing(), {"run_id": "generated-test"}, 42, resume=run)
    # Test node-level verified reuse, not optimizer-state resume.
    spec = NodeSpec.from_dict(expected["node"])
    base = GeneratedDataset(data, "train", Preprocessing())
    validation = GeneratedDataset(data, "val", Preprocessing())
    parent_contract = {k: v for k, v in expected.items() if k not in {"node", "training_image_ids", "selection_image_ids"}}
    assert training.train_node(config, spec, base, validation, parent_contract, run / "nodes") == first
    (first / "history.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        audit_node(first)
