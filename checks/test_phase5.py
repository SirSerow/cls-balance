"""Shared-feature correctness checks; synthetic pixels are never research results."""

import pytest
import torch
from torch import nn

from cls_balance.artifacts import file_hash, object_hash, seed_everything
from cls_balance.checkpoints import load_checkpoint, save_checkpoint, state_hash
from cls_balance.embeddings import training_features
from cls_balance.models.cascade import node_plan
from cls_balance.models.shared import assert_frozen, build_head
from cls_balance.report_flat import read_json
from cls_balance.train_independent import compatibility
from cls_balance.train_shared import SharedConfig, load_shared, read_config, source_encoder, train_seed
from cls_balance.transforms import Preprocessing


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    seed_everything(42)


@pytest.mark.parametrize("kwargs", [{"depths": ()}, {"depths": (2, 2)}, {"depths": (True,)},
    {"backbone": "bad"}, {"hidden_dimension": 0}, {"hidden_dimension": True}, {"backbone": "FT-FROZEN"}])
def test_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        SharedConfig("d", "p", "o", **({"backbone": "FROZEN"} | kwargs))


def test_config_paths(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("data_root: d\nprepared_run: p\noutput_root: o\ncache_root: cache\nflat_runs: {42: flat}\nseeds: [42]\ndepths: [2, 3]\n")
    config = read_config(path)
    assert config.flat_runs[42] == str(tmp_path / "flat")
    assert config.cache_root == str(tmp_path / "cache")
    assert config.depths == (2, 3)


@pytest.mark.parametrize("changes", [{}, {"seed": 43}, {"training_image_ids": ["test-id"]},
    {"selection_image_ids": ["test-id"]}, {"parent_flat_sha256": "wrong"}, {"run_id": "wrong"}])
def test_source_encoder_requires_seed_matched_official_provenance(tmp_path, tiny_data, monkeypatch, changes):
    import cls_balance.train_shared as training
    (tmp_path / "flat.pt").write_bytes(b"source reference")
    (tmp_path / "encoder.pt").write_bytes(b"source reference")
    metadata = {"seed": 42, "run_id": "flat-source", "parent_flat_sha256": file_hash(tmp_path / "flat.pt"),
                "training_image_ids": [r.image_id for r in tiny_data.splits["train"]],
                "selection_image_ids": [r.image_id for r in tiny_data.splits["val"]]} | changes
    monkeypatch.setattr("cls_balance.report_flat.audit_training_run", lambda path:
                        {"seed": 42, "run_id": "flat-source", "resources": {}})
    monkeypatch.setattr(training, "load_checkpoint", lambda *a, **k: (TinyEncoder(), metadata))
    config = SharedConfig("d", "p", "o", seeds=(42,), flat_runs={42: str(tmp_path)})
    if changes:
        with pytest.raises(ValueError, match="provenance"):
            source_encoder(config, tiny_data, Preprocessing(), 42)
    else:
        _, source = source_encoder(config, tiny_data, Preprocessing(), 42)
        assert source["source_run_id"] == "flat-source"


@pytest.mark.parametrize("values", [(1, 256, 1280), (30, 0, 1280), (30, 256, True)])
def test_head_dimensions(values):
    with pytest.raises(ValueError, match="dimensions"):
        build_head(*values)


class TinyEncoder(nn.Module):
    dimension = 1280

    def __init__(self, unused=None):
        super().__init__()
        self.register_buffer("scale", torch.ones(1))
        self.calls = 0
        self.eval()

    def forward(self, images):
        self.calls += 1
        return images[:, 0, 0, 0].unsqueeze(1).expand(-1, 1280) * self.scale


def test_frozen_guard_catches_buffers_modes_and_gradients():
    model = TinyEncoder()
    fingerprint = state_hash(model)
    assert_frozen(model, fingerprint)
    model.train()
    with pytest.raises(ValueError, match="Frozen"):
        assert_frozen(model, fingerprint)
    model.eval()
    model.scale.add_(1)
    with pytest.raises(ValueError, match="Frozen"):
        assert_frozen(model, fingerprint)
    model = nn.Linear(3, 2).eval()
    with pytest.raises(ValueError, match="Frozen"):
        assert_frozen(model, state_hash(model))


def test_head_checkpoint_structure_and_version(tmp_path, tiny_data):
    spec = node_plan(tiny_data.taxonomy, 2)[0]
    hashes = compatibility(tiny_data, Preprocessing())
    model = build_head(len(spec.global_ids), 8)
    metadata = {**hashes, "node": spec.to_dict(), "hidden_dimension": 8, "input_dimension": 1280}
    save_checkpoint(tmp_path / "head.pt", model, metadata, kind="shared-head")
    loaded, _ = load_checkpoint(tmp_path / "head.pt", **hashes, kind="shared-head")
    assert state_hash(model) == state_hash(loaded)
    with pytest.raises(ValueError, match="preprocessing"):
        load_checkpoint(tmp_path / "head.pt", **(hashes | {"preprocessing_sha256": "wrong"}), kind="shared-head")


class GeneratedDataset:
    def __init__(self, data, split, preprocessing, training=False, seed=42):
        assert split in ("train", "val"), "No test images may be processed"
        self.rows, self.taxonomy, self.preprocessing = data.splits[split], data.taxonomy, preprocessing
        self.training, self.seed, self.epoch = training, seed, 0
        self.image = torch.zeros(3, 224, 224)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        v, f, m = self.taxonomy.ids(row)
        return {"image": self.image if self.epoch == 0 else self.image + (self.epoch * .01), "image_id": row.image_id,
                "variant_id": v, "family_id": f, "manufacturer_id": m}


def test_training_cache_epoch_version_corruption_and_split(tmp_path, tiny_data):
    dataset = GeneratedDataset(tiny_data, "train", Preprocessing(), training=True)
    encoder = TinyEncoder()
    contract = {**compatibility(tiny_data, Preprocessing()), "backbone_sha256": state_hash(encoder),
                "split": "train", "image_ids": [r.image_id for r in dataset.rows], "seed": 42,
                "epoch": 0, "augmentation": "seeded epoch"}
    kwargs = dict(device=torch.device("cpu"), batch_size=2, workers=0, cache_root=tmp_path)
    first = training_features(encoder, dataset, contract=contract, **kwargs)
    calls = encoder.calls
    assert torch.equal(first, training_features(encoder, dataset, contract=contract, **kwargs))
    assert encoder.calls == calls
    with pytest.raises(ValueError, match="mismatch"):
        training_features(encoder, dataset, contract=contract | {"epoch": 1}, **kwargs)
    dataset.set_epoch(1)
    second = training_features(encoder, dataset, contract=contract | {"epoch": 1}, **kwargs)
    assert not torch.equal(first, second)
    for split in ("val", "test"):
        with pytest.raises(ValueError, match="training"):
            training_features(encoder, dataset, contract=contract | {"split": split}, **kwargs)
    dataset.set_epoch(0)
    cached = tmp_path / (object_hash(contract) + ".pt")
    cached.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        training_features(encoder, dataset, contract=contract, **kwargs)


@pytest.mark.parametrize("backbone", ["FROZEN", "FT-FROZEN"])
def test_train_reload_route_replace_and_publish(tmp_path, full_metadata, monkeypatch, backbone):
    import cls_balance.train_shared as training
    import cls_balance.checkpoints as checkpoints
    import cls_balance.evaluate as evaluation
    from cls_balance.data import AircraftData

    splits, taxonomy, _ = full_metadata
    data = AircraftData(tmp_path, splits, taxonomy, {})
    monkeypatch.setattr(training, "AircraftDataset", GeneratedDataset)
    monkeypatch.setattr(evaluation, "AircraftDataset", GeneratedDataset)
    monkeypatch.setattr(training, "source_encoder", lambda *args: (TinyEncoder(), {"source": "generated fixture"}))
    monkeypatch.setattr(checkpoints, "FrozenEncoder", TinyEncoder)
    monkeypatch.setattr(checkpoints, "build_flat", lambda **kwargs: nn.Identity())
    config = SharedConfig(str(tmp_path), str(tmp_path), str(tmp_path / "runs"), seeds=(42,),
        backbone=backbone, flat_runs={2: "unused", 10: "unused", 42: "fixture"}, depths=(2, 3), hidden_dimension=8,
        epochs=2, patience=1, workers=0, batch_size=256, cpu_threads=2)
    run = train_seed(config, data, Preprocessing(), {"run_id": "generated"}, 42)
    summary = read_json(run / "summary.json")
    assert summary["test_evaluated"] is False and summary["frozen_weights_and_buffers_unchanged"]
    assert len(summary["evaluations"]) == 4
    for depth in (2, 3):
        model = load_shared(run, depth)
        assert model.predict(torch.zeros(3, 224, 224)).backbone_calls == 1
        assert model.backbone.calls == 1
        untouched = {k: (state_hash(n), n(torch.ones(1, 1280))) for k, n in
                     ((k, v.classifier) for k, v in model.nodes.items())}
        spec = next(s for s in node_plan(taxonomy, depth) if s.stage == "variant")
        head_path = run / "nodes" / spec.key / "head.pt"
        metadata = read_json(head_path.with_suffix(".json"))["metadata"]
        replacement = build_head(len(spec.global_ids), 8)
        save_checkpoint(tmp_path / f"replacement{depth}.pt", replacement, metadata, kind="shared-head")
        updated = load_shared(run, depth, replacements={spec.key: tmp_path / f"replacement{depth}.pt"})
        for key, node in updated.nodes.items():
            if key != spec.adapter_key:
                assert state_hash(node.classifier) == untouched[key][0]
                assert torch.equal(node.classifier(torch.ones(1, 1280)), untouched[key][1])
        assert state_hash(updated.backbone) == summary["backbone_sha256"]
        assert state_hash(updated.nodes[spec.adapter_key].classifier) != untouched[spec.adapter_key][0]
        bad = metadata | {"backbone_sha256": "wrong"}
        save_checkpoint(tmp_path / f"bad{depth}.pt", replacement, bad, kind="shared-head")
        with pytest.raises(ValueError, match="compatibility"):
            load_shared(run, depth, replacements={spec.key: tmp_path / f"bad{depth}.pt"})
        leaked = metadata | {"training_image_ids": [data.splits["test"][0].image_id]}
        save_checkpoint(tmp_path / f"leaked{depth}.pt", replacement, leaked, kind="shared-head")
        with pytest.raises(ValueError, match="provenance"):
            load_shared(run, depth, replacements={spec.key: tmp_path / f"leaked{depth}.pt"})
    from cls_balance.report_shared import audit_training_run, publish
    assert audit_training_run(run) == summary
    monkeypatch.setattr("cls_balance.report_shared.draw_figures", lambda *args: None)
    publication = publish([run], tmp_path / "published")
    assert not list(publication.rglob("*.pt"))
    assert not any("training_image_ids" in p.read_text(encoding="utf-8") for p in publication.rglob("*.json"))
    node = next((run / "nodes").glob("*/history.csv"))
    node.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        audit_training_run(run)
