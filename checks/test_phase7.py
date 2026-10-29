"""Class-introduction leakage, immutable replacement and artifact regression checks."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from cls_balance import incremental as inc
from cls_balance.artifacts import file_hash, write_json
from cls_balance.checkpoints import state_hash
from cls_balance.data import AircraftData
from cls_balance.infer import HierarchicalAdapter, LogitNode
from cls_balance.taxonomy import Taxonomy
from cls_balance.transforms import Preprocessing


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


class TinyB0(nn.Module):
    """Cheap shape-compatible stand-in; never used for experimental metrics."""

    def __init__(self, classes=100, **kwargs):
        super().__init__()
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3, 1280, 1), nn.SiLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(1280, classes))

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)).flatten(1))


@pytest.fixture
def one_branch(tiny_data):
    splits = {s: [replace(r, manufacturer="Only maker") for r in rows] for s, rows in tiny_data.splits.items()}
    return AircraftData(tiny_data.root, splits, Taxonomy.from_training(splits["train"]), {})


def test_selection_uses_training_taxonomy_only(one_branch):
    selected = inc.select_introduction(one_branch.taxonomy)
    assert selected["variant_id"] == 1
    assert len(selected["branch_variant_ids"]) == 3
    assert selected == inc.select_introduction(Taxonomy.from_training(reversed(one_branch.splits["train"])))
    assert len(one_branch.taxonomy.branches()["family_to_variant"][one_branch.taxonomy.variant_to_family[selected["variant_id"]]]) > 1


def test_no_eligible_branch_rejected(tiny_data):
    with pytest.raises(ValueError, match="three"):
        inc.select_introduction(tiny_data.taxonomy)


def test_initial_view_excludes_train_and_val_preserves_test_and_global_ids(one_branch):
    original = {s: list(rows) for s, rows in one_branch.splits.items()}
    view = inc.supervised_data(one_branch, 0)
    assert view.taxonomy is one_branch.taxonomy
    for split in ("train", "val"):
        assert len(view.splits[split]) == 2
        assert all(view.taxonomy.ids(r)[0] != 0 for r in view.splits[split])
    assert view.splits["test"] == original["test"]
    assert one_branch.splits == original
    specs = inc.initial_specs(view.taxonomy, 0)
    assert len(specs) == 1 and specs[0].global_ids == (1, 2)


@pytest.mark.parametrize("value", [-1, 3, True, "0"])
def test_bad_holdout(one_branch, value):
    with pytest.raises(ValueError):
        inc.supervised_data(one_branch, value)


@pytest.mark.parametrize("ids", [(), (0, 0), (2, 1), (9,), (True,)])
def test_bad_active_logit_maps(one_branch, ids):
    with pytest.raises(ValueError):
        HierarchicalAdapter(one_branch.taxonomy, {}, active_variant_ids=ids)


def test_active_subset_keeps_manufacturers_and_masks_predictions(one_branch):
    node = LogitNode(nn.Linear(3, 2), (1, 2))
    encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    model = HierarchicalAdapter(one_branch.taxonomy, {"variant:0": node}, backbone=encoder,
                                active_variant_ids=(1, 2))
    prediction = model.predict(torch.zeros(3, 224, 224))
    assert prediction.variant_id in (1, 2) and prediction.backbone_calls == 1
    assert prediction.singleton_bypasses == 1
    assert all(not p.requires_grad for p in encoder.parameters())


def test_masking_cannot_remove_manufacturer(tiny_data):
    with pytest.raises(ValueError, match="preserve"):
        HierarchicalAdapter(tiny_data.taxonomy, {}, active_variant_ids=(1, 2))


@pytest.mark.parametrize("cnn", [False, True])
def test_expansion_copies_rows_by_global_id_and_preserves_other_tensors(cnn):
    model = TinyB0(classes=2) if cnn else nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    before = {k: t.clone() for k, t in model.state_dict().items()}
    old = model.classifier[-1] if cnn else model[-1]
    weight, bias = old.weight.detach().clone(), old.bias.detach().clone()
    inc.expand_classifier(model, (1, 3), (0, 1, 3), seed=42)
    new = model.classifier[-1] if cnn else model[-1]
    torch.testing.assert_close(new.weight[1:], weight, rtol=0, atol=0)
    torch.testing.assert_close(new.bias[1:], bias, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        if before[key].shape == value.shape:
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert new.out_features == 3


def test_expansion_rejects_multiple_or_wrong_outputs():
    model = nn.Sequential(nn.Linear(3, 2))
    with pytest.raises(ValueError):
        inc.expand_classifier(model, (0, 1), (0, 1, 2, 3), seed=42)
    with pytest.raises(ValueError):
        inc.expand_classifier(model, (0,), (0, 1), seed=42)


def test_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        inc.local_path(tmp_path, "../elsewhere.pt")


@pytest.mark.parametrize("kwargs", [{"head_learning_rate": float("nan")}, {"hidden_dimension": 0},
                                     {"head_learning_rate": 0}, {"epochs": 0}, {"seeds": (42, 42)}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        inc.IncrementalConfig("data", "prep", "out", **kwargs)


@pytest.fixture
def trained(tmp_path, one_branch, monkeypatch):
    from cls_balance import checkpoints, train_independent
    for module in (inc, checkpoints, train_independent):
        monkeypatch.setattr(module, "build_flat", lambda *, classes=100, **kw: TinyB0(classes=classes))
    root = tmp_path / "suite"
    root.mkdir()
    config = inc.IncrementalConfig(str(one_branch.root), "unused", str(root), seeds=(42,),
                                    epochs=2, patience=1, batch_size=2, workers=0, cpu_threads=2, hidden_dimension=4)
    write_json(root / "source_hashes.json", {"synthetic": True})
    write_json(root / "taxonomy.json", one_branch.taxonomy.to_dict())
    write_json(root / "compatibility.json", inc.compatibility(one_branch, Preprocessing()))
    write_json(root / "config.resolved.json", inc.asdict(config))
    write_json(root / "selection.json", inc.select_introduction(one_branch.taxonomy))
    # Keep every family populated: variant 1 shares its family with variant 2.
    inc.train_seed(root, config, one_branch, Preprocessing(), inc.select_introduction(one_branch.taxonomy), 42)
    write_json(root / "summary.json", {"status": "complete", "seeds": [42], "artifacts_sha256": inc.inventory(root)})
    return root, config, one_branch


def test_full_reduced_training_and_single_specialist_replacement(trained):
    root, config, data = trained
    summary = inc.audit_inventory(root / "seed-42")
    assert summary["test_evaluated"] is False
    bundle = inc.read_json(root / "seed-42/bundle.json")
    assert bundle["original_artifacts_unchanged"]
    protected = bundle["encoder"]
    for architecture in ("IND-2", "FT-FROZEN-2"):
        before = inc.load_system(root, 42, architecture, "before")
        after = inc.load_system(root, 42, architecture, "after")
        assert before.nodes["variant:0"].global_ids == (0, 2)
        assert after.nodes["variant:0"].global_ids == (0, 1, 2)
        assert before.predict(torch.zeros(3, 224, 224)).variant_id != 1
        for stage in ("initial", "replacement"):
            entries = bundle[architecture][stage].values() if stage == "initial" else [bundle[architecture][stage]]
            for entry in entries:
                meta = inc.read_json(inc.local_path(root, entry["path"]).with_suffix(".json"))["metadata"]
                train_ids = {r.image_id for r in data.splits["train"]}
                val_ids = {r.image_id for r in data.splits["val"]}
                assert set(meta["training_image_ids"]) <= train_ids
                assert set(meta["selection_image_ids"]) <= val_ids
                if stage == "initial":
                    heldout_ids = {r.image_id for rows in data.splits.values() for r in rows if data.taxonomy.ids(r)[0] == 1}
                    assert not heldout_ids.intersection(meta["training_image_ids"] + meta["selection_image_ids"])
        if architecture == "FT-FROZEN-2":
            assert state_hash(before.backbone) == state_hash(after.backbone) == protected["state_sha256"]
    original = inc.inventory(root / "seed-42")
    inc.train_seed(root, config, data, Preprocessing(), inc.select_introduction(data.taxonomy), 42)
    assert original == inc.inventory(root / "seed-42")


def test_corruption_rejected(trained):
    root, _, _ = trained
    bundle = inc.read_json(root / "seed-42/bundle.json")
    target = inc.local_path(root, bundle["encoder"]["path"])
    with target.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        inc.load_system(root, 42, "FT-FROZEN-2", "after")
    with pytest.raises(ValueError, match="corrupt"):
        inc.audit_inventory(root / "seed-42")


def test_semantic_training_audit_rederives_exact_supervised_ids(trained):
    root, _, data = trained
    audit = inc.audit_training(root, data)
    assert audit["checkpoints_verified"] == 6
    assert audit["test_leakage"] is False
    assert audit["heldout_excluded_from_all_initial_train_and_val"]


def test_shared_bundle_rejects_other_encoder_version(trained):
    from cls_balance.checkpoints import save_checkpoint
    root, _, _ = trained
    target = root / "seed-42/bundle.json"
    bundle = inc.read_json(target)
    model, metadata = inc.load_entry(root, bundle["FT-FROZEN-2"]["initial"]["variant:0"],
                                     inc.read_json(root / "compatibility.json"))
    metadata["backbone_sha256"] = "f" * 64
    forged = root / "other-version.pt"
    save_checkpoint(forged, model, metadata, kind="shared-head")
    bundle["FT-FROZEN-2"]["initial"]["variant:0"] = inc.checkpoint_entry(root, forged, "shared-head")
    target.unlink()
    write_json(target, bundle)
    with pytest.raises(ValueError, match="head/encoder version"):
        inc.load_system(root, 42, "FT-FROZEN-2", "before")


@pytest.fixture
def grouped_records(full_metadata):
    from cls_balance.infer import Decision, Prediction
    from cls_balance.metrics import PredictionRecord
    _, taxonomy, _ = full_metadata
    selection = inc.select_introduction(taxonomy)
    def record(v, predicted, route=None):
        f = taxonomy.variant_to_family[v]
        m = taxonomy.family_to_manufacturer[f]
        pf = taxonomy.variant_to_family[predicted]
        pm = taxonomy.family_to_manufacturer[pf]
        if route is not None:
            assert route == pm
        decisions = (Decision("manufacturer", None, pm, "classifier"),
                     Decision("variant", pm, predicted, "classifier"))
        return PredictionRecord(f"{v:07d}", "test", "IND-2", "predicted", v, f, m,
                                Prediction(predicted, pm, pf, decisions, 2))
    heldout = selection["variant_id"]
    other = next(v for v in selection["branch_variant_ids"] if v != heldout)
    before = [record(v, v if v != heldout else other) for v in range(100)]
    after = [record(v, v) for v in range(100)]
    return taxonomy, selection, record, before, after


def test_group_metrics_fixed_denominators_and_unknown_before(grouped_records):
    from cls_balance.report_incremental import compare_records
    taxonomy, selection, _, before, after = grouped_records
    result = compare_records(before, after, selection, taxonomy)
    assert result["groups"]["new"]["before"]["macro_recall"] == 0
    assert result["groups"]["new"]["after"]["macro_recall"] == 1
    assert result["groups"]["known"]["before"]["macro_recall"] == 1
    assert result["groups"]["all"]["before"]["macro_recall"] == .99
    assert result["functional_locality"]["changed_predictions_routed_outside"] == 0


def test_true_outside_branch_can_change_when_misrouted(grouped_records):
    from cls_balance.report_incremental import compare_records
    taxonomy, selection, record, before, after = grouped_records
    outside = next(v for v in range(100) if v not in selection["branch_variant_ids"])
    old_choice = next(v for v in selection["branch_variant_ids"] if v != selection["variant_id"])
    before[outside] = record(outside, old_choice)
    after[outside] = record(outside, selection["variant_id"])
    result = compare_records(before, after, selection, taxonomy)
    assert result["functional_locality"]["changed_predictions_with_true_manufacturer_outside"] == 1
    assert result["groups"]["outside_branch"]["before"]["macro_recall"] == result["groups"]["outside_branch"]["after"]["macro_recall"]


def test_router_change_rejected(grouped_records):
    from cls_balance.report_incremental import compare_records
    taxonomy, selection, record, before, after = grouped_records
    v = next(v for v in range(100) if v not in selection["branch_variant_ids"])
    after[v] = record(v, selection["variant_id"])
    with pytest.raises(ValueError, match="Router changed"):
        compare_records(before, after, selection, taxonomy)


def test_group_missing_class_and_image_id_mismatch_rejected(grouped_records):
    from cls_balance.report_incremental import group_metrics, compare_records
    taxonomy, selection, _, before, after = grouped_records
    with pytest.raises(ValueError, match="support"):
        group_metrics(before[1:], (0, 1))
    with pytest.raises(ValueError, match="test IDs"):
        compare_records(before, list(reversed(after)), selection, taxonomy)


def test_full_networks_reflect_mask_and_replacement_epochs(tmp_path, full_metadata):
    from cls_balance.network_diagrams import draw_network, network_inventory
    _, taxonomy, _ = full_metadata
    selection = inc.select_introduction(taxonomy)
    ids = tuple(v for v in range(100) if v != selection["variant_id"])
    inventory = network_inventory(taxonomy, "IND-2", active_variant_ids=ids)
    outputs = [v for b in inventory["branches"] for v in b["global_variant_ids"]]
    assert sorted(outputs) == list(ids)
    trained = [{"architecture": "IND-2", "seed": 42, "key": s.key, "best_epoch": 3}
               for s in inc.initial_specs(taxonomy, selection["variant_id"])]
    draw_network(tmp_path, taxonomy, "IND-2", trained, active_variant_ids=ids, stage="before",
                 introduced_variant_id=selection["variant_id"])
    text = (tmp_path / "network-IND-2.svg").read_text(encoding="utf-8")
    assert "s42: epoch 3" in text and "excluded from ALL initial" in text
    saved = inc.read_json(tmp_path / "network-IND-2.json")
    assert saved["active_variant_ids"] == list(ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional real CUDA class-introduction integration")
def test_real_cuda_reduced_backbone_and_replacement(tmp_path, one_branch, monkeypatch):
    """Actual EfficientNet kernels on generated pixels; never download weights."""
    from cls_balance import train_independent
    from cls_balance.models.flat import build_flat
    for module in (inc, train_independent):
        monkeypatch.setattr(module, "build_flat", lambda *, classes=100, **kw:
                            build_flat(pretrained=False, classes=classes))
    root = tmp_path / "cuda-synthetic-check-only"
    root.mkdir()
    config = inc.IncrementalConfig(str(one_branch.root), "unused", str(root), seeds=(42,), device="cuda:0",
                                  epochs=1, patience=1, batch_size=2, workers=0, cpu_threads=2, hidden_dimension=4)
    write_json(root / "source_hashes.json", {"synthetic_cuda": True})
    write_json(root / "taxonomy.json", one_branch.taxonomy.to_dict())
    write_json(root / "compatibility.json", inc.compatibility(one_branch, Preprocessing()))
    write_json(root / "config.resolved.json", inc.asdict(config))
    write_json(root / "selection.json", inc.select_introduction(one_branch.taxonomy))
    inc.train_seed(root, config, one_branch, Preprocessing(), inc.select_introduction(one_branch.taxonomy), 42)
    write_json(root / "summary.json", {"status": "complete", "seeds": [42], "artifacts_sha256": inc.inventory(root)})
    assert inc.audit_training(root, one_branch)["checkpoints_verified"] == 6
    before = inc.load_system(root, 42, "FT-FROZEN-2", "before").to("cuda:0")
    after = inc.load_system(root, 42, "FT-FROZEN-2", "after").to("cuda:0")
    assert state_hash(before.backbone) == state_hash(after.backbone)
    image = torch.zeros(3, 224, 224, device="cuda:0")
    assert before.predict(image).backbone_calls == after.predict(image).backbone_calls == 1
