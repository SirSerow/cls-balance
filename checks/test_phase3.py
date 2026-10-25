"""Phase-three extension suite; the validated Phase 2 suite stays byte-for-byte frozen."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cls_balance.artifacts import file_hash, object_hash, seed_everything, write_json
from cls_balance.checkpoints import load_checkpoint, save_checkpoint, state_hash
from cls_balance.models.flat import FrozenEncoder, build_flat
from cls_balance.train import CheckpointSelection, TrainingConfig, read_config, run_epoch, validate_preparation
from cls_balance.prepare import write_csv
from cls_balance.transforms import Preprocessing


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    seed_everything(42)


def contract():
    return {"taxonomy_sha256": "taxonomy", "preprocessing_sha256": "preprocessing",
            "dataset_manifest_sha256": "dataset"}


def test_selection_uses_strict_improvement_and_keeps_earliest_tie():
    selection = CheckpointSelection()
    assert selection.observe(1, 0.0)
    assert not selection.observe(2, 0.0)
    assert selection.best_epoch == 1 and selection.stale == 1
    assert selection.observe(3, 0.2)
    assert selection.best_epoch == 3 and selection.stale == 0
    for epoch in range(4, 11):
        assert not selection.observe(epoch, 0.1)
    assert selection.stale == 7 and selection.best_score == 0.2


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1])
def test_selection_rejects_invalid_validation_scores(value):
    with pytest.raises(ValueError, match="selection"):
        CheckpointSelection().observe(1, value)


@pytest.mark.parametrize("kind", ["FLAT", "FT-FROZEN-encoder"])
def test_real_efficientnet_reload_and_frozen_encoder(tmp_path, kind):
    flat = build_flat(pretrained=False)
    assert flat.classifier[1].out_features == 100
    model = flat if kind == "FLAT" else FrozenEncoder(flat)
    model.eval()
    image = torch.randn(1, 3, 224, 224)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, contract(), kind=kind)
    loaded, _ = load_checkpoint(path, **contract(), kind=kind)
    with torch.inference_mode():
        assert torch.equal(model(image), loaded(image))
    assert state_hash(model) == state_hash(loaded)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, contract(), kind=kind)
    if kind == "FT-FROZEN-encoder":
        before = state_hash(loaded)
        loaded.train()
        assert not any(m.training for m in loaded.modules())
        assert not any(p.requires_grad for p in loaded.parameters())
        features = loaded(image)
        assert features.shape == (1, 1280) and not features.requires_grad
        head = nn.Linear(1280, 2)
        optimizer = torch.optim.AdamW(head.parameters())
        head(features).sum().backward()
        optimizer.step()
        assert state_hash(loaded) == before
        assert all(p.grad is None for p in loaded.parameters())


@pytest.mark.parametrize("field", list(contract()))
def test_checkpoint_rejects_wrong_compatibility(tmp_path, field):
    model = build_flat(pretrained=False)
    path = tmp_path / "flat.pt"
    save_checkpoint(path, model, contract(), kind="FLAT")
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(path, **{**contract(), field: "wrong"})


def test_checkpoint_rejects_corruption_and_kind(tmp_path):
    path = tmp_path / "flat.pt"
    save_checkpoint(path, build_flat(pretrained=False), contract(), kind="FLAT")
    with pytest.raises(ValueError, match="kind"):
        load_checkpoint(path, **contract(), kind="FT-FROZEN-encoder")
    with path.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, **contract())


def test_encoder_matches_pooled_flat_and_owns_separate_storage():
    flat = build_flat(pretrained=False).eval()
    encoder = FrozenEncoder(flat)
    image = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        assert torch.equal(encoder(image), flat.avgpool(flat.features(image)).flatten(1))
    assert next(flat.parameters()).data_ptr() != next(encoder.parameters()).data_ptr()


def test_epoch_uses_all_samples_macro_recall_and_no_validation_updates():
    # Unequal supports distinguish macro recall (1/2) from top-1 (3/4).
    samples = [{"image": torch.tensor([1.0, 0.0]), "variant_id": y} for y in [0, 0, 0, 1]]
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    before = state_hash(model)
    result = run_epoch(model, DataLoader(samples, batch_size=3), torch.device("cpu"), classes=2)
    assert result["macro_recall"] == 0.5 and result["top1"] == 0.75
    assert result["images"] == 4 and result["optimizer_steps"] == 0
    assert state_hash(model) == before
    trained = run_epoch(model, DataLoader(samples, batch_size=3), torch.device("cpu"),
                        torch.optim.AdamW(model.parameters()), classes=2)
    assert trained["optimizer_steps"] == 2 and trained["images"] == 4
    assert state_hash(model) != before


def test_epoch_rejects_missing_class_and_nonfinite_loss():
    samples = [{"image": torch.ones(2), "variant_id": 0}]
    with pytest.raises(ValueError, match="support"):
        run_epoch(nn.Linear(2, 2), DataLoader(samples), torch.device("cpu"), classes=2)
    samples[0]["image"][:] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        run_epoch(nn.Linear(2, 2), DataLoader(samples), torch.device("cpu"), classes=2)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"patience": 0}, {"batch_size": 0},
    {"workers": -1}, {"seeds": (42, 42)}, {"seeds": (-1,)}, {"learning_rate": float("nan")},
    {"weight_decay": -1}, {"device": "meta"}, {"cpu_threads": 0}])
def test_invalid_training_configs(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig("data", "prepared", "runs", **kwargs)


def test_config_paths_resolve_against_yaml_and_unknown_fields_fail(tmp_path):
    config = tmp_path / "flat.yaml"
    config.write_text("data_root: data\nprepared_run: prepared\noutput_root: runs\nseeds: [42, 43, 44]\n")
    result = read_config(config)
    assert result.data_root == str(tmp_path / "data")
    assert result.seeds == (42, 43, 44)
    config.write_text(config.read_text() + "test_split: test\n")
    with pytest.raises(TypeError):
        read_config(config)


@pytest.mark.parametrize("tamper", [None, "image", "manifest", "taxonomy", "inventory"])
def test_preparation_rejects_changed_inputs(tiny_data, tmp_path, tamper):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    preprocessing = Preprocessing()
    manifests = {}
    for split in tiny_data.splits:
        path = prepared / f"manifest_{split}.csv"
        write_csv(path, tiny_data.manifest(split))
        manifests[split] = file_hash(path)
    inventory = [{"image_id": row.image_id, "sha256": file_hash(tiny_data.image_path(row.image_id))}
                 for rows in tiny_data.splits.values() for row in rows]
    write_csv(prepared / "image_checksums.csv", inventory)
    summary = {"status": "validated", "edition": "torchvision_100_variants",
               "preprocessing_sha256": preprocessing.sha256,
               "taxonomy_sha256": object_hash(tiny_data.taxonomy.to_dict()),
               "source_annotations_sha256": object_hash(tiny_data.source_hashes),
               "manifest_sha256": manifests,
               "image_checksums_sha256": file_hash(prepared / "image_checksums.csv")}
    if tamper == "taxonomy":
        summary["taxonomy_sha256"] = "wrong"
    write_json(prepared / "summary.json", summary)
    if tamper == "image":
        tiny_data.image_path(tiny_data.splits["train"][0].image_id).write_bytes(b"modified")
    if tamper in {"manifest", "inventory"}:
        target = prepared / ("manifest_val.csv" if tamper == "manifest" else "image_checksums.csv")
        target.write_text("tampered")
    if tamper is None:
        assert validate_preparation(tiny_data, preprocessing, prepared) == summary
    else:
        with pytest.raises(ValueError, match="mismatch"):
            validate_preparation(tiny_data, preprocessing, prepared)
