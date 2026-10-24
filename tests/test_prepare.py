import csv
import json

import pytest
import torch
import yaml

from cls_balance.artifacts import file_hash, seed_everything, write_json
from cls_balance.data import AircraftData
from cls_balance.prepare import prepare, read_config


def test_immutable_json(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_deterministic_torch_seed():
    seed_everything(42)
    first = torch.rand(5)
    seed_everything(42)
    assert torch.equal(first, torch.rand(5))
    assert torch.are_deterministic_algorithms_enabled()


def test_config_relative_to_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dataset": {"root": "data", "edition": "torchvision_100_variants", "download": False},
                                   "output_root": "runs", "seed": 42, "preprocessing": {}}))
    config, _ = read_config(path)
    assert config["dataset"]["root"] == str((tmp_path / "data").resolve())
    assert config["output_root"] == str((tmp_path / "runs").resolve())


def test_prepare_artifact_roundtrip(tmp_path, tiny_data, monkeypatch):
    # Only acquisition is replaced; real parser, dataset adapter, image validation,
    # preprocessing, serializers and artifact hashing run on generated fixtures.
    monkeypatch.setattr(AircraftData, "load", lambda *args, **kwargs: tiny_data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dataset": {"root": str(tiny_data.root), "edition": "torchvision_100_variants", "download": False},
                                   "output_root": str(tmp_path / "runs"), "seed": 42, "preprocessing": {}}))
    one = prepare(path)
    two = prepare(path)
    assert one != two
    summary = json.loads((one / "summary.json").read_text())
    other = json.loads((two / "summary.json").read_text())
    assert summary["manifest_sha256"] == other["manifest_sha256"]
    assert summary["taxonomy_sha256"] == other["taxonomy_sha256"]
    for split in ("train", "val", "test"):
        manifest = one / f"manifest_{split}.csv"
        assert file_hash(manifest) == summary["manifest_sha256"][split]
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]["image_id"].startswith("00000")
        assert len(rows) == 3
    assert summary["images"]["decoded_images"] == 9
    assert (one / "spot_checks.jpg").is_file()
