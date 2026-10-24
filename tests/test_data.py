from dataclasses import replace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cls_balance.data import (AircraftData, AircraftDataset, inspect_images, read_annotations,
                              read_catalog, read_splits, validate_edition)
from cls_balance.transforms import Preprocessing


def test_join_by_id_not_row_order(tiny_data):
    rows = tiny_data.splits["train"]
    assert [(r.variant, r.family, r.manufacturer) for r in rows] == [
        ("Zulu jet", "Alpha", "Second maker"), ("Alpha jet", "Beta", "First maker"),
        ("Middle jet", "Alpha", "Second maker")]


@pytest.mark.parametrize("content", ["0000001 V\n0000001 V\n", "../bad V\n", "0000001\n", "\n", ""])
def test_bad_annotations(tmp_path, content):
    path = tmp_path / "labels.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_annotations(path)


@pytest.mark.parametrize("content", ["", "V\nV\n", "V\n\n"])
def test_bad_catalog(tmp_path, content):
    path = tmp_path / "classes.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_catalog(path)


def test_missing_annotation_level_id(tiny_data):
    directory = tiny_data.root / "fgvc-aircraft-2013b" / "data"
    (directory / "images_family_val.txt").write_text("0000010 Alpha\n")
    with pytest.raises(ValueError, match="mismatched label IDs"):
        read_splits(directory)


def test_split_leakage(tiny_data):
    directory = tiny_data.root / "fgvc-aircraft-2013b" / "data"
    for path in directory.glob("*val.txt"):
        path.write_text(path.read_text().replace("0000010", "0000000"))
    with pytest.raises(ValueError, match="Split leakage"):
        read_splits(directory)


def test_official_id_list_mismatch(tiny_data):
    directory = tiny_data.root / "fgvc-aircraft-2013b" / "data"
    (directory / "images_val.txt").write_text("1234567\n")
    with pytest.raises(ValueError, match="Official image list"):
        read_splits(directory)


def test_complete_edition(full_metadata):
    validate_edition(*full_metadata)


def test_missing_official_split(full_metadata):
    splits, taxonomy, catalogs = full_metadata
    del splits["test"]
    with pytest.raises(ValueError, match="three official splits"):
        validate_edition(splits, taxonomy, catalogs)


def test_duplicate_manifest_ids(full_metadata):
    splits, taxonomy, catalogs = full_metadata
    splits["test"][1] = splits["test"][0]
    with pytest.raises(ValueError, match="Duplicate image IDs"):
        validate_edition(splits, taxonomy, catalogs)


def test_reject_other_edition(tiny_data):
    with pytest.raises(ValueError, match="Unexpected dataset edition"):
        AircraftData.load(tiny_data.root)


def test_reject_subset(full_metadata):
    splits, taxonomy, catalogs = full_metadata
    splits["test"].pop()
    with pytest.raises(ValueError, match="test image count"):
        validate_edition(splits, taxonomy, catalogs)


def test_reject_unsupported_path(full_metadata):
    splits, taxonomy, catalogs = full_metadata
    splits["test"][0] = replace(splits["test"][0], family="F001")
    with pytest.raises(ValueError, match="Hierarchy mismatch"):
        validate_edition(splits, taxonomy, catalogs)


def test_reject_per_class_count_change(full_metadata):
    splits, taxonomy, catalogs = full_metadata
    splits["test"][0] = replace(splits["test"][0], variant="V070")  # same family/manufacturer
    with pytest.raises(ValueError, match="100 images per variant"):
        validate_edition(splits, taxonomy, catalogs)


def test_reject_missing_variant_support(full_metadata):
    splits, taxonomy, catalogs = full_metadata
    splits["test"] = [replace(r, variant="V070") if r.variant == "V000" else r for r in splits["test"]]
    with pytest.raises(ValueError, match="Missing variant support"):
        validate_edition(splits, taxonomy, catalogs)


def test_reject_catalog_missing_train_class(full_metadata):
    splits, taxonomy, catalogs = full_metadata
    catalogs = (tuple(["unknown"] + list(catalogs[0][1:])), *catalogs[1:])
    with pytest.raises(ValueError, match="Catalog labels"):
        validate_edition(splits, taxonomy, catalogs)


def test_torchvision_loader_global_ids(tiny_data):
    dataset = AircraftDataset(tiny_data, "train", Preprocessing())
    for i, row in enumerate(tiny_data.splits["train"]):
        sample = dataset[i]
        assert sample["image_id"] == row.image_id
        assert sample["variant_id"] == tiny_data.taxonomy.variants.index(row.variant)
        assert sample["image"].shape == (3, 224, 224)


@pytest.mark.parametrize("split", ["test", "val", "trainval"])
def test_forbid_eval_augmentation(tiny_data, split):
    with pytest.raises(ValueError):
        AircraftDataset(tiny_data, split, Preprocessing(), training=True)


def test_worker_order_reproducibility(tiny_data):
    dataset = AircraftDataset(tiny_data, "train", Preprocessing(), training=True)
    dataset.set_epoch(3)
    expected = {dataset[i]["image_id"]: dataset[i]["image"] for i in range(len(dataset))}
    for batch in DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True):
        assert torch.equal(batch["image"][0], expected[batch["image_id"][0]])


def test_verify_every_image(tiny_data):
    summary, hashes = inspect_images(tiny_data)
    assert summary["decoded_images"] == 9
    assert len(hashes) == 9 and all(len(h["sha256"]) == 64 for h in hashes)


def test_reject_unannotated_images(tiny_data):
    Image.new("RGB", (30, 40)).save(tiny_data.image_path("9999999"))
    with pytest.raises(ValueError, match="unexpected 1"):
        inspect_images(tiny_data)


def test_archive_integrity(tiny_data):
    assert tiny_data.archive_integrity()["available"] is False
    (tiny_data.root / "fgvc-aircraft-2013b.tar.gz").write_bytes(b"corrupt archive")
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        tiny_data.archive_integrity()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "short"])
def test_invalid_image(tiny_data, failure):
    path = tiny_data.image_path("0000000")
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_bytes(b"not a JPEG")
    else:
        Image.new("RGB", (30, 20)).save(path)
    with pytest.raises((ValueError, OSError)):
        inspect_images(tiny_data)
