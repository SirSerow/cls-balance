"""Read official FGVC annotations, join by image ID, and expose global targets."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import FGVCAircraft

from .artifacts import file_hash
from .taxonomy import Labels, Taxonomy
from .transforms import Preprocessing

SPLITS = ("train", "val", "test")
LEVELS = ("variant", "family", "manufacturer")
CATALOGS = ("variants.txt", "families.txt", "manufacturers.txt")
EXPECTED_CLASSES = (100, 70, 30)
EXPECTED_SPLITS = {"train": 3334, "val": 3333, "test": 3333}
ARCHIVE_NAME = "fgvc-aircraft-2013b.tar.gz"
ARCHIVE_MD5 = "32eca553f897e747706144f3bb444df0"


def read_annotations(path: Path) -> dict[str, str]:
    """Read seven-digit IDs and complete label names, rejecting duplicates."""
    result: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9]{7}", parts[0]):
            raise ValueError(f"Invalid annotation at {path}:{line_no}")
        image_id, label = parts
        if image_id in result:
            raise ValueError(f"Duplicate image ID {image_id} in {path}")
        result[image_id] = label
    if not result:
        raise ValueError(f"Empty annotation file: {path}")
    return result


def read_catalog(path: Path) -> tuple[str, ...]:
    names = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError(f"Empty or duplicate class names in {path}")
    return names


def read_splits(data_dir: Path) -> dict[str, list[Labels]]:
    """Join the three annotation files by ID, ignoring their row ordering."""
    result = {}
    seen: set[str] = set()
    for split in SPLITS:
        levels = [read_annotations(data_dir / f"images_{level}_{split}.txt") for level in LEVELS]
        ids = set(levels[0])
        if any(set(level) != ids for level in levels[1:]):
            raise ValueError(f"Missing or mismatched label IDs in {split}")
        official_ids = data_dir / f"images_{split}.txt"
        if official_ids.exists():
            lines = official_ids.read_text(encoding="utf-8").splitlines()
            if len(lines) != len(set(lines)) or set(lines) != ids:
                raise ValueError(f"Official image list differs from annotations in {split}")
        if seen & ids:
            raise ValueError(f"Split leakage: duplicate IDs in {split}: {sorted(seen & ids)[:5]}")
        seen.update(ids)
        result[split] = [Labels(image_id, *(level[image_id] for level in levels)) for image_id in sorted(ids)]
    return result


def validate_edition(splits: dict[str, list[Labels]], taxonomy: Taxonomy,
                     catalogs: tuple[tuple[str, ...], ...]) -> None:
    """Require the complete Torchvision edition, including every official split."""
    if set(splits) != set(SPLITS):
        raise ValueError("All three official splits are required")
    seen: set[str] = set()
    for split, rows in splits.items():
        ids = [r.image_id for r in rows]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Duplicate image IDs or split leakage in {split}")
        seen.update(ids)
    if tuple(map(len, catalogs)) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected dataset edition: class counts {tuple(map(len, catalogs))}; expected {EXPECTED_CLASSES}")
    for names, train_names in zip(catalogs, (taxonomy.variants, taxonomy.families, taxonomy.manufacturers), strict=True):
        if set(names) != set(train_names):
            raise ValueError("Catalog labels must all occur in training; no extra/unknown labels allowed")
    for split, rows in splits.items():
        if len(rows) != EXPECTED_SPLITS[split]:
            raise ValueError(f"Unexpected {split} image count: {len(rows)}; expected {EXPECTED_SPLITS[split]}")
        for row in rows:
            taxonomy.ids(row)
        if {r.variant for r in rows} != set(taxonomy.variants):
            raise ValueError(f"Missing variant support in {split}")
    counts = Counter(r.variant for rows in splits.values() for r in rows)
    if set(counts.values()) != {100}:
        raise ValueError("Expected exactly 100 images per variant across the official splits")


@dataclass(frozen=True)
class AircraftData:
    root: Path
    splits: dict[str, list[Labels]]
    taxonomy: Taxonomy
    source_hashes: dict[str, str]

    def archive_integrity(self) -> dict:
        """Check the authors' published archive checksum when the archive is local."""
        archive = self.root / ARCHIVE_NAME
        if not archive.exists():
            return {"available": False, "note": "Existing extracted dataset; no archive supplied"}
        actual = file_hash(archive, "md5")
        if actual != ARCHIVE_MD5:
            raise ValueError(f"Official archive checksum mismatch: {actual}; expected {ARCHIVE_MD5}")
        return {"available": True, "filename": ARCHIVE_NAME, "bytes": archive.stat().st_size,
                "md5": actual, "expected_md5": ARCHIVE_MD5,
                "checksum_source": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html"}

    @classmethod
    def load(cls, root: Path, *, download: bool = False) -> AircraftData:
        """Use Torchvision acquisition/loading, but construct hierarchy by ID."""
        FGVCAircraft(str(root), split="train", annotation_level="variant", download=download)
        data_dir = root / "fgvc-aircraft-2013b" / "data"
        splits = read_splits(data_dir)
        catalogs = tuple(read_catalog(data_dir / name) for name in CATALOGS)
        taxonomy = Taxonomy.from_training(splits["train"])
        validate_edition(splits, taxonomy, catalogs)
        for split in SPLITS:
            for level, catalog in zip(LEVELS, catalogs, strict=True):
                loader = FGVCAircraft(str(root), split=split, annotation_level=level)
                if len(loader) != len(splits[split]) or tuple(loader.classes) != catalog:
                    raise ValueError(f"Torchvision metadata mismatch for {split}/{level}")
        hashes = {p.name: file_hash(p) for p in sorted(data_dir.glob("*.txt"))}
        return cls(root, splits, taxonomy, hashes)

    def image_path(self, image_id: str) -> Path:
        return self.root / "fgvc-aircraft-2013b" / "data" / "images" / f"{image_id}.jpg"

    def manifest(self, split: str) -> list[dict]:
        rows = []
        for row in self.splits[split]:
            v, f, m = self.taxonomy.ids(row)
            rows.append({"image_id": row.image_id,
                         "relative_path": self.image_path(row.image_id).relative_to(self.root).as_posix(),
                         "split": split, "variant_id": v, "variant_name": row.variant,
                         "family_id": f, "family_name": row.family,
                         "manufacturer_id": m, "manufacturer_name": row.manufacturer})
        return rows


class AircraftDataset(Dataset):
    """Torchvision-backed images with manifest order and canonical global targets.

    Training is permitted only on train. Call set_epoch before creating the
    DataLoader iterator; use persistent_workers=False so workers see the epoch.
    Validation/test tensors are deterministic and ignore seed and epoch.
    """

    def __init__(self, data: AircraftData, split: str, preprocessing: Preprocessing,
                 *, training: bool = False, seed: int = 42) -> None:
        if split not in SPLITS or (training and split != "train"):
            raise ValueError("Training augmentation is allowed only for the official train split")
        self.rows = data.splits[split]
        self.taxonomy = data.taxonomy
        self.preprocessing = preprocessing
        self.training, self.seed, self.epoch = training, seed, 0
        self.loader = FGVCAircraft(str(data.root), split=split, annotation_level="variant")
        order = read_annotations(data.root / "fgvc-aircraft-2013b" / "data" / f"images_variant_{split}.txt")
        self.indices = {image_id: i for i, image_id in enumerate(order)}

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.rows[index]
        image, target = self.loader[self.indices[row.image_id]]
        if self.loader.classes[target] != row.variant:
            raise ValueError(f"Torchvision target mismatch for {row.image_id}")
        v, f, m = self.taxonomy.ids(row)
        return {"image": self.preprocessing.tensor(image, training=self.training, seed=self.seed,
                                                    epoch=self.epoch, image_id=row.image_id),
                "image_id": row.image_id, "variant_id": v, "family_id": f, "manufacturer_id": m}


def inspect_images(data: AircraftData) -> tuple[dict, list[dict]]:
    """Fully decode and checksum every image, rejecting missing/corrupt files."""
    expected_names = {f"{r.image_id}.jpg" for rows in data.splits.values() for r in rows}
    image_dir = data.root / "fgvc-aircraft-2013b" / "data" / "images"
    actual_names = {p.name for p in image_dir.glob("*.jpg")}
    if actual_names != expected_names:
        raise ValueError(f"Image inventory mismatch: missing {len(expected_names - actual_names)}, "
                         f"unexpected {len(actual_names - expected_names)}")
    sizes = []
    checksums = []
    for split in SPLITS:
        for row in data.splits[split]:
            path = data.image_path(row.image_id)
            if not path.is_file():
                raise ValueError(f"Missing image: {path}")
            with Image.open(path) as image:
                image.load()
                if image.height <= 20:
                    raise ValueError(f"Image too short for banner removal: {row.image_id}")
                sizes.append(image.size)
            checksums.append({"image_id": row.image_id, "sha256": file_hash(path)})
    return {"decoded_images": len(sizes), "min_width": min(w for w, h in sizes),
            "max_width": max(w for w, h in sizes), "min_height": min(h for w, h in sizes),
            "max_height": max(h for w, h in sizes)}, checksums
