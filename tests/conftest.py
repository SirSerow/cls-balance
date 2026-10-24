"""Generated metadata and tiny images only; never download the dataset in tests."""

from pathlib import Path

import pytest
from PIL import Image

from cls_balance.data import AircraftData, CATALOGS, LEVELS, SPLITS, read_splits
from cls_balance.taxonomy import Taxonomy


@pytest.fixture
def tiny_data(tmp_path: Path) -> AircraftData:
    directory = tmp_path / "fgvc-aircraft-2013b" / "data"
    (directory / "images").mkdir(parents=True)
    # Label names deliberately have no textual hierarchy relationship.
    labels = [("Zulu jet", "Alpha", "Second maker"), ("Alpha jet", "Beta", "First maker"),
              ("Middle jet", "Alpha", "Second maker")]
    for j, filename in enumerate(CATALOGS):
        (directory / filename).write_text("\n".join(sorted({r[j] for r in labels}, reverse=True)) + "\n")
    for split_no, split in enumerate(SPLITS):
        ids = [f"{split_no * 10 + i:07d}" for i in range(3)]
        (directory / f"images_{split}.txt").write_text("\n".join(ids) + "\n")
        for level_no, level in enumerate(LEVELS):
            order = [(i + level_no) % 3 for i in range(3)]
            (directory / f"images_{level}_{split}.txt").write_text(
                "".join(f"{ids[i]} {labels[i][level_no]}\n" for i in order))
        for i, image_id in enumerate(ids):
            Image.new("RGB", (80, 60), (i * 80, 30, 150)).save(directory / "images" / f"{image_id}.jpg")
    splits = read_splits(directory)
    return AircraftData(tmp_path, splits, Taxonomy.from_training(splits["train"]), {})


@pytest.fixture
def full_metadata():
    """Protocol-sized synthetic metadata for exact edition guards, no image data."""
    from cls_balance.taxonomy import Labels

    splits = {s: [] for s in SPLITS}
    for v in range(100):
        f = v % 70
        labels = (f"V{v:03d}", f"F{f:03d}", f"M{f % 30:03d}")
        extra_split = 0 if v < 34 else 1 if v < 67 else 2
        image = 0
        for split_no, split in enumerate(SPLITS):
            for _ in range(33 + int(split_no == extra_split)):
                splits[split].append(Labels(f"{v * 100 + image:07d}", *labels))
                image += 1
    taxonomy = Taxonomy.from_training(splits["train"])
    return splits, taxonomy, (taxonomy.variants, taxonomy.families, taxonomy.manufacturers)
