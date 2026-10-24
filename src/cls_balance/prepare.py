"""Separately invocable stage-one preparation; creates immutable local runs."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from .artifacts import environment, file_hash, object_hash, seed_everything, write_json
from .data import AircraftData, AircraftDataset, SPLITS, inspect_images
from .transforms import Preprocessing


def read_config(path: Path) -> tuple[dict, Preprocessing]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"dataset", "output_root", "seed", "preprocessing"}:
        raise ValueError("Use the operational configs/data.yaml schema (unknown/missing keys)")
    dataset = config["dataset"]
    if set(dataset) != {"root", "edition", "download"} or dataset["edition"] != "torchvision_100_variants":
        raise ValueError("Only the complete torchvision_100_variants edition is supported")
    if type(dataset["download"]) is not bool or type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("download must be boolean and seed an integer in [0, 2**32)")
    config["dataset"]["root"] = str((path.parent / dataset["root"]).resolve())
    config["output_root"] = str((path.parent / config["output_root"]).resolve())
    params = dict(config["preprocessing"])
    if "pad_rgb" in params:
        params["pad_rgb"] = tuple(params["pad_rgb"])
    return config, Preprocessing(**params)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def spot_checks(data: AircraftData, preprocessing: Preprocessing, output: Path) -> list[str]:
    """Keep six training-only original/processed image pairs local for visual QA."""
    rows = data.splits["train"]
    selected = [rows[i * (len(rows) - 1) // 5] for i in range(6)]
    canvas = Image.new("RGB", (900, 6 * 260), "#dddddd")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(selected):
        with Image.open(data.image_path(row.image_id)) as image:
            original = image.convert("RGB").crop((0, 0, image.width, image.height - 20))
            original.thumbnail((620, 230))
            canvas.paste(original, (5, i * 260 + 25))
            canvas.paste(preprocessing.image(image), (660, i * 260 + 25))
        draw.text((5, i * 260 + 5), f"{row.image_id}: banner removed original | primary 224x224", fill="black")
    canvas.save(output, quality=95)
    return [r.image_id for r in selected]


def prepare(config_path: Path, *, download_override: bool | None = None) -> Path:
    config, preprocessing = read_config(config_path.resolve())
    if download_override is not None:
        config["dataset"]["download"] = download_override
    seed_everything(config["seed"])
    run_id = datetime.now(timezone.utc).strftime("data-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    output = Path(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.resolved.json", config)
    repo = Path(__file__).resolve().parents[2]
    write_json(output / "environment.json", environment(repo))
    print(f"Run {run_id}: acquiring official dataset", flush=True)
    data = AircraftData.load(Path(config["dataset"]["root"]), download=config["dataset"]["download"])
    archive_integrity = data.archive_integrity()
    print("Valid hierarchy and splits; decoding/checksumming every image", flush=True)
    image_summary, checksums = inspect_images(data)
    write_csv(output / "image_checksums.csv", checksums)
    taxonomy_dict = data.taxonomy.to_dict()
    write_json(output / "taxonomy.json", taxonomy_dict)
    write_json(output / "preprocessing.json", preprocessing.to_dict())
    write_json(output / "source_annotations.json", data.source_hashes)
    manifests = {}
    for split in SPLITS:
        path = output / f"manifest_{split}.csv"
        write_csv(path, data.manifest(split))
        manifests[split] = file_hash(path)
        dataset = AircraftDataset(data, split, preprocessing)
        for i in (0, len(dataset) - 1):
            sample = dataset[i]
            if tuple(sample["image"].shape) != (3, 224, 224):
                raise ValueError("Invalid transformed tensor dimensions")
    checks = spot_checks(data, preprocessing, output / "spot_checks.jpg")
    summary = {
        "schema_version": 1, "run_id": run_id, "status": "validated",
        "scope": "stage-one data validation; no training or classification/latency measurements",
        "edition": "torchvision_100_variants", "seed": config["seed"],
        "counts": {"splits": {s: len(r) for s, r in data.splits.items()},
                   "variants": len(data.taxonomy.variants), "families": len(data.taxonomy.families),
                   "manufacturers": len(data.taxonomy.manufacturers)},
        "per_variant_support": {s: dict(sorted(Counter(r.variant for r in rows).items())) for s, rows in data.splits.items()},
        "branches": data.taxonomy.branch_summary(data.splits), "images": image_summary,
        "archive_integrity": archive_integrity,
        "manifest_sha256": manifests, "dataset_manifest_sha256": object_hash(manifests),
        "taxonomy_sha256": file_hash(output / "taxonomy.json"),
        "preprocessing_sha256": preprocessing.sha256,
        "image_checksums_sha256": file_hash(output / "image_checksums.csv"),
        "source_annotations_sha256": object_hash(data.source_hashes),
        "config_sha256": object_hash(config), "environment_sha256": file_hash(output / "environment.json"),
        "visual_check_train_ids": checks,
        "visual_review": "pending human/agent inspection of local spot_checks.jpg",
        "bypass_note": "True-route taxonomy potential only; predicted-route execution rates require trained models.",
        "protocol_deviations": [],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"run_id": run_id, "output": str(output), "counts": summary["counts"]}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--no-download", action="store_true", help="Require an existing official dataset")
    args = parser.parse_args()
    prepare(args.config, download_override=False if args.no_download else None)


if __name__ == "__main__":
    main()
