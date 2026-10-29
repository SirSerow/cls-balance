"""Strict, immutable per-image CSV interchange, with JSON path diagnostics."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .infer import Decision, Prediction
from .metrics import PredictionRecord, validate_record
from .taxonomy import Taxonomy

COLUMNS = ("schema_version", "image_id", "split", "architecture", "routing", "true_variant_id",
           "true_family_id", "true_manufacturer_id", "predicted_variant_id", "selected_family_id",
           "selected_manufacturer_id", "decisions_json", "backbone_calls", "conditional_family_id")


def write_predictions(path: Path, records: Sequence[PredictionRecord], taxonomy: Taxonomy) -> None:
    if not records or len({r.image_id for r in records}) != len(records):
        raise ValueError("Predictions must be nonempty and have unique IDs")
    for r in records:
        validate_record(r, taxonomy)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in records:
            p = r.prediction
            writer.writerow(dict(zip(COLUMNS, (1, r.image_id, r.split, r.architecture, r.routing,
                r.true_variant_id, r.true_family_id, r.true_manufacturer_id, p.variant_id,
                p.family_id, p.manufacturer_id, json.dumps([asdict(d) for d in p.decisions],
                sort_keys=True, separators=(",", ":")), p.backbone_calls,
                "" if r.conditional_family_id is None else r.conditional_family_id), strict=True)))


def read_predictions(path: Path, taxonomy: Taxonomy) -> tuple[PredictionRecord, ...]:
    result = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError("Unexpected prediction CSV schema")
        for row in reader:
            try:
                if set(row) != set(COLUMNS) or any(v is None for v in row.values()) or row["schema_version"] != "1":
                    raise ValueError("Invalid prediction CSV row")
                p = Prediction(int(row["predicted_variant_id"]), int(row["selected_manufacturer_id"]),
                               int(row["selected_family_id"]), tuple(Decision(**d) for d in json.loads(row["decisions_json"])),
                               int(row["backbone_calls"]))
                r = PredictionRecord(row["image_id"], row["split"], row["architecture"], row["routing"],
                                     int(row["true_variant_id"]), int(row["true_family_id"]), int(row["true_manufacturer_id"]),
                                     p, int(row["conditional_family_id"]) if row["conditional_family_id"] else None)
                validate_record(r, taxonomy)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction at {path}:{reader.line_num}") from exc
            result.append(r)
    if not result or len({r.image_id for r in result}) != len(result):
        raise ValueError("Predictions must be nonempty and have unique IDs")
    return tuple(result)
