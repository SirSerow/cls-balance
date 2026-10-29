"""Recheck all original FGVC bytes and bind them to a Phase 7 training suite."""

import argparse
from pathlib import Path

from cls_balance.artifacts import file_hash, object_hash, write_json
from cls_balance.data import AircraftData
from cls_balance.incremental import audit_training
from cls_balance.report_flat import read_json
from cls_balance.train import validate_preparation
from cls_balance.transforms import Preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.training / "config.resolved.json")
    data, preprocessing = AircraftData.load(Path(config["data_root"])), Preprocessing()
    prepared = validate_preparation(data, preprocessing, Path(config["prepared_run"]))
    if object_hash(prepared) != object_hash(read_json(args.training / "preparation.json")):
        raise ValueError("Phase 7 preparation binding mismatch")
    audit = audit_training(args.training, data)
    write_json(args.output, {"status": "verified", "training_run_id": args.training.name,
        "training_summary_sha256": file_hash(args.training / "summary.json"),
        "prepared_run_id": prepared["run_id"], "images_checked": sum(map(len, data.splits.values())),
        "split_counts": {s: len(rows) for s, rows in data.splits.items()},
        "manifest_records_sha256": {s: object_hash(data.manifest(s)) for s in data.splits},
        "taxonomy_sha256": object_hash(data.taxonomy.to_dict()), "preprocessing_sha256": preprocessing.sha256,
        "semantic_training_audit": audit, "scope": "all image bytes rechecked; no pixels or supervised IDs exported"})
    print(args.output)


if __name__ == "__main__":
    main()
