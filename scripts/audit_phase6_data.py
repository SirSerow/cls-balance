"""Recheck official dataset bytes after benchmarking; publish hashes/counts only."""

import argparse
from pathlib import Path

from cls_balance.artifacts import file_hash, object_hash, write_json
from cls_balance.benchmark import read_config
from cls_balance.data import AircraftData, SPLITS
from cls_balance.report_flat import read_json
from cls_balance.train import validate_preparation
from cls_balance.transforms import Preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    data = AircraftData.load(Path(config.data_root))
    prepared = Path(config.prepared_run)
    summary = validate_preparation(data, Preprocessing(), prepared)
    for run in map(Path, config.runs):
        training_data = read_json(run / "dataset_manifest_hashes.json")
        if training_data["prepared_summary_sha256"] != object_hash(summary):
            raise ValueError("Training run used a different dataset preparation")
    write_json(args.output, {"status": "verified", "prepared_run_id": summary["run_id"],
        "prepared_summary_sha256": file_hash(prepared / "summary.json"),
        "image_checksum_inventory_sha256": file_hash(prepared / "image_checksums.csv"),
        "split_counts": {s: len(data.splits[s]) for s in SPLITS},
        "images_checked": sum(len(data.splits[s]) for s in SPLITS),
        "training_preparation_binding_verified": True, "training_runs_checked": len(config.runs),
        "manifest_records_sha256": {s: object_hash(data.manifest(s)) for s in SPLITS},
        "taxonomy_sha256": object_hash(data.taxonomy.to_dict()),
        "preprocessing_sha256": Preprocessing().sha256,
        "scope": "post-benchmark recheck of every image against original preparation; no pixels or training IDs"})
    print(args.output)


if __name__ == "__main__":
    main()
