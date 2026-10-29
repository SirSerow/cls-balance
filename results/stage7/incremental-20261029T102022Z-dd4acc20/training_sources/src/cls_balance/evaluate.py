"""Complete-split evaluation API and separately invocable saved-run audit.

No training or latency benchmarking occurs here. Normal inference receives only
an image tensor. Labels enter the model only through the explicit oracle and
conditional-family diagnostic APIs. Saved runs are immutable and self-auditing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import torch

from .artifacts import canonical_json, environment, file_hash, object_hash, seed_everything, write_json
from .data import AircraftData, AircraftDataset, SPLITS, validate_edition
from .infer import Ancestors, HierarchicalAdapter, InferenceAdapter
from .metrics import PredictionRecord, paired_scores, score
from .predictions import read_predictions, write_predictions
from .prepare import write_csv
from .taxonomy import Taxonomy
from .transforms import Preprocessing


def model_hash(model: InferenceAdapter) -> str:
    """Fingerprint tensor bytes, architecture, module types and logit-ID maps."""
    digest = hashlib.sha256()
    digest.update(canonical_json({"architecture": model.architecture, "taxonomy": model.taxonomy.to_dict(),
        "modules": {name: f"{type(m).__module__}.{type(m).__qualname__}" for name, m in model.named_modules()},
        "logit_ids": {name: m.global_ids for name, m in model.named_modules() if hasattr(m, "global_ids")}}))
    if getattr(model, "active_variant_ids", None) is not None:
        digest.update(canonical_json({"active_variant_ids": model.active_variant_ids}))
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(canonical_json({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelProvenance:
    """Declared training/selection IDs are checked against the official splits.

    Tensor fingerprints are verified before evaluation; checkpoint loading and
    training code will supply this contract in later phases. Provenance declares
    how an artifact was trained; it cannot independently prove its history.
    """

    model_id: str
    model_sha256: str
    taxonomy_sha256: str
    preprocessing_sha256: str
    seed: int
    training_image_ids: tuple[str, ...]
    selection_image_ids: tuple[str, ...]
    precision: str = "fp32"
    residency: str = "all-resident"


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[PredictionRecord, ...]
    metrics: dict
    metadata: dict
    taxonomy: Taxonomy
    preprocessing: Preprocessing

    def save(self, output_root: Path) -> Path:
        """Write immutable artifacts; summary.json marks successful completion."""
        if score(self.records, self.taxonomy) != self.metrics:
            raise ValueError("Evaluation metrics disagree with per-image predictions")
        run_id = datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        output = output_root / run_id
        output.mkdir(parents=True, exist_ok=False)
        write_predictions(output / "predictions.csv", self.records, self.taxonomy)
        write_json(output / "metrics.json", self.metrics)
        write_json(output / "routing_diagnostics.json", self.metrics["routing_diagnostics"])
        write_csv(output / "classwise_metrics.csv", self.metrics["classwise"])
        write_json(output / "taxonomy.json", self.taxonomy.to_dict())
        write_json(output / "preprocessing.json", self.preprocessing.to_dict())
        write_json(output / "metadata.json", self.metadata)
        write_json(output / "environment.json", self.metadata["environment"])
        hashes = {p.name: file_hash(p) for p in sorted(output.iterdir())}
        write_json(output / "summary.json", {"schema_version": 1, "run_id": run_id, "status": "complete",
                   "artifacts_sha256": hashes, "metadata_sha256": object_hash(self.metadata),
                   "scope": "classification evaluation; no latency/throughput measurements"})
        return output


class Evaluator:
    """Evaluate every image in an official split with deterministic preprocessing."""

    def __init__(self, data: AircraftData, preprocessing: Preprocessing,
                 provenance: ModelProvenance, *, device: str = "cpu") -> None:
        self.data, self.preprocessing, self.provenance = data, preprocessing, provenance
        self.device = torch.device(device)
        if set(data.splits) != set(SPLITS):
            raise ValueError("Evaluation requires all official splits to check leakage")
        seen: set[str] = set()
        for rows in data.splits.values():
            ids = [r.image_id for r in rows]
            if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError("Empty split, duplicate IDs or split leakage")
            seen.update(ids)
            for row in rows:
                data.taxonomy.ids(row)
        validate_edition(data.splits, data.taxonomy,
                         (data.taxonomy.variants, data.taxonomy.families, data.taxonomy.manufacturers))
        self.manifests = {s: object_hash(data.manifest(s)) for s in SPLITS}
        for declared, split in ((provenance.training_image_ids, "train"), (provenance.selection_image_ids, "val")):
            if (not declared or len(declared) != len(set(declared))
                    or not set(declared) <= {r.image_id for r in data.splits[split]}):
                raise ValueError(f"Model provenance contains invalid {split} IDs or test leakage")
        if (not provenance.model_id or type(provenance.seed) is not int or not 0 <= provenance.seed < 2**32
                or provenance.precision != "fp32" or provenance.residency != "all-resident"):
            raise ValueError("Require a model ID, valid seed and explicit FP32/all-resident policy")
        if provenance.taxonomy_sha256 != object_hash(data.taxonomy.to_dict()):
            raise ValueError("Model taxonomy hash mismatch")
        if provenance.preprocessing_sha256 != preprocessing.sha256:
            raise ValueError("Model preprocessing hash mismatch")

    def evaluate(self, model: InferenceAdapter, *, split: str = "val", routing: str = "predicted",
                 conditional_family: bool = False) -> EvaluationResult:
        """No subsets; optional family diagnostics run separately from prediction."""
        if split not in SPLITS or routing not in ("predicted", "oracle"):
            raise ValueError("Invalid official split or routing mode")
        if {s: object_hash(self.data.manifest(s)) for s in SPLITS} != self.manifests:
            raise ValueError("Dataset manifests changed after validation")
        if (routing == "oracle" and model.depth == 1) or (conditional_family and model.depth != 3):
            raise ValueError("Requested diagnostic is unavailable for this architecture")
        if object_hash(model.taxonomy.to_dict()) != self.provenance.taxonomy_sha256:
            raise ValueError("Adapter taxonomy hash mismatch")
        if model_hash(model) != self.provenance.model_sha256:
            raise ValueError("Model/checkpoint fingerprint mismatch")
        if any(t.is_floating_point() and t.dtype != torch.float32
               for t in (*model.parameters(), *model.buffers())):
            raise ValueError("Primary evaluation requires FP32 weights and buffers")
        if {r.variant for r in self.data.splits[split]} != set(self.data.taxonomy.variants):
            raise ValueError("Missing variant support in evaluation split")
        seed_everything(self.provenance.seed)
        model.to(self.device).eval()
        dataset = AircraftDataset(self.data, split, self.preprocessing)
        records = []
        for sample in dataset:
            image = sample["image"].to(self.device)
            truth = Ancestors(sample["manufacturer_id"], sample["family_id"])
            # Predicted inference never receives any label argument.
            p = (model.predict(image) if routing == "predicted" else
                 model.predict(image, routing="oracle", truth=truth))
            conditional = None
            if conditional_family:
                if not isinstance(model, HierarchicalAdapter):
                    raise ValueError("Conditional diagnostic requires a hierarchical adapter")
                conditional = model.diagnose_family(image, truth.manufacturer_id).global_id
            records.append(PredictionRecord(sample["image_id"], split, model.architecture, routing,
                           sample["variant_id"], sample["family_id"], sample["manufacturer_id"], p, conditional))
        if model_hash(model) != self.provenance.model_sha256:
            raise ValueError("Model state changed during evaluation")
        manifests = self.manifests
        env = environment(Path(__file__).resolve().parents[2])
        env.update(device=str(self.device), runtime_policy="all-resident model; FP32 batch-1 evaluation")
        config = {"split": split, "routing": routing, "conditional_family": conditional_family,
                  "seed": self.provenance.seed, "device": str(self.device), "precision": "fp32",
                  "residency": "all-resident", "architecture": model.architecture}
        metadata = {"schema_version": 1, "config": config, "config_sha256": object_hash(config),
                    "model_id": self.provenance.model_id, "model_sha256": self.provenance.model_sha256,
                    "taxonomy_sha256": self.provenance.taxonomy_sha256,
                    "preprocessing_sha256": self.preprocessing.sha256,
                    "manifest_records_sha256": manifests, "dataset_manifest_records_sha256": object_hash(manifests),
                    "manifest_hash_policy": "canonical JSON manifest rows; not CSV file-byte hashes from preparation",
                    "source_annotations_sha256": object_hash(self.data.source_hashes),
                    "evaluation_ids_sha256": object_hash([r.image_id for r in records]),
                    "training_ids_sha256": object_hash(sorted(self.provenance.training_image_ids)),
                    "selection_ids_sha256": object_hash(sorted(self.provenance.selection_image_ids)),
                    "environment": env}
        return EvaluationResult(tuple(records), score(records, self.data.taxonomy), metadata,
                                self.data.taxonomy, self.preprocessing)


def compare_evaluations(predicted: EvaluationResult, oracle: EvaluationResult) -> dict:
    """Require matching artifact/data/preprocessing contracts before pairing."""
    for key in ("model_id", "model_sha256", "taxonomy_sha256", "preprocessing_sha256",
                "dataset_manifest_records_sha256", "training_ids_sha256", "selection_ids_sha256"):
        if predicted.metadata[key] != oracle.metadata[key]:
            raise ValueError(f"Cannot compare evaluations with mismatched {key}")
    for key in ("split", "architecture", "seed", "device", "precision", "residency"):
        if predicted.metadata["config"][key] != oracle.metadata["config"][key]:
            raise ValueError(f"Cannot compare evaluations with mismatched {key}")
    for key in ("git_sha", "torch", "cuda_runtime", "cudnn", "platform", "gpus", "packages"):
        if predicted.metadata["environment"].get(key) != oracle.metadata["environment"].get(key):
            raise ValueError(f"Cannot compare evaluations with mismatched environment {key}")
    return paired_scores(predicted.records, oracle.records, predicted.taxonomy)


def audit_run(path: Path) -> dict:
    """Verify saved bytes, schema/contracts and recomputed per-image metrics."""
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    required = {"predictions.csv", "metrics.json", "routing_diagnostics.json", "classwise_metrics.csv",
                "taxonomy.json", "preprocessing.json", "metadata.json", "environment.json"}
    if (summary.get("schema_version") != 1 or summary.get("status") != "complete"
            or set(summary.get("artifacts_sha256", {})) != required):
        raise ValueError("Incomplete evaluation run or unexpected artifact inventory")
    for name, expected in summary["artifacts_sha256"].items():
        if file_hash(path / name) != expected:
            raise ValueError(f"Evaluation artifact checksum mismatch: {name}")
    taxonomy = Taxonomy.from_dict(json.loads((path / "taxonomy.json").read_text(encoding="utf-8")))
    preprocessing = json.loads((path / "preprocessing.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    records = read_predictions(path / "predictions.csv", taxonomy)
    metrics = score(records, taxonomy)
    if (object_hash(metadata) != summary["metadata_sha256"]
            or object_hash(taxonomy.to_dict()) != metadata["taxonomy_sha256"]
            or object_hash(preprocessing) != metadata["preprocessing_sha256"]
            or object_hash(metadata["config"]) != metadata["config_sha256"]
            or object_hash([r.image_id for r in records]) != metadata["evaluation_ids_sha256"]
            or object_hash(metadata["manifest_records_sha256"]) != metadata["dataset_manifest_records_sha256"]):
        raise ValueError("Evaluation metadata contract mismatch")
    saved = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if object_hash(metrics) != object_hash(saved):
        raise ValueError("Saved metrics disagree with per-image predictions")
    if any(metadata["config"][key] != metrics[key] for key in ("split", "architecture", "routing")):
        raise ValueError("Evaluation configuration disagrees with predictions")
    diagnostics = json.loads((path / "routing_diagnostics.json").read_text(encoding="utf-8"))
    saved_environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    with (path / "classwise_metrics.csv").open(encoding="utf-8", newline="") as stream:
        classwise = list(csv.DictReader(stream))
    expected_classwise = [{key: str(value) for key, value in row.items()} for row in metrics["classwise"]]
    if (object_hash(diagnostics) != object_hash(metrics["routing_diagnostics"])
            or object_hash(saved_environment) != object_hash(metadata["environment"])
            or classwise != expected_classwise):
        raise ValueError("Redundant evaluation artifacts disagree")
    return {"run_id": summary["run_id"], "status": "verified", "images": len(records),
            "routing": metrics["routing"], "metrics_sha256": file_hash(path / "metrics.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Verify and rescore an existing evaluation run")
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
