"""Immutable full-test, batch-one benchmarks of selected deployment checkpoints.

The primary boundary starts at opening an image file and ends at a Python global
variant ID. Model-path timing is nested after input transfer synchronization.
There are no inference caches, test-driven selection, or subset CLI switches.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from .artifacts import environment, file_hash, object_hash, seed_everything, write_json
from .checkpoints import load_checkpoint
from .data import AircraftData, SPLITS
from .evaluate import model_hash
from .infer import Ancestors, FlatAdapter, InferenceAdapter, LogitNode, Prediction
from .metrics import PredictionRecord, paired_scores, score
from .predictions import read_predictions, write_predictions
from .prepare import write_csv
from .report_flat import read_json
from .taxonomy import Taxonomy
from .train import synchronize, validate_preparation
from .train_independent import compatibility, unique_id
from .transforms import Preprocessing


@dataclass(frozen=True)
class BenchmarkConfig:
    data_root: str
    prepared_run: str
    output_root: str
    runs: tuple[str, ...]
    device: str = "cuda:0"
    precision: str = "fp32"
    passes: int = 5
    warmup_iterations: int = 32
    cpu_threads: int = 6
    depth: int = 2

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("Precision must be fp32 or separate fp16 deployment ablation")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU/CUDA supported")
        if self.precision == "fp16" and torch.device(self.device).type != "cuda":
            raise ValueError("FP16 deployment requires CUDA")
        for key, minimum in (("passes", 5), ("warmup_iterations", 1), ("cpu_threads", 1)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f"Invalid {key}; minimum {minimum}")
        if self.depth not in (2, 3) or not self.runs or len(set(self.runs)) != len(self.runs):
            raise ValueError("Require unique completed runs and depth 2 or 3")


def read_config(path: Path) -> BenchmarkConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("data_root", "prepared_run", "output_root"):
        raw[key] = str((path.resolve().parent / raw[key]).resolve())
    raw["runs"] = tuple(str((path.resolve().parent / p).resolve()) for p in raw["runs"])
    return BenchmarkConfig(**raw)


def distribution(values: list[float]) -> dict:
    """Milliseconds; NumPy linear quantiles, never reciprocal-derived throughput."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("Timings must be positive finite measurements")
    return {"p50": float(np.percentile(a, 50)), "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def telemetry() -> dict:
    """Snapshot outside timed regions; unavailable fields stay explicit."""
    query = "name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used"
    try:
        value = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv"],
                                        text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        value = f"unavailable: {type(exc).__name__}"
    try:
        power = subprocess.check_output(["powercfg", "/getactivescheme"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        power = "unavailable on this platform"
    return {"nvidia_smi": value, "os_power_scheme": power,
            "clocks_policy": "OS/driver defaults; unlocked; no power/clock changes",
            "ambient_temperature": "not measured", "system_load": "interactive desktop; no exclusive GPU reservation"}


@contextmanager
def awake():
    """Temporary Windows process-thread idle-sleep guard, released on exit."""
    import sys
    request = None
    if sys.platform == "win32":
        import ctypes
        request = ctypes.windll.kernel32.SetThreadExecutionState
        request.argtypes, request.restype = [ctypes.c_uint], ctypes.c_uint
        if not request(0x80000001):
            raise OSError("Cannot acquire idle-sleep guard")
    try:
        yield
    finally:
        if request:
            request(0x80000000)


def load_selected(run: Path, data: AircraftData, preprocessing: Preprocessing,
                  depth: int) -> tuple[InferenceAdapter, dict]:
    """Audit completed training evidence and explicit supervised split provenance."""
    summary = read_json(run / "summary.json")
    if (run / "flat.pt").exists():
        from .report_flat import audit_training_run
        audit_training_run(run)
        model, meta = load_checkpoint(run / "flat.pt", **compatibility(data, preprocessing))
        adapter = FlatAdapter(data.taxonomy, LogitNode(model, tuple(range(len(data.taxonomy.variants)))))
        checkpoints = [run / "flat.pt"]
        metas = [meta]
    else:
        shared = (run / f"bundle-shared{depth}.json").exists()
        if shared:
            from .report_shared import audit_training_run
            from .train_shared import load_shared
            audit_training_run(run)
            adapter = load_shared(run, depth)
        else:
            from .report_independent import audit_training_run
            from .train_independent import load_cascade
            audit_training_run(run)
            adapter = load_cascade(run, depth)
        bundle = read_json(run / f"bundle-{'shared' if shared else 'ind'}{depth}.json")
        if bundle["compatibility"] != compatibility(data, preprocessing):
            raise ValueError("Benchmark checkpoint/data/preprocessing mismatch")
        checkpoints = ([run / bundle["encoder_path"]] if shared else [])
        checkpoints += [run / n["path"] / "node.pt" if not shared else run / n["path"]
                        for n in bundle["nodes"].values()]
        metas = [read_json(p.with_suffix(".json"))["metadata"] for p in checkpoints]
        if shared:
            # Shared exports bind an encoder source instead of duplicating its
            # supervised ID lists. Audit that original source explicitly.
            metas = metas[1:]
            if bundle["backbone"] == "FT-FROZEN":
                from .report_flat import audit_training_run as audit_flat
                resolved = read_json(run / "config.resolved.json")
                source = Path(resolved["flat_runs"][str(summary["seed"])])
                source_summary = audit_flat(source)
                source_info = read_json(run / "backbone_source.json")
                original = read_json(source / "encoder.json")
                if (source_summary["seed"] != summary["seed"] or source.name != source_info["source_run_id"]
                        or file_hash(source / "encoder.pt") != source_info["source_encoder_sha256"]
                        or file_hash(source / "flat.pt") != source_info["parent_flat_sha256"]
                        or original["state_sha256"] != read_json(checkpoints[0].with_suffix(".json"))["state_sha256"]):
                    raise ValueError("Shared encoder source provenance mismatch")
                metas.append(original["metadata"])
    for meta in metas:
        if meta["seed"] != summary["seed"]:
            raise ValueError("Checkpoint seed mismatch")
        for key, split in (("training_image_ids", "train"), ("selection_image_ids", "val")):
            ids = meta[key]
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= {r.image_id for r in data.splits[split]}:
                raise ValueError("Invalid checkpoint supervised IDs / test leakage")
    training_resources = (read_json(run / "resources.json") if (run / "resources.json").exists()
                          else summary["bundles"][f"IND-{depth}"]["resources"])
    return adapter, {"training_run_id": run.name, "seed": summary["seed"],
        "training_summary_sha256": file_hash(run / "summary.json"),
        "architecture": adapter.architecture, "model_sha256_fp32": model_hash(adapter),
        "checkpoint_files": [{"path": p.relative_to(run).as_posix(), "sha256": file_hash(p),
                              "bytes": p.stat().st_size} for p in checkpoints],
        "stored_checkpoint_bytes": sum(p.stat().st_size for p in checkpoints),
        "total_model_parameters": sum(p.numel() for p in adapter.parameters()),
        "training_resources": training_resources,
        "backbone_source": read_json(run / "backbone_source.json") if (run / "backbone_source.json").exists() else None}


def module_macs(module: nn.Module, value: torch.Tensor) -> int:
    """Conv2d + Linear MAC estimate, one multiply-accumulate counts as one MAC.

    Includes grouped/depthwise and squeeze/excitation convolutions; excludes BN,
    activations, pooling, elementwise operations, routing and memory transfers.
    Hooks run only in untimed setup and are always removed.
    """
    total = 0
    def hook(layer, inputs, output):
        nonlocal total
        if isinstance(layer, nn.Conv2d):
            total += output.numel() * (layer.in_channels // layer.groups) * math.prod(layer.kernel_size)
        elif isinstance(layer, nn.Linear):
            total += output.numel() * layer.in_features
    handles = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode():
            module(value)
    finally:
        for handle in handles:
            handle.remove()
    return total


def warmup(model: InferenceAdapter, image: torch.Tensor, iterations: int) -> dict:
    """Exercise EVERY trained node directly, then label-free predicted routes."""
    costs = {}
    with torch.inference_mode():
        if isinstance(model, FlatAdapter):
            costs["variant"] = module_macs(model.node.classifier, image.unsqueeze(0))
        else:
            value = image.unsqueeze(0)
            if model.backbone is not None:
                costs["backbone"] = module_macs(model.backbone, value)
                value = model.backbone(value)
            for key, node in model.nodes.items():
                costs[key] = module_macs(node.classifier, value)
                for _ in range(iterations):
                    node.classifier(value)
        for _ in range(iterations):
            model.predict(image)
    synchronize(image.device)
    return costs


def path_macs(prediction: Prediction, costs: dict) -> int:
    return costs.get("backbone", 0) + sum(costs[d.stage if d.parent_id is None else f"{d.stage}:{d.parent_id}"]
        for d in prediction.decisions if d.source == "classifier")


@torch.inference_mode()
def timed_image(model: InferenceAdapter, path: Path, preprocessing: Preprocessing,
                device: torch.device, dtype: torch.dtype) -> tuple[Prediction, torch.Tensor, dict]:
    """Nested synchronized wall timers; truth is absent from this API.

    H2D completion sync is charged to full pipeline. The ready-tensor path timer
    includes adapter validation, routing, device-to-host argmax and Python ID.
    File open/decode is repeated; normal OS file caching is retained.
    """
    synchronize(device)
    start = time.perf_counter()
    with Image.open(path) as source:
        image = preprocessing.tensor(source).to(device=device, dtype=dtype)
    synchronize(device)
    path_start = time.perf_counter()
    prediction = model.predict(image)
    synchronize(device)
    end = time.perf_counter()
    return prediction, image, {"full_pipeline_ms": 1000 * (end - start),
                               "model_path_ms": 1000 * (end - path_start)}


def aggregate_timings(rows: list[dict], passes: list[dict]) -> dict:
    result = {k: distribution([float(r[k]) for r in rows]) for k in ("full_pipeline_ms", "model_path_ms")}
    result["passes"] = [{**p, **{k: distribution([float(r[k]) for r in rows if int(r["pass"]) == p["pass"]])
                                for k in ("full_pipeline_ms", "model_path_ms")}} for p in passes]
    result["sustained_fps"] = len(rows) / sum(p["elapsed_seconds"] for p in passes)
    result["mean_executed_macs"] = sum(int(r["executed_macs"]) for r in rows) / len(rows)
    result["mean_backbone_calls"] = sum(int(r["backbone_calls"]) for r in rows) / len(rows)
    result["singleton_bypass_rate"] = sum(int(r["singleton_bypasses"]) > 0 for r in rows) / len(rows)
    result["paths"] = {}
    for route in sorted({r["route"] for r in rows}):
        selected = [r for r in rows if r["route"] == route]
        result["paths"][route] = {"observations": len(selected), "full_pipeline_ms": distribution([float(r["full_pipeline_ms"]) for r in selected]),
            "model_path_ms": distribution([float(r["model_path_ms"]) for r in selected])}
    return result


def measure(config: BenchmarkConfig, data: AircraftData, run: Path, output_root: Path) -> Path:
    """One immutable architecture/seed/dtype run; summary only follows all checks."""
    preprocessing, device = Preprocessing(), torch.device(config.device)
    model, provenance = load_selected(run, data, preprocessing, config.depth)
    seed_everything(provenance["seed"])
    dtype = torch.float32 if config.precision == "fp32" else torch.float16
    output = output_root / unique_id(f"bench-{model.architecture.lower()}-s{provenance['seed']}-{config.precision}")
    output.mkdir(parents=True, exist_ok=False)
    resolved = {**asdict(config), "runs": [run.name], "seed": provenance["seed"],
                "residency": "all-resident", "batch_size": 1, "split": "test", "routing": "predicted"}
    write_json(output / "config.resolved.json", resolved)
    write_json(output / "provenance.json", provenance)
    repo = Path(__file__).resolve().parents[2]
    write_json(output / "source_hashes.json", {p.relative_to(repo).as_posix(): file_hash(p)
                                              for p in sorted((repo / "src/cls_balance").rglob("*.py"))})
    env = environment(repo)
    env.update(device=str(device), runtime_policy=f"all-resident {model.architecture}; {config.precision}; batch 1; TF32 disabled",
               cpu_threads=config.cpu_threads, workers=0, telemetry_before=telemetry())
    write_json(output / "environment.json", env)
    manifests = {s: object_hash(data.manifest(s)) for s in SPLITS}
    contract = {"manifest_records_sha256": manifests, "taxonomy_sha256": object_hash(data.taxonomy.to_dict()),
                "preprocessing_sha256": preprocessing.sha256,
                "test_ids": [r.image_id for r in data.splits["test"]], "test_count": len(data.splits["test"]),
                "source_annotations_sha256": object_hash(data.source_hashes)}
    write_json(output / "data_contract.json", contract)
    write_json(output / "taxonomy.json", data.taxonomy.to_dict())
    write_json(output / "preprocessing.json", preprocessing.to_dict())
    try:
        model.to(device=device, dtype=dtype).eval()
        fingerprint = model_hash(model)
        image = torch.zeros(3, 224, 224, dtype=dtype, device=device)
        costs = warmup(model, image, config.warmup_iterations)
        del image
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        records, latencies, pass_results, reference = [], [], [], {}
        manifest = data.manifest("test")
        for pass_id in range(config.passes):
            # Identical deterministic order for all architectures/seeds/dtypes;
            # change between passes to expose branch-order and cache effects.
            order = np.random.default_rng(6000 + pass_id).permutation(len(manifest))
            synchronize(device)
            start = time.perf_counter()
            for index in order:
                row = manifest[index]
                p, image, timing = timed_image(model, data.image_path(row["image_id"]), preprocessing, device, dtype)
                del image
                if pass_id == 0:
                    reference[row["image_id"]] = p
                    records.append(PredictionRecord(row["image_id"], "test", model.architecture, "predicted",
                        row["variant_id"], row["family_id"], row["manufacturer_id"], p))
                elif reference[row["image_id"]] != p:
                    raise ValueError("Repeated inference changed prediction/path")
                route = "/".join(f"{d.stage}:{d.global_id}:{d.source}" for d in p.decisions[:-1]) or "flat"
                latencies.append({"pass": pass_id, "image_id": row["image_id"], "variant_id": p.variant_id,
                    "route": route, "backbone_calls": p.backbone_calls, "classifier_calls": p.classifier_calls,
                    "singleton_bypasses": p.singleton_bypasses, "executed_macs": path_macs(p, costs), **timing})
            synchronize(device)
            elapsed = time.perf_counter() - start
            pass_results.append({"pass": pass_id, "images": len(manifest), "elapsed_seconds": elapsed,
                                 "sustained_fps": len(manifest) / elapsed})
            write_json(output / f"pass-{pass_id:02d}.json", {**pass_results[-1], "telemetry_after": telemetry()})
            print(json.dumps({"run": output.name, **pass_results[-1]}), flush=True)
        resources = {**provenance, "macs_by_node": costs,
            "mac_convention": "Conv2d/Linear only; 1 multiply-accumulate = 1 MAC; other ops excluded",
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "peak_process_resident_bytes": None, "oom": False, "model_state_unchanged": model_hash(model) == fingerprint}
        if not resources["model_state_unchanged"]:
            raise ValueError("Model state changed during benchmark")
        records.sort(key=lambda r: r.image_id)
        write_predictions(output / "predictions_test.csv", records, data.taxonomy)
        metrics = score(records, data.taxonomy)
        write_json(output / "metrics.json", metrics)
        write_csv(output / "classwise_metrics.csv", metrics["classwise"])
        write_csv(output / "latency_test.csv", latencies)
        write_json(output / "timing_summary.json", aggregate_timings(latencies, pass_results))
        # Separate untimed oracle diagnostics: labels never enter measured paths.
        if model.depth > 1:
            oracle = []
            for row in manifest:
                with Image.open(data.image_path(row["image_id"])) as source:
                    image = preprocessing.tensor(source).to(device=device, dtype=dtype)
                truth = Ancestors(row["manufacturer_id"], row["family_id"])
                p = model.predict(image, routing="oracle", truth=truth)
                oracle.append(PredictionRecord(row["image_id"], "test", model.architecture, "oracle",
                    row["variant_id"], row["family_id"], row["manufacturer_id"], p))
            oracle.sort(key=lambda r: r.image_id)
            write_predictions(output / "predictions_oracle_test.csv", oracle, data.taxonomy)
            write_json(output / "oracle_metrics.json", score(oracle, data.taxonomy))
            write_json(output / "routing_comparison.json", paired_scores(records, oracle, data.taxonomy))
        write_json(output / "resources.json", resources)
        hashes = {p.name: file_hash(p) for p in sorted(output.iterdir())}
        write_json(output / "summary.json", {"status": "complete", "run_id": output.name,
            "architecture": model.architecture, "seed": provenance["seed"], "precision": config.precision,
            "passes": config.passes, "images_per_pass": len(manifest), "model_sha256": fingerprint,
            "config_sha256": object_hash(resolved), "data_contract_sha256": object_hash(contract),
            "artifacts_sha256": hashes,
            "full_pipeline_boundary": "file open/decode + banner removal/letterbox/normalization + dtype/H2D + H2D sync + predicted adapter + D2H/Python global ID + completion sync; OS file cache retained",
            "model_path_boundary": "ready device tensor through complete predicted adapter, D2H/Python ID and completion sync",
            "fps_boundary": "continuous timed-pass loop including decode, preprocessing, transfers, inference, synchronization, timing instrumentation and in-memory result collection; excludes CSV writes, telemetry, setup, warmup and oracle",
            "latency_aggregation": "pooled image/pass observations within seed; linear percentiles; pass distributions also retained",
            "warmup": "every trained node directly plus predicted adapter; synthetic zero tensor; no test labels",
            "peak_memory_boundary": "post-warmup timed passes only; all deployment weights resident; oracle excluded"})
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                   "error": str(exc), "oom": isinstance(exc, torch.cuda.OutOfMemoryError)})
        raise
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def audit_run(path: Path) -> dict:
    """Recompute all classification/timing summaries from retained observations."""
    summary = read_json(path / "summary.json")
    inventory = {p.name for p in path.iterdir() if p.is_file() and p.name != "summary.json"}
    if summary.get("status") != "complete" or inventory != set(summary["artifacts_sha256"]):
        raise ValueError("Incomplete/extra benchmark artifacts")
    for name, digest in summary["artifacts_sha256"].items():
        if Path(name).name != name or file_hash(path / name) != digest:
            raise ValueError(f"Benchmark checksum mismatch: {name}")
    config = read_json(path / "config.resolved.json")
    contract = read_json(path / "data_contract.json")
    taxonomy = Taxonomy.from_dict(read_json(path / "taxonomy.json"))
    if (object_hash(config) != summary["config_sha256"] or object_hash(contract) != summary["data_contract_sha256"]
            or object_hash(taxonomy.to_dict()) != contract["taxonomy_sha256"]
            or object_hash(read_json(path / "preprocessing.json")) != contract["preprocessing_sha256"]
            or summary["passes"] < 5 or summary["images_per_pass"] != contract["test_count"]):
        raise ValueError("Benchmark metadata contract mismatch")
    records = read_predictions(path / "predictions_test.csv", taxonomy)
    if (set(r.image_id for r in records) != set(contract["test_ids"])
            or any(r.split != "test" or r.routing != "predicted" or r.architecture != summary["architecture"] for r in records)
            or object_hash(score(records, taxonomy)) != object_hash(read_json(path / "metrics.json"))):
        raise ValueError("Test predictions/metrics mismatch")
    with (path / "latency_test.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    references = {r.image_id: r.prediction for r in records}
    costs = read_json(path / "resources.json")["macs_by_node"]
    if len(rows) != summary["passes"] * len(records) or {int(r["pass"]) for r in rows} != set(range(summary["passes"])):
        raise ValueError("Missing timed passes")
    for pass_id in range(summary["passes"]):
        selected = [r for r in rows if int(r["pass"]) == pass_id]
        if len(selected) != len(records) or {r["image_id"] for r in selected} != set(references):
            raise ValueError("Missing/duplicate timed image")
    for row in rows:
        p = references[row["image_id"]]
        route = "/".join(f"{d.stage}:{d.global_id}:{d.source}" for d in p.decisions[:-1]) or "flat"
        if (int(row["variant_id"]) != p.variant_id or int(row["backbone_calls"]) != p.backbone_calls
                or int(row["classifier_calls"]) != p.classifier_calls or int(row["singleton_bypasses"]) != p.singleton_bypasses
                or int(row["executed_macs"]) != path_macs(p, costs) or row["route"] != route
                or float(row["model_path_ms"]) > float(row["full_pipeline_ms"])):
            raise ValueError("Timing/predicted path mismatch")
    passes = [{k: v for k, v in read_json(path / f"pass-{i:02d}.json").items() if k != "telemetry_after"}
              for i in range(summary["passes"])]
    if aggregate_timings(rows, passes) != read_json(path / "timing_summary.json"):
        raise ValueError("Saved timing summary disagrees with per-image measurements")
    if summary["architecture"] != "FLAT":
        oracle = read_predictions(path / "predictions_oracle_test.csv", taxonomy)
        if (object_hash(score(oracle, taxonomy)) != object_hash(read_json(path / "oracle_metrics.json"))
                or object_hash(paired_scores(records, oracle, taxonomy)) != object_hash(read_json(path / "routing_comparison.json"))):
            raise ValueError("Oracle diagnostic mismatch")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_run(args.audit), indent=2))
        return
    config = read_config(args.config)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, preprocessing = AircraftData.load(Path(config.data_root)), Preprocessing()
    validate_preparation(data, preprocessing, Path(config.prepared_run))
    # Freeze and validate the complete checkpoint matrix BEFORE opening test images.
    matrix = []
    for run in config.runs:
        print(f"Auditing selected checkpoint: {Path(run).name}", flush=True)
        model, provenance = load_selected(Path(run), data, preprocessing, config.depth)
        matrix.append(provenance)
        del model
    seeds = {m["seed"] for m in matrix}
    architectures = {m["architecture"] for m in matrix}
    if len(matrix) != len(seeds) * len(architectures) or any(
            {m["seed"] for m in matrix if m["architecture"] == a} != seeds for a in architectures):
        raise ValueError("Architectures require identical seed sets, one checkpoint per architecture/seed")
    for entry in matrix:
        if entry["architecture"].startswith("FT-FROZEN") and "FLAT" in architectures:
            flat = next(m for m in matrix if m["architecture"] == "FLAT" and m["seed"] == entry["seed"])
            if (entry["backbone_source"]["source_run_id"] != flat["training_run_id"]
                    or entry["backbone_source"]["parent_flat_sha256"] != flat["checkpoint_files"][0]["sha256"]):
                raise ValueError("Shared encoder does not originate from the paired FLAT checkpoint")
    root = Path(config.output_root) / unique_id("benchmark-suite")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "frozen_matrix.json", {"config": asdict(config), "models": matrix})
    print(f"Frozen benchmark suite: {root}", flush=True)
    outputs = []
    with awake():
        for run in config.runs:
            output = measure(config, data, Path(run), root)
            audit_run(output)
            outputs.append(output.name)
    write_json(root / "summary.json", {"status": "complete", "runs": outputs,
        "frozen_matrix_sha256": file_hash(root / "frozen_matrix.json"),
        "run_summaries_sha256": {name: file_hash(root / name / "summary.json") for name in outputs}})
    print(root, flush=True)


if __name__ == "__main__":
    main()
