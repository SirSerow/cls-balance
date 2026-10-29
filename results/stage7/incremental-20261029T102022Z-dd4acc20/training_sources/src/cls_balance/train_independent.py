"""Train full independent EfficientNets; select each node using local validation.

Usage: python -m cls_balance.train_independent --config configs/independent.yaml
Completed nodes can be reused after interruption only with identical contracts.
Incomplete nodes are never resumed; a retry writes a new immutable attempt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import torch
from torch.utils.data import DataLoader
import yaml

from .artifacts import environment, file_hash, object_hash, seed_everything, write_json
from .checkpoints import load_checkpoint, save_checkpoint, state_hash
from .data import AircraftData, AircraftDataset, SPLITS
from .evaluate import Evaluator, ModelProvenance, audit_run, compare_evaluations, model_hash
from .infer import HierarchicalAdapter, LogitNode
from .models.cascade import LocalDataset, NodeSpec, node_plan
from .models.flat import build_flat
from .prepare import write_csv
from .train import CheckpointSelection, TrainingConfig, run_epoch, synchronize, validate_preparation
from .transforms import Preprocessing


@dataclass(frozen=True)
class IndependentConfig(TrainingConfig):
    depths: tuple[int, ...] = (2,)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (not self.depths or len(set(self.depths)) != len(self.depths)
                or any(type(d) is not int or d not in (2, 3) for d in self.depths)):
            raise ValueError("Depths must be distinct values from 2, 3")


def read_config(path: Path) -> IndependentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("data_root", "prepared_run", "output_root"):
        raw[key] = str((path.resolve().parent / raw[key]).resolve())
    for key, default in (("seeds", (42, 43, 44)), ("depths", (2,))):
        raw[key] = tuple(raw.get(key, default))
    return IndependentConfig(**raw)


def unique_id(prefix: str) -> str:
    return datetime.now(timezone.utc).strftime(prefix + "-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def compatibility(data: AircraftData, preprocessing: Preprocessing) -> dict:
    return {"taxonomy_sha256": object_hash(data.taxonomy.to_dict()),
            "preprocessing_sha256": preprocessing.sha256,
            "dataset_manifest_sha256": object_hash({s: object_hash(data.manifest(s)) for s in SPLITS})}


def audit_node(path: Path, expected: dict | None = None) -> dict:
    from .report_flat import read_json

    summary = read_json(path / "summary.json")
    required = {"node.pt", "node.json", "history.csv", "contract.json", "resources.json"}
    if summary.get("status") != "complete" or set(summary.get("artifacts_sha256", {})) != required:
        raise ValueError("Incomplete node")
    for name, digest in summary["artifacts_sha256"].items():
        if Path(name).name != name or file_hash(path / name) != digest:
            raise ValueError("Node artifact checksum mismatch")
    contract = read_json(path / "contract.json")
    if object_hash(contract) != summary["contract_sha256"] or (expected is not None and contract != expected):
        raise ValueError("Node contract mismatch")
    manifest = read_json(path / "node.json")
    if (manifest["metadata"]["contract_sha256"] != summary["contract_sha256"]
            or manifest["file_sha256"] != file_hash(path / "node.pt")):
        raise ValueError("Node checkpoint binding mismatch")
    return summary


def train_node(config: IndependentConfig, spec: NodeSpec, train: AircraftDataset,
               val: AircraftDataset, contract: dict, output: Path) -> Path:
    """Independent initialization, optimizer and checkpoint; no feature sharing."""
    seed_everything(contract["seed"])
    device = torch.device(config.device)
    local_train = LocalDataset(train, spec, split="train")
    local_val = LocalDataset(val, spec, split="val")
    node_contract = {**contract, "node": spec.to_dict(),
                     "training_image_ids": local_train.image_ids, "selection_image_ids": local_val.image_ids}
    # JSON roundtrip normalizes tuples before equality checks on retry.
    node_contract = json.loads(json.dumps(node_contract))
    attempts = output / spec.key
    for previous in sorted(attempts.glob("*/summary.json")):
        audit_node(previous.parent, node_contract)
        print(f"Reusing verified node {previous.parent}", flush=True)
        return previous.parent
    path = attempts / unique_id("node")
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "contract.json", node_contract)
    # Large branches overlap image decoding with GPU work. Tiny branches avoid
    # spawning fresh Windows workers at every epoch; sample order is unchanged.
    actual_workers = config.workers if len(local_train) >= 256 else 0
    loaders = [DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                          generator=torch.Generator().manual_seed(contract["seed"]) if training else None,
                          num_workers=actual_workers, persistent_workers=False)
               for dataset, training in ((local_train, True), (local_val, False))]
    model = build_flat(classes=len(spec.global_ids)).to(device)
    # This is the pretrained feature tensor hash, excluding the random local head.
    feature_initialization = state_hash(model.features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    selection = CheckpointSelection()
    history, best_state = [], None
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train.set_epoch(epoch - 1)
        training = run_epoch(model, loaders[0], device, optimizer, classes=len(spec.global_ids))
        validation = run_epoch(model, loaders[1], device, classes=len(spec.global_ids))
        improved = selection.observe(epoch, validation["macro_recall"])
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in training.items()},
               **{f"val_{k}": v for k, v in validation.items()}, "selected": improved}
        history.append(row)
        write_json(path / f"epoch-{epoch:03d}.json", row)
        print(json.dumps({"seed": contract["seed"], "node": spec.key, **row}), flush=True)
        if selection.stale >= config.patience:
            break
    synchronize(device)
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    metadata = {**node_contract, "contract_sha256": object_hash(node_contract),
                "best_epoch": selection.best_epoch, "validation_macro_recall": selection.best_score,
                "initialization_features_sha256": feature_initialization}
    manifest = save_checkpoint(path / "node.pt", model, metadata, kind="IND-node")
    loaded, _ = load_checkpoint(path / "node.pt", **{k: contract[k] for k in compatibility_keys()}, kind="IND-node")
    loaded.to(device)
    fixed = local_val[0]["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        torch.testing.assert_close(model(fixed), loaded(fixed), rtol=0, atol=0)
    write_csv(path / "history.csv", history)
    resources = {"training_wall_seconds": elapsed,
                 "training_gpu_hours": elapsed / 3600 if device.type == "cuda" else None,
                 "optimizer_steps": sum(r["train_optimizer_steps"] for r in history),
                 "training_image_exposures": sum(r["train_images"] for r in history),
                 "train_images": len(local_train), "validation_images": len(local_val),
                 "data_loader_workers": actual_workers,
                 "trainable_parameters": sum(p.numel() for p in model.parameters()),
                 "deployment_bytes": (path / "node.pt").stat().st_size,
                 "timing_scope": "epoch train+validation+selection; excludes initialization, export and reload"}
    write_json(path / "resources.json", resources)
    write_json(path / "summary.json", {"status": "complete", "node": spec.to_dict(),
               "contract_sha256": object_hash(node_contract), "seed": contract["seed"],
               "best_epoch": selection.best_epoch, "validation_macro_recall": selection.best_score,
               "epochs_completed": len(history), "exact_reload_logits": True,
               "checkpoint_sha256": manifest["file_sha256"], "resources": resources,
               "initialization_features_sha256": feature_initialization,
               "artifacts_sha256": {name: file_hash(path / name) for name in
                   ("node.pt", "node.json", "history.csv", "contract.json", "resources.json")}})
    return path


def compatibility_keys() -> tuple[str, ...]:
    return ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256")


def load_cascade(path: Path, depth: int) -> HierarchicalAdapter:
    """Load a completed bundle (or training-stage bundle) with checked local maps."""
    from .report_flat import read_json
    from .taxonomy import Taxonomy

    bundle = read_json(path / f"bundle-ind{depth}.json")
    taxonomy = Taxonomy.from_dict(read_json(path / "taxonomy.json"))
    specs = node_plan(taxonomy, depth)
    if set(bundle["nodes"]) != {s.key for s in specs}:
        raise ValueError("Incomplete/extra classifier bundle")
    nodes = {}
    for spec in specs:
        entry = bundle["nodes"][spec.key]
        node_path = (path / entry["path"]).resolve()
        if not node_path.is_relative_to(path.resolve()):
            raise ValueError("Node path escapes run")
        audit_node(node_path)
        if file_hash(node_path / "node.pt") != entry["checkpoint_sha256"]:
            raise ValueError("Bundle checkpoint checksum mismatch")
        model, metadata = load_checkpoint(node_path / "node.pt", **bundle["compatibility"], kind="IND-node")
        if (NodeSpec.from_dict(metadata["node"]) != spec or metadata["seed"] != bundle["seed"]
                or metadata["run_id"] != bundle["run_id"]):
            raise ValueError("Node checkpoint identity mismatch")
        nodes[spec.adapter_key] = LogitNode(model, spec.global_ids)
    return HierarchicalAdapter(taxonomy, nodes, depth=depth)


def train_seed(config: IndependentConfig, data: AircraftData, preprocessing: Preprocessing,
               prepared: dict, seed: int, *, resume: Path | None = None) -> Path:
    from .report_flat import read_json

    seed_everything(seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.device(config.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    resolved = {**asdict(config), "seed": seed, "precision": "fp32", "optimizer": "AdamW",
                "initialization": "EfficientNet_B0_Weights.IMAGENET1K_V1",
                "selection": "local validation macro recall; strict improvement; earliest tie",
                "sampling": "shuffle without replacement; all branch images; drop_last=False",
                "scheduler": None, "betas": [0.9, 0.999], "eps": 1e-8,
                "worker_policy": "configured workers for train branches >=256 images; otherwise zero"}
    repo = Path(__file__).resolve().parents[2]
    sources = {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted((repo / "src/cls_balance").rglob("*.py"))}
    hashes = compatibility(data, preprocessing)
    if resume is None:
        path = Path(config.output_root) / unique_id(f"ind-s{seed}")
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "config.resolved.json", resolved)
        write_json(path / "source_hashes.json", sources)
        write_json(path / "taxonomy.json", data.taxonomy.to_dict())
        write_json(path / "preprocessing.json", preprocessing.to_dict())
        write_json(path / "dataset_manifest_hashes.json", {"canonical_records":
            {s: object_hash(data.manifest(s)) for s in SPLITS}, "prepared_run_id": prepared["run_id"],
            "prepared_summary_sha256": object_hash(prepared)})
        env = environment(repo)
        env.update(device=config.device, runtime_policy="all-resident; independent CNNs; FP32; TF32 disabled",
                   cpu_threads=config.cpu_threads, power_policy="OS/driver defaults; clocks not locked")
        write_json(path / "environment.json", env)
    else:
        path = resume.resolve()
        if (path / "summary.json").exists():
            raise ValueError("Cannot resume a published/completed run")
        if (object_hash(read_json(path / "config.resolved.json")) != object_hash(resolved)
                or read_json(path / "source_hashes.json") != sources
                or object_hash(read_json(path / "taxonomy.json")) != hashes["taxonomy_sha256"]
                or read_json(path / "dataset_manifest_hashes.json")["prepared_summary_sha256"] != object_hash(prepared)):
            raise ValueError("Resume configuration/source/data mismatch")
        previous_env, current_env = read_json(path / "environment.json"), environment(repo)
        for key in ("torch", "cuda_runtime", "gpus", "packages", "platform"):
            if previous_env[key] != current_env[key]:
                raise ValueError(f"Resume environment mismatch: {key}")
    print(f"Training run {path}", flush=True)
    contract = {**hashes, "run_id": path.name, "seed": seed, "config_sha256": object_hash(resolved),
                "source_hashes_sha256": object_hash(sources)}
    train = AircraftDataset(data, "train", preprocessing, training=True, seed=seed)
    val = AircraftDataset(data, "val", preprocessing)
    specs = {s.key: s for depth in config.depths for s in node_plan(data.taxonomy, depth)}
    nodes = {key: train_node(config, spec, train, val, contract, path / "nodes") for key, spec in specs.items()}
    if len({read_json(p / "summary.json")["initialization_features_sha256"] for p in nodes.values()}) != 1:
        raise ValueError("Independent CNNs were not initialized from identical ImageNet features")
    evaluations, bundles = {}, {}
    for depth in config.depths:
        subset = {s.key: nodes[s.key] for s in node_plan(data.taxonomy, depth)}
        bundle = {"architecture": f"IND-{depth}", "run_id": path.name, "seed": seed, "compatibility": hashes,
                  "residency": "all-resident", "nodes": {key: {"path": p.relative_to(path).as_posix(),
                      "checkpoint_sha256": file_hash(p / "node.pt")} for key, p in subset.items()}}
        bundle_path = path / f"bundle-ind{depth}.json"
        if bundle_path.exists():
            if read_json(bundle_path) != bundle:
                raise ValueError("Existing bundle differs on resume")
        else:
            write_json(bundle_path, bundle)
        adapter = load_cascade(path, depth)
        provenance = ModelProvenance(path.name, model_hash(adapter), hashes["taxonomy_sha256"],
            preprocessing.sha256, seed, tuple(r.image_id for r in data.splits["train"]),
            tuple(r.image_id for r in data.splits["val"]))
        evaluator = Evaluator(data, preprocessing, provenance, device=config.device)
        paired = []
        for routing in ("predicted", "oracle"):
            result = evaluator.evaluate(adapter, split="val", routing=routing,
                                        conditional_family=depth == 3 and routing == "predicted")
            saved = result.save(path / "evaluations")
            evaluations[f"IND-{depth}/{routing}"] = {"path": saved.relative_to(path).as_posix(), "audit": audit_run(saved)}
            paired.append(result)
        pair = compare_evaluations(*paired)
        resources = {key: sum(read_json(p / "resources.json")[key] for p in subset.values()) for key in
            ("training_wall_seconds", "optimizer_steps", "training_image_exposures", "trainable_parameters", "deployment_bytes")}
        resources["training_gpu_hours"] = resources["training_wall_seconds"] / 3600 if config.device.startswith("cuda") else None
        bundles[f"IND-{depth}"] = {"nodes": len(subset), "resources": resources,
            "routing_penalty_macro_recall_pp": pair["routing_penalty_macro_recall_pp"]}
        del adapter, evaluator, paired, result, pair
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(path / "summary.json", {"schema_version": 1, "status": "complete", **contract,
               "depths": config.depths, "test_evaluated": False, "evaluations": evaluations, "bundles": bundles,
               "unique_trained_nodes": len(nodes),
               "router_reuse": "same selected manufacturer router reused across depths within seed",
               "artifacts_sha256": {p.name: file_hash(p) for p in sorted(path.iterdir()) if p.is_file()}})
    print(f"Completed {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/independent.yaml"))
    parser.add_argument("--resume", type=Path, help="Retry incomplete bundle, reuse verified completed nodes only")
    args = parser.parse_args()
    config = read_config(args.config)
    data = AircraftData.load(Path(config.data_root), download=False)
    preprocessing = Preprocessing()
    print("Verifying all prepared annotations and image checksums", flush=True)
    prepared = validate_preparation(data, preprocessing, Path(config.prepared_run))
    if args.resume:
        from .report_flat import read_json
        seed = read_json(args.resume / "config.resolved.json")["seed"]
        if seed not in config.seeds:
            raise ValueError("Resume seed is outside configured seeds")
        train_seed(config, data, preprocessing, prepared, seed, resume=args.resume)
    else:
        for seed in config.seeds:
            train_seed(config, data, preprocessing, prepared, seed)


if __name__ == "__main__":
    main()
