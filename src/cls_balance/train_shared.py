"""Train independently selected heads on one immutable frozen EfficientNet.

Usage: python -m cls_balance.train_shared --config configs/shared.yaml
Only train/validation images are processed. Interrupted runs restart with new IDs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml

from .artifacts import environment, file_hash, object_hash, seed_everything, write_json
from .checkpoints import load_checkpoint, save_checkpoint, state_hash
from .data import AircraftData, AircraftDataset, SPLITS
from .embeddings import extract_features, training_features
from .evaluate import Evaluator, ModelProvenance, audit_run, compare_evaluations, model_hash
from .infer import HierarchicalAdapter, LogitNode
from .models.cascade import LocalDataset, NodeSpec, node_plan
from .models.flat import FrozenEncoder, build_flat
from .models.shared import FeatureDataset, assert_frozen, build_head
from .prepare import write_csv
from .report_flat import read_json
from .taxonomy import Taxonomy
from .train import CheckpointSelection, TrainingConfig, run_epoch, synchronize, validate_preparation
from .train_independent import compatibility, unique_id
from .transforms import Preprocessing


@dataclass(frozen=True)
class SharedConfig(TrainingConfig):
    learning_rate: float = 1e-3
    depths: tuple[int, ...] = (2,)
    backbone: str = "FT-FROZEN"
    flat_runs: dict[int, str] = field(default_factory=dict)
    hidden_dimension: int = 256
    cache_root: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (not self.depths or len(set(self.depths)) != len(self.depths)
                or any(type(d) is not int or d not in (2, 3) for d in self.depths)):
            raise ValueError("Depths must be distinct values from 2, 3")
        if self.backbone not in {"FROZEN", "FT-FROZEN"}:
            raise ValueError("Unknown frozen backbone")
        if type(self.hidden_dimension) is not int or self.hidden_dimension < 1:
            raise ValueError("Invalid hidden dimension")
        if self.backbone == "FT-FROZEN" and not set(self.seeds) <= self.flat_runs.keys():
            raise ValueError("Require a corresponding FLAT run for every seed")


def read_config(path: Path) -> SharedConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("data_root", "prepared_run", "output_root", "cache_root"):
        if raw.get(key) is not None:
            raw[key] = str((path.resolve().parent / raw[key]).resolve())
    raw["flat_runs"] = {int(k): str((path.resolve().parent / v).resolve()) for k, v in raw.get("flat_runs", {}).items()}
    for key, default in (("seeds", (42, 43, 44)), ("depths", (2,))):
        raw[key] = tuple(raw.get(key, default))
    return SharedConfig(**raw)


def source_encoder(config: SharedConfig, data: AircraftData, preprocessing: Preprocessing,
                   seed: int) -> tuple[torch.nn.Module, dict]:
    """Verify selected FLAT provenance, including exact official train/val IDs."""
    if config.backbone == "FROZEN":
        encoder = FrozenEncoder(build_flat())
        return encoder, {"initialization": "EfficientNet_B0_Weights.IMAGENET1K_V1",
                         "backbone_training_cost": "external ImageNet pretraining; not measured here"}
    from .report_flat import audit_training_run

    path = Path(config.flat_runs[seed])
    summary = audit_training_run(path)
    encoder, metadata = load_checkpoint(path / "encoder.pt", **compatibility(data, preprocessing), kind="FT-FROZEN-encoder")
    if (summary["seed"] != seed or metadata["seed"] != seed or metadata["run_id"] != summary["run_id"]
            or metadata["parent_flat_sha256"] != file_hash(path / "flat.pt")
            or metadata["training_image_ids"] != [r.image_id for r in data.splits["train"]]
            or metadata["selection_image_ids"] != [r.image_id for r in data.splits["val"]]):
        raise ValueError("FLAT seed/source/split provenance mismatch")
    return encoder, {"source_run_id": summary["run_id"], "source_encoder_sha256": file_hash(path / "encoder.pt"),
                     "parent_flat_sha256": metadata["parent_flat_sha256"],
                     "flat_resources": summary["resources"],
                     "backbone_training_cost": "one Phase 3 FLAT run per seed; reused across shared depths"}


def checked_head(path: Path, spec: NodeSpec, hashes: dict, backbone_hash: str,
                 seed: int, hidden: int, dimension: int) -> torch.nn.Module:
    head, metadata = load_checkpoint(path, **hashes, kind="shared-head")
    if (NodeSpec.from_dict(metadata["node"]) != spec or metadata["backbone_sha256"] != backbone_hash
            or metadata["seed"] != seed or metadata["hidden_dimension"] != hidden
            or metadata["input_dimension"] != dimension):
        raise ValueError("Head backbone/seed/structure/local-map compatibility mismatch")
    return head


def load_shared(path: Path, depth: int, *, replacements: dict[str, Path] | None = None) -> HierarchicalAdapter:
    """Reload an immutable bundle, optionally replacing specified compatible heads.

    Replacement keys are canonical NodeSpec keys. Unrelated modules are never
    changed. The caller must record overrides/model_hash in any new evaluation.
    """
    bundle = read_json(path / f"bundle-shared{depth}.json")
    taxonomy = Taxonomy.from_dict(read_json(path / "taxonomy.json"))
    if object_hash(taxonomy.to_dict()) != bundle["compatibility"]["taxonomy_sha256"]:
        raise ValueError("Bundle taxonomy mismatch")
    specs = node_plan(taxonomy, depth)
    if set(bundle["nodes"]) != {s.key for s in specs} or set(replacements or {}) - set(bundle["nodes"]):
        raise ValueError("Missing/extra bundle or replacement heads")
    def local(relative: str) -> Path:
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("Artifact path escapes run")
        return target
    encoder_path = local(bundle["encoder_path"])
    if file_hash(encoder_path) != bundle["encoder_file_sha256"]:
        raise ValueError("Encoder checkpoint mismatch")
    encoder, _ = load_checkpoint(encoder_path, **bundle["compatibility"], kind=bundle["encoder_kind"])
    if state_hash(encoder) != bundle["backbone_sha256"]:
        raise ValueError("Backbone version mismatch")
    nodes = {}
    for spec in specs:
        entry = bundle["nodes"][spec.key]
        target = local(entry["path"])
        if file_hash(target) != entry["checkpoint_sha256"]:
            raise ValueError("Head checkpoint mismatch")
        target = (replacements or {}).get(spec.key, target)
        head = checked_head(target, spec, bundle["compatibility"], bundle["backbone_sha256"],
                            bundle["seed"], bundle["hidden_dimension"], bundle["input_dimension"])
        nodes[spec.adapter_key] = LogitNode(head, spec.global_ids)
    return HierarchicalAdapter(taxonomy, nodes, depth=depth, backbone=encoder,
                               fine_tuned=bundle["backbone"] == "FT-FROZEN")


def train_seed(config: SharedConfig, data: AircraftData, preprocessing: Preprocessing,
               prepared: dict, seed: int) -> Path:
    """One shared image pass per epoch, isolated optimizers and local selection.

    Heads share read-only augmented inputs but no parameters, optimizer state,
    shuffle generator, checkpoint selection or early-stopping state.
    """
    seed_everything(seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.device)
    encoder, source = source_encoder(config, data, preprocessing, seed)
    encoder.to(device).eval()
    fingerprint = state_hash(encoder)
    assert_frozen(encoder, fingerprint)
    path = Path(config.output_root) / unique_id(f"shared-{config.backbone.lower()}-s{seed}")
    path.mkdir(parents=True, exist_ok=False)
    print(f"Training run {path}", flush=True)
    resolved = {**asdict(config), "seed": seed, "precision": "fp32", "optimizer": "AdamW",
                "augmentation": "fresh seeded view per image per epoch; shared read-only across heads",
                "selection": "local validation macro recall; strict improvement; earliest tie",
                "sampling": "all branch images; independent shuffle generators; no replacement",
                "runtime_policy": "all-resident; one frozen encoder; independent heads; TF32 disabled"}
    repo = Path(__file__).resolve().parents[2]
    sources = {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted((repo / "src/cls_balance").rglob("*.py"))}
    hashes = compatibility(data, preprocessing)
    contract = {**hashes, "run_id": path.name, "seed": seed, "config_sha256": object_hash(resolved),
                "source_hashes_sha256": object_hash(sources), "backbone_sha256": fingerprint}
    for name, value in {"config.resolved.json": resolved, "source_hashes.json": sources,
                        "taxonomy.json": data.taxonomy.to_dict(), "preprocessing.json": preprocessing.to_dict(),
                        "backbone_source.json": source,
                        "dataset_manifest_hashes.json": {"canonical_records": {s: object_hash(data.manifest(s)) for s in SPLITS},
                            "prepared_run_id": prepared["run_id"], "prepared_summary_sha256": object_hash(prepared)}}.items():
        write_json(path / name, value)
    env = environment(repo)
    env.update(device=str(device), runtime_policy=resolved["runtime_policy"], cpu_threads=config.cpu_threads,
               power_policy="OS/driver defaults; clocks not locked")
    write_json(path / "environment.json", env)
    encoder_kind = config.backbone + "-encoder"
    save_checkpoint(path / "encoder.pt", encoder, {**contract, **source}, kind=encoder_kind)
    train = AircraftDataset(data, "train", preprocessing, training=True, seed=seed)
    val = AircraftDataset(data, "val", preprocessing)
    reloaded_encoder, _ = load_checkpoint(path / "encoder.pt", **hashes, kind=encoder_kind)
    reloaded_encoder.to(device)
    with torch.no_grad():
        fixed_image = val[0]["image"].unsqueeze(0).to(device)
        torch.testing.assert_close(encoder(fixed_image), reloaded_encoder(fixed_image), rtol=0, atol=0)
    del reloaded_encoder
    specs = {s.key: s for d in config.depths for s in node_plan(data.taxonomy, d)}
    states = {}
    for key, spec in specs.items():
        node_seed = int(object_hash([seed, key])[:8], 16)
        seed_everything(node_seed)
        head = build_head(len(spec.global_ids), config.hidden_dimension, encoder.dimension).to(device)
        states[key] = {"head": head, "optimizer": torch.optim.AdamW(head.parameters(), lr=config.learning_rate,
                       weight_decay=config.weight_decay), "selection": CheckpointSelection(), "history": [],
                       "train": LocalDataset(train, spec, split="train"), "val": LocalDataset(val, spec, split="val"),
                       "generator": torch.Generator().manual_seed(node_seed), "done": False, "best": None}
    # Feature extraction is counted once across all nodes/depths, separately from
    # head optimizer/selection time and the source FLAT training cost.
    synchronize(device)
    started = time.perf_counter()
    extraction_seconds = 0.0
    extraction_started = time.perf_counter()
    val_features = extract_features(encoder, val, device=device, batch_size=config.batch_size, workers=config.workers)
    extraction_seconds += time.perf_counter() - extraction_started
    for epoch in range(1, config.epochs + 1):
        train.set_epoch(epoch - 1)
        extraction_started = time.perf_counter()
        cache_contract = {**hashes, "backbone_sha256": fingerprint, "split": "train",
                          "image_ids": [r.image_id for r in train.rows], "seed": seed, "epoch": epoch - 1,
                          "transform_source_sha256": sources["src/cls_balance/transforms.py"],
                          "augmentation": resolved["augmentation"]}
        features = training_features(encoder, train, contract=cache_contract,
                    cache_root=Path(config.cache_root) if config.cache_root else None,
                    device=device, batch_size=config.batch_size, workers=config.workers)
        extraction_seconds += time.perf_counter() - extraction_started
        assert_frozen(encoder, fingerprint)
        for key, state in states.items():
            if state["done"]:
                continue
            spec = specs[key]
            loaders = [DataLoader(FeatureDataset(values, state[split].indices, state[split].targets),
                        batch_size=config.batch_size, shuffle=split == "train",
                        generator=state["generator"] if split == "train" else None)
                       for split, values in (("train", features), ("val", val_features))]
            training = run_epoch(state["head"], loaders[0], device, state["optimizer"], classes=len(spec.global_ids))
            validation = run_epoch(state["head"], loaders[1], device, classes=len(spec.global_ids))
            improved = state["selection"].observe(epoch, validation["macro_recall"])
            if improved:
                state["best"] = {k: v.detach().cpu().clone() for k, v in state["head"].state_dict().items()}
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in training.items()},
                   **{f"val_{k}": v for k, v in validation.items()}, "selected": improved}
            state["history"].append(row)
            node = path / "nodes" / key
            node.mkdir(parents=True, exist_ok=True)
            write_json(node / f"epoch-{epoch:03d}.json", row)
            state["done"] = state["selection"].stale >= config.patience or epoch == config.epochs
            if state["done"]:
                state["head"].load_state_dict(state["best"])
                state["head"].eval()
                metadata = {**contract, "node": spec.to_dict(), "hidden_dimension": config.hidden_dimension,
                            "input_dimension": encoder.dimension, "best_epoch": state["selection"].best_epoch,
                            "validation_macro_recall": state["selection"].best_score,
                            "training_image_ids": state["train"].image_ids, "selection_image_ids": state["val"].image_ids}
                save_checkpoint(node / "head.pt", state["head"], metadata, kind="shared-head")
                loaded = checked_head(node / "head.pt", spec, hashes, fingerprint, seed, config.hidden_dimension, encoder.dimension).to(device)
                fixed = val_features[state["val"].indices[:1]].to(device)
                with torch.no_grad():
                    torch.testing.assert_close(state["head"](fixed), loaded(fixed), rtol=0, atol=0)
                write_csv(node / "history.csv", state["history"])
                write_json(node / "summary.json", {"node": spec.to_dict(), "best_epoch": metadata["best_epoch"],
                    "validation_macro_recall": metadata["validation_macro_recall"], "epochs_completed": len(state["history"]),
                    "exact_reload_logits": True, "trainable_parameters": sum(p.numel() for p in state["head"].parameters()),
                    "optimizer_steps": sum(h["train_optimizer_steps"] for h in state["history"]),
                    "training_image_exposures": sum(h["train_images"] for h in state["history"]),
                    "head_epoch_seconds": sum(h["train_wall_seconds"] + h["val_wall_seconds"] for h in state["history"]),
                    "artifacts_sha256": {n: file_hash(node / n) for n in ("head.pt", "head.json", "history.csv")}})
        print(f"Seed {seed} epoch {epoch}: {sum(s['done'] for s in states.values())}/{len(states)} heads selected", flush=True)
        if all(s["done"] for s in states.values()):
            break
    synchronize(device)
    elapsed = time.perf_counter() - started
    assert_frozen(encoder, fingerprint)
    resources = {"shared_training_wall_seconds": elapsed, "feature_extraction_seconds": extraction_seconds,
                 "training_gpu_hours": elapsed / 3600 if device.type == "cuda" else None,
                 "unique_train_feature_images": epoch * len(train), "validation_feature_images": len(val),
                 "encoder_bytes": (path / "encoder.pt").stat().st_size,
                 "head_bytes": sum(p.stat().st_size for p in (path / "nodes").glob("*/head.pt")),
                 "timing_scope": "shared train/val extraction + head epochs + selection/export; excludes source FLAT and final evaluation",
                 "cache_enabled": config.cache_root is not None}
    node_summaries = [read_json(path / "nodes" / key / "summary.json") for key in specs]
    resources.update({key: sum(n[key] for n in node_summaries) for key in
                      ("optimizer_steps", "training_image_exposures", "trainable_parameters", "head_epoch_seconds")})
    write_json(path / "resources.json", resources)
    del states, features, val_features, encoder
    evaluations, penalties = {}, {}
    for depth in config.depths:
        bundle = {**contract, "backbone": config.backbone, "encoder_kind": encoder_kind,
                  "encoder_path": "encoder.pt", "encoder_file_sha256": file_hash(path / "encoder.pt"),
                  "hidden_dimension": config.hidden_dimension, "input_dimension": 1280,
                  "compatibility": hashes, "nodes": {s.key: {"path": f"nodes/{s.key}/head.pt",
                    "checkpoint_sha256": file_hash(path / "nodes" / s.key / "head.pt")} for s in node_plan(data.taxonomy, depth)}}
        write_json(path / f"bundle-shared{depth}.json", bundle)
        adapter = load_shared(path, depth)
        provenance = ModelProvenance(path.name, model_hash(adapter), hashes["taxonomy_sha256"], preprocessing.sha256,
                                    seed, tuple(r.image_id for r in train.rows), tuple(r.image_id for r in val.rows))
        evaluator = Evaluator(data, preprocessing, provenance, device=config.device)
        paired = []
        for routing in ("predicted", "oracle"):
            result = evaluator.evaluate(adapter, split="val", routing=routing,
                                        conditional_family=depth == 3 and routing == "predicted")
            saved = result.save(path / "evaluations")
            evaluations[f"{config.backbone}-{depth}/{routing}"] = {"path": saved.relative_to(path).as_posix(), "audit": audit_run(saved)}
            paired.append(result)
        penalties[str(depth)] = compare_evaluations(*paired)["routing_penalty_macro_recall_pp"]
    write_json(path / "summary.json", {"schema_version": 1, "status": "complete", **contract,
        "backbone": config.backbone, "depths": config.depths, "evaluations": evaluations,
        "routing_penalties_pp": penalties, "test_evaluated": False, "frozen_weights_and_buffers_unchanged": True,
        "exact_reload_embeddings": True,
        "unique_trained_heads": len(specs), "resources": resources,
        "artifacts_sha256": {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}})
    print(f"Completed {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shared.yaml"))
    args = parser.parse_args()
    config = read_config(args.config)
    data, preprocessing = AircraftData.load(Path(config.data_root), download=False), Preprocessing()
    print("Verifying complete prepared dataset and image checksums", flush=True)
    prepared = validate_preparation(data, preprocessing, Path(config.prepared_run))
    for seed in config.seeds:
        train_seed(config, data, preprocessing, prepared, seed)


if __name__ == "__main__":
    main()
