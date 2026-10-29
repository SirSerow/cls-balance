"""Leak-free, two-stage class introduction; training and evaluation are separate.

The original taxonomy/official test set remain intact. Initial deployment masks
one preselected global variant ID, and every initial supervised dataset excludes
its train AND validation images, including the task-adapted encoder's datasets.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from .artifacts import environment, file_hash, object_hash, seed_everything, write_json
from .checkpoints import load_checkpoint, save_checkpoint, state_hash
from .data import AircraftData, AircraftDataset
from .embeddings import extract_features
from .evaluate import Evaluator, ModelProvenance, audit_run, model_hash
from .infer import HierarchicalAdapter, LogitNode
from .models.cascade import LocalDataset, NodeSpec, node_plan
from .models.flat import FrozenEncoder, build_flat
from .models.shared import FeatureDataset, assert_frozen, build_head
from .prepare import write_csv
from .report_flat import read_json
from .train import CheckpointSelection, TrainingConfig, run_epoch, synchronize, validate_preparation
from .train_independent import compatibility, train_node, IndependentConfig, unique_id
from .transforms import Preprocessing


@dataclass(frozen=True)
class IncrementalConfig(TrainingConfig):
    """Fixed before training/test access; same recipe as Phases 3–5."""

    head_learning_rate: float = 1e-3
    hidden_dimension: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        import math
        if (not math.isfinite(self.head_learning_rate) or self.head_learning_rate <= 0
                or type(self.hidden_dimension) is not int or self.hidden_dimension < 1):
            raise ValueError("Invalid incremental head settings")


def read_config(path: Path) -> IncrementalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("data_root", "prepared_run", "output_root"):
        raw[key] = str((path.resolve().parent / raw[key]).resolve())
    raw["seeds"] = tuple(raw.get("seeds", (42, 43, 44)))
    return IncrementalConfig(**raw)


def select_introduction(taxonomy) -> dict:
    """Lowest canonical variant ID in a branch with >=3 variants; no scores.

    Requiring two old children preserves a trained specialist before/after,
    isolating replacement cost from the separate singleton-to-classifier case.
    """
    branches = taxonomy.branches()["manufacturer_to_variant"]
    family_branches = taxonomy.branches()["family_to_variant"]
    candidates = [v for ids in branches if len(ids) >= 3 for v in ids
                  if len(family_branches[taxonomy.variant_to_family[v]]) > 1]
    if not candidates:
        raise ValueError("Need a manufacturer with at least three variants and a retained family")
    v = min(candidates)
    f = taxonomy.variant_to_family[v]
    m = taxonomy.family_to_manufacturer[f]
    return {"variant_id": v, "variant": taxonomy.variants[v], "manufacturer_id": m,
            "manufacturer": taxonomy.manufacturers[m], "branch_variant_ids": list(branches[m]),
            "selection_rule": "lowest training-taxonomy global variant ID in a manufacturer with >=3 variants and a family with >=2 variants"}


def supervised_data(data: AircraftData, heldout: int) -> AircraftData:
    """Explicit train/val view; never mutate or subset official test data."""
    if type(heldout) is not int or not 0 <= heldout < len(data.taxonomy.variants):
        raise ValueError("Invalid held-out global variant")
    splits = {s: [r for r in rows if data.taxonomy.ids(r)[0] != heldout] if s != "test" else list(rows)
              for s, rows in data.splits.items()}
    return AircraftData(data.root, splits, data.taxonomy, data.source_hashes)


def initial_specs(taxonomy, heldout: int) -> tuple[NodeSpec, ...]:
    """Retain original global IDs; singleton branches need no classifier."""
    specs = []
    for spec in node_plan(taxonomy, 2):
        ids = tuple(v for v in spec.global_ids if spec.stage != "variant" or v != heldout)
        if len(ids) > 1:
            specs.append(NodeSpec(spec.key, spec.stage, spec.parent_stage, spec.parent_id, ids))
    return tuple(specs)


class FlatTargets(Dataset):
    """99-way supervised targets with an explicit global-to-local map."""

    def __init__(self, base: AircraftDataset, ids: tuple[int, ...]) -> None:
        self.base, self.ids = base, ids
        self.local = {v: i for i, v in enumerate(ids)}
        if {base.taxonomy.ids(r)[0] for r in base.rows} != set(ids):
            raise ValueError("Missing/extra reduced FLAT class support")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        return {**sample, "variant_id": self.local[sample["variant_id"]]}


def local_path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Incremental artifact path escapes run")
    return result


def inventory(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p != root / "summary.json"}


def audit_inventory(root: Path) -> dict:
    summary = read_json(root / "summary.json")
    if summary.get("status") != "complete" or inventory(root) != summary["artifacts_sha256"]:
        raise ValueError("Incomplete/corrupt incremental artifacts")
    return summary


def fit_classifier(config: IncrementalConfig, model: nn.Module, train, val, *,
                   classes: int, metadata: dict, kind: str, output: Path,
                   base_train: AircraftDataset, encoder: nn.Module | None = None) -> Path:
    """Fit a reduced FLAT or one replacement; augmented views recomputed each epoch.

    Update wall time includes feature extraction, optimization, validation,
    checkpoint selection/export and exact reload. Initialization/loading and final
    test evaluation are outside this explicitly declared boundary.
    """
    contract = {**metadata, "kind": kind, "classes": classes, "recipe": asdict(config)}
    for previous in sorted(output.glob("*/summary.json")):
        summary = audit_inventory(previous.parent)
        if object_hash(read_json(previous.parent / "contract.json")) != object_hash(contract):
            raise ValueError("Completed fit contract mismatch")
        return previous.parent
    path = output / unique_id("fit")
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "contract.json", contract)
    seed_everything(metadata["seed"])
    device = torch.device(config.device)
    model.to(device)
    initialization_sha256 = state_hash(model)
    fingerprint = state_hash(encoder) if encoder is not None else None
    if encoder is not None:
        encoder.to(device).eval()
        assert_frozen(encoder, fingerprint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.head_learning_rate if encoder is not None
                                 else config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(metadata["seed"])
    workers = config.workers if len(train) >= 256 else 0
    selection, history, best = CheckpointSelection(), [], None
    synchronize(device)
    start = time.perf_counter()
    val_features = extract_features(encoder, val, device=device, batch_size=config.batch_size,
                                   workers=workers) if encoder is not None else None
    for epoch in range(1, config.epochs + 1):
        base_train.set_epoch(epoch - 1)
        if encoder is not None:
            features = extract_features(encoder, train, device=device, batch_size=config.batch_size, workers=workers)
            train_values = FeatureDataset(features, list(range(len(train))), train.targets)
            val_values = FeatureDataset(val_features, list(range(len(val))), val.targets)
            assert_frozen(encoder, fingerprint)
        else:
            train_values, val_values = train, val
        loaders = [DataLoader(values, batch_size=config.batch_size, shuffle=is_train,
                              generator=generator if is_train else None,
                              num_workers=0 if encoder is not None else workers, persistent_workers=False)
                   for values, is_train in ((train_values, True), (val_values, False))]
        training = run_epoch(model, loaders[0], device, optimizer, classes=classes)
        validation = run_epoch(model, loaders[1], device, classes=classes)
        improved = selection.observe(epoch, validation["macro_recall"])
        if improved:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in training.items()},
               **{f"val_{k}": v for k, v in validation.items()}, "selected": improved}
        history.append(row)
        write_json(path / f"epoch-{epoch:03d}.json", row)
        print(f"{path} epoch {epoch}: val macro {validation['macro_recall']:.5f}", flush=True)
        if selection.stale >= config.patience:
            break
    model.load_state_dict(best)
    model.eval()
    selected = {**metadata, "best_epoch": selection.best_epoch, "validation_macro_recall": selection.best_score,
                "initialization_state_sha256": initialization_sha256}
    save_checkpoint(path / "model.pt", model, selected, kind=kind)
    loaded, _ = load_checkpoint(path / "model.pt", **{k: metadata[k] for k in
                         ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256")}, kind=kind)
    loaded.to(device)
    fixed = val_features[:1].to(device) if encoder is not None else val[0]["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        torch.testing.assert_close(model(fixed), loaded(fixed), rtol=0, atol=0)
    if encoder is not None:
        assert_frozen(encoder, fingerprint)
    synchronize(device)
    elapsed = time.perf_counter() - start
    write_csv(path / "history.csv", history)
    resources = {"update_wall_seconds": elapsed, "trainable_parameters": sum(p.numel() for p in model.parameters()),
                 "replacement_bytes": (path / "model.pt").stat().st_size,
                 "optimizer_steps": sum(h["train_optimizer_steps"] for h in history),
                 "training_image_exposures": sum(h["train_images"] for h in history),
                 "train_images": len(train), "validation_images": len(val),
                 "timing_scope": "feature extraction if shared + epochs + selection + export + exact reload; initialization/loading excluded"}
    write_json(path / "resources.json", resources)
    write_json(path / "summary.json", {"status": "complete", "best_epoch": selection.best_epoch,
               "validation_macro_recall": selection.best_score, "epochs_completed": len(history),
               "exact_reload_logits": True, "frozen_encoder_unchanged": encoder is not None,
               "resources": resources, "artifacts_sha256": inventory(path)})
    return path


def expand_classifier(model: nn.Module, old_ids: tuple[int, ...], new_ids: tuple[int, ...], *, seed: int) -> nn.Module:
    """Warm start both architectures; preserve old rows and initialize the new row.

    This mutates only a freshly loaded affected specialist, never a live bundle.
    The single-child case has no old classifier and is initialized by the caller.
    """
    if not set(old_ids) < set(new_ids) or len(set(new_ids) - set(old_ids)) != 1:
        raise ValueError("Expected exactly one introduced output")
    seed_everything(seed)
    parent = model.classifier if hasattr(model, "classifier") else model
    old = parent[-1]
    if not isinstance(old, nn.Linear) or old.out_features != len(old_ids):
        raise ValueError("Affected specialist output mismatch")
    replacement = nn.Linear(old.in_features, len(new_ids)).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        for old_index, v in enumerate(old_ids):
            new_index = new_ids.index(v)
            replacement.weight[new_index].copy_(old.weight[old_index])
            replacement.bias[new_index].copy_(old.bias[old_index])
    parent[-1] = replacement
    return model


def fit_shared_initial(config: IncrementalConfig, data: AircraftData, preprocessing: Preprocessing,
                       specs: tuple[NodeSpec, ...], encoder: nn.Module, metadata: dict, output: Path) -> Path:
    """Independent optimizers/selection with one augmented extraction per epoch."""
    contract = {**metadata, "specs": [s.to_dict() for s in specs], "recipe": asdict(config)}
    for previous in sorted(output.glob("*/summary.json")):
        audit_inventory(previous.parent)
        if object_hash(read_json(previous.parent / "contract.json")) != object_hash(contract):
            raise ValueError("Initial shared contract mismatch")
        return previous.parent
    path = output / unique_id("heads")
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "contract.json", contract)
    device = torch.device(config.device)
    encoder.to(device).eval()
    fingerprint = state_hash(encoder)
    assert_frozen(encoder, fingerprint)
    train = AircraftDataset(data, "train", preprocessing, training=True, seed=metadata["seed"])
    val = AircraftDataset(data, "val", preprocessing)
    states = {}
    for spec in specs:
        seed = int(object_hash([metadata["seed"], spec.key])[:8], 16)
        seed_everything(seed)
        head = build_head(len(spec.global_ids), config.hidden_dimension).to(device)
        states[spec.key] = {"head": head, "optimizer": torch.optim.AdamW(head.parameters(), lr=config.head_learning_rate,
                           weight_decay=config.weight_decay), "selection": CheckpointSelection(), "history": [],
                           "train": LocalDataset(train, spec, split="train"), "val": LocalDataset(val, spec, split="val"),
                           "generator": torch.Generator().manual_seed(seed), "done": False, "best": None}
    synchronize(device)
    started = time.perf_counter()
    val_features = extract_features(encoder, val, device=device, batch_size=config.batch_size, workers=config.workers)
    for epoch in range(1, config.epochs + 1):
        train.set_epoch(epoch - 1)
        features = extract_features(encoder, train, device=device, batch_size=config.batch_size, workers=config.workers)
        assert_frozen(encoder, fingerprint)
        for spec in specs:
            state = states[spec.key]
            if state["done"]:
                continue
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
            state["done"] = state["selection"].stale >= config.patience or epoch == config.epochs
            node = path / spec.key
            node.mkdir(exist_ok=True)
            write_json(node / f"epoch-{epoch:03d}.json", row)
            if state["done"]:
                state["head"].load_state_dict(state["best"])
                state["head"].eval()
                selected = {**metadata, "node": spec.to_dict(), "backbone_sha256": fingerprint,
                            "input_dimension": 1280, "hidden_dimension": config.hidden_dimension,
                            "best_epoch": state["selection"].best_epoch,
                            "validation_macro_recall": state["selection"].best_score,
                            "training_image_ids": state["train"].image_ids, "selection_image_ids": state["val"].image_ids}
                save_checkpoint(node / "model.pt", state["head"], selected, kind="shared-head")
                loaded, _ = load_checkpoint(node / "model.pt", **{k: metadata[k] for k in
                            ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256")}, kind="shared-head")
                with torch.inference_mode():
                    fixed = val_features[state["val"].indices[:1]].to(device)
                    torch.testing.assert_close(state["head"](fixed), loaded.to(device)(fixed), rtol=0, atol=0)
                write_csv(node / "history.csv", state["history"])
                write_json(node / "selection.json", {"best_epoch": selected["best_epoch"],
                    "validation_macro_recall": selected["validation_macro_recall"], "exact_reload_logits": True,
                    "trainable_parameters": sum(p.numel() for p in state["head"].parameters()),
                    "optimizer_steps": sum(h["train_optimizer_steps"] for h in state["history"]),
                    "training_image_exposures": sum(h["train_images"] for h in state["history"])})
        print(f"Initial shared s{metadata['seed']} epoch {epoch}: {sum(s['done'] for s in states.values())}/{len(states)} heads", flush=True)
        if all(s["done"] for s in states.values()):
            break
    assert_frozen(encoder, fingerprint)
    synchronize(device)
    write_json(path / "summary.json", {"status": "complete", "frozen_encoder_unchanged": True,
               "training_wall_seconds": time.perf_counter() - started, "artifacts_sha256": inventory(path)})
    return path


def checkpoint_entry(root: Path, target: Path, kind: str) -> dict:
    return {"path": target.relative_to(root).as_posix(), "kind": kind, "sha256": file_hash(target),
            "state_sha256": read_json(target.with_suffix(".json"))["state_sha256"]}


def load_entry(root: Path, entry: dict, hashes: dict) -> tuple[nn.Module, dict]:
    target = local_path(root, entry["path"])
    if file_hash(target) != entry["sha256"]:
        raise ValueError("Incremental checkpoint checksum mismatch")
    model, metadata = load_checkpoint(target, **hashes, kind=entry["kind"])
    if state_hash(model) != entry["state_sha256"]:
        raise ValueError("Incremental state checksum mismatch")
    return model, metadata


def load_system(root: Path, seed: int, architecture: str, stage: str) -> HierarchicalAdapter:
    """Load the exact before/after inventory; old artifacts are never overwritten."""
    from .taxonomy import Taxonomy
    if architecture not in {"IND-2", "FT-FROZEN-2"} or stage not in {"before", "after"}:
        raise ValueError("Unknown incremental architecture/stage")
    bundle = read_json(root / f"seed-{seed}" / "bundle.json")
    taxonomy = Taxonomy.from_dict(read_json(root / "taxonomy.json"))
    hashes = read_json(root / "compatibility.json")
    if object_hash(taxonomy.to_dict()) != hashes["taxonomy_sha256"]:
        raise ValueError("Incremental bundle taxonomy mismatch")
    entries = dict(bundle[architecture]["initial"])
    if stage == "after":
        entries[bundle["affected_key"]] = bundle[architecture]["replacement"]
    nodes = {}
    for key, entry in entries.items():
        model, meta = load_entry(root, entry, hashes)
        spec = NodeSpec.from_dict(meta["node"])
        if (key != spec.adapter_key or meta["seed"] != seed or meta["run_id"] != root.name
                or entry["kind"] != ("shared-head" if architecture == "FT-FROZEN-2" else "IND-node")):
            raise ValueError("Incremental node identity mismatch")
        if architecture == "FT-FROZEN-2" and meta["backbone_sha256"] != bundle["encoder"]["state_sha256"]:
            raise ValueError("Incremental head/encoder version mismatch")
        if stage == "after" and key == bundle["affected_key"] and meta["initial_inventory_sha256"] != bundle["initial_inventory_sha256"]:
            raise ValueError("Replacement initial inventory mismatch")
        nodes[key] = LogitNode(model, spec.global_ids)
    encoder = None
    if architecture == "FT-FROZEN-2":
        encoder, encoder_meta = load_entry(root, bundle["encoder"], hashes)
        if (encoder_meta["seed"] != seed or encoder_meta["run_id"] != root.name
                or encoder_meta["parent_flat_sha256"] != bundle["flat"]["sha256"]):
            raise ValueError("Incremental encoder source mismatch")
    active = tuple(v for v in range(len(taxonomy.variants)) if v != bundle["heldout_variant_id"]) if stage == "before" else None
    return HierarchicalAdapter(taxonomy, nodes, backbone=encoder, fine_tuned=encoder is not None,
                               active_variant_ids=active)


def train_seed(root: Path, config: IncrementalConfig, data: AircraftData, preprocessing: Preprocessing,
               selection: dict, seed: int) -> None:
    path = root / f"seed-{seed}"
    if (path / "summary.json").exists():
        audit_inventory(path)
        return
    path.mkdir(exist_ok=True)
    seed_everything(seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reduced = supervised_data(data, selection["variant_id"])
    ids = tuple(v for v in range(len(data.taxonomy.variants)) if v != selection["variant_id"])
    hashes = compatibility(data, preprocessing)
    contract = {**hashes, "run_id": root.name, "seed": seed,
                "config_sha256": object_hash(asdict(config)), "source_hashes_sha256": file_hash(root / "source_hashes.json"),
                "heldout_variant_id": selection["variant_id"]}
    train = AircraftDataset(reduced, "train", preprocessing, training=True, seed=seed)
    val = AircraftDataset(reduced, "val", preprocessing)
    flat_meta = {**contract, "variant_ids": list(ids), "training_image_ids": [r.image_id for r in train.rows],
                 "selection_image_ids": [r.image_id for r in val.rows]}
    seed_everything(seed)
    flat_path = fit_classifier(config, build_flat(classes=len(ids)), FlatTargets(train, ids), FlatTargets(val, ids),
              classes=len(ids), metadata=flat_meta, kind="FLAT", output=path / "flat", base_train=train)
    flat, _ = load_checkpoint(flat_path / "model.pt", **hashes)
    encoder = FrozenEncoder(flat)
    encoder_path = path / "encoder.pt"
    if not encoder_path.exists():
        save_checkpoint(encoder_path, encoder, {**flat_meta, "parent_flat_sha256": file_hash(flat_path / "model.pt")},
                        kind="FT-FROZEN-encoder")
    else:
        loaded, _ = load_checkpoint(encoder_path, **hashes, kind="FT-FROZEN-encoder")
        if state_hash(loaded) != state_hash(encoder):
            raise ValueError("Resumed encoder mismatch")
    del flat
    specs = initial_specs(data.taxonomy, selection["variant_id"])
    independent_config = IndependentConfig(**{k: v for k, v in asdict(config).items()
                                             if k not in {"head_learning_rate", "hidden_dimension"}})
    independent = {s.adapter_key: checkpoint_entry(root, train_node(independent_config, s, train, val, contract,
                   path / "independent-initial") / "node.pt", "IND-node") for s in specs}
    shared_path = fit_shared_initial(config, reduced, preprocessing, specs, encoder, contract, path / "shared-initial")
    shared = {s.adapter_key: checkpoint_entry(root, shared_path / s.key / "model.pt", "shared-head") for s in specs}
    affected = next(s for s in node_plan(data.taxonomy, 2)
                    if s.stage == "variant" and s.parent_id == selection["manufacturer_id"])
    full_train = AircraftDataset(data, "train", preprocessing, training=True, seed=seed)
    full_val = AircraftDataset(data, "val", preprocessing)
    local_train, local_val = LocalDataset(full_train, affected, split="train"), LocalDataset(full_val, affected, split="val")
    # Freeze the entire pre-update artifact inventory before either replacement.
    protected = {"encoder": checkpoint_entry(root, encoder_path, "FT-FROZEN-encoder"),
                 "IND-2": independent, "FT-FROZEN-2": shared}
    protected_hash = object_hash(protected)
    replacements = {}
    for architecture, entries in (("IND-2", independent), ("FT-FROZEN-2", shared)):
        is_shared = architecture == "FT-FROZEN-2"
        seed_everything(seed)
        if affected.adapter_key in entries:
            model, old_meta = load_entry(root, entries[affected.adapter_key], hashes)
            old_ids = tuple(old_meta["node"]["global_ids"])
            expand_classifier(model, old_ids, affected.global_ids, seed=seed)
        else:
            model = build_head(len(affected.global_ids), config.hidden_dimension) if is_shared else build_flat(classes=len(affected.global_ids))
        metadata = {**contract, "node": affected.to_dict(), "training_image_ids": local_train.image_ids,
                    "selection_image_ids": local_val.image_ids, "update_policy": "warm-start existing output rows; random new row; replay all old branch training images",
                    "initial_inventory_sha256": protected_hash}
        if is_shared:
            metadata.update(backbone_sha256=state_hash(encoder), input_dimension=1280, hidden_dimension=config.hidden_dimension)
        kind = "shared-head" if is_shared else "IND-node"
        update = fit_classifier(config, model, local_train, local_val, classes=len(affected.global_ids), metadata=metadata,
                     kind=kind, output=path / f"update-{architecture}", base_train=full_train, encoder=encoder if is_shared else None)
        replacements[architecture] = checkpoint_entry(root, update / "model.pt", kind)
        del model
    # Re-read all original bytes after both fits, including the affected old node.
    for entries in (independent, shared, {"encoder": protected["encoder"]}):
        for entry in entries.values():
            if file_hash(local_path(root, entry["path"])) != entry["sha256"]:
                raise ValueError("Original checkpoint modified by update")
    assert_frozen(encoder, protected["encoder"]["state_sha256"])
    bundle = {"seed": seed, "heldout_variant_id": selection["variant_id"], "affected_key": affected.adapter_key,
              "encoder": protected["encoder"], "flat": checkpoint_entry(root, flat_path / "model.pt", "FLAT"),
              **{a: {"initial": entries, "replacement": replacements[a]} for a, entries in
                 (("IND-2", independent), ("FT-FROZEN-2", shared))},
              "original_artifacts_unchanged": True, "initial_inventory_sha256": protected_hash}
    write_json(path / "bundle.json", bundle)
    write_json(path / "summary.json", {"status": "complete", "seed": seed, "test_evaluated": False,
               "artifacts_sha256": inventory(path)})
    print(f"Completed incremental seed {seed}: {path}", flush=True)


def train(config: IncrementalConfig, *, resume: Path | None = None) -> Path:
    seed_everything(config.seeds[0])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, preprocessing = AircraftData.load(Path(config.data_root)), Preprocessing()
    print("Verifying all 10,000 original image checksums", flush=True)
    prepared = validate_preparation(data, preprocessing, Path(config.prepared_run))
    selection = select_introduction(data.taxonomy)
    repo = Path(__file__).resolve().parents[2]
    sources = {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted((repo / "src/cls_balance").rglob("*.py"))}
    if resume is None:
        root = Path(config.output_root) / unique_id("incremental")
        root.mkdir(parents=True, exist_ok=False)
        for name, value in {"config.resolved.json": asdict(config), "selection.json": selection,
                            "taxonomy.json": data.taxonomy.to_dict(), "preprocessing.json": preprocessing.to_dict(),
                            "compatibility.json": compatibility(data, preprocessing), "source_hashes.json": sources,
                            "preparation.json": prepared,
                            "dataset_manifest_hashes.json": {s: object_hash(data.manifest(s)) for s in data.splits}}.items():
            write_json(root / name, value)
        env = environment(repo)
        env.update(device=config.device, precision="fp32", cpu_threads=config.cpu_threads,
                   runtime_policy="all-resident; sequential training; TF32 disabled; one frozen shared encoder",
                   power_policy="OS/driver defaults; clocks not locked")
        write_json(root / "environment.json", env)
        for name, digest in sources.items():
            content = (repo / name).read_bytes()
            import hashlib
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Training source changed during snapshot")
            target = root / "training_sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
    else:
        root = resume.resolve()
        if (root / "summary.json").exists():
            raise ValueError("Cannot resume a completed training suite")
        for name, value in (("config.resolved.json", asdict(config)), ("source_hashes.json", sources),
                            ("selection.json", selection), ("preparation.json", prepared)):
            if object_hash(read_json(root / name)) != object_hash(value):
                raise ValueError("Resume config/source/data mismatch")
        current, previous = environment(repo), read_json(root / "environment.json")
        if any(current[k] != previous[k] for k in ("torch", "cuda_runtime", "gpus", "packages", "platform")):
            raise ValueError("Resume environment mismatch")
    print(f"Incremental suite {root}; introduce {selection['variant']} / {selection['manufacturer']}", flush=True)
    for seed in config.seeds:
        train_seed(root, config, data, preprocessing, selection, seed)
    write_json(root / "summary.json", {"status": "complete", "seeds": config.seeds, "test_evaluated": False,
               "artifacts_sha256": inventory(root)})
    return root


def audit_training(root: Path, data: AircraftData) -> dict:
    """Re-derive exclusion, branch IDs, source encoder and replacement contracts.

    This audits declared provenance and saved tensor bytes; it does not claim
    cryptographic proof of historical training execution.
    """
    import csv

    summary = audit_inventory(root)
    if (root / "training_sources").exists():
        snapshots = {p.relative_to(root / "training_sources").as_posix(): file_hash(p)
                     for p in (root / "training_sources").rglob("*.py")}
        if snapshots != read_json(root / "source_hashes.json"):
            raise ValueError("Exact training source snapshot mismatch")
    preprocessing = Preprocessing()
    hashes = compatibility(data, preprocessing)
    if read_json(root / "compatibility.json") != hashes:
        raise ValueError("Incremental dataset/preprocessing compatibility mismatch")
    selection = select_introduction(data.taxonomy)
    if read_json(root / "selection.json") != selection:
        raise ValueError("Incremental training-only selection mismatch")
    reduced = supervised_data(data, selection["variant_id"])
    specs = {s.adapter_key: s for s in initial_specs(data.taxonomy, selection["variant_id"])}
    affected = next(s for s in node_plan(data.taxonomy, 2) if s.adapter_key == f"variant:{selection['manufacturer_id']}")
    expected_ids = tuple(v for v in range(len(data.taxonomy.variants)) if v != selection["variant_id"])
    config = read_json(root / "config.resolved.json")
    if summary["seeds"] != config["seeds"]:
        raise ValueError("Incremental seed inventory mismatch")
    checked = 0
    for seed in config["seeds"]:
        bundle = read_json(root / f"seed-{seed}/bundle.json")
        protected = {"encoder": bundle["encoder"], "IND-2": bundle["IND-2"]["initial"],
                     "FT-FROZEN-2": bundle["FT-FROZEN-2"]["initial"]}
        if (bundle["initial_inventory_sha256"] != object_hash(protected)
                or bundle["seed"] != seed or bundle["heldout_variant_id"] != selection["variant_id"]
                or bundle["affected_key"] != affected.adapter_key or not bundle["original_artifacts_unchanged"]):
            raise ValueError("Initial immutable inventory mismatch")
        initial_features = {read_json(local_path(root, entry["path"]).with_suffix(".json"))["metadata"]["initialization_features_sha256"]
                            for entry in bundle["IND-2"]["initial"].values()}
        if len(initial_features) != 1:
            raise ValueError("Independent initial CNN feature initializations differ")
        entries = [(bundle["flat"], None, reduced), (bundle["encoder"], None, reduced)]
        for architecture in ("IND-2", "FT-FROZEN-2"):
            if set(bundle[architecture]["initial"]) != set(specs):
                raise ValueError("Initial specialist inventory mismatch")
            entries += [(entry, specs[key], reduced) for key, entry in bundle[architecture]["initial"].items()]
            entries.append((bundle[architecture]["replacement"], affected, data))
        for entry, spec, view in entries:
            model, meta = load_entry(root, entry, hashes)
            del model
            checked += 1
            expected = {}
            for split, key in (("train", "training_image_ids"), ("val", "selection_image_ids")):
                rows = view.splits[split]
                if spec is not None and spec.parent_stage is not None:
                    parent_index = {"manufacturer": 2, "family": 1}[spec.parent_stage]
                    rows = [r for r in rows if view.taxonomy.ids(r)[parent_index] == spec.parent_id]
                expected[key] = [r.image_id for r in rows]
            if (any(meta[key] != value for key, value in expected.items()) or meta["seed"] != seed
                    or meta["run_id"] != root.name or meta["config_sha256"] != object_hash(config)
                    or meta["source_hashes_sha256"] != file_hash(root / "source_hashes.json")
                    or meta["heldout_variant_id"] != selection["variant_id"]):
                raise ValueError("Supervised exclusion/provenance mismatch or test leakage")
            if spec is None:
                if tuple(meta["variant_ids"]) != expected_ids:
                    raise ValueError("Initial backbone label exclusion mismatch")
            elif NodeSpec.from_dict(meta["node"]) != spec:
                raise ValueError("Initial/update global logit map mismatch")
            if entry["kind"] == "FT-FROZEN-encoder":
                if meta["parent_flat_sha256"] != bundle["flat"]["sha256"]:
                    raise ValueError("Reduced encoder source mismatch")
                continue
            if entry["kind"] == "shared-head" and meta["backbone_sha256"] != bundle["encoder"]["state_sha256"]:
                raise ValueError("Updated/shared encoder compatibility mismatch")
            target = local_path(root, entry["path"])
            with (target.parent / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            best = max(history, key=lambda h: float(h["val_macro_recall"]))
            if (int(best["epoch"]) != meta["best_epoch"]
                    or float(best["val_macro_recall"]) != meta["validation_macro_recall"]
                    or any(int(h["train_images"]) != len(expected["training_image_ids"])
                           or int(h["val_images"]) != len(expected["selection_image_ids"]) for h in history)):
                raise ValueError("Validation selection or supervised exposure mismatch")
            if view is data and meta.get("initial_inventory_sha256") != object_hash(protected):
                raise ValueError("Replacement not bound to initial inventory")
    return {"training_run_id": root.name, "checkpoints_verified": checked,
            "heldout_excluded_from_all_initial_train_and_val": True,
            "all_initial_artifacts_unchanged": True, "test_leakage": False,
            "complete_official_test_images": len(data.splits["test"])}


def evaluate(root: Path, *, device: str | None = None) -> Path:
    """Freeze all seeds first, then evaluate both stages on ALL official test IDs."""
    audit_inventory(root)
    config = IncrementalConfig(**{**read_json(root / "config.resolved.json"),
                                  "seeds": tuple(read_json(root / "config.resolved.json")["seeds"])})
    data, preprocessing = AircraftData.load(Path(config.data_root)), Preprocessing()
    validate_preparation(data, preprocessing, Path(config.prepared_run))
    audit = audit_training(root, data)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = root.parent / unique_id("incremental-evaluation")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "training_source.json", {"path": str(root.resolve()), "run_id": root.name,
               "summary_sha256": file_hash(root / "summary.json")})
    write_json(output / "training_audit.json", audit)
    repo = Path(__file__).resolve().parents[2]
    sources = {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted((repo / "src/cls_balance").rglob("*.py"))}
    write_json(output / "source_hashes.json", sources)
    for name, digest in sources.items():
        content = (repo / name).read_bytes()
        import hashlib
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Inference source changed during snapshot")
        target = output / "inference_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    reduced = supervised_data(data, read_json(root / "selection.json")["variant_id"])
    hashes = compatibility(data, preprocessing)
    entries = []
    for seed in config.seeds:
        for architecture in ("IND-2", "FT-FROZEN-2"):
            for stage in ("before", "after"):
                model = load_system(root, seed, architecture, stage)
                provenance = ModelProvenance(f"{root.name}/s{seed}/{architecture}/{stage}", model_hash(model),
                    hashes["taxonomy_sha256"], preprocessing.sha256, seed,
                    tuple(r.image_id for r in (reduced if stage == "before" else data).splits["train"]),
                    tuple(r.image_id for r in (reduced if stage == "before" else data).splits["val"]))
                evaluator = Evaluator(data, preprocessing, provenance, device=device or config.device)
                for routing in ("predicted", "oracle"):
                    result = evaluator.evaluate(model, split="test", routing=routing)
                    result.metadata.update(inference_source_sha256=object_hash(sources),
                                           training_summary_sha256=file_hash(root / "summary.json"))
                    saved = result.save(output)
                    entries.append({"seed": seed, "architecture": architecture, "stage": stage, "routing": routing,
                                    "path": saved.relative_to(output).as_posix(), "audit": audit_run(saved)})
                    print(f"Evaluated s{seed} {architecture} {stage} {routing}: {saved}", flush=True)
                del model, evaluator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_json(output / "entries.json", entries)
    write_json(output / "summary.json", {"status": "complete", "training_run_id": root.name,
               "artifacts_sha256": inventory(output)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--config", type=Path, default=Path("configs/incremental.yaml"))
    fitting.add_argument("--resume", type=Path)
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--suite", type=Path, required=True)
    scoring.add_argument("--device")
    args = parser.parse_args()
    result = train(read_config(args.config), resume=args.resume) if args.command == "train" else evaluate(args.suite, device=args.device)
    print(f"Completed: {result}", flush=True)


if __name__ == "__main__":
    main()
