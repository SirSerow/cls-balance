"""Train FLAT on official train only; select on complete validation macro recall.

Run ``python -m cls_balance.train --config configs/flat.yaml``. No test inference,
embedding cache, mixed precision, scheduler or training subset is exposed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .artifacts import environment, file_hash, object_hash, seed_everything, write_json
from .checkpoints import load_checkpoint, save_checkpoint, state_hash
from .data import AircraftData, AircraftDataset, SPLITS
from .evaluate import Evaluator, ModelProvenance, audit_run, model_hash
from .infer import FlatAdapter, LogitNode
from .models.flat import FrozenEncoder, build_flat
from .prepare import write_csv
from .transforms import Preprocessing


@dataclass(frozen=True)
class TrainingConfig:
    data_root: str
    prepared_run: str
    output_root: str
    seeds: tuple[int, ...] = (42, 43, 44)
    device: str = "cpu"
    epochs: int = 30
    patience: int = 7
    batch_size: int = 16
    workers: int = 2
    cpu_threads: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds)):
            raise ValueError("Seeds must be distinct uint32 integers")
        for key in ("epochs", "patience", "batch_size", "cpu_threads"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("Invalid workers")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("Invalid AdamW hyperparameters")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU/CUDA FP32 is supported")


def read_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("data_root", "prepared_run", "output_root"):
        raw[key] = str((path.resolve().parent / raw[key]).resolve())
    raw["seeds"] = tuple(raw.get("seeds", (42, 43, 44)))
    return TrainingConfig(**raw)


def validate_preparation(data: AircraftData, preprocessing: Preprocessing, prepared: Path) -> dict:
    """Bind actual annotations and image bytes to the completed Phase 1 audit."""
    import csv

    summary = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "validated" or summary.get("edition") != "torchvision_100_variants"
            or summary["preprocessing_sha256"] != preprocessing.sha256
            or summary["taxonomy_sha256"] != object_hash(data.taxonomy.to_dict())
            or summary["source_annotations_sha256"] != object_hash(data.source_hashes)):
        raise ValueError("Prepared dataset/preprocessing/taxonomy mismatch")
    for split in SPLITS:
        if file_hash(prepared / f"manifest_{split}.csv") != summary["manifest_sha256"][split]:
            raise ValueError("Prepared manifest checksum mismatch")
        with (prepared / f"manifest_{split}.csv").open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        expected = [{k: str(v) for k, v in row.items()} for row in data.manifest(split)]
        if saved != expected:
            raise ValueError("Prepared manifests differ from current official annotations")
    checksums = prepared / "image_checksums.csv"
    if file_hash(checksums) != summary["image_checksums_sha256"]:
        raise ValueError("Prepared image checksum inventory mismatch")
    with checksums.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_ids = {r.image_id for split in data.splits.values() for r in split}
    if len(rows) != len(expected_ids) or {r["image_id"] for r in rows} != expected_ids:
        raise ValueError("Image checksum inventory must cover all official IDs exactly once")
    for row in rows:
        if file_hash(data.image_path(row["image_id"])) != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['image_id']}")
    return summary


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class CheckpointSelection:
    """Strict validation-only improvement with earliest-tie selection."""

    best_score: float = -1.0
    best_epoch: int = 0
    stale: int = 0

    def observe(self, epoch: int, validation_macro_recall: float) -> bool:
        if epoch < 1 or not math.isfinite(validation_macro_recall) or not 0 <= validation_macro_recall <= 1:
            raise ValueError("Invalid validation selection observation")
        if validation_macro_recall > self.best_score:
            self.best_score, self.best_epoch, self.stale = validation_macro_recall, epoch, 0
            return True
        self.stale += 1
        return False


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None, *, classes: int = 100) -> dict:
    """Process every supplied sample; fixed-class macro recall rejects missing support."""
    model.train(optimizer is not None)
    confusion = torch.zeros((classes, classes), dtype=torch.int64, device=device)
    total_loss = torch.zeros((), device=device)
    count = steps = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["variant_id"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training/validation loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
                steps += 1
            total_loss += loss.detach() * targets.numel()
            # bincount on CUDA is nondeterministic; integer scatter_add is deterministic.
            index = targets * classes + logits.detach().argmax(1)
            confusion.view(-1).scatter_add_(0, index, torch.ones_like(index))
            count += targets.numel()
    synchronize(device)
    elapsed = time.perf_counter() - started
    matrix = confusion.cpu()
    support = matrix.sum(1)
    if not count or not torch.all(support > 0):
        raise ValueError("Missing class support in epoch")
    return {"loss": total_loss.item() / count, "macro_recall": (matrix.diag().double() / support).mean().item(),
            "top1": matrix.diag().sum().item() / count, "images": count,
            "optimizer_steps": steps, "wall_seconds": elapsed}


def train_seed(config: TrainingConfig, data: AircraftData, preprocessing: Preprocessing,
               prepared: dict, seed: int) -> Path:
    """One immutable run; summary is written only after reload and evaluation checks."""
    seed_everything(seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but this PyTorch environment has no CUDA device")
    run_id = datetime.now(timezone.utc).strftime(f"flat-s{seed}-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    output = Path(config.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    resolved = {**asdict(config), "seed": seed, "architecture": "FLAT", "precision": "fp32",
                "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                "selection": "validation variant macro recall; strict improvement; earliest tie",
                "sampling": "shuffle train once per epoch; no replacement; drop_last=False",
                "initialization": "EfficientNet_B0_Weights.IMAGENET1K_V1", "scheduler": None,
                "early_stopping": "consecutive epochs without strict macro-recall improvement"}
    write_json(output / "config.resolved.json", resolved)
    env = environment(Path(__file__).resolve().parents[2])
    env.update(device=str(device), runtime_policy="all-resident FLAT; FP32; TF32 disabled",
               cpu_threads=config.cpu_threads, power_policy="OS/driver defaults; clocks not locked")
    write_json(output / "environment.json", env)
    repo = Path(__file__).resolve().parents[2]
    sources = [p for p in (repo / "src/cls_balance").rglob("*.py")]
    write_json(output / "source_hashes.json", {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted(sources)})
    write_json(output / "taxonomy.json", data.taxonomy.to_dict())
    write_json(output / "preprocessing.json", preprocessing.to_dict())
    manifests = {s: object_hash(data.manifest(s)) for s in SPLITS}
    write_json(output / "dataset_manifest_hashes.json", {"canonical_records": manifests,
               "prepared_run_id": prepared["run_id"], "prepared_summary_sha256": object_hash(prepared)})
    contract = {"run_id": run_id, "seed": seed, "config_sha256": object_hash(resolved),
                "taxonomy_sha256": object_hash(data.taxonomy.to_dict()),
                "preprocessing_sha256": preprocessing.sha256,
                "dataset_manifest_sha256": object_hash(manifests),
                "training_image_ids": [r.image_id for r in data.splits["train"]],
                "selection_image_ids": [r.image_id for r in data.splits["val"]]}
    train = AircraftDataset(data, "train", preprocessing, training=True, seed=seed)
    val = AircraftDataset(data, "val", preprocessing)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator,
                              num_workers=config.workers, persistent_workers=False)
    val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.workers, persistent_workers=False)
    model = build_flat().to(device)
    initialization_sha256 = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict] = []
    selection = CheckpointSelection()
    best_state = None
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train.set_epoch(epoch - 1)
        training = run_epoch(model, train_loader, device, optimizer)
        validation = run_epoch(model, val_loader, device)
        improved = selection.observe(epoch, validation["macro_recall"])
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in training.items()},
               **{f"val_{k}": v for k, v in validation.items()}, "selected": improved}
        history.append(row)
        write_json(output / f"epoch-{epoch:03d}.json", row)
        print(json.dumps({"run_id": run_id, **row}), flush=True)
        if selection.stale >= config.patience:
            break
    synchronize(device)
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    metadata = {**contract, "best_epoch": selection.best_epoch, "validation_macro_recall": selection.best_score,
                "initialization_state_sha256": initialization_sha256,
                "source_hashes_sha256": file_hash(output / "source_hashes.json")}
    full_manifest = save_checkpoint(output / "flat.pt", model, metadata, kind="FLAT")
    encoder = FrozenEncoder(model)
    encoder_manifest = save_checkpoint(output / "encoder.pt", encoder,
                      {**metadata, "parent_flat_sha256": full_manifest["file_sha256"]}, kind="FT-FROZEN-encoder")
    expected = {k: contract[k] for k in ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256")}
    reloaded, _ = load_checkpoint(output / "flat.pt", **expected)
    reloaded_encoder, _ = load_checkpoint(output / "encoder.pt", **expected, kind="FT-FROZEN-encoder")
    fixed = val[0]["image"].unsqueeze(0).to(device)
    reloaded.to(device)
    reloaded_encoder.to(device)
    with torch.inference_mode():
        torch.testing.assert_close(model(fixed), reloaded(fixed), rtol=0, atol=0)
        torch.testing.assert_close(encoder(fixed), reloaded_encoder(fixed), rtol=0, atol=0)
    del encoder, reloaded_encoder, model, optimizer, best_state
    adapter = FlatAdapter(data.taxonomy, LogitNode(reloaded, tuple(range(100))))
    provenance = ModelProvenance(run_id, model_hash(adapter), contract["taxonomy_sha256"],
                preprocessing.sha256, seed, tuple(contract["training_image_ids"]), tuple(contract["selection_image_ids"]))
    evaluator = Evaluator(data, preprocessing, provenance, device=str(device))
    evaluations = {}
    for split in ("train", "val"):
        path = evaluator.evaluate(adapter, split=split).save(output / "evaluations")
        evaluations[split] = {"path": path.relative_to(output).as_posix(), "audit": audit_run(path)}
    write_csv(output / "history.csv", history)
    resources = {"training_wall_seconds": training_seconds,
                 "training_gpu_hours": training_seconds / 3600 if device.type == "cuda" else None,
                 "optimizer_steps": sum(r["train_optimizer_steps"] for r in history),
                 "training_image_exposures": sum(r["train_images"] for r in history),
                 "trainable_parameters": sum(p.numel() for p in reloaded.parameters()),
                 "deployment_bytes": (output / "flat.pt").stat().st_size,
                 "encoder_bytes": (output / "encoder.pt").stat().st_size,
                 "timing_scope": "epoch training+validation+selection; excludes initialization and final evaluation"}
    write_json(output / "resources.json", resources)
    write_json(output / "checkpoint_manifest.json", {"flat": full_manifest, "encoder": encoder_manifest,
               "exact_reload_logits": True, "exact_reload_embeddings": True})
    write_json(output / "summary.json", {"schema_version": 1, "status": "complete", **metadata,
               "epochs_completed": len(history), "stopped_early": len(history) < config.epochs,
               "evaluations": evaluations, "resources": resources, "test_evaluated": False,
               "artifacts_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(f"Completed {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/flat.yaml"))
    args = parser.parse_args()
    config = read_config(args.config)
    data = AircraftData.load(Path(config.data_root), download=False)
    preprocessing = Preprocessing()
    print("Verifying complete prepared dataset and image checksums", flush=True)
    prepared = validate_preparation(data, preprocessing, Path(config.prepared_run))
    for seed in config.seeds:
        train_seed(config, data, preprocessing, prepared, seed)


if __name__ == "__main__":
    main()
