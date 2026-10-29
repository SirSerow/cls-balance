"""Optional training-only feature caches, bound to exact encoder and transforms.

Augmented caches contain a separate, reproducibly seeded view for each epoch.
Validation features are held in memory. Deployment always consumes real images.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import file_hash, object_hash, write_json


def extract_features(backbone: nn.Module, dataset, *, device: torch.device,
                     batch_size: int, workers: int) -> torch.Tensor:
    """Extract every image in manifest order with no gradient or dropped samples."""
    backbone.eval()
    rows = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=workers, persistent_workers=False):
            value = backbone(batch["image"].to(device)).detach().cpu().float()
            if value.ndim != 2 or not torch.isfinite(value).all():
                raise ValueError("Invalid frozen features")
            rows.append(value)
    features = torch.cat(rows)
    if len(features) != len(dataset):
        raise ValueError("Incomplete feature extraction")
    return features


def training_features(backbone: nn.Module, dataset, *, contract: dict, cache_root: Path | None,
                      device: torch.device, batch_size: int, workers: int) -> torch.Tensor:
    """Read or write an immutable training cache; reject corruption and test IDs."""
    if (not dataset.training or contract.get("split") != "train"
            or [r.image_id for r in dataset.rows] != contract.get("image_ids")):
        raise ValueError("Feature cache accepts exact training IDs only")
    if not {"backbone_sha256", "preprocessing_sha256", "dataset_manifest_sha256", "augmentation", "seed", "epoch"} <= contract.keys():
        raise ValueError("Incomplete cache version contract")
    from .checkpoints import state_hash
    if (state_hash(backbone) != contract["backbone_sha256"]
            or dataset.preprocessing.sha256 != contract["preprocessing_sha256"]
            or dataset.seed != contract["seed"] or dataset.epoch != contract["epoch"]):
        raise ValueError("Feature cache encoder/transform/seed/epoch mismatch")
    if cache_root is None:
        return extract_features(backbone, dataset, device=device, batch_size=batch_size, workers=workers)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / (object_hash(contract) + ".pt")
    manifest = path.with_suffix(".json")
    if path.exists() or manifest.exists():
        import json
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["contract"] != contract or saved["file_sha256"] != file_hash(path):
            raise ValueError("Feature cache version/checksum mismatch")
        features = torch.load(path, weights_only=True, map_location="cpu")
        if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32
                or features.ndim != 2 or list(features.shape) != saved["shape"]
                or len(features) != len(dataset) or not torch.isfinite(features).all()):
            raise ValueError("Invalid cached features")
        return features
    features = extract_features(backbone, dataset, device=device, batch_size=batch_size, workers=workers)
    with path.open("xb") as stream:
        torch.save(features, stream)
    write_json(manifest, {"contract": contract, "shape": list(features.shape), "file_sha256": file_hash(path)})
    return features
