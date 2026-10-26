"""Immutable, hash-checked FLAT and encoder artifacts (no pretrained download)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .artifacts import canonical_json, file_hash, object_hash, write_json
from .models.flat import FrozenEncoder, build_flat


def state_hash(module: nn.Module) -> str:
    """Hash names, shapes, dtypes and tensor bytes, independently of serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(canonical_json([name, str(tensor.dtype), list(tensor.shape)]))
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, model: nn.Module, metadata: dict, *, kind: str) -> dict:
    """Write once; an adjacent manifest binds file bytes and semantic metadata."""
    if kind not in {"FLAT", "FT-FROZEN-encoder", "IND-node"}:
        raise ValueError("Unknown checkpoint kind")
    payload = {"schema_version": 1, "kind": kind, "metadata": metadata,
               "metadata_sha256": object_hash(metadata), "state_sha256": state_hash(model),
               "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    with path.open("xb") as stream:
        torch.save(payload, stream)
    manifest = {k: v for k, v in payload.items() if k != "state_dict"}
    manifest["file_sha256"] = file_hash(path)
    write_json(path.with_suffix(".json"), manifest)
    return manifest


def load_checkpoint(path: Path, *, taxonomy_sha256: str, preprocessing_sha256: str,
                    dataset_manifest_sha256: str, kind: str = "FLAT") -> tuple[nn.Module, dict]:
    """Reject corrupted/incompatible checkpoints before using their predictions."""
    import json

    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["file_sha256"] != file_hash(path):
        raise ValueError("Checkpoint file checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema_version") != 1 or payload.get("kind") != kind
            or kind not in {"FLAT", "FT-FROZEN-encoder", "IND-node"}
            or payload["metadata_sha256"] != object_hash(payload["metadata"])
            or any(manifest.get(k) != v for k, v in payload.items() if k != "state_dict")):
        raise ValueError("Checkpoint metadata/kind mismatch")
    metadata = payload["metadata"]
    for key, expected in (("taxonomy_sha256", taxonomy_sha256),
                          ("preprocessing_sha256", preprocessing_sha256),
                          ("dataset_manifest_sha256", dataset_manifest_sha256)):
        if metadata.get(key) != expected:
            raise ValueError(f"Checkpoint {key} mismatch")
    if kind == "IND-node":
        from .models.cascade import NodeSpec

        spec = NodeSpec.from_dict(metadata["node"])
        flat = build_flat(pretrained=False, classes=len(spec.global_ids))
    else:
        flat = build_flat(pretrained=False)
    model = FrozenEncoder(flat) if kind == "FT-FROZEN-encoder" else flat
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if state_hash(model) != payload["state_sha256"]:
        raise ValueError("Checkpoint state checksum mismatch")
    return model, metadata
