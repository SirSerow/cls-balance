"""Independently check trained exports and their committed training sources.

Run from the repository root with the three ignored local training directories:
python results/stage3/validation-20261025T042101Z-bcc17404/verify_exports.py --runs <run42> <run43> <run44>
The zero input is a functional check, not an aircraft-accuracy measurement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from cls_balance.checkpoints import load_checkpoint, state_hash
from cls_balance.report_flat import audit_training_run


def verify(path: Path) -> dict:
    summary = audit_training_run(path)
    env = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    sources = json.loads((path / "source_hashes.json").read_text(encoding="utf-8"))
    for name, digest in sources.items():
        committed = subprocess.check_output(["git", "show", f"{env['git_sha']}:{name}"])
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Training source differs from recorded commit: {name}")
    manifest = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if not manifest["exact_reload_logits"] or not manifest["exact_reload_embeddings"]:
        raise ValueError("Training-device reload checks failed")
    metadata = manifest["flat"]["metadata"]
    expected = {k: metadata[k] for k in ("taxonomy_sha256", "preprocessing_sha256", "dataset_manifest_sha256")}
    flat, _ = load_checkpoint(path / "flat.pt", **expected)
    encoder, encoder_metadata = load_checkpoint(path / "encoder.pt", **expected, kind="FT-FROZEN-encoder")
    if encoder_metadata["parent_flat_sha256"] != manifest["flat"]["file_sha256"]:
        raise ValueError("Encoder parent mismatch")
    before = state_hash(encoder)
    encoder.train(True)
    if any(m.training for m in encoder.modules()) or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Encoder is not frozen in eval mode")
    image = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        features = encoder(image)
        torch.testing.assert_close(features, flat.avgpool(flat.features(image)).flatten(1), rtol=0, atol=0)
    if state_hash(encoder) != before or tuple(features.shape) != (1, 1280):
        raise ValueError("Encoder state or shape changed")
    return {"run_id": summary["run_id"], "seed": summary["seed"], "selected_epoch": summary["best_epoch"],
            "training_git_sha": env["git_sha"], "training_sources_match_commit": len(sources),
            "training_device_exact_reload_logits": True, "training_device_exact_reload_embeddings": True,
            "independent_cpu_encoder_matches_flat_features": True,
            "encoder_frozen_and_unchanged": True, "encoder_parent_matches": True,
            "fixed_reload_validation_image_id": metadata["selection_image_ids"][0],
            "train_evaluation_images": summary["evaluations"]["train"]["audit"]["images"],
            "validation_evaluation_images": summary["evaluations"]["val"]["audit"]["images"],
            "test_evaluated": summary["test_evaluated"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps({"scope": "artifact/source/encoder contract checks; no accuracy measurements",
                      "runs": [verify(path) for path in args.runs]}, indent=2, sort_keys=True))
