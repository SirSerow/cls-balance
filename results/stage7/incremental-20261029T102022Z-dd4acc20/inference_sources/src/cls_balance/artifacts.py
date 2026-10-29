"""Stable serialization, immutable artifacts, and execution provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def canonical_json(value: Any) -> bytes:
    """Serialize semantic metadata independently of dictionary insertion order."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Hash file bytes; MD5 is used only to match the dataset authors' checksum."""
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Create an artifact once; never replace an existing result."""
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def seed_everything(seed: int) -> None:
    """Fail on nondeterministic Torch kernels rather than silently accepting them."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment(repo: Path) -> dict[str, Any]:
    """Record installed distributions and hardware without requiring CUDA."""
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {
        "python": sys.version, "platform": platform.platform(),
        "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "nvidia_smi": driver, "device": "cpu", "runtime_policy": "data preparation; no model resident",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "determinism_note": "Unsupported deterministic Torch kernels raise; transforms use per-image local RNGs.",
    }
