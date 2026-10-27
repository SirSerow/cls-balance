"""Frozen shared features and independent, fixed-capacity specialist heads."""

from __future__ import annotations

import torch
from torch import nn


def build_head(classes: int, hidden: int = 256, dimension: int = 1280) -> nn.Sequential:
    """Classifier columns are bound to a NodeSpec by the checkpoint metadata."""
    if any(type(v) is not int or v < 1 for v in (classes, hidden, dimension)) or classes < 2:
        raise ValueError("Invalid MLP dimensions")
    return nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, classes)).float()


def assert_frozen(backbone: nn.Module, expected_hash: str) -> None:
    """Check parameters AND buffers, including BatchNorm running statistics."""
    from ..checkpoints import state_hash

    if (any(p.requires_grad or p.grad is not None for p in backbone.parameters())
            or any(m.training for m in backbone.modules()) or state_hash(backbone) != expected_hash):
        raise ValueError("Frozen backbone weights, buffers, mode or gradients changed")


class FeatureDataset(torch.utils.data.Dataset):
    """Branch-local features in official manifest order; never inference routing."""

    def __init__(self, features: torch.Tensor, indices: list[int], targets: list[int]) -> None:
        self.features = features
        self.indices, self.targets = indices, targets

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        return {"image": self.features[self.indices[index]], "variant_id": self.targets[index]}
