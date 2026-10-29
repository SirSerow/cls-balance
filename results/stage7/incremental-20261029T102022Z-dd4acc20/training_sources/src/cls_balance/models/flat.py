"""The primary FP32 EfficientNet-B0 baseline and pooled encoder."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_flat(*, pretrained: bool = True, classes: int = 100) -> nn.Module:
    """Construct independent weights; downloading is explicit at construction."""
    if type(classes) is not int or classes < 2:
        raise ValueError("Require at least two classes")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    return model.float()


class FrozenEncoder(nn.Module):
    """Independent copy of selected FLAT features, always frozen and in eval mode."""

    dimension = 1280

    def __init__(self, flat: nn.Module) -> None:
        super().__init__()
        self.features = copy.deepcopy(flat.features)
        self.avgpool = copy.deepcopy(flat.avgpool)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> FrozenEncoder:
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.avgpool(self.features(images)).flatten(1)
