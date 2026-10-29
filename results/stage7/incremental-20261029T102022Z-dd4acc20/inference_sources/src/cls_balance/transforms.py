"""Full-content letterboxing and worker/order-independent train augmentation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass

import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import functional as TF

from .artifacts import object_hash


@dataclass(frozen=True)
class Preprocessing:
    image_size: int = 224
    remove_bottom_banner_px: int = 20
    pad_rgb: tuple[int, int, int] = (0, 0, 0)
    interpolation: str = "bicubic"
    normalization: str = "EfficientNet_B0_Weights.IMAGENET1K_V1"
    horizontal_flip_probability: float = 0.5
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.image_size != 224 or self.remove_bottom_banner_px != 20:
            raise ValueError("Primary protocol requires 224 pixels and 20-pixel banner removal")
        if self.interpolation != "bicubic" or self.normalization != "EfficientNet_B0_Weights.IMAGENET1K_V1":
            raise ValueError("Unsupported primary preprocessing profile")
        if len(self.pad_rgb) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in self.pad_rgb):
            raise ValueError("pad_rgb must contain three integer bytes")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("Flip probability must be in [0, 1]")
        if not 0 <= self.brightness_jitter <= 1 or not 0 <= self.contrast_jitter <= 1:
            raise ValueError("Jitter must be in [0, 1]")

    def to_dict(self) -> dict:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        return {**asdict(self), "mean": weights.mean, "std": weights.std,
                "schema_version": 1, "geometry": "crop banner; aspect fit; round half up; center pad; extra pixel right/bottom",
                "train_order": "letterbox, flip, brightness, contrast, tensor, normalize",
                "rng": "SHA256(seed:epoch:image_id) -> Python Random; epoch set before creating workers"}

    @property
    def sha256(self) -> str:
        return object_hash(self.to_dict())

    def image(self, image: Image.Image) -> Image.Image:
        """Remove only the banner, then fit all remaining content inside the square."""
        image = image.convert("RGB")
        w, h = image.size
        if h <= self.remove_bottom_banner_px:
            raise ValueError(f"Image too short to remove copyright banner: {image.size}")
        image = image.crop((0, 0, w, h - self.remove_bottom_banner_px))
        scale = self.image_size / max(image.size)
        size = tuple(max(1, min(self.image_size, int(d * scale + 0.5))) for d in image.size)
        image = image.resize(size, Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.image_size, self.image_size), self.pad_rgb)
        canvas.paste(image, ((self.image_size - size[0]) // 2, (self.image_size - size[1]) // 2))
        return canvas

    def tensor(self, image: Image.Image, *, training: bool = False,
               seed: int = 42, epoch: int = 0, image_id: str = "") -> torch.Tensor:
        image = self.image(image)
        if training:
            if not image_id:
                raise ValueError("Training augmentation requires an image ID")
            rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{image_id}".encode()).digest(), "big"))
            if rng.random() < self.horizontal_flip_probability:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(1 - self.brightness_jitter, 1 + self.brightness_jitter))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(1 - self.contrast_jitter, 1 + self.contrast_jitter))
        profile = self.to_dict()
        return TF.normalize(TF.to_tensor(image), profile["mean"], profile["std"])
