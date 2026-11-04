"""Image preprocessing and cached-image dataset for the DINOv2 encoder."""

from __future__ import annotations

import os
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..octip_cache import cache_path

N_VIEWS = 2
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: np.ndarray, img_size_hw: Sequence[int] = (196, 518)) -> np.ndarray:
    """Resize a grayscale B-scan, min-max scale it, expand to RGB, and apply ImageNet normalization."""
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim != 2 or img.size == 0:
        raise ValueError(f"expected a non-empty grayscale image, got shape {img.shape}")
    height, width = (int(s) for s in img_size_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"image dimensions must be positive, got {img_size_hw}")
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    lo, hi = float(resized.min()), float(resized.max())
    scaled = (resized - lo) / (hi - lo) if hi > lo else np.zeros_like(resized)
    rgb = np.broadcast_to(scaled[None, :, :], (3, height, width)).copy()
    return (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def apply_view(x, view: int):
    """Apply identity (0) or horizontal-flip (1) view to a CHW or BCHW image."""
    if view == 0:
        return x
    if view != 1:
        raise ValueError(f"view must be 0 or 1, got {view}")
    if isinstance(x, torch.Tensor):
        return torch.flip(x, dims=[-1])
    return np.flip(x, axis=-1).copy()


def all_views(x: torch.Tensor) -> torch.Tensor:
    """Return identity and horizontal-flip views as a (B, 2, C, H, W) tensor."""
    return torch.stack((apply_view(x, 0), apply_view(x, 1)), dim=1)


def check_cached(images: Sequence[str], cache_dir: str, mode: str) -> list[str]:
    """Return the requested image names whose preprocessed cache file is absent."""
    return [image for image in images if not os.path.isfile(cache_path(cache_dir, mode, image))]


class CachedImageDataset(Dataset):
    """Loads preprocessed cache PNGs and applies the encoder's standard image transform."""

    def __init__(
        self,
        images: Sequence[str],
        cache_dir: str,
        mode: str,
        img_size_hw: Sequence[int] = (196, 518),
    ):
        self.images = list(images)
        self.cache_dir = cache_dir
        self.mode = mode
        self.img_size_hw = tuple(int(s) for s in img_size_hw)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        path = cache_path(self.cache_dir, self.mode, self.images[index])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"cannot read cached image {path!r}")
        tensor = torch.from_numpy(preprocess(image, self.img_size_hw))
        return index, tensor
