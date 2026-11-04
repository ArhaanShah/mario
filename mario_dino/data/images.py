"""Shared encoder input specs, legacy/new transforms, cached images and paired-image datasets."""

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


def resolve_input_spec(enc_cfg) -> dict:
    """Serializable transform identity shared by all image paths."""
    preset = getattr(enc_cfg, "preprocessing", "legacy_dino")
    if preset not in ("legacy_dino", "imagenet", "visionfm_oct"):
        raise ValueError(f"unknown preprocessing preset {preset!r}")
    mean, std = IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()
    if preset == "visionfm_oct":
        mean = [0.21091926, 0.21091926, 0.21091919]
        std = [0.17598894, 0.17598891, 0.17598893]
    return {"version": 1, "preset": preset, "img_size_hw": list(enc_cfg.img_size),
            "scaling": "per_image_minmax" if preset == "legacy_dino" else "uint8_div_255",
            "interpolation": "opencv_linear" if preset == "legacy_dino" else "opencv_cubic",
            "channels": "grayscale_repeat_3", "mean": mean, "std": std}


def preprocess_input(img, spec):
    if spec["version"] != 1:
        raise ValueError(f"unsupported input transform version {spec['version']}")
    if spec["preset"] == "legacy_dino":
        return preprocess(img, spec["img_size_hw"])
    if img.dtype != np.uint8:
        raise ValueError("imagenet and visionfm_oct transforms require uint8 cached images")
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim != 2 or img.size == 0:
        raise ValueError(f"expected nonempty grayscale image, got {img.shape}")
    h, w = spec["img_size_hw"]
    scaled = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    rgb = np.broadcast_to(scaled[None], (3, h, w)).copy()
    mean = np.asarray(spec["mean"], np.float32)[:, None, None]
    std = np.asarray(spec["std"], np.float32)[:, None, None]
    return (rgb - mean) / std


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
        input_spec: dict | None = None,
    ):
        self.input_spec = input_spec
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
        arr = preprocess(image, self.img_size_hw) if self.input_spec is None else preprocess_input(image, self.input_spec)
        tensor = torch.from_numpy(arr)
        return index, tensor


class PairedImageDataset(Dataset):
    """On-demand image pairs; a training view is applied equally to both members."""
    def __init__(self, frame, cache_dir, mode, input_spec):
        self.frame = frame.reset_index(drop=True)
        self.images = CachedImageDataset(
            self.frame["image_at_ti"].tolist() + self.frame["image_at_ti+1"].tolist(),
            cache_dir, mode, input_spec["img_size_hw"], input_spec)

    def __len__(self):
        return len(self.frame)

    def pair(self, index, view=0):
        _, a = self.images[index]
        _, b = self.images[index + len(self)]
        row = self.frame.iloc[index]
        return apply_view(a, int(view)), apply_view(b, int(view)), float(row["delta_t"]), int(row["label"])

    def __getitem__(self, index):
        return self.pair(index)

    def batch(self, rows, views, device):
        from torch.utils.data import default_collate
        batch = default_collate([self.pair(int(i), int(v)) for i, v in zip(rows, views)])
        a, b, dt, y = [t.to(device) for t in batch]
        return a, b, dt.float(), y.long()
