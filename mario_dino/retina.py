"""Weight-free OCTIP preprocessing: classical retina localization + DF41's own OCTIP PreProcessor.

OCTIP's two FPN retina localizers (``FPN_efficientnetb6_384x384.hdf5``, ``FPN_efficientnetb7_320x320.hdf5``)
are only distributed on request from their author, so ``preprocess.mode: classic`` replaces them with two
independent learning-free estimators of the retina's top boundary:

* **threshold**: Gaussian blur, Otsu threshold, morphological opening, largest connected component;
* **gradient**: Gaussian blur, vertical Sobel (dark-to-bright going down), first row per A-scan above a
  per-column fraction of its maximum.

Both are median-filtered across A-scans and turned into masks (retina = top boundary down to the image
bottom). OCTIP's PreProcessor only uses the first in-retina pixel of each A-scan from each mask, and keeps an
A-scan only when the masks agree within ``max_diff_tops`` (5 px), exactly as it does with the two FPN models.
Flattening, cropping, A-scan rejection and intensity normalisation are DF41's unmodified OCTIP code.

``load_preprocessor`` imports ``octip/pre_processor.py`` from the pinned DF41 clone without running
``octip/__init__.py`` (which needs TensorFlow + segmentation_models). The module's ``import tensorflow`` is
unused, so a stub is supplied when TensorFlow is not installed.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import os
import sys
import types
import warnings

import cv2
import numpy as np
from scipy import ndimage

from .octip_cache import check_flattened
from .utils import THIRD_PARTY

_PKG = "_df41_octip"


@functools.lru_cache(maxsize=1)
def load_preprocessor_cls(third_party: str = THIRD_PARTY):
    """DF41's ``octip.pre_processor.PreProcessor`` class, importable in the main env."""
    octip_dir = os.path.join(third_party, "utils", "octip_segmentation_2d", "octip")
    if not os.path.isfile(os.path.join(octip_dir, "pre_processor.py")):
        raise FileNotFoundError(f"OCTIP not found at {octip_dir}; run `git submodule update --init`")
    # SciPy 2.0 removes the deprecated ndimage.morphology namespace that PreProcessor calls
    if not hasattr(ndimage, "morphology"):
        ndimage.morphology = types.SimpleNamespace(binary_closing=ndimage.binary_closing)
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [octip_dir]  # a package over octip/ that skips its __init__.py
        sys.modules[_PKG] = pkg
    stub = importlib.util.find_spec("tensorflow") is None
    if stub:
        sys.modules["tensorflow"] = types.ModuleType("tensorflow")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(f"{_PKG}.pre_processor")
    finally:
        if stub:
            sys.modules.pop("tensorflow", None)
    return module.PreProcessor


def _median_fill(tops: np.ndarray, size: int) -> np.ndarray:
    """Median-filter valid tops (>= 0) across A-scans; invalid A-scans stay -1."""
    valid = tops >= 0
    if valid.sum() < 3:
        return np.full_like(tops, -1)
    filled = np.interp(np.arange(len(tops)), np.where(valid)[0], tops[valid]).astype(np.float32)
    smooth = ndimage.median_filter(filled, size=size, mode="nearest")
    return np.where(valid, np.round(smooth), -1).astype(np.int64)


def top_by_threshold(img: np.ndarray, sigma: float = 3.0, frac: float = 0.3) -> np.ndarray:
    s = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    s8 = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    otsu, _ = cv2.threshold(s8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu alone can skip a dim inner layer for a brighter one below it; cap it at background + frac * range
    bg, hi = np.percentile(s8, 20), np.percentile(s8, 99)
    fg = (s8 > min(otsu, bg + frac * (hi - bg))).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=4)
    if n <= 1:
        return np.full(img.shape[1], -1, dtype=np.int64)
    # keep every tissue component that is not speckle (>= 5% of the largest), so a dark gap between
    # layers cannot make the top estimate jump to a lower band
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= 0.05 * areas.max())[0]
    fg = np.isin(labels, keep).astype(np.uint8)
    has = fg.any(axis=0)
    return np.where(has, fg.argmax(axis=0), -1).astype(np.int64)


def top_by_gradient(img: np.ndarray, sigma: float = 2.0, frac: float = 0.4) -> np.ndarray:
    s = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    g = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=5)  # > 0 where intensity increases downward
    col_max = g.max(axis=0)
    floor = 0.2 * np.percentile(col_max, 90)  # A-scans with no clear edge (e.g. black padding) are invalid
    above = g >= frac * np.maximum(col_max, 1e-6)[None, :]
    tops = above.argmax(axis=0).astype(np.int64)
    return np.where((col_max > floor) & (col_max > 0), tops, -1)


def tops_to_mask(tops: np.ndarray, height: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    return ((rows >= tops[None, :]) & (tops[None, :] >= 0)).astype(np.uint8) * 255


def classic_segmentations(img: np.ndarray, median_size: int = 15) -> list[np.ndarray]:
    """Two uint8 (0/255) retina masks at B-scan resolution, one per estimator."""
    h = img.shape[0]
    return [
        tops_to_mask(_median_fill(top_by_threshold(img), median_size), h),
        tops_to_mask(_median_fill(top_by_gradient(img), median_size), h),
    ]


class ClassicBackend:
    """``octip_cache`` backend: classical localizers + DF41's OCTIP PreProcessor. No weights, no TensorFlow."""

    def __init__(self, third_party: str = THIRD_PARTY, min_valid_fraction: float = 0.25):
        self.pre = load_preprocessor_cls(third_party)(200, min_height=100, normalize_intensities=True)
        self.min_valid_fraction = min_valid_fraction

    def process(self, paths: list[str]) -> list[tuple[np.ndarray | None, list[tuple[str, str, str]]]]:
        out = []
        for p in paths:
            raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                out.append((None, [("read", "cv2.imread returned None", "skipped")]))
                continue
            try:
                with warnings.catch_warnings():  # DF41 calls the deprecated scipy.ndimage.morphology
                    warnings.simplefilter("ignore", DeprecationWarning)
                    img = np.asarray(self.pre(p, classic_segmentations(raw), output_directory=""), np.uint8)
                check_flattened(img, self.min_valid_fraction)
                out.append((img, []))
            except Exception as e:
                out.append((raw, [("preprocessor", repr(e), "raw_fallback")]))
        return out

