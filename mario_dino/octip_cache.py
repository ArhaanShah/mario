"""OCTIP preprocessing cache: unique images from the CSVs -> ``<cache_dir>/<mode>/<image>.png``.

* ``--mode classic`` (default) runs DF41's OCTIP PreProcessor on masks from two learning-free retina
  localizers (``mario_dino.retina``). Needs no weights and no TensorFlow; runs in the main env.
* ``--mode octip`` runs DF41's OCTIP (two FPN retina localizers + PreProcessor). It needs the octip env
  (TensorFlow 2.10, segmentation_models 1.0.1) and the two weight files. Localizers run in batches over many
  paths instead of DF41's per-sample calls. If PreProcessor fails on an image, the raw image is cached
  instead, as DF41 does, and the failure is recorded.
* ``--mode raw`` copies the original images; it needs no TensorFlow and no weights.

Existing cache files are skipped, so the step is idempotent and resumable. Failures are appended to
``<cache_dir>/<mode>/failures.csv``. TensorFlow is imported only inside ``OctipBackend``.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np
import pandas as pd

from .cli import base_parser, load_from_args
from .config import MODES
from .data import schema
from .utils import THIRD_PARTY, get_logger, write_provenance

log = get_logger(__name__)

OCTIP_WEIGHT_FILES = ("FPN_efficientnetb6_384x384.hdf5", "FPN_efficientnetb7_320x320.hdf5")
FAILURE_COLUMNS = ["time", "image", "stage", "error", "action"]


def cache_path(cache_dir: str, mode: str, image: str) -> str:
    return os.path.join(cache_dir, mode, os.path.splitext(str(image))[0] + ".png")


def missing_weight_files(weights_dir: str) -> list[str]:
    return [f for f in OCTIP_WEIGHT_FILES if not os.path.isfile(os.path.join(weights_dir, f))]


def collect_images(csv_paths: Sequence[str]) -> list[str]:
    """Sorted unique B-scan names over Task 1 and Task 2 CSVs (task detected from columns)."""
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        task = schema.detect_task(df)
        dfs.append((task, schema.validate(df, task, require_label=False, check_images=False, source=p)))
    return schema.unique_images(dfs)


def _import_octip(third_party: str = THIRD_PARTY):
    """Import OCTIP from the pinned DF41 clone (never DF41's utils.dataset, which pulls TF via relative imports)."""
    octip_root = os.path.join(third_party, "utils", "octip_segmentation_2d")
    if not os.path.isdir(os.path.join(octip_root, "octip")):
        raise FileNotFoundError(f"OCTIP not found at {octip_root}; run `git submodule update --init`")
    if octip_root not in sys.path:
        sys.path.insert(0, octip_root)
    try:
        import segmentation_models  # noqa: F401
    except Exception:  # sm 1.0.1 defaults to standalone keras; fall back to tf.keras
        for m in [m for m in sys.modules if m.split(".")[0] in ("segmentation_models", "efficientnet")]:
            del sys.modules[m]
        os.environ["SM_FRAMEWORK"] = "tf.keras"
        import segmentation_models  # noqa: F401
    from octip.pre_processor import PreProcessor
    from octip.retina_localizer import RetinaLocalizationDataset, RetinaLocalizer

    return RetinaLocalizer, RetinaLocalizationDataset, PreProcessor


@contextlib.contextmanager
def _no_imagenet_download():
    """Build OCTIP's FPN models with ``encoder_weights=None``.

    RetinaLocalizer asks segmentation_models for ImageNet encoder weights (a download from GitHub) and then
    overwrites every weight with the .hdf5 file, so skipping the download changes nothing and keeps the
    step offline. DF41's code is not edited: ``sm.FPN`` is wrapped for the duration of construction.
    """
    import segmentation_models as sm

    original = sm.FPN

    @functools.wraps(original)
    def fpn(*args, **kwargs):
        kwargs["encoder_weights"] = None
        return original(*args, **kwargs)

    sm.FPN = fpn
    try:
        yield
    finally:
        sm.FPN = original


def check_flattened(img: np.ndarray, min_valid_fraction: float = 0.25) -> None:
    """Raise if PreProcessor kept too few A-scans (an almost black image is worse than the raw B-scan)."""
    frac = float((img > 0).any(axis=0).mean()) if img.size else 0.0
    if frac < min_valid_fraction:
        raise ValueError(f"only {frac:.0%} of A-scans kept by PreProcessor")


def _seg_to_u8(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim == 4:
        seg = seg[..., 0]
    return (seg * 255).astype(np.uint8)


class OctipBackend:
    """Batched OCTIP: both localizers over a list of paths, then PreProcessor per image."""

    def __init__(self, weights_dir: str, batch_size: int = 16, third_party: str = THIRD_PARTY):
        missing = missing_weight_files(weights_dir)
        if missing:
            raise FileNotFoundError(
                f"OCTIP weight file(s) {missing} not found in {weights_dir!r}. "
                "Use --mode raw to run without them (see RUNBOOK.md)."
            )
        RetinaLocalizer, self._Dataset, PreProcessor = _import_octip(third_party)
        self.batch_size = batch_size
        with _no_imagenet_download():
            self.loc1 = RetinaLocalizer("FPN", "efficientnetb6", (384, 384), model_directory=weights_dir)
            self.loc2 = RetinaLocalizer("FPN", "efficientnetb7", (320, 320), model_directory=weights_dir)
        self.pre = PreProcessor(200, min_height=100, normalize_intensities=True)

    def segment(self, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
        _, s1 = self.loc1(self._Dataset(paths, self.batch_size, self.loc1))
        _, s2 = self.loc2(self._Dataset(paths, self.batch_size, self.loc2))
        return _seg_to_u8(s1), _seg_to_u8(s2)

    def process(self, paths: list[str]) -> list[tuple[np.ndarray | None, list[tuple[str, str, str]]]]:
        """Returns (image or None, [(stage, error, action)]) per path."""
        results: list = [None] * len(paths)
        raws = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        ok = [i for i, r in enumerate(raws) if r is not None]
        for i, r in enumerate(raws):
            if r is None:
                results[i] = (None, [("read", "cv2.imread returned None", "skipped")])
        segs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        try:
            s1, s2 = self.segment([paths[i] for i in ok])
            segs = {i: (s1[k], s2[k]) for k, i in enumerate(ok)}
        except Exception as e:  # one bad image fails the whole batch; retry one by one
            log.warning("batch segmentation failed (%s); retrying per image", e)
            for i in ok:
                try:
                    s1, s2 = self.segment([paths[i]])
                    segs[i] = (s1[0], s2[0])
                except Exception as e1:
                    results[i] = (raws[i], [("localizer", repr(e1), "raw_fallback")])
        for i, (a, b) in segs.items():
            try:
                img = np.asarray(self.pre(paths[i], [a, b], output_directory=""), np.uint8)
                check_flattened(img)
                results[i] = (img, [])
            except Exception as e:
                results[i] = (raws[i], [("preprocessor", repr(e), "raw_fallback")])
        return results


def _atomic_imwrite(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.png"
    if not cv2.imwrite(tmp, img):
        raise OSError(f"cv2.imwrite failed for {path}")
    os.replace(tmp, path)


def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + ".tmp.png"
    if src.lower().endswith(".png"):
        shutil.copyfile(src, tmp)
    else:
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None or not cv2.imwrite(tmp, img):
            raise OSError(f"cannot convert {src} to PNG")
    os.replace(tmp, dst)


def _append_failures(path: str, rows: list[list[str]]) -> None:
    if not rows:
        return
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FAILURE_COLUMNS)
        w.writerows(rows)


def build_cache(
    images: Sequence[str],
    image_root: str,
    cache_dir: str,
    mode: str = "raw",
    weights_dir: str | None = None,
    batch_size: int = 16,
    backend_factory: Callable[[], OctipBackend] | None = None,
) -> dict:
    """Cache every image not already cached; returns counts."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    out_dir = os.path.join(cache_dir, mode)
    os.makedirs(out_dir, exist_ok=True)
    failures_csv = os.path.join(out_dir, "failures.csv")
    pending = [im for im in images if not os.path.isfile(cache_path(cache_dir, mode, im))]
    stats = {"total": len(images), "cached_before": len(images) - len(pending), "written": 0, "fallback": 0,
             "failed": 0}
    log.info("%s cache: %d images, %d already cached, %d to process", mode, len(images),
             stats["cached_before"], len(pending))
    if not pending:
        return stats
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    if mode == "raw":
        fails = []
        for im in pending:
            src = schema.image_path(image_root, im)
            if not os.path.isfile(src) or cv2.imread(src, cv2.IMREAD_GRAYSCALE) is None:
                fails.append([now, im, "read", "missing or unreadable", "skipped"])
                continue
            _atomic_copy(src, cache_path(cache_dir, mode, im))
            stats["written"] += 1
        stats["failed"] = len(fails)
        _append_failures(failures_csv, fails)
        return stats

    if backend_factory:
        backend = backend_factory()
    elif mode == "classic":
        from .retina import ClassicBackend

        backend = ClassicBackend()
    else:
        backend = OctipBackend(weights_dir or "", batch_size)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results = backend.process([schema.image_path(image_root, im) for im in batch])
        fails = []
        for im, (img, notes) in zip(batch, results):
            for stage, err, action in notes:
                fails.append([now, im, stage, err, action])
            if img is None:
                stats["failed"] += 1
                continue
            _atomic_imwrite(cache_path(cache_dir, mode, im), img)
            stats["written"] += 1
            stats["fallback"] += int(bool(notes))
        _append_failures(failures_csv, fails)
        log.info("%s cache: %d / %d", mode, min(start + batch_size, len(pending)), len(pending))
    return stats


def main(argv: list[str] | None = None) -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--mode", choices=list(MODES), default=None, help="overrides preprocess.mode")
    p.add_argument("--csv", action="append", default=None,
                   help="CSV(s) to take images from (default: all training + extra CSVs in the config)")
    p.add_argument("--batch-size", type=int, default=None, help="localizer batch (paths per call)")
    args = p.parse_args(argv)
    cfg = load_from_args(args)
    mode = args.mode or cfg.preprocess.mode
    csvs = args.csv or (cfg.data.task1_train_csvs + cfg.data.task2_train_csvs + cfg.data.extra_csvs)
    images = collect_images(csvs)
    stats = build_cache(
        images,
        cfg.data.image_root,
        cfg.preprocess.cache_dir,
        mode,
        weights_dir=cfg.preprocess.octip_weights_dir,
        batch_size=args.batch_size or cfg.preprocess.octip_batch_size,
    )
    write_provenance(os.path.join(cfg.preprocess.cache_dir, mode), cfg, {"csvs": csvs, "stats": stats})
    log.info("done: %s", stats)


if __name__ == "__main__":
    main()
