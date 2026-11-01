import filecmp
import importlib.util
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from mario_dino import octip_cache
from mario_dino.octip_cache import build_cache, cache_path, collect_images, missing_weight_files
from mario_dino.utils import REPO_ROOT


def _images(synth, n=None):
    imgs = collect_images([synth["csv"]["task1_train"], synth["csv"]["task2_train"]])
    return imgs[:n] if n else imgs


def test_collect_images_unique_over_tasks(synth):
    t1 = pd.read_csv(synth["csv"]["task1_train"])
    imgs = _images(synth)
    assert imgs == sorted(set(t1["image_at_ti"]) | set(t1["image_at_ti+1"]))


def test_collect_images_skips_rows_with_unreadable_scans(synth, tmp_path):
    import cv2

    df = pd.read_csv(synth["csv"]["task1_train"])
    bad_image = df.iloc[0]["image_at_ti+1"]
    (tmp_path / bad_image).write_bytes(b"not a readable image")
    image_root = str(tmp_path)
    for column in ("image_at_ti", "image_at_ti+1"):
        for image in df[column].unique():
            source = os.path.join(synth["images"], image)
            destination = os.path.join(image_root, image)
            if image != bad_image:
                shutil.copyfile(source, destination)
    csv_path = str(tmp_path / "task1.csv")
    df.to_csv(csv_path, index=False)

    images = collect_images([csv_path], image_root)

    assert bad_image not in images
    assert all(cv2.imread(os.path.join(image_root, image), cv2.IMREAD_GRAYSCALE) is not None for image in images)


def test_raw_mode_copies_and_skips_existing(synth, tmp_path):
    imgs = _images(synth, 6)
    cache = str(tmp_path / "cache")
    s1 = build_cache(imgs, synth["images"], cache, "raw")
    assert s1["written"] == 6 and s1["failed"] == 0
    for im in imgs:
        assert filecmp.cmp(cache_path(cache, "raw", im), os.path.join(synth["images"], im), shallow=False)
    s2 = build_cache(imgs, synth["images"], cache, "raw")
    assert s2["cached_before"] == 6 and s2["written"] == 0


def test_raw_mode_records_failures(synth, tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    good = _images(synth, 1)[0]
    shutil.copy(os.path.join(synth["images"], good), root / good)
    (root / "BROKEN.png").write_bytes(b"not a png")
    cache = str(tmp_path / "cache")
    s = build_cache([good, "BROKEN.png", "ABSENT.png"], str(root), cache, "raw")
    assert (s["written"], s["failed"]) == (1, 2)
    fails = pd.read_csv(os.path.join(cache, "raw", "failures.csv"))
    assert sorted(fails["image"]) == ["ABSENT.png", "BROKEN.png"]
    assert not os.path.exists(cache_path(cache, "raw", "BROKEN.png"))


class _FakeBackend:
    """Stands in for OCTIP: flips the image, and fails PreProcessor on one image."""

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.calls = 0

    def process(self, paths):
        import cv2

        self.calls += 1
        out = []
        for p in paths:
            raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if os.path.basename(p) == self.fail_on:
                out.append((raw, [("preprocessor", "boom", "raw_fallback")]))
            else:
                out.append((raw[::-1].copy(), []))
        return out


def test_octip_mode_batches_and_falls_back(synth, tmp_path):
    import cv2

    imgs = _images(synth, 5)
    cache = str(tmp_path / "cache")
    fake = _FakeBackend(fail_on=imgs[2])
    s = build_cache(imgs, synth["images"], cache, "octip", batch_size=2, backend_factory=lambda: fake)
    assert fake.calls == 3 and s["written"] == 5 and s["fallback"] == 1
    raw = cv2.imread(os.path.join(synth["images"], imgs[2]), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(cv2.imread(cache_path(cache, "octip", imgs[2]), cv2.IMREAD_GRAYSCALE), raw)
    fails = pd.read_csv(os.path.join(cache, "octip", "failures.csv"))
    assert list(fails["image"]) == [imgs[2]] and fails["action"].iloc[0] == "raw_fallback"


def test_octip_mode_real_weights(synth, tmp_path):
    weights = os.environ.get("OCTIP_WEIGHTS_DIR", os.path.join(REPO_ROOT, "weights", "octip"))
    if missing_weight_files(weights):
        pytest.skip(f"OCTIP weight files not in {weights!r} (set OCTIP_WEIGHTS_DIR)")
    if importlib.util.find_spec("tensorflow") is None or importlib.util.find_spec("segmentation_models") is None:
        pytest.skip("octip env (tensorflow, segmentation_models) not installed")
    imgs = _images(synth, 3)
    cache = str(tmp_path / "cache")
    s = build_cache(imgs, synth["images"], cache, "octip", weights_dir=weights, batch_size=2)
    assert s["written"] == 3
    for im in imgs:
        assert os.path.isfile(octip_cache.cache_path(cache, "octip", im))
