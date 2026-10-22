import os
import subprocess
import sys
import types

import cv2
import numpy as np
import pandas as pd
import pytest

from mario_dino.octip_cache import (
    _no_imagenet_download,
    build_cache,
    cache_path,
    check_flattened,
    collect_images,
)
from mario_dino.retina import (
    ClassicBackend,
    _median_fill,
    classic_segmentations,
    load_preprocessor_cls,
    top_by_gradient,
    top_by_threshold,
)
from mario_dino.synthetic import synthetic_bscan
from mario_dino.utils import REPO_ROOT


def test_estimators_find_the_retina_top():
    rng = np.random.default_rng(0)
    for _ in range(10):
        img, top = synthetic_bscan(rng, 512)
        a = _median_fill(top_by_threshold(img), 15)
        b = _median_fill(top_by_gradient(img), 15)
        assert np.median(np.abs(a - top)) <= 3 and np.median(np.abs(b - top)) <= 4
        # OCTIP keeps an A-scan only when the two localizers agree within 5 px
        assert (np.abs(a - b) <= 5).mean() > 0.9


def test_estimators_reject_empty_ascans():
    rng = np.random.default_rng(1)
    img, _ = synthetic_bscan(rng, 512)
    img[:, :100] = 0  # black padding, as at the edges of some B-scans
    masks = classic_segmentations(img)
    assert all(m.shape == img.shape and m.dtype == np.uint8 for m in masks)
    # blurring bleeds ~2 px of tissue into the padding edge; PreProcessor's FOV mask removes it anyway
    assert not masks[0][:, :95].any() and not masks[1][:, :95].any()


def test_preprocessor_loads_without_tensorflow():
    code = (
        "import sys\n"
        "from mario_dino.retina import load_preprocessor_cls\n"
        "cls = load_preprocessor_cls()\n"
        "assert cls.__name__ == 'PreProcessor'\n"
        "assert 'tensorflow' not in sys.modules or sys.modules['tensorflow'].__spec__ is not None\n"
        "assert 'segmentation_models' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def test_classic_backend_flattens(tmp_path):
    rng = np.random.default_rng(2)
    img, top = synthetic_bscan(rng, 768)
    assert top.max() - top.min() > 5  # curved retina
    p = str(tmp_path / "b.png")
    cv2.imwrite(p, img)
    (out, notes), = ClassicBackend().process([p])
    assert notes == [] and out.shape == (200, 768) and out.dtype == np.uint8
    # flattened: the retina starts at row 0 in (almost) every A-scan
    first = (out > 0).argmax(axis=0)
    assert (out > 0).any(axis=0).mean() > 0.9 and np.median(first) == 0
    # DF41's PreProcessor is what did the flattening
    assert load_preprocessor_cls().__module__.endswith("pre_processor")


def test_classic_backend_falls_back_to_raw(tmp_path):
    flat = np.full((200, 300), 30, np.uint8)  # no retina at all
    p = str(tmp_path / "flat.png")
    cv2.imwrite(p, flat)
    (out, notes), = ClassicBackend().process([p])
    assert np.array_equal(out, flat) and notes[0][2] == "raw_fallback"
    with pytest.raises(ValueError):
        check_flattened(np.zeros((200, 50), np.uint8))


def test_classic_cache_mode(synth, tmp_path):
    imgs = collect_images([synth["csv"]["task2_train"]])[:4]
    cache = str(tmp_path / "cache")
    s = build_cache(imgs, synth["images"], cache, "classic", batch_size=2)
    assert s["written"] == 4 and s["fallback"] == 0 and s["failed"] == 0
    for im in imgs:
        assert cv2.imread(cache_path(cache, "classic", im), cv2.IMREAD_GRAYSCALE).shape[0] == 200
    assert not os.path.exists(os.path.join(cache, "classic", "failures.csv"))
    assert build_cache(imgs, synth["images"], cache, "classic")["written"] == 0


def test_octip_fpn_built_without_imagenet_download(monkeypatch):
    calls = []
    fake = types.ModuleType("segmentation_models")
    fake.FPN = lambda *a, **k: calls.append(k)
    monkeypatch.setitem(sys.modules, "segmentation_models", fake)
    original = fake.FPN
    with _no_imagenet_download():
        fake.FPN("efficientnetb6", classes=1, encoder_weights="imagenet")
    assert calls == [{"classes": 1, "encoder_weights": None}]
    assert fake.FPN is original


def test_synthetic_dataset_images_are_classic_compatible(synth):
    t2 = pd.read_csv(synth["csv"]["task2_train"])
    img = cv2.imread(os.path.join(synth["images"], t2["image"].iloc[0]), cv2.IMREAD_GRAYSCALE)
    a, b = classic_segmentations(img)
    assert (a > 0).any() and (b > 0).any()
