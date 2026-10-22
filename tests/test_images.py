import numpy as np
import pytest
import torch

from mario_dino.data.images import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CachedImageDataset,
    all_views,
    apply_view,
    check_cached,
    preprocess,
)
from mario_dino.octip_cache import build_cache, collect_images


def _gray(h=200, w=768, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 200, size=(h, w), dtype=np.uint8)


def test_output_shape_and_dtype():
    for w in (512, 768):
        x = preprocess(_gray(w=w))
        assert x.shape == (3, 196, 518) and x.dtype == np.float32


def test_normalisation():
    x = preprocess(_gray())
    unnorm = x * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    # min-max to the full 0-255 range, then identical channels
    np.testing.assert_allclose(unnorm.min(axis=(1, 2)), 0.0, atol=1e-6)
    np.testing.assert_allclose(unnorm.max(axis=(1, 2)), 1.0, atol=1e-6)
    np.testing.assert_allclose(unnorm[0], unnorm[1], atol=1e-6)
    np.testing.assert_allclose(unnorm[0], unnorm[2], atol=1e-6)


def test_view1_is_exact_flip_of_view0():
    x = preprocess(_gray())
    v0, v1 = apply_view(x, 0), apply_view(x, 1)
    assert np.array_equal(v1, v0[:, :, ::-1])
    t = torch.from_numpy(x)[None]
    views = all_views(t)
    assert views.shape == (1, 2, 3, 196, 518)
    assert torch.equal(views[:, 1], torch.flip(views[:, 0], dims=[-1]))
    assert torch.equal(views[0, 1], torch.from_numpy(v1))
    with pytest.raises(ValueError):
        apply_view(x, 2)


def test_dataset_reads_cache(synth, tmp_path):
    imgs = collect_images([synth["csv"]["task2_train"]])[:3]
    cache = str(tmp_path / "cache")
    assert check_cached(imgs, cache, "raw") == imgs
    build_cache(imgs, synth["images"], cache, "raw")
    assert check_cached(imgs, cache, "raw") == []
    ds = CachedImageDataset(imgs, cache, "raw", (196, 518))
    i, x = ds[1]
    assert i == 1 and x.shape == (3, 196, 518)
