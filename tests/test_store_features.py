import numpy as np
import pytest
import torch

from mario_dino.features import extract_features
from mario_dino.models.encoder import DinoEncoder, build_timm_model
from mario_dino.octip_cache import build_cache, collect_images
from mario_dino.store import FeatureStore, StoreMismatch


def test_store_append_reopen_and_identity(tmp_path):
    root = str(tmp_path / "s")
    s = FeatureStore(root, (2, 4), "float16", identity={"enc": "a"})
    rows = s.ensure(["x", "y"])
    assert list(rows) == [0, 1] and list(s.pending()) == [0, 1]
    s.write(rows, np.ones((2, 2, 4)))
    s.mark_done(rows)
    assert list(s.ensure(["y", "z"])) == [1, 2]  # appends only z
    s.write([2], np.full((1, 2, 4), 2.0))
    s.mark_done([2])
    s.close()

    s2 = FeatureStore(root, (2, 4), "float16", identity={"enc": "a"})
    assert s2.images == ["x", "y", "z"] and s2.done.all()
    np.testing.assert_array_equal(s2.get(["z", "x"])[:, 0, 0], [2.0, 1.0])
    assert s2.get(["x"]).dtype == np.float16
    s2.close()
    with pytest.raises(StoreMismatch):
        FeatureStore(root, (2, 4), "float16", identity={"enc": "b"})
    with pytest.raises(StoreMismatch):
        FeatureStore(root, (2, 4), "float32", identity={"enc": "a"})


def test_store_refuses_unfinished_rows(tmp_path):
    s = FeatureStore(str(tmp_path / "s"), (3,), "float32")
    s.ensure(["a"])
    with pytest.raises(RuntimeError, match="not computed"):
        s.get(["a"])
    with pytest.raises(KeyError):
        s.rows(["nope"])


class CountingEncoder(DinoEncoder):
    """Real (tiny) encoder that counts forward calls and can fail on the n-th one."""

    def __init__(self, model, fail_at=None):
        super().__init__(model)
        self.calls = 0
        self.fail_at = fail_at

    def forward(self, x):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise KeyboardInterrupt("simulated interruption")
        return super().forward(x)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return build_timm_model("vit_small_patch14_reg4_dinov2", (28, 56), depth=1).eval()


def test_interrupted_extraction_resumes(synth, tmp_path, tiny_model):
    images = collect_images([synth["csv"]["task2_train"]])[:10]
    cache = str(tmp_path / "cache")
    build_cache(images, synth["images"], cache, "raw")
    kw = dict(cache_dir=cache, mode="raw", img_size_hw=(28, 56), device=torch.device("cpu"), batch_size=2,
              chunk_size=4, amp=False)
    # 10 images, chunks of 4 -> 3 chunks; each batch of 2 images costs 2 forward calls (one per view)
    root = str(tmp_path / "store")
    enc = CountingEncoder(tiny_model, fail_at=5)  # dies on the first call of chunk 2
    store = FeatureStore(root, (2, enc.out_dim), "float32")
    with pytest.raises(KeyboardInterrupt):
        extract_features(enc, images, store, **kw)
    store.close()

    store = FeatureStore(root, (2, enc.out_dim), "float32")
    assert int(store.done.sum()) == 4
    enc2 = CountingEncoder(tiny_model)
    stats = extract_features(enc2, images, store, **kw)
    assert stats == {"images": 10, "already_done": 4, "encoded": 6}
    assert enc2.calls == 2 * 2 + 1 * 2  # chunk 2 (2 batches) + chunk 3 (1 batch), 2 views each
    resumed = store.get(images)
    store.close()

    clean_store = FeatureStore(str(tmp_path / "clean"), (2, enc.out_dim), "float32")
    extract_features(CountingEncoder(tiny_model), images, clean_store, **kw)
    np.testing.assert_allclose(resumed, clean_store.get(images), atol=1e-6)
    # running again encodes nothing
    enc3 = CountingEncoder(tiny_model)
    assert extract_features(enc3, images, clean_store, **kw)["encoded"] == 0 and enc3.calls == 0
    clean_store.close()


def test_view1_features_are_flip_features(synth, tmp_path, tiny_model):
    from mario_dino.data.images import CachedImageDataset, apply_view

    images = collect_images([synth["csv"]["task2_train"]])[:2]
    cache = str(tmp_path / "cache")
    build_cache(images, synth["images"], cache, "raw")
    store = FeatureStore(str(tmp_path / "s"), (2, 768), "float32")
    enc = DinoEncoder(tiny_model)
    extract_features(enc, images, store, cache, "raw", (28, 56), torch.device("cpu"), amp=False)
    _, x = CachedImageDataset(images, cache, "raw", (28, 56))[0]
    with torch.no_grad():
        v1 = enc(apply_view(x[None], 1))[0].numpy()
    np.testing.assert_allclose(store.get(images[:1])[0, 1], v1, atol=1e-5)
    store.close()
