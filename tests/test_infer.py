import os

import numpy as np
import pandas as pd
import pytest
import torch

from mario_dino.config import load_config
from mario_dino.data import schema
from mario_dino.data.images import CachedImageDataset
from mario_dino.data.splits import make_splits, save_splits
from mario_dino.features import extract_features, open_feature_store, store_identity
from mario_dino.infer import check_identity, run_inference
from mario_dino.models.encoder import load_encoder, save_weights
from mario_dino.models.full import MarioModelDINO
from mario_dino.octip_cache import build_cache, collect_images
from mario_dino.store import FeatureStore
from mario_dino.train_heads import train_all
from mario_dino.utils import REPO_ROOT


@pytest.fixture(scope="module")
def pipeline(synth, tmp_path_factory):
    """Synthetic data -> raw cache -> fp32 features (tiny random ViT) -> heads for 4 folds."""
    tmp = tmp_path_factory.mktemp("pipe")
    c = synth["csv"]
    cfg = load_config(
        os.path.join(REPO_ROOT, "configs", "smoke.yaml"),
        [
            f"data.image_root={synth['images']}",
            f"data.task1_train_csvs=[{c['task1_train']}]",
            f"data.task2_train_csvs=[{c['task2_train']}]",
            f"data.extra_csvs=[{c['task1_val']}, {c['task2_val']}]",
            f"data.splits={tmp / 'splits.json'}",
            f"preprocess.cache_dir={tmp / 'cache'}",
            f"encoder.weights={tmp / 'enc.pth'}",
            "encoder.img_size=[42, 112]",
            f"features.store_dir={tmp / 'store'}",
            "features.dtype=float32",
            f"train.out_dir={tmp / 'heads'}",
            "train.max_epochs=2",
            f"inference.run_dir={tmp / 'heads'}",
            f"inference.store_dir={tmp / 'store_infer'}",
            f"inference.out_dir={tmp / 'infer'}",
        ],
    )
    save_weights(cfg.encoder, pretrained=False, seed=0)
    t1, t2 = pd.read_csv(c["task1_train"]), pd.read_csv(c["task2_train"])
    save_splits(make_splits(t1, t2, 4), cfg.data.splits)
    images = collect_images([c["task1_train"], c["task2_train"], c["task1_val"], c["task2_val"]])
    build_cache(images, cfg.data.image_root, cfg.preprocess.cache_dir, "raw")
    enc = load_encoder(cfg.encoder)
    store = open_feature_store(cfg, cfg.features.store_dir, enc.out_dim)
    extract_features(enc, images, store, cfg.preprocess.cache_dir, "raw", cfg.encoder.img_size,
                     torch.device("cpu"), batch_size=8, chunk_size=64, amp=False)
    store.close()
    train_all(cfg)
    return cfg, enc


def _images_tensor(cfg, names):
    ds = CachedImageDataset(names, cfg.preprocess.cache_dir, cfg.preprocess.mode, cfg.encoder.img_size)
    return torch.stack([ds[i][1] for i in range(len(names))])


def test_full_model_matches_heads_on_cached_vectors(pipeline):
    cfg, enc = pipeline
    t1 = pd.read_csv(cfg.data.task1_train_csvs[0]).head(5)
    store = FeatureStore.open_existing(cfg.features.store_dir)
    va, vb = (torch.from_numpy(store.get(t1[c]).astype(np.float32)) for c in ("image_at_ti", "image_at_ti+1"))
    store.close()

    for task in (1, 2):
        model, _ = MarioModelDINO.from_run(cfg.train.out_dir, task, enc)
        assert model.n_folds == 4
        x_t = _images_tensor(cfg, list(t1["image_at_ti"]))
        if task == 1:
            full = model.forward_task1(x_t, _images_tensor(cfg, list(t1["image_at_ti+1"])))
        else:
            full = model.forward_task2(x_t)
        # heads applied directly to the stored vectors, averaged over folds and both views
        ref = 0
        with torch.no_grad():
            for h in model.heads:
                for v in range(2):
                    ref = ref + (h.task1_logits(va[:, v], vb[:, v]) if task == 1 else h.task2_logits(va[:, v]))
        ref = ref / 8
        torch.testing.assert_close(full, ref, rtol=0, atol=1e-4)


def test_infer_writes_challenge_csvs(pipeline, synth):
    cfg, _ = pipeline
    c = synth["csv"]
    r1 = run_inference(cfg, 1, c["task1_test"])
    out1 = pd.read_csv(r1["output"])
    assert list(out1.columns) == ["y_pred", "cases"]
    assert len(out1) == len(pd.read_csv(c["task1_test"])) and out1["y_pred"].between(0, 3).all()
    n_after_t1 = len(FeatureStore.open_existing(cfg.inference.store_dir))

    r2 = run_inference(cfg, 2, c["task2_test"])
    out2 = pd.read_csv(r2["output"])
    assert list(out2.columns) == ["y_pred", "cases"] and out2["y_pred"].between(0, 2).all()
    assert os.path.basename(r2["output"]) == "results_task2_team_dino_method_1.csv"
    # Task 2 images all start a Task 1 pair: the store gains no rows
    assert len(FeatureStore.open_existing(cfg.inference.store_dir)) == n_after_t1

    r3 = run_inference(cfg, 2, c["task2_val"], method=2)
    out3 = pd.read_csv(r3["output"])
    assert list(out3.columns) == ["y_true", "y_pred", "cases"]
    assert set(r3["scores"]) == {"F1_score", "Rk-correlation", "Specificity", "Quadratic-weighted_Kappa"}
    assert os.path.isfile(os.path.join(cfg.inference.out_dir, "provenance.json"))


def test_infer_rejects_features_from_other_preprocessing(pipeline):
    cfg, enc = pipeline
    _, ckpts = MarioModelDINO.from_run(cfg.train.out_dir, 1, enc)
    check_identity(ckpts, store_identity(cfg))
    cfg_octip = load_config(None, [])
    cfg_octip.encoder, cfg_octip.preprocess.mode = cfg.encoder, "octip"
    with pytest.raises(ValueError, match="preprocess_mode"):
        check_identity(ckpts, store_identity(cfg_octip))


def test_infer_validates_csv(pipeline, tmp_path):
    cfg, _ = pipeline
    bad = tmp_path / "bad.csv"
    pd.read_csv(cfg.data.task2_train_csvs[0]).drop(columns=["case"]).to_csv(bad, index=False)
    with pytest.raises(schema.SchemaError):
        run_inference(cfg, 2, str(bad))
