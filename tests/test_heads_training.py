import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from mario_dino.config import load_config
from mario_dino.data.splits import make_splits, save_splits
from mario_dino.losses import class_weights, forecast_loss
from mario_dino.models.heads import Forecaster, FusionHead, MarioHeads, load_heads_checkpoint
from mario_dino.scoring import score, specificity
from mario_dino.synthetic import make_planted_feature_store, make_synthetic_dataset
from mario_dino.train_heads import train_all
from mario_dino.utils import REPO_ROOT


def test_forecaster_is_identity_at_init():
    torch.manual_seed(0)
    fc = Forecaster(16, hidden=8, dt_dim=4)
    f = torch.randn(5, 16)
    torch.testing.assert_close(fc(f, torch.full((5,), 30.0)), f)
    torch.testing.assert_close(fc(f, torch.full((5,), 30.0), steps=3), f)


def test_head_shapes_and_task2_slice():
    h = MarioHeads(16, hidden=8, dt_dim=4, delta_t0=42.0).eval()
    a, b = torch.randn(3, 16), torch.randn(3, 16)
    assert h.task1_logits(a, b).shape == (3, 4)
    full = h.head(a, h.forecast(a, torch.full((3,), 42.0)))
    torch.testing.assert_close(h.task2_logits(a), full[:, :3])
    assert FusionHead(16).fc[2].in_features == 48
    ft = MarioHeads(16, head_input="ft_only").eval()
    torch.testing.assert_close(ft.task2_logits(a), ft.head(a)[:, :3])
    assert not ft.has_forecast_loss and not MarioHeads(16, forecaster="identity").has_forecast_loss


def test_class_weights_inverse_frequency():
    w = class_weights(np.array([0, 1, 1, 1, 2, 2]), 4)
    np.testing.assert_allclose(w.numpy(), [6 / 3, 6 / 9, 6 / 6, 0.0], rtol=1e-6)


def test_forecast_target_is_detached():
    f_hat = torch.randn(4, 8, requires_grad=True)
    f_next = torch.randn(4, 8, requires_grad=True)
    forecast_loss(f_hat, f_next).backward()
    assert f_hat.grad is not None and f_next.grad is None


def test_df41_specificity_wrapper():
    y = np.array([0, 1, 2, 2, 1, 0, 3])
    p = np.array([0, 2, 2, 1, 1, 0, 3])
    cm = pd.crosstab(y, p).reindex(index=range(4), columns=range(4), fill_value=0).to_numpy()
    fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(1) - fp
    assert specificity(y, p) == pytest.approx(np.mean(tn / (tn + fp + 1e-6)))
    s2 = score(y[y < 3], p[y < 3], task=2)
    assert set(s2) == {"F1_score", "Rk-correlation", "Specificity", "Quadratic-weighted_Kappa"}
    assert set(score(y, p, task=1)) == {"F1_score", "Rk-correlation", "Specificity"}
    # the plotting-only imports of DF41's scoring.py were stubbed, not left behind
    assert "seaborn" not in sys.modules or getattr(sys.modules["seaborn"], "__file__", None)


@pytest.fixture(scope="module")
def big_csvs(tmp_path_factory):
    """CSV-only synthetic data (no images), large enough for a stable fold-validation MCC."""
    root = str(tmp_path_factory.mktemp("big"))
    return make_synthetic_dataset(root, seed=1, n_patients=40, n_heldout_patients=0, write_images=False)


def _planted_cfg(csvs, tmp_path, *overrides):
    t1 = pd.read_csv(csvs["task1_train"])
    t2 = pd.read_csv(csvs["task2_train"])
    store_dir = str(tmp_path / "store")
    make_planted_feature_store(store_dir, t1, t2, dim=32, signal=3.0, noise=1.0).close()
    splits = str(tmp_path / "splits.json")
    save_splits(make_splits(t1, t2, 4), splits)
    return load_config(
        os.path.join(REPO_ROOT, "configs", "smoke.yaml"),
        [
            f"data.task1_train_csvs=[{csvs['task1_train']}]",
            f"data.task2_train_csvs=[{csvs['task2_train']}]",
            f"data.splits={splits}",
            f"features.store_dir={store_dir}",
            f"train.out_dir={tmp_path / 'heads'}",
            "train.max_epochs=40",
            "train.patience=8",
            "train.batch_size=32",
            "model.hidden=32",
            *overrides,
        ],
    )


def test_stage_a_learns_planted_signal_and_stage_b_runs(big_csvs, tmp_path):
    cfg = _planted_cfg(big_csvs, tmp_path)
    metrics = train_all(cfg)
    mcc = metrics["task1"]["oof"]["Rk-correlation"]
    assert mcc > 0.5, f"stage A OOF MCC {mcc:.3f} should clearly beat the majority-class MCC of 0"
    out = cfg.train.out_dir
    for k in range(4):
        for t in (1, 2):
            heads, ckpt = load_heads_checkpoint(os.path.join(out, f"fold{k}", f"task{t}.pt"))
            assert ckpt["task"] == t and heads.hparams["dt_scale"] > 0
    oof2 = pd.read_csv(os.path.join(out, "oof_task2.csv"))
    assert len(oof2) == len(pd.read_csv(big_csvs["task2_train"]))
    assert oof2["y_pred"].between(0, 2).all()
    m = json.load(open(os.path.join(out, "metrics.json")))
    assert "Quadratic-weighted_Kappa" in m["task2"]["oof"]
    assert os.path.isfile(os.path.join(out, "config.resolved.yaml"))
    assert os.path.isfile(os.path.join(out, "provenance.json"))


@pytest.mark.parametrize(
    "overrides",
    [
        ["model.head_input=ft_only"],
        ["model.forecaster=identity"],
        ["train.lambda_pred=0"],
        ["train.task2_aux_from_task1_pairs=true", "train.delta_t0=30", "train.select_metric=loss"],
    ],
)
def test_ablations_and_flags_run(big_csvs, tmp_path, overrides):
    cfg = _planted_cfg(big_csvs, tmp_path, "train.max_epochs=2", "train.folds=[0]", *overrides)
    metrics = train_all(cfg)
    assert metrics["folds"] == [0]
    heads, ckpt = load_heads_checkpoint(os.path.join(cfg.train.out_dir, "fold0", "task2.pt"))
    if cfg.train.delta_t0 is not None:
        assert float(heads.delta_t0) == 30.0
