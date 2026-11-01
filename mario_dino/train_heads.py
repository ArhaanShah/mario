"""CPU training of the forecaster and fusion head from cached features, per fold.

Stage A (Task 1 pairs):  CE_w(head(f_t, f_next), y) + lambda_pred * MSE(LN(f_hat), LN(f_next.detach()))
Stage B (Task 2 rows):   starts from the stage A checkpoint; CE_w(head(f_t, forecaster(f_t, dt0))[:, :3], y)

One view is sampled per pair per step (the same view for t and t+dt); evaluation averages logits over both
views. Checkpoints are selected on fold-validation ``train.select_metric`` (MCC by default) with early
stopping. Ablations are config flags: ``model.head_input=ft_only``, ``model.forecaster=identity``,
``train.lambda_pred=0``.

    python -m mario_dino.train_heads --config configs/base.yaml
    python -m mario_dino.train_heads --config configs/base.yaml --set model.forecaster=identity \
        --set train.out_dir=outputs/abl_identity

Outputs in ``train.out_dir``: ``fold{k}/task1.pt``, ``fold{k}/task2.pt``, ``fold{k}/history.json``,
``oof_task{1,2}.csv``, ``metrics.json``, the resolved config and provenance.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cli import base_parser, load_from_args
from .config import Config, config_to_dict
from .data import schema
from .data.images import N_VIEWS
from .data.splits import assign_folds, load_splits
from .losses import class_weights, stage_a_loss, stage_b_loss
from .models.heads import N_TASK1, N_TASK2, MarioHeads
from .scoring import score, selection_value
from .store import FeatureStore
from .utils import get_logger, resolve_device, write_json, write_provenance

log = get_logger(__name__)


@dataclass
class Task1Data:
    t_rows: np.ndarray
    n_rows: np.ndarray
    dt: np.ndarray
    y: np.ndarray
    case: np.ndarray
    pid: np.ndarray
    fold: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Task2Data:
    t_rows: np.ndarray
    y: np.ndarray
    case: np.ndarray
    pid: np.ndarray
    fold: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def build_task1(df: pd.DataFrame, store: FeatureStore, splits: dict) -> Task1Data:
    return Task1Data(
        t_rows=store.rows(df["image_at_ti"]),
        n_rows=store.rows(df["image_at_ti+1"]),
        dt=df["delta_t"].to_numpy(np.float32),
        y=df["label"].to_numpy(np.int64),
        case=df["case"].to_numpy(),
        pid=df["id_patient"].to_numpy(),
        fold=assign_folds(df, splits),
    )


def build_task2(df: pd.DataFrame, store: FeatureStore, splits: dict) -> Task2Data:
    return Task2Data(
        t_rows=store.rows(df["image"]),
        y=df["label"].to_numpy(np.int64),
        case=df["case"].to_numpy(),
        pid=df["id_patient"].to_numpy(),
        fold=assign_folds(df, splits),
    )


def load_training_data(cfg: Config, store_dir: str | None = None):
    """(features [n_images, 2, d] fp16 tensor, Task1Data | None, Task2Data | None, store meta)."""
    store = FeatureStore.open_existing(store_dir or cfg.features.store_dir)
    splits = load_splits(cfg.data.splits)
    kw = dict(image_root=cfg.data.image_root, check_images=False)
    t1 = t2 = None
    task1_df = None
    task1_source_df = None
    if cfg.data.task1_train_csvs:
        task1_source_df = schema.read_csvs(cfg.data.task1_train_csvs, 1, **kw)
        task1_df, unreadable = schema.exclude_unreadable_rows(task1_source_df, 1, cfg.data.image_root)
        if unreadable:
            log.warning(
                "excluding %d Task 1 row(s) referencing %d unreadable image(s): %s",
                len(task1_source_df) - len(task1_df),
                len(unreadable),
                unreadable,
            )
        t1 = build_task1(task1_df, store, splits)
    if cfg.data.task2_train_csvs:
        task2_source_df = schema.read_csvs(cfg.data.task2_train_csvs, 2, **kw)
        if task1_source_df is None:
            raise ValueError("Task 2 training needs Task 1 rows to derive patient-grouped folds")
        task2_df = schema.attach_task2_patients(task2_source_df, task1_source_df)
        task2_df, unreadable = schema.exclude_unreadable_rows(task2_df, 2, cfg.data.image_root)
        if unreadable:
            log.warning(
                "excluding %d Task 2 row(s) referencing %d unreadable image(s): %s",
                len(task2_source_df) - len(task2_df),
                len(unreadable),
                unreadable,
            )
        t2 = build_task2(task2_df, store, splits)
    parts = ([t1.t_rows, t1.n_rows] if t1 else []) + ([t2.t_rows] if t2 else [])
    used = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
    if not store.done[used].all():
        raise RuntimeError(f"{int((~store.done[used]).sum())} training images have no features yet in "
                           f"{store.root!r}; run python -m mario_dino.features")
    feats = torch.from_numpy(store.load_all())
    meta = store.meta
    store.close()
    return feats, t1, t2, meta


# ---- prediction ---------------------------------------------------------------------------------


def _gather(feats: torch.Tensor, rows, views, device) -> torch.Tensor:
    return feats[torch.as_tensor(rows), torch.as_tensor(views)].to(device).float()


@torch.no_grad()
def predict_task1(heads: MarioHeads, feats, t_rows, n_rows, device, batch: int = 4096) -> np.ndarray:
    """Logits [n, 4], averaged over both views."""
    heads.eval()
    out = []
    for s in range(0, len(t_rows), batch):
        tr, nr = t_rows[s : s + batch], n_rows[s : s + batch]
        logits = 0
        for v in range(N_VIEWS):
            vv = np.full(len(tr), v)
            logits = logits + heads.task1_logits(_gather(feats, tr, vv, device), _gather(feats, nr, vv, device))
        out.append((logits / N_VIEWS).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, N_TASK1), np.float32)


@torch.no_grad()
def predict_task2(heads: MarioHeads, feats, t_rows, device, batch: int = 4096) -> np.ndarray:
    """Logits [n, 3], averaged over both views, at the checkpoint's delta_t0."""
    heads.eval()
    out = []
    for s in range(0, len(t_rows), batch):
        tr = t_rows[s : s + batch]
        logits = 0
        for v in range(N_VIEWS):
            logits = logits + heads.task2_logits(_gather(feats, tr, np.full(len(tr), v), device))
        out.append((logits / N_VIEWS).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, N_TASK2), np.float32)


# ---- generic early-stopping loop ----------------------------------------------------------------


def fit(
    module: torch.nn.Module,
    opt: torch.optim.Optimizer,
    n_train: int,
    step_fn: Callable[[np.ndarray, torch.Tensor], tuple[torch.Tensor, dict]],
    eval_fn: Callable[[], tuple[float | None, dict]],
    *,
    batch_size: int,
    max_epochs: int,
    patience: int,
    gen: torch.Generator,
    tag: str,
    n_views: int = N_VIEWS,
    max_steps: int | None = None,
    on_epoch: Callable[[dict], None] | None = None,
) -> tuple[dict, list[dict], int]:
    """Minibatch training with early stopping on ``eval_fn``'s value (higher is better).

    ``step_fn(idx, views)`` gets training indices and one sampled view per sample. Returns (best
    ``module.state_dict()``, history, best epoch). If ``eval_fn`` returns None (no validation rows), the last
    epoch is kept. ``max_steps`` caps optimiser steps (smoke tests).
    """
    best_value, best_state, best_epoch, bad = -np.inf, copy.deepcopy(module.state_dict()), 0, 0
    steps = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        module.train()
        perm = torch.randperm(n_train, generator=gen).numpy()
        sums: dict[str, float] = {}
        n_batches = 0
        for s in range(0, n_train, batch_size):
            if max_steps is not None and steps >= max_steps:
                break
            idx = perm[s : s + batch_size]
            views = torch.randint(0, n_views, (len(idx),), generator=gen)
            loss, parts = step_fn(idx, views)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            sums["total"] = sums.get("total", 0.0) + loss.item()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1
        module.eval()
        value, metrics = eval_fn()
        rec = {"epoch": epoch, **{f"train_{k}": v / max(n_batches, 1) for k, v in sums.items()}, **metrics}
        history.append(rec)
        if on_epoch is not None:
            # A callback keeps notebook dashboards responsive while retaining the command-line workflow.
            # It runs only after validation, which is deliberately evaluated once per epoch by the plan.
            on_epoch(dict(rec))
        if value is None or value > best_value:
            best_value = -np.inf if value is None else value
            best_state, best_epoch, bad = copy.deepcopy(module.state_dict()), epoch, 0
        else:
            bad += 1
        log.info("%s epoch %d: %s", tag, epoch, ", ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                                       if isinstance(v, float)))
        if bad >= patience or (max_steps is not None and steps >= max_steps):
            break
    return best_state, history, best_epoch


def _eval_value(y_true, logits, task: int, metric: str, weights: torch.Tensor) -> tuple[float, dict]:
    y_pred = logits.argmax(1)
    m = {f"val_{k}": v for k, v in score(y_true, y_pred, task).items()}
    val_loss = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y_true), weight=weights))
    m["val_loss"] = val_loss
    value = -val_loss if metric == "loss" else selection_value(y_true, y_pred, task, metric)
    return value, m


# ---- one fold -----------------------------------------------------------------------------------


def train_fold(
    cfg: Config,
    fold: int,
    feats: torch.Tensor,
    t1: Task1Data,
    t2: Task2Data | None,
    out_dir: str,
    device: torch.device,
    store_meta: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    d = feats.shape[-1]
    tc, mc = cfg.train, cfg.model
    gen = torch.Generator().manual_seed(cfg.seed * 1000 + fold)
    torch.manual_seed(cfg.seed * 1000 + fold)
    fold_dir = os.path.join(out_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    tr1, va1 = np.where(t1.fold != fold)[0], np.where(t1.fold == fold)[0]
    dt_median = float(np.median(t1.dt[tr1]))
    delta_t0 = float(tc.delta_t0) if tc.delta_t0 is not None else dt_median
    heads = MarioHeads(
        d,
        head_input=mc.head_input,
        forecaster=mc.forecaster,
        hidden=mc.hidden,
        dt_dim=mc.dt_dim,
        dt_scale=dt_median,
        dropout=mc.dropout,
        forecast_steps=mc.forecast_steps,
        delta_t0=delta_t0,
    ).to(device)
    common = {"fold": fold, "hparams": heads.hparams, "config": config_to_dict(cfg), "store_meta": store_meta,
              "dt_median_train": dt_median}

    def emit(stage: str, rec: dict) -> None:
        if progress_callback is not None:
            progress_callback({"time": time.time(), "fold": fold, "stage": stage, **rec})

    # ---- stage A: Task 1 pairs
    w1 = class_weights(t1.y[tr1], N_TASK1).to(device)
    y1 = torch.from_numpy(t1.y).to(device)
    dt1 = torch.from_numpy(t1.dt).to(device)

    def step_a(idx, views):
        rows = tr1[idx]
        a = _gather(feats, t1.t_rows[rows], views, device)
        b = _gather(feats, t1.n_rows[rows], views, device)
        return stage_a_loss(heads, a, b, dt1[rows], y1[rows], w1, tc.lambda_pred)

    def eval_a():
        if len(va1) == 0:
            return None, {}
        logits = predict_task1(heads, feats, t1.t_rows[va1], t1.n_rows[va1], device)
        return _eval_value(t1.y[va1], logits, 1, tc.select_metric, w1.cpu())

    loop = dict(batch_size=tc.batch_size, max_epochs=tc.max_epochs, patience=tc.patience, gen=gen)
    opt = torch.optim.AdamW(heads.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    state_a, hist_a, ep_a = fit(
        heads, opt, len(tr1), step_a, eval_a, tag=f"fold{fold} A", on_epoch=lambda rec: emit("stage_a", rec), **loop
    )
    heads.load_state_dict(state_a)
    torch.save({"kind": "heads", "task": 1, "heads": state_a, "best_epoch": ep_a, **common},
               os.path.join(fold_dir, "task1.pt"))
    logits_a = predict_task1(heads, feats, t1.t_rows[va1], t1.n_rows[va1], device)
    oof1 = _oof_frame(t1.case[va1], t1.pid[va1], fold, t1.y[va1], logits_a)
    result = {"fold": fold, "oof_task1": oof1, "history": {"stage_a": hist_a}, "best_epoch_a": ep_a}

    # ---- stage B: Task 2 rows, from the stage A checkpoint
    if t2 is not None and len(t2) > 0:
        tr2, va2 = np.where(t2.fold != fold)[0], np.where(t2.fold == fold)[0]
        w2 = class_weights(t2.y[tr2], N_TASK2).to(device)
        y2 = torch.from_numpy(t2.y).to(device)
        aux_rows = tr1[t1.y[tr1] < N_TASK2] if tc.task2_aux_from_task1_pairs else np.zeros(0, np.int64)

        def step_b(idx, views):
            rows = tr2[idx]
            a = _gather(feats, t2.t_rows[rows], views, device)
            aux = None
            if len(aux_rows):
                # an equal-sized random draw of Task 1 pairs with y < 3, forecast at their own delta_t
                pick = aux_rows[torch.randint(0, len(aux_rows), (len(idx),), generator=gen).numpy()]
                aux = (_gather(feats, t1.t_rows[pick], views, device), dt1[pick], y1[pick])
            return stage_b_loss(heads, a, y2[rows], w2, aux)

        def eval_b():
            if len(va2) == 0:
                return None, {}
            logits = predict_task2(heads, feats, t2.t_rows[va2], device)
            return _eval_value(t2.y[va2], logits, 2, tc.select_metric, w2.cpu())

        opt = torch.optim.AdamW(heads.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        state_b, hist_b, ep_b = fit(
            heads, opt, len(tr2), step_b, eval_b, tag=f"fold{fold} B", on_epoch=lambda rec: emit("stage_b", rec), **loop
        )
        heads.load_state_dict(state_b)
        torch.save({"kind": "heads", "task": 2, "heads": state_b, "best_epoch": ep_b, **common},
                   os.path.join(fold_dir, "task2.pt"))
        logits_b = predict_task2(heads, feats, t2.t_rows[va2], device)
        result["oof_task2"] = _oof_frame(t2.case[va2], t2.pid[va2], fold, t2.y[va2], logits_b)
        result["history"]["stage_b"] = hist_b
        result["best_epoch_b"] = ep_b
    write_json(os.path.join(fold_dir, "history.json"), result["history"])
    return result


def _oof_frame(case, pid, fold, y, logits) -> pd.DataFrame:
    df = pd.DataFrame({"case": case, "id_patient": pid, "fold": fold, "y_true": y, "y_pred": logits.argmax(1)})
    for c in range(logits.shape[1]):
        df[f"logit_{c}"] = logits[:, c]
    return df


def _metrics(frames: list[pd.DataFrame], task: int) -> dict:
    if not frames:
        return {}
    oof = pd.concat(frames, ignore_index=True)
    per_fold = {int(k): score(g["y_true"], g["y_pred"], task) for k, g in oof.groupby("fold") if len(g)}
    return {"per_fold": per_fold, "oof": score(oof["y_true"], oof["y_pred"], task), "n": len(oof)}


def train_all(
    cfg: Config,
    store_dir: str | None = None,
    folds: list[int] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Train every requested fold and persist one training/validation record after every epoch.

    ``progress_callback`` receives dictionaries containing the fold, stage, epoch, mean training losses and
    validation metrics.  It is intended for the notebook dashboard; command-line runs get the same data in
    ``training_progress.jsonl`` under ``train.out_dir``.
    """
    device = resolve_device(cfg.device)
    feats, t1, t2, meta = load_training_data(cfg, store_dir)
    if t1 is None:
        raise ValueError("data.task1_train_csvs is empty: stage A needs Task 1 pairs")
    folds = folds if folds is not None else (cfg.train.folds or list(range(cfg.data.n_folds)))
    out_dir = cfg.train.out_dir
    os.makedirs(out_dir, exist_ok=True)
    write_provenance(out_dir, cfg, {"store_meta": meta, "folds": folds, "device": str(device)})
    progress_path = os.path.join(out_dir, "training_progress.jsonl")
    # This file describes this invocation, rather than accidentally mixing records from a previous run.
    with open(progress_path, "w", encoding="utf-8"):
        pass

    def report(event: dict) -> None:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
            f.flush()
        if progress_callback is not None:
            progress_callback(event)

    results = [train_fold(cfg, k, feats, t1, t2, out_dir, device, meta, report) for k in folds]

    oof1 = [r["oof_task1"] for r in results]
    oof2 = [r["oof_task2"] for r in results if "oof_task2" in r]
    pd.concat(oof1, ignore_index=True).to_csv(os.path.join(out_dir, "oof_task1.csv"), index=False)
    if oof2:
        pd.concat(oof2, ignore_index=True).to_csv(os.path.join(out_dir, "oof_task2.csv"), index=False)
    metrics = {
        "folds": folds,
        "select_metric": cfg.train.select_metric,
        "ablation": {"head_input": cfg.model.head_input, "forecaster": cfg.model.forecaster,
                     "lambda_pred": cfg.train.lambda_pred},
        "best_epochs": {r["fold"]: {"a": r["best_epoch_a"], "b": r.get("best_epoch_b")} for r in results},
        "task1": _metrics(oof1, 1),
        "task2": _metrics(oof2, 2),
    }
    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    log.info("task1 OOF %s", metrics["task1"].get("oof"))
    if oof2:
        log.info("task2 OOF %s", metrics["task2"].get("oof"))
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = base_parser("Train the forecaster + fusion head on cached features (stages A and B, per fold).")
    p.add_argument("--folds", type=int, nargs="+", default=None, help="overrides train.folds")
    p.add_argument("--store", default=None, help="overrides features.store_dir")
    args = p.parse_args(argv)
    cfg = load_from_args(args)
    train_all(cfg, args.store, args.folds)


if __name__ == "__main__":
    main()
