"""Joint Task 1 encoder/head training on cached images, without feature or token stores.

python -m mario_dino.train_finetune --config configs/task1_convnext.yaml --device cuda
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
import torch

from .cli import base_parser, load_from_args
from .config import config_to_dict
from .data import schema
from .data.images import N_VIEWS, PairedImageDataset, check_cached, resolve_input_spec
from .data.splits import assign_folds, load_splits
from .features import store_identity
from .losses import class_weights, stage_a_loss
from .models.encoder import load_encoder, source_provenance
from .models.finetuned import FORMAT_VERSION, encoder_spec
from .models.full import Task1Net
from .models.heads import N_TASK1, MarioHeads, load_heads_checkpoint
from .octip_cache import build_cache
from .train_heads import _eval_value, _metrics, _oof_frame, fit
from .utils import get_logger, resolve_device, write_json, write_provenance

log = get_logger(__name__)


def training_autocast(cfg, device):
    # Full precision when CUDA bf16 is unavailable; fp16 needs a GradScaler.
    if device.type == "cuda" and cfg.features.amp and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_pairs(net, dataset, rows, cfg, device):
    """Mean logits over the same two paired views used by legacy validation."""
    net.eval()
    result = []
    for start in range(0, len(rows), cfg.finetune.batch_size):
        selected = rows[start : start + cfg.finetune.batch_size]
        total = 0
        for view in range(N_VIEWS):
            a, b, _, _ = dataset.batch(selected, np.full(len(selected), view), device)
            with training_autocast(cfg, device):
                fa, fb = net.features(a, b)
            total = total + net.heads.task1_logits(fa.float(), fb.float())
        result.append((total / N_VIEWS).cpu())
    return torch.cat(result).numpy() if result else np.zeros((0, N_TASK1), np.float32)


def train_all_finetune(cfg, folds=None):
    if cfg.encoder.kind not in ("convnext", "visionfm"):
        raise ValueError("train_finetune supports convnext and visionfm; use train_partial for DINOv2")
    device = resolve_device(cfg.device)
    fc, mc = cfg.finetune, cfg.model
    frame = schema.read_csvs(cfg.data.task1_train_csvs, 1, image_root=cfg.data.image_roots, check_images=False)
    frame, unreadable = schema.exclude_unreadable_rows(frame, 1, cfg.data.image_roots)
    if unreadable:
        log.warning("excluded pairs referencing unreadable images: %s", unreadable)
    if not len(frame):
        raise ValueError("no readable Task 1 training pairs")
    frame = frame.reset_index(drop=True)
    assignment = assign_folds(frame, load_splits(cfg.data.splits))
    folds = folds if folds is not None else (cfg.train.folds or list(range(cfg.data.n_folds)))
    if not folds or len(set(folds)) != len(folds) or any(k < 0 or k >= cfg.data.n_folds for k in folds):
        raise ValueError("folds must be a nonempty list of distinct fold indices")
    for k in folds:
        if not np.any(assignment == k) or not np.any(assignment != k):
            raise ValueError(f"fold {k} must have nonempty patient-held-out training and validation sets")
    images = schema.unique_images([(1, frame)])
    build_cache(images, cfg.data.image_roots, cfg.preprocess.cache_dir, cfg.preprocess.mode,
                cfg.preprocess.octip_weights_dir, cfg.preprocess.octip_batch_size)
    missing = check_cached(images, cfg.preprocess.cache_dir, cfg.preprocess.mode)
    if missing:
        raise FileNotFoundError(f"retinal cache failed for {len(missing)} images: {missing[:5]}")
    dataset = PairedImageDataset(frame, cfg.preprocess.cache_dir, cfg.preprocess.mode, resolve_input_spec(cfg.encoder))
    identity = store_identity(cfg)
    write_provenance(fc.out_dir, cfg, {"folds": folds, "device": str(device), "identity": identity})
    oof, summary = [], {}
    labels = frame["label"].to_numpy(np.int64)
    gaps = frame["delta_t"].to_numpy(np.float32)
    for k in folds:
        tr, va = np.where(assignment != k)[0], np.where(assignment == k)[0]
        # Reload the same source initialization independently for each fold.
        torch.manual_seed(cfg.seed)
        encoder = load_encoder(cfg.encoder, device, trainable=True)
        torch.manual_seed(cfg.seed * 1000 + k)
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + k)
        median = float(np.median(gaps[tr]))
        heads = MarioHeads(encoder.out_dim, mc.head_input, mc.forecaster, mc.hidden, mc.dt_dim,
                           median, mc.dropout, mc.forecast_steps,
                           cfg.train.delta_t0 if cfg.train.delta_t0 is not None else median)
        if fc.init_heads_from:
            path = os.path.join(fc.init_heads_from, f"fold{k}", "task1.pt")
            initial, source = load_heads_checkpoint(path)
            if source.get("kind", "heads") != "heads" or source.get("task") != 1 or source.get("fold") != k:
                raise ValueError("head initialization must be a frozen Task 1 checkpoint for the same fold")
            if (source.get("store_meta") or {}).get("identity") != identity:
                raise ValueError("head initialization encoder/preprocessing identity differs; train fresh heads")
            if initial.hparams["d"] != encoder.out_dim:
                raise ValueError("head initialization feature dimension differs")
            # Preserve training-fold time scale/horizon and configured head architecture.
            hp = dict(initial.hparams)
            hp["dt_scale"], hp["delta_t0"] = median, heads.hparams["delta_t0"]
            if hp != heads.hparams:
                raise ValueError("head initialization architecture differs from model config")
            heads.load_state_dict(initial.state_dict(), strict=True)
            heads.set_delta_t0(hp["delta_t0"])
            # dt_scale is a scalar attribute of the forecaster, initialized above from this fold.
            del initial, source
        net = Task1Net(encoder, heads).to(device)
        opt = torch.optim.AdamW([{"params": net.encoder.parameters(), "lr": fc.lr_encoder},
                                 {"params": net.heads.parameters(), "lr": fc.lr_heads}],
                                weight_decay=fc.weight_decay)
        weights = class_weights(labels[tr], N_TASK1).to(device)

        def step(idx, views, tr=tr, net=net, weights=weights):
            a, b, dt, y = dataset.batch(tr[idx], views, device)
            with training_autocast(cfg, device):
                fa, fb = net.features(a, b)
            return stage_a_loss(net.heads, fa.float(), fb.float(), dt, y, weights, cfg.train.lambda_pred)

        def evaluate(net=net, va=va, weights=weights):
            logits = predict_pairs(net, dataset, va, cfg, device)
            return _eval_value(labels[va], logits, 1, cfg.train.select_metric, weights.cpu())

        best, history, best_epoch = fit(net, opt, len(tr), step, evaluate,
                                        batch_size=fc.batch_size, max_epochs=fc.max_epochs,
                                        patience=fc.patience, gen=gen, tag=f"fold{k} finetune",
                                        max_steps=fc.max_steps)
        net.load_state_dict(best, strict=True)
        spec = encoder_spec(cfg, encoder, identity)
        fold_dir = os.path.join(fc.out_dir, f"fold{k}")
        os.makedirs(fold_dir, exist_ok=True)
        checkpoint = {"kind": "finetuned", "format_version": FORMAT_VERSION, "task": 1, "fold": k,
                      "encoder": {n: t.detach().cpu() for n, t in net.encoder.state_dict().items()},
                      "heads": {n: t.detach().cpu() for n, t in net.heads.state_dict().items()},
                      "hparams": net.heads.hparams, "encoder_spec": spec,
                      "source": source_provenance(cfg.encoder),
                      "store_meta": {"identity": identity}, "config": config_to_dict(cfg),
                      "best_epoch": best_epoch, "dt_median_train": median}
        torch.save(checkpoint, os.path.join(fold_dir, "task1.pt"))
        write_json(os.path.join(fold_dir, "history.json"), history)
        logits = predict_pairs(net, dataset, va, cfg, device)
        predictions = _oof_frame(frame["case"].to_numpy()[va], frame["id_patient"].to_numpy()[va],
                                 k, labels[va], logits)
        predictions.to_csv(os.path.join(fold_dir, "oof_task1.csv"), index=False)
        oof.append(predictions)
        summary[k] = {"best_epoch": best_epoch, "last": history[-1] if history else None}
        # Closures otherwise retain the previous fold's model/optimizer until reassignment.
        del step, evaluate, net, opt, encoder, heads, best, checkpoint, weights
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.concat(oof, ignore_index=True).to_csv(os.path.join(fc.out_dir, "oof_task1.csv"), index=False)
    metrics = {"folds": folds, "select_metric": cfg.train.select_metric, "task1": _metrics(oof, 1)}
    write_json(os.path.join(fc.out_dir, "summary.json"), summary)
    write_json(os.path.join(fc.out_dir, "metrics.json"), metrics)
    return metrics


def main(argv=None):
    p = base_parser("Fine-tune ConvNeXt or VisionFM and MARIO Task 1 heads on image pairs.")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    args = p.parse_args(argv)
    train_all_finetune(load_from_args(args), args.folds)


if __name__ == "__main__":
    main()
