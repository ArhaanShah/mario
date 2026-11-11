"""Opt-in Task 1 inner selection, fresh refit, and one outer evaluation.

This module executes models ONLY when explicitly invoked through train_finetune
with research.protocol=grouped_holdout. Study preparation does not import it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import config_to_dict, validate
from .data import schema
from .data.images import PairedImageDataset, check_cached, resolve_input_spec
from .features import store_identity
from .losses import class_weights, stage_a_loss
from .models.encoder import load_encoder, source_provenance
from .models.finetuned import FORMAT_VERSION, encoder_spec
from .models.full import Task1Net
from .models.heads import MarioHeads
from .octip_cache import build_cache
from .study import check_cohort, json_hash, sha256
from .train_heads import _eval_value, _metrics, _oof_frame, fit
from .utils import resolve_device, seed_everything, write_json, write_provenance


def train_research(cfg, folds=None):
    from .train_finetune import predict_pairs, training_autocast

    validate(cfg)
    if cfg.research.protocol != "grouped_holdout":
        raise ValueError("train_research requires grouped_holdout")
    frame = schema.read_csvs(cfg.data.task1_train_csvs, 1, image_root=cfg.data.image_roots, check_images=False)
    frame, _ = schema.exclude_unreadable_rows(frame, 1, cfg.data.image_roots)
    frame = frame.reset_index(drop=True)
    cohort, outer, inner = check_cohort(cfg, frame)
    folds = folds if folds is not None else (cfg.train.folds or list(range(4)))
    if not folds or len(set(folds)) != len(folds) or any(k not in range(4) for k in folds):
        raise ValueError("research folds must be distinct indices in [0, 3]")
    root = Path(cfg.finetune.out_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("research output is nonempty; use a separate directory (resume is not implemented)")
    device = resolve_device(cfg.device)
    identity = store_identity(cfg)
    provenance = {"protocol": cfg.research.protocol, "seed": cfg.seed, "folds": folds,
                  "dataset_hash": json_hash(cohort), "inner_split_hash": json_hash(inner),
                  "outer_splits_sha256": sha256(cfg.data.splits), "config_hash": json_hash(config_to_dict(cfg)),
                  "primary_endpoint": "MCC", "identity": identity}
    write_provenance(str(root), cfg, provenance)
    images = schema.unique_images([(1, frame)])
    build_cache(images, cfg.data.image_roots, cfg.preprocess.cache_dir, cfg.preprocess.mode,
                cfg.preprocess.octip_weights_dir, cfg.preprocess.octip_batch_size)
    missing = check_cached(images, cfg.preprocess.cache_dir, cfg.preprocess.mode)
    if missing:
        raise FileNotFoundError(f"retinal cache missing {len(missing)} images")
    dataset = PairedImageDataset(frame, cfg.preprocess.cache_dir, cfg.preprocess.mode, resolve_input_spec(cfg.encoder))
    labels, gaps, patients = frame.label.to_numpy(np.int64), frame.delta_t.to_numpy(np.float32), frame.id_patient.astype(str)
    fc, mc = cfg.finetune, cfg.model

    def fresh_net(rows, fold):
        # Same pretrained source and deterministic independent initialization in BOTH phases.
        seed_everything(cfg.seed)
        enc = load_encoder(cfg.encoder, device, trainable=True)
        seed_everything(cfg.seed * 1000 + fold)
        median = float(np.median(gaps[rows]))
        heads = MarioHeads(enc.out_dim, mc.head_input, mc.forecaster, mc.hidden, mc.dt_dim,
                           median, mc.dropout, mc.forecast_steps,
                           cfg.train.delta_t0 if cfg.train.delta_t0 is not None else median,
                           fusion=mc.fusion, fusion_hidden=mc.fusion_hidden)
        net = Task1Net(enc, heads).to(device)
        opt = torch.optim.AdamW([{"params": net.encoder.parameters(), "lr": fc.lr_encoder},
                                 {"params": net.heads.parameters(), "lr": fc.lr_heads}], weight_decay=fc.weight_decay)
        return net, opt, class_weights(labels[rows], 4).to(device), median

    def save_checkpoint(net, fold, epoch, median, path, phase):
        ckpt = {"kind": "finetuned", "format_version": FORMAT_VERSION, "task": 1, "fold": fold,
                "encoder": {n: t.detach().cpu() for n, t in net.encoder.state_dict().items()},
                "heads": {n: t.detach().cpu() for n, t in net.heads.state_dict().items()},
                "hparams": net.heads.hparams, "encoder_spec": encoder_spec(cfg, net.encoder, identity),
                "source": source_provenance(cfg.encoder), "store_meta": {"identity": identity},
                "config": config_to_dict(cfg), "best_epoch": epoch, "dt_median_train": median,
                "research": {**provenance, "phase": phase}}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        return sha256(path)

    all_oof, summary = [], {}
    for fold in folds:
        groups = inner["folds"][str(fold)]
        fit_rows = np.flatnonzero(patients.isin(groups["fit"]))
        selection = np.flatnonzero(patients.isin(groups["selection"]))
        test = np.flatnonzero(patients.isin(groups["test"]))
        outer_train = np.flatnonzero(~patients.isin(groups["test"]))
        fold_dir = root / f"fold{fold}"
        phase_histories = {}
        best_epoch = None
        for phase, rows in (("selection", fit_rows), ("refit", outer_train)):
            net, opt, weights, median = fresh_net(rows, fold)

            def step(idx, views, rows=rows, net=net, weights=weights):
                a, b, dt, y = dataset.batch(rows[idx], views, device)
                with training_autocast(cfg, device):
                    fa, fb = net.features(a, b)
                return stage_a_loss(net.heads, fa.float(), fb.float(), dt, y, weights, cfg.train.lambda_pred)

            def evaluate(net=net, weights=weights, phase=phase, selection=selection):
                if phase == "refit":
                    # No outer labels, predictions or metrics enter the fitting loop.
                    return None, {}
                logits = predict_pairs(net, dataset, selection, cfg, device)
                return _eval_value(labels[selection], logits, 1, "mcc", weights.cpu())

            epochs = fc.max_epochs if phase == "selection" else best_epoch
            best, history, epoch = fit(net, opt, len(rows), step, evaluate,
                                       batch_size=fc.batch_size, max_epochs=epochs,
                                       patience=fc.patience if phase == "selection" else epochs + 1,
                                       gen=torch.Generator().manual_seed(cfg.seed * 1000 + fold),
                                       tag=f"fold{fold} {phase}")
            net.load_state_dict(best, strict=True)
            phase_histories[phase] = history
            phase_dir = fold_dir / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            write_json(str(phase_dir / "history.json"), history)
            if phase == "selection":
                best_epoch = epoch
                digest = save_checkpoint(net, fold, epoch, median, phase_dir / "task1.pt", phase)
            else:
                digest = save_checkpoint(net, fold, epoch, median, fold_dir / "task1.pt", phase)
                # Exactly one final outer evaluation per refit model; no fold ensemble here.
                logits = predict_pairs(net, dataset, test, cfg, device)
                predictions = _oof_frame(frame.case.to_numpy()[test], frame.id_patient.to_numpy()[test],
                                         fold, labels[test], logits)
                predictions.to_csv(fold_dir / "oof_task1.csv", index=False)
                all_oof.append(predictions)
            write_json(str(phase_dir / "provenance.json"), {
                **provenance, "phase": phase, "fold": fold, "epoch_count": epoch,
                "fit_patients": sorted(set(patients.iloc[rows])), "dt_median_train": median,
                "class_support": np.bincount(labels[rows], minlength=4).tolist(),
                "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
                "checkpoint": "task1.pt" if phase == "selection" else "../task1.pt", "checkpoint_sha256": digest,
            })
            # Release phase-specific closures before constructing the next full encoder.
            del step, evaluate, net, opt, weights, best
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_json(str(fold_dir / "history.json"), phase_histories)
        summary[fold] = {"selected_epochs": best_epoch, "outer_test_patients": groups["test"]}
    pd.concat(all_oof, ignore_index=True).to_csv(root / "oof_task1.csv", index=False)
    metrics = {"folds": folds, "select_metric": "mcc", "protocol": cfg.research.protocol,
               "ablation": {"fusion": mc.fusion, "lambda_pred": cfg.train.lambda_pred},
               "task1": _metrics(all_oof, 1)}
    write_json(str(root / "metrics.json"), metrics)
    write_json(str(root / "summary.json"), summary)
    return metrics
