"""Challenge inference: CSV -> OCTIP cache -> features (once per unique image and view) -> fold ensemble.

    python -m mario_dino.infer --config configs/task1.yaml --task 1
    python -m mario_dino.infer --config configs/task2.yaml --task 2

Writes ``results_task{t}_team_{team}_method_{m}.csv`` with DF41's columns: ``y_true`` (only when the CSV has
labels), ``y_pred``, ``cases``. With labels it also scores (DF41's metric keys) into ``scores_task{t}.json``.

With frozen-encoder checkpoints, features go to the feature store (``inference.store_dir``, defaulting to
``features.store_dir``), so images shared between Task 1 and Task 2, or already encoded during training, are
not encoded again. Partial fine-tune checkpoints carry per-fold encoder suffixes; their features are computed
in memory (the prefix once per image and view, the suffix once per fold).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .cli import base_parser, load_from_args
from .config import Config
from .data import schema
from .data.images import N_VIEWS, CachedImageDataset
from .features import autocast_ctx, extract_features, open_feature_store, store_identity
from .models.encoder import load_encoder
from .models.full import MarioModelDINO
from .octip_cache import build_cache
from .scoring import score
from .utils import get_logger, resolve_device, write_json, write_provenance

log = get_logger(__name__)

_IDENTITY_KEYS = ("encoder", "encoder_kwargs", "img_size_hw", "weights_sha256", "preprocess_mode", "pooling")


def check_identity(ckpts: list[dict], identity: dict) -> None:
    """Refuse to apply heads to features from a different encoder/preprocessing than they were trained on."""
    for c in ckpts:
        trained = ((c.get("store_meta") or {}).get("identity")) or {}
        diff = {k: (trained.get(k), identity.get(k)) for k in _IDENTITY_KEYS if trained.get(k) != identity.get(k)}
        if diff:
            raise ValueError(
                f"fold {c.get('fold')} was trained on features that differ from this config: "
                f"{ {k: {'trained': a, 'now': b} for k, (a, b) in diff.items()} }. "
                "Pass --allow-identity-mismatch to override."
            )


@torch.no_grad()
def _features_in_memory(model: MarioModelDINO, images, cfg: Config, device, batch_size: int) -> np.ndarray:
    """(G, n_images, V, D) for partial checkpoints."""
    ds = CachedImageDataset(images, cfg.preprocess.cache_dir, cfg.preprocess.mode, cfg.encoder.img_size)
    out = None
    for idx, x in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg.features.num_workers):
        with autocast_ctx(device, cfg.features.amp):
            f = model.encode(x.to(device)).float().cpu().numpy()
        if out is None:
            out = np.zeros((f.shape[0], len(images), N_VIEWS, f.shape[-1]), dtype=np.float32)
        out[:, idx.numpy()] = f
    return out


def run_inference(
    cfg: Config,
    task: int,
    csv_path: str,
    run_dir: str | None = None,
    out_dir: str | None = None,
    method: int | None = None,
    allow_identity_mismatch: bool = False,
) -> dict:
    device = resolve_device(cfg.device)
    ic = cfg.inference
    run_dir = run_dir or ic.run_dir
    out_dir = out_dir or ic.out_dir
    method = ic.method if method is None else method
    df = schema.read_csv(csv_path, task, image_root=cfg.data.image_root, require_label=False)
    labelled = schema.has_labels(df)
    images = schema.unique_images([(task, df)])
    log.info("Task %d: %d rows, %d unique images, labels: %s", task, len(df), len(images), labelled)

    # 1. OCTIP cache for this CSV's images (skips cached files)
    build_cache(images, cfg.data.image_root, cfg.preprocess.cache_dir, cfg.preprocess.mode,
                cfg.preprocess.octip_weights_dir, cfg.preprocess.octip_batch_size)

    # 2. model and per-image features
    encoder = load_encoder(cfg.encoder, device)
    model, ckpts = MarioModelDINO.from_run(run_dir, task, encoder, device)
    identity = store_identity(cfg)
    if not allow_identity_mismatch:
        check_identity(ckpts, identity)
    if model.per_fold_features:
        feats = _features_in_memory(model, images, cfg, device, ic.batch_size)
        row = {im: i for i, im in enumerate(images)}
    else:
        store_dir = ic.store_dir or cfg.features.store_dir
        store = open_feature_store(cfg, store_dir, encoder.out_dim)
        extract_features(encoder, images, store, cfg.preprocess.cache_dir, cfg.preprocess.mode,
                         cfg.encoder.img_size, device, ic.batch_size, cfg.features.chunk_size,
                         cfg.features.num_workers, cfg.features.amp)
        feats = store.get(images).astype(np.float32)[None]  # (1, n, V, D)
        store.close()
        row = {im: i for i, im in enumerate(images)}

    # 3. fold x view ensemble on features
    feats_t = torch.from_numpy(feats)
    preds = []
    for s in range(0, len(df), 4096):
        part = df.iloc[s : s + 4096]
        if task == 1:
            a = feats_t[:, [row[str(i)] for i in part["image_at_ti"]]].to(device)
            b = feats_t[:, [row[str(i)] for i in part["image_at_ti+1"]]].to(device)
            logits = model.task1_from_features(a, b)
        else:
            a = feats_t[:, [row[str(i)] for i in part["image"]]].to(device)
            logits = model.task2_from_features(a)
        preds.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.zeros(0, np.int64)

    # 4. challenge CSV (DF41 format) and scores when labels exist
    os.makedirs(out_dir, exist_ok=True)
    out = {"y_pred": y_pred.astype(int), "cases": df["case"].to_numpy()}
    if labelled:
        out = {"y_true": df["label"].to_numpy().astype(int), **out}
    out_csv = os.path.join(out_dir, f"results_task{task}_team_{ic.team}_method_{method}.csv")
    pd.DataFrame(out).to_csv(out_csv, index=False)
    result = {"task": task, "csv": csv_path, "output": out_csv, "n_rows": len(df), "n_folds": model.n_folds,
              "partial": model.per_fold_features}
    if labelled:
        result["scores"] = score(df["label"].to_numpy(), y_pred, task)
        write_json(os.path.join(out_dir, f"scores_task{task}_method_{method}.json"), result["scores"])
        log.info("scores: %s", result["scores"])
    write_provenance(out_dir, cfg, {"inference": result, "run_dir": run_dir})
    log.info("wrote %s", out_csv)
    return result


def main(argv: list[str] | None = None) -> None:
    p = base_parser("Run challenge inference for Task 1 or Task 2.")
    p.add_argument("--task", type=int, choices=[1, 2], required=True)
    p.add_argument("--csv", default=None, help="challenge CSV (default: inference.task{1,2}_csv)")
    p.add_argument("--run-dir", default=None, help="overrides inference.run_dir")
    p.add_argument("--out-dir", default=None, help="overrides inference.out_dir")
    p.add_argument("--method", type=int, default=None, help="overrides inference.method (output file name)")
    p.add_argument("--allow-identity-mismatch", action="store_true")
    args = p.parse_args(argv)
    cfg = load_from_args(args)
    csv_path = args.csv or (cfg.inference.task1_csv if args.task == 1 else cfg.inference.task2_csv)
    if not csv_path:
        p.error(f"no CSV: pass --csv or set inference.task{args.task}_csv")
    run_inference(cfg, args.task, csv_path, args.run_dir, args.out_dir, args.method, args.allow_identity_mismatch)


if __name__ == "__main__":
    main()
