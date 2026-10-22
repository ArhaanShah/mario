"""Optional GPU partial fine-tune: the last N ViT blocks + final norm + heads, with stage A's loss.

1. Token cache: blocks 0 .. depth-N-1 run once per Task 1 training image (and ``partial.token_views``) and
   their output tokens are stored in fp16 (~0.8 MB per image-view for ViT-B at 196x518). Resumable.
2. Per fold: copies of the last N blocks and the norm are trained on the cached tokens together with the
   forecaster and head (AdamW, lr ``partial.lr_blocks`` for blocks, ``partial.lr_heads`` for heads). The
   forecasting target is ``f_next.detach()``; that detach now matters because f_next carries gradients.
   Heads start from ``partial.init_heads_from``/fold{k}/task1.pt when present.

Run only after the frozen-encoder results are scored (RUNBOOK step 5): its GPU cost grows with epochs.

    python -m mario_dino.train_partial --config configs/base.yaml --device cuda

Each ``fold{k}/task1.pt`` holds the heads and the fine-tuned suffix; ``infer.py --run-dir outputs/partial
--task 1`` uses them. Task 2 is not trained here.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
from torch import nn

from .cli import base_parser, load_from_args
from .config import Config, config_to_dict
from .data import schema
from .data.images import apply_view
from .data.splits import load_splits
from .features import autocast_ctx, extract_features, store_identity
from .losses import class_weights, stage_a_loss
from .models.encoder import DinoEncoder, load_encoder
from .models.full import EncoderSuffix
from .models.heads import N_TASK1, MarioHeads, load_heads_checkpoint
from .store import FeatureStore
from .train_heads import Task1Data, _eval_value, build_task1, fit
from .utils import get_logger, resolve_device, write_json, write_provenance

log = get_logger(__name__)


class PartialNet(nn.Module):
    """Trainable copy of the encoder suffix + heads; the frozen prefix lives in the token cache."""

    def __init__(self, encoder: DinoEncoder, split: int, heads: MarioHeads):
        super().__init__()
        self.suffix = EncoderSuffix(encoder, split)
        self.heads = heads
        self.num_prefix_tokens = encoder.num_prefix_tokens

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Same pooling as DinoEncoder.pool, applied after the trainable suffix."""
        x = self.suffix(tokens)
        return torch.cat([x[:, 0], x[:, self.num_prefix_tokens :].mean(dim=1)], dim=1)


def token_split(encoder: DinoEncoder, n_blocks: int) -> int:
    if not 1 <= n_blocks <= encoder.depth:
        raise ValueError(f"partial.n_blocks must be in [1, {encoder.depth}], got {n_blocks}")
    return encoder.depth - n_blocks


def open_token_store(cfg: Config, encoder: DinoEncoder, split: int, root: str | None = None) -> FeatureStore:
    h, w = cfg.encoder.img_size
    n_tokens = (h // 14) * (w // 14) + encoder.num_prefix_tokens
    identity = {**store_identity(cfg), "kind": "prefix_tokens", "split": split,
                "token_views": list(cfg.partial.token_views)}
    return FeatureStore(root or cfg.partial.token_store_dir,
                        row_shape=(len(cfg.partial.token_views), n_tokens, encoder.embed_dim),
                        dtype="float16", identity=identity)


def build_token_cache(cfg: Config, encoder: DinoEncoder, split: int, images, device, store: FeatureStore) -> dict:
    views = list(cfg.partial.token_views)

    @torch.inference_mode()
    def encode_tokens(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        with autocast_ctx(device, cfg.features.amp):
            out = [encoder.forward_prefix(apply_view(x, v), split).float() for v in views]
        return torch.stack(out, dim=1).cpu()

    return extract_features(encoder, images, store, cfg.preprocess.cache_dir, cfg.preprocess.mode,
                            cfg.encoder.img_size, device, batch_size=cfg.features.batch_size,
                            chunk_size=cfg.partial.token_chunk_size, num_workers=cfg.features.num_workers,
                            amp=cfg.features.amp, encode_fn=encode_tokens)


def _tokens(store: FeatureStore, rows: np.ndarray, views, device) -> torch.Tensor:
    """(B, T, C) float32 tokens, one chosen view index per row."""
    arr = store.get_rows(rows)  # (B, V, T, C) fp16, read from the memmap
    return torch.from_numpy(arr[np.arange(len(rows)), np.asarray(views)]).to(device).float()


def train_partial_fold(
    cfg: Config,
    encoder: DinoEncoder,
    split: int,
    tokens: FeatureStore,
    t1: Task1Data,
    tr: np.ndarray,
    va: np.ndarray,
    heads: MarioHeads,
    device: torch.device,
    gen: torch.Generator,
    tag: str = "partial",
) -> tuple[PartialNet, list[dict], int]:
    """Train one fold; returns the PartialNet at its best epoch, the history and the best epoch.

    ``encoder`` itself is never modified: training runs on a copy of its suffix.
    """
    pc, tc = cfg.partial, cfg.train
    net = PartialNet(encoder, split, heads).to(device)
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [{"params": net.suffix.parameters(), "lr": pc.lr_blocks},
         {"params": net.heads.parameters(), "lr": pc.lr_heads}],
        weight_decay=pc.weight_decay,
    )
    w = class_weights(t1.y[tr], N_TASK1).to(device)
    y = torch.from_numpy(t1.y).to(device)
    dt = torch.from_numpy(t1.dt).to(device)
    n_views = len(pc.token_views)
    use_amp = device.type == "cuda" and cfg.features.amp and torch.cuda.is_bf16_supported()

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    def step(idx, views):
        rows = tr[idx]
        with amp():
            f_t = net.features(_tokens(tokens, t1.t_rows[rows], views, device))
            f_n = net.features(_tokens(tokens, t1.n_rows[rows], views, device))
        return stage_a_loss(net.heads, f_t.float(), f_n.float(), dt[rows], y[rows], w, tc.lambda_pred)

    @torch.no_grad()
    def predict(rows_t, rows_n) -> np.ndarray:
        out = []
        for s in range(0, len(rows_t), pc.batch_size):
            a, b = rows_t[s : s + pc.batch_size], rows_n[s : s + pc.batch_size]
            logits = 0
            for v in range(n_views):
                vv = np.full(len(a), v)
                with amp():
                    fa = net.features(_tokens(tokens, a, vv, device)).float()
                    fb = net.features(_tokens(tokens, b, vv, device)).float()
                logits = logits + net.heads.task1_logits(fa, fb)
            out.append((logits / n_views).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, N_TASK1), np.float32)

    def evaluate():
        if len(va) == 0:
            return None, {}
        return _eval_value(t1.y[va], predict(t1.t_rows[va], t1.n_rows[va]), 1, tc.select_metric, w.cpu())

    best, history, best_epoch = fit(net, opt, len(tr), step, evaluate, batch_size=pc.batch_size,
                                    max_epochs=pc.max_epochs, patience=pc.patience, gen=gen, tag=tag,
                                    n_views=n_views, max_steps=pc.max_steps)
    net.load_state_dict(best)
    return net.eval(), history, best_epoch


def train_all_partial(cfg: Config, folds: list[int] | None = None) -> dict:
    device = resolve_device(cfg.device)
    pc, mc = cfg.partial, cfg.model
    encoder = load_encoder(cfg.encoder, device)
    split = token_split(encoder, pc.n_blocks)
    t1_df = schema.read_csvs(cfg.data.task1_train_csvs, 1, image_root=cfg.data.image_root, check_images=False)
    images = schema.unique_images([(1, t1_df)])
    tokens = open_token_store(cfg, encoder, split)
    log.info("token cache: %d images x %d view(s), frozen blocks 0..%d", len(images), len(pc.token_views),
             split - 1)
    build_token_cache(cfg, encoder, split, images, device, tokens)
    t1 = build_task1(t1_df, tokens, load_splits(cfg.data.splits))
    folds = folds if folds is not None else (cfg.train.folds or list(range(cfg.data.n_folds)))
    out_dir = pc.out_dir
    write_provenance(out_dir, cfg, {"split": split, "folds": folds, "device": str(device)})
    identity = store_identity(cfg)
    summary = {}
    for k in folds:
        torch.manual_seed(cfg.seed * 1000 + k)
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + k)
        tr, va = np.where(t1.fold != k)[0], np.where(t1.fold == k)[0]
        dt_median = float(np.median(t1.dt[tr]))
        init = os.path.join(pc.init_heads_from, f"fold{k}", "task1.pt") if pc.init_heads_from else None
        if init and os.path.isfile(init):
            heads, _ = load_heads_checkpoint(init)
            log.info("fold %d: heads warm-started from %s", k, init)
        else:
            heads = MarioHeads(encoder.out_dim, mc.head_input, mc.forecaster, mc.hidden, mc.dt_dim, dt_median,
                               mc.dropout, mc.forecast_steps,
                               cfg.train.delta_t0 if cfg.train.delta_t0 is not None else dt_median)
        net, history, best_epoch = train_partial_fold(cfg, encoder, split, tokens, t1, tr, va, heads, device, gen,
                                                      tag=f"fold{k} partial")
        fold_dir = os.path.join(out_dir, f"fold{k}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(
            {"kind": "partial", "task": 1, "fold": k, "split": split, "n_blocks": pc.n_blocks,
             "heads": net.heads.state_dict(), "hparams": net.heads.hparams,
             "suffix": {n: t.cpu() for n, t in net.suffix.state_dict().items()},
             "store_meta": {"identity": identity}, "best_epoch": best_epoch, "config": config_to_dict(cfg)},
            os.path.join(fold_dir, "task1.pt"),
        )
        write_json(os.path.join(fold_dir, "history.json"), history)
        summary[k] = {"best_epoch": best_epoch, "last": history[-1] if history else None}
    tokens.close()
    write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    p = base_parser("Optional partial fine-tune of the last N encoder blocks (GPU).")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    args = p.parse_args(argv)
    train_all_partial(load_from_args(args), args.folds)


if __name__ == "__main__":
    main()
