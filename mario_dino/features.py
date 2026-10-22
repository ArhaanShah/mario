"""Feature cache: each unique image -> pooled DINOv2 vector per view, into a resumable FeatureStore.

The only mandatory GPU step. Store rows have shape ``[2 views, 2D]``. Work runs in chunks of
``features.chunk_size`` images; a chunk is marked done only after its rows are flushed, so an interrupted
run resumes at the first unfinished chunk.

    python -m mario_dino.features --config configs/base.yaml --device cuda
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cli import base_parser, load_from_args
from .data.images import N_VIEWS, CachedImageDataset, apply_view, check_cached
from .models.encoder import DinoEncoder, load_encoder, weights_identity
from .octip_cache import collect_images
from .store import FeatureStore
from .utils import get_logger, resolve_device, write_provenance

log = get_logger(__name__)

POOLING = "concat(cls, mean(patch tokens excl. prefix/registers)) after final norm"


def autocast_ctx(device: torch.device, amp: bool):
    if device.type == "cuda" and amp:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def store_identity(cfg) -> dict:
    """What a feature store's vectors depend on; a store refuses to open with a different identity."""
    return {
        "encoder": cfg.encoder.name,
        "encoder_kwargs": cfg.encoder.kwargs,
        "img_size_hw": list(cfg.encoder.img_size),
        "weights_sha256": weights_identity(cfg.encoder),
        "preprocess_mode": cfg.preprocess.mode,
        "pooling": POOLING,
        "views": ["identity", "hflip"],
    }


def open_feature_store(cfg, root: str, out_dim: int) -> FeatureStore:
    return FeatureStore(root, row_shape=(N_VIEWS, out_dim), dtype=cfg.features.dtype, identity=store_identity(cfg))


@torch.inference_mode()
def encode_views(encoder: DinoEncoder, x: torch.Tensor, device: torch.device, amp: bool = True) -> torch.Tensor:
    """(B, 3, H, W) view-0 batch -> (B, N_VIEWS, D) float32 on CPU."""
    x = x.to(device, non_blocking=True)
    out = []
    with autocast_ctx(device, amp):
        for v in range(N_VIEWS):
            out.append(encoder(apply_view(x, v)).float())
    return torch.stack(out, dim=1).cpu()


def extract_features(
    encoder: DinoEncoder,
    images: Sequence[str],
    store: FeatureStore,
    cache_dir: str,
    mode: str,
    img_size_hw: Sequence[int],
    device: torch.device,
    batch_size: int = 64,
    chunk_size: int = 1024,
    num_workers: int = 0,
    amp: bool = True,
    on_chunk: Callable[[int, int], None] | None = None,
    encode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    """Encode every image in ``images`` whose row is not done yet. Returns counts.

    ``encode_fn`` maps a view-0 batch (B, 3, H, W) to rows (B, *store.row_shape); the default is the pooled
    two-view vector. ``train_partial`` passes a token-prefix encoder to build its token cache.
    """
    if encode_fn is None:
        def encode_fn(x):
            return encode_views(encoder, x, device, amp)
    rows = store.ensure(images)
    pending = store.pending(np.unique(rows))
    stats = {"images": len(set(images)), "already_done": len(set(rows)) - len(pending), "encoded": 0}
    if len(pending) == 0:
        return stats
    missing = check_cached([store.images[r] for r in pending], cache_dir, mode)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) missing from the {mode} cache in {cache_dir!r}, e.g. {missing[:5]}; "
            f"run python -m mario_dino.octip_cache --mode {mode} first"
        )
    n_chunks = (len(pending) + chunk_size - 1) // chunk_size
    t0 = time.time()
    for c in range(n_chunks):
        chunk = pending[c * chunk_size : (c + 1) * chunk_size]
        names = [store.images[r] for r in chunk]
        loader = DataLoader(
            CachedImageDataset(names, cache_dir, mode, img_size_hw),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        out = np.empty((len(chunk), *store.row_shape), dtype=np.float32)
        for idx, x in loader:
            out[idx.numpy()] = encode_fn(x).numpy()
        store.write(chunk, out)
        store.mark_done(chunk)
        stats["encoded"] += len(chunk)
        log.info("chunk %d/%d done (%d images, %.1fs elapsed)", c + 1, n_chunks, stats["encoded"], time.time() - t0)
        if on_chunk:
            on_chunk(c, n_chunks)
    return stats


def main(argv: list[str] | None = None) -> None:
    p = base_parser("Encode unique images (2 views each) into the feature store.")
    p.add_argument("--csv", action="append", default=None,
                   help="CSV(s) whose images to encode (default: training + extra CSVs in the config)")
    p.add_argument("--store", default=None, help="overrides features.store_dir")
    args = p.parse_args(argv)
    cfg = load_from_args(args)
    device = resolve_device(cfg.device)
    csvs = args.csv or (cfg.data.task1_train_csvs + cfg.data.task2_train_csvs + cfg.data.extra_csvs)
    images = collect_images(csvs)
    encoder = load_encoder(cfg.encoder, device)
    root = args.store or cfg.features.store_dir
    store = open_feature_store(cfg, root, encoder.out_dim)
    log.info("%d unique images, store %s holds %d rows (%d done)", len(images), root, len(store),
             int(store.done.sum()))
    stats = extract_features(
        encoder,
        images,
        store,
        cfg.preprocess.cache_dir,
        cfg.preprocess.mode,
        cfg.encoder.img_size,
        device,
        batch_size=cfg.features.batch_size,
        chunk_size=cfg.features.chunk_size,
        num_workers=cfg.features.num_workers,
        amp=cfg.features.amp,
    )
    store.close()
    write_provenance(root, cfg, {"csvs": csvs, "stats": stats, "device": str(device)})
    log.info("done: %s", stats)


if __name__ == "__main__":
    main()
