"""Explicit offline-runtime weight preparation for DINOv2, ConvNeXt, and local VisionFM sources.

DINOv2 and ConvNeXt download only here, via timm pretrained=True. Their constructor
kwargs match runtime, including headless ConvNeXt native pooling. VisionFM uses
--source to strictly import manually downloaded OCT weights. --random-init is for
offline fixtures only. Output sidecars record the SHA-256 and source provenance.
"""

import argparse
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mario_dino.config import load_config  # noqa: E402
from mario_dino.models.encoder import load_encoder, save_weights, sha256_file  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--random-init", action="store_true", help="save random weights (no download)")
    p.add_argument("--source", default=None, help="import local VisionFM weights using configured checkpoint_key")
    p.add_argument("--out", default=None, help="output path (default: encoder.weights)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = args.out or cfg.encoder.weights
    if out and os.path.isfile(out) and not args.force:
        print(f"{out} exists; pass --force to overwrite")
        return
    if args.source:
        if cfg.encoder.kind != "visionfm" or args.random_init:
            p.error("--source is for pretrained VisionFM imports only")
        if not out or os.path.abspath(args.source) == os.path.abspath(out):
            p.error("--source requires a distinct output path")
        enc = load_encoder(dataclasses.replace(cfg.encoder, weights=args.source))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        torch.save(enc.model.state_dict(), out + ".tmp")
        os.replace(out + ".tmp", out)
        with open(out + ".json", "w", encoding="utf-8") as f:
            json.dump({"model": cfg.encoder.name, "sha256": sha256_file(out),
                       "source_sha256": sha256_file(args.source), "source_checkpoint_key": cfg.encoder.checkpoint_key,
                       "checkpoint_key": None, "native_grid_size": enc.native_grid_size,
                       "pretrained": True}, f, indent=2)
        path = out
        print("Canonical VisionFM backbone saved; use encoder.checkpoint_key=null for this file")
    else:
        path = save_weights(cfg.encoder, pretrained=not args.random_init, out_path=out, seed=cfg.seed)
    print(f"saved {cfg.encoder.name} ({'random init' if args.random_init else 'pretrained'}) to {path}")


if __name__ == "__main__":
    main()
