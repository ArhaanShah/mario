"""Save the DINOv2 encoder state_dict to ``encoder.weights`` so nothing downloads at runtime.

    python scripts/fetch_weights.py --config configs/base.yaml                 # needs network (timm/HF hub)
    python scripts/fetch_weights.py --config configs/smoke.yaml --random-init  # no network; smoke/tests only

The model is created with the same kwargs as ``mario_dino.models.encoder`` (num_classes=0, img_size from
the config, dynamic_img_size=True), so the positional embedding is resampled once here and the runtime
load is ``strict=True``. A JSON sidecar records the sha256, timm version and whether weights are pretrained.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mario_dino.config import load_config  # noqa: E402
from mario_dino.models.encoder import save_weights  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--random-init", action="store_true", help="save random weights (no download)")
    p.add_argument("--out", default=None, help="output path (default: encoder.weights)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = args.out or cfg.encoder.weights
    if out and os.path.isfile(out) and not args.force:
        print(f"{out} exists; pass --force to overwrite")
        return
    path = save_weights(cfg.encoder, pretrained=not args.random_init, out_path=out, seed=cfg.seed)
    print(f"saved {cfg.encoder.name} ({'random init' if args.random_init else 'pretrained'}) to {path}")


if __name__ == "__main__":
    main()
