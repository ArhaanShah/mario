"""Arguments shared by every CLI: --config, --device and --set overrides."""

from __future__ import annotations

import argparse

from .config import Config, load_config
from .utils import seed_everything


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/base.yaml", help="YAML config (may set `base:`)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="overrides config.device")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. --set model.forecaster=identity (repeatable)",
    )
    return p


def load_from_args(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, args.overrides)
    if args.device is not None:
        cfg.device = args.device
    seed_everything(cfg.seed)
    return cfg
