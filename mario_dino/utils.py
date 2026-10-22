"""Seeding, device selection, provenance and small file helpers. Imports no TensorFlow."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from importlib import metadata

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(REPO_ROOT, "third_party", "mario_challenge_code")
DF41_COMMIT = "4d4a11bc8fa1ee0ccf54d3337a2797390ff32ad0"

_TRACKED_PACKAGES = (
    "torch",
    "timm",
    "numpy",
    "pandas",
    "scikit-learn",
    "opencv-python-headless",
    "opencv-python",
    "PyYAML",
    "tensorflow-cpu",
    "tensorflow",
    "segmentation-models",
)


def get_logger(name: str = "mario_dino") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # the octip env has no torch
        pass


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, got {device!r}")
    return torch.device(device)


def git_commit(path: str = REPO_ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        # no git (e.g. the Docker image): the build passes the commit in
        return os.environ.get("MARIO_DINO_GIT_COMMIT")


def package_versions() -> dict[str, str]:
    out = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def write_provenance(out_dir: str, cfg=None, extra: dict | None = None) -> None:
    """Store the resolved config, git commits and package versions next to any output."""
    from .config import config_to_dict

    os.makedirs(out_dir, exist_ok=True)
    if cfg is not None:
        with open(os.path.join(out_dir, "config.resolved.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
    prov = {
        "git_commit": git_commit(),
        "third_party_commit": git_commit(THIRD_PARTY) if os.path.isdir(THIRD_PARTY) else None,
        "third_party_expected_commit": DF41_COMMIT,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "argv": sys.argv,
    }
    if extra:
        prov.update(extra)
    write_json(os.path.join(out_dir, "provenance.json"), prov)


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
