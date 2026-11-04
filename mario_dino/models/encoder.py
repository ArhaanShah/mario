"""Offline encoder factory plus DINOv2 pooling and its partial-training prefix/suffix split.

Weights are never downloaded at runtime: ``scripts/fetch_weights.py`` creates the timm model with
``pretrained=True`` and the same kwargs as here and saves its ``state_dict``; this module builds with
``pretrained=False`` and loads that file with ``strict=True``.

DINOv2 pooled vector = concat(CLS token, mean of patch tokens), prefix and register tokens excluded from the mean,
both after the final norm (the DINOv2 linear-evaluation pooling). For ViT-B that is 2 x 768 = 1536 values.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence

import torch
from torch import nn

from ..utils import get_logger

log = get_logger(__name__)


def build_timm_model(name: str, img_size_hw: Sequence[int], pretrained: bool = False, **kwargs) -> nn.Module:
    import timm

    return timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        img_size=tuple(int(s) for s in img_size_hw),
        dynamic_img_size=True,
        **kwargs,
    )


class DinoEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.num_prefix_tokens = int(model.num_prefix_tokens)
        self.embed_dim = int(model.embed_dim)
        self.depth = len(model.blocks)

    @property
    def out_dim(self) -> int:
        return 2 * self.embed_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat([tokens[:, 0], tokens[:, self.num_prefix_tokens :].mean(dim=1)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_features(x))

    def forward_prefix(self, x: torch.Tensor, upto: int) -> torch.Tensor:
        """Tokens after blocks[0 .. upto-1] (no final norm)."""
        m = self.model
        x = m.patch_embed(x)
        x = m._pos_embed(x)
        x = m.patch_drop(x)
        x = m.norm_pre(x)
        for blk in m.blocks[:upto]:
            x = blk(x)
        return x

    def forward_suffix(self, tokens: torch.Tensor, start: int) -> torch.Tensor:
        """blocks[start ..] then the final norm; forward_suffix(forward_prefix(x, k), k) == forward_features(x)."""
        x = tokens
        for blk in self.model.blocks[start:]:
            x = blk(x)
        return self.model.norm(x)

    def suffix_parameter_names(self, start: int) -> list[str]:
        names = []
        for n, _ in self.named_parameters():
            parts = n.split(".")
            if parts[:2] == ["model", "blocks"] and int(parts[2]) >= start:
                names.append(n)
            elif parts[:2] == ["model", "norm"]:
                names.append(n)
        return names


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_encoder(enc_cfg, *, native_grid_size=14):
    """Construct only: never open weights or download. Used by self-contained checkpoints."""
    kind = getattr(enc_cfg, "kind", "dinov2")
    if kind == "dinov2":
        if "dinov2" not in enc_cfg.name:
            raise ValueError("dinov2 requires a DINOv2 model name")
        return DinoEncoder(build_timm_model(enc_cfg.name, enc_cfg.img_size, **enc_cfg.kwargs))
    if kind == "convnext":
        from .convnext import ConvNeXtEncoder, build_convnext
        return ConvNeXtEncoder(build_convnext(enc_cfg.name, **enc_cfg.kwargs))
    if kind == "visionfm":
        from .visionfm import VisionFMEncoder
        return VisionFMEncoder(enc_cfg.name, native_grid_size, **enc_cfg.kwargs)
    raise ValueError(f"unknown encoder.kind {kind!r}")


def load_encoder(enc_cfg, device: torch.device | str = "cpu", *, trainable=False) -> nn.Module:
    """Load local pretrained weights strictly; frozen/eval unless explicitly trainable."""
    state, grid = None, 14
    kind = getattr(enc_cfg, "kind", "dinov2")
    if enc_cfg.weights is None:
        log.warning("encoder.weights is null: using RANDOM-INIT %s (tests only)", enc_cfg.name)
    else:
        if not os.path.isfile(enc_cfg.weights):
            raise FileNotFoundError(
                f"encoder weights {enc_cfg.weights!r} not found; prepare weights explicitly with "
                "scripts/fetch_weights.py (VisionFM OCT sources must be downloaded manually)"
            )
        if kind == "visionfm":
            from .visionfm import load_visionfm_file
            state = load_visionfm_file(enc_cfg.weights, enc_cfg.checkpoint_key)
        else:
            state = torch.load(enc_cfg.weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and state.get("kind") == "finetuned":
            raise ValueError("MARIO finetuned checkpoints are run checkpoints, not pretrained encoder weights")
        if kind == "visionfm":
            from .visionfm import backbone_state, native_grid
            state = backbone_state(state, enc_cfg.checkpoint_key)
            grid = native_grid(state)
    enc = build_encoder(enc_cfg, native_grid_size=grid)
    if state is not None:
        enc.model.load_state_dict(state, strict=True)
    enc.to(device).train(trainable)
    enc.requires_grad_(trainable)
    return enc


def weights_identity(enc_cfg) -> str | None:
    if enc_cfg.weights is None:
        return None
    side = enc_cfg.weights + ".json"
    if os.path.isfile(side):
        with open(side, encoding="utf-8") as f:
            sha = json.load(f).get("sha256")
        if sha:
            return sha
    return sha256_file(enc_cfg.weights)


def save_weights(enc_cfg, pretrained: bool, out_path: str | None = None, seed: int = 0) -> str:
    """Explicit weight preparation; VisionFM pretrained sources are imported by fetch_weights --source."""
    import timm

    out_path = out_path or enc_cfg.weights
    if out_path is None:
        raise ValueError("no output path: set encoder.weights or pass out_path")
    if not pretrained:
        torch.manual_seed(seed)
    kind = getattr(enc_cfg, "kind", "dinov2")
    if kind == "convnext":
        from .convnext import build_convnext
        model = build_convnext(enc_cfg.name, pretrained=pretrained, **enc_cfg.kwargs)
    elif kind == "visionfm":
        if pretrained:
            raise ValueError("download official OCT weights manually; use --source to import a local checkpoint")
        model = build_encoder(enc_cfg).model
    else:
        model = build_timm_model(enc_cfg.name, enc_cfg.img_size, pretrained=pretrained, **enc_cfg.kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, out_path)
    sidecar = {
        "model": enc_cfg.name,
        "img_size_hw": list(enc_cfg.img_size),
        "kwargs": enc_cfg.kwargs,
        "pretrained": pretrained,
        "random_init_seed": None if pretrained else seed,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "sha256": sha256_file(out_path),
    }
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    return out_path


def source_provenance(enc_cfg):
    """Keep the original external source identity after optional canonical import."""
    source = {"sha256": weights_identity(enc_cfg), "checkpoint_key": enc_cfg.checkpoint_key}
    if enc_cfg.weights and os.path.isfile(enc_cfg.weights + ".json"):
        with open(enc_cfg.weights + ".json", encoding="utf-8") as f:
            side = json.load(f)
        if side.get("source_sha256"):
            source = {"sha256": side["source_sha256"], "checkpoint_key": side.get("source_checkpoint_key")}
    return source
