"""Strict VisionFM checkpoint import and differentiable final-normalized CLS adapter."""
import argparse
import contextlib
import importlib
import math
import pickle
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from ..utils import get_logger
from .vendor.visionfm_vit import VisionTransformer

log = get_logger(__name__)


def load_visionfm_file(path, checkpoint_key=None):
    # Upstream pretraining saves argparse.Namespace under "args", beside teacher/student.
    # Permit known metadata classes without unrestricted pickle fallback.
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if checkpoint_key and safe_globals:
        try:
            multiarray = importlib.import_module("numpy._core.multiarray")
        except ModuleNotFoundError:
            multiarray = importlib.import_module("numpy.core.multiarray")
        allowed = [
            argparse.Namespace,
            multiarray.scalar,
            (multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        ctx = safe_globals(allowed)
    else:
        ctx = contextlib.nullcontext()
    try:
        with ctx:
            return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("cannot safely read VisionFM checkpoint; select checkpoint_key explicitly and use "
                         "PyTorch >=2.4 for upstream metadata, or import a raw backbone state dict") from exc


def backbone_state(checkpoint, checkpoint_key=None):
    if checkpoint_key is not None:
        if not isinstance(checkpoint, Mapping) or checkpoint_key not in checkpoint:
            raise ValueError(f"VisionFM checkpoint has no requested key {checkpoint_key!r}")
        checkpoint = checkpoint[checkpoint_key]
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("expected a nonempty VisionFM backbone state dict")
    state, discarded, seen = {}, [], set()
    for original, tensor in checkpoint.items():
        if not isinstance(original, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("expected tensor state dict; select the wrapper using encoder.checkpoint_key")
        name = original
        while name.startswith(("module.", "backbone.")):
            name = name.split(".", 1)[1]
        if name in seen:
            raise ValueError(f"VisionFM prefix normalization collision: {original!r} -> {name!r}")
        seen.add(name)
        # Upstream iBOT pretraining heads, downstream classifier, and unused MIM token only.
        if name == "masked_embed" or name.startswith(("head.", "head_dist.", "ibot_head.", "dino_head.")):
            discarded.append(original)
        else:
            state[name] = tensor
    if discarded:
        log.info("discarded permitted VisionFM non-backbone tensors: %s", sorted(discarded))
    native_grid(state)
    return state


def native_grid(state):
    pos = state.get("pos_embed")
    if pos is None or pos.ndim != 3 or pos.shape[0] != 1 or pos.shape[2] != 768:
        raise ValueError("VisionFM requires pos_embed [1, square_grid + 1, 768]")
    n = pos.shape[1] - 1
    side = math.isqrt(n) if n > 0 else 0
    if side == 0 or side * side != n:
        raise ValueError("VisionFM pretrained positional grid must be square")
    return side


class VisionFMEncoder(nn.Module):
    def __init__(self, name="vit_base_patch16", native_grid_size=14, **kwargs):
        super().__init__()
        if name != "vit_base_patch16":
            raise ValueError(f"unsupported VisionFM name: {name!r}")
        # Architecture overrides are deliberately excluded: released weights are ViT-B/16.
        allowed = {"drop_rate", "attn_drop_rate", "drop_path_rate"}
        if set(kwargs) - allowed:
            raise ValueError(f"unsupported VisionFM kwargs: {sorted(set(kwargs) - allowed)}")
        if not isinstance(native_grid_size, int) or native_grid_size < 1:
            raise ValueError("native positional grid must be a positive integer")
        self.native_grid_size = native_grid_size
        self.model = VisionTransformer(img_size=[16 * native_grid_size], patch_size=16,
                                       embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
                                       masked_im_modeling=False, use_flash=False, **kwargs)
        self.out_dim = self.model.embed_dim

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or any(s % 16 for s in images.shape[-2:]):
            raise ValueError("VisionFM expects [B, 3, H, W] with H and W divisible by 16")
        return self.model(images)
