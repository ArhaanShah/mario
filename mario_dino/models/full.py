"""MarioModelDINO: encoder + pooling + per-fold forecaster/head, for inference.

Logits are averaged over folds and over both views, as DF41's ``simple_ensemble_inference`` averages its
fold models. The frozen-encoder path computes one feature vector per image and view, shared by every fold.
Checkpoints from ``train_partial`` also carry fine-tuned encoder suffixes (last N blocks + final norm); then
the prefix runs once per image and view and each fold runs its own suffix.
"""

from __future__ import annotations

import copy
import glob
import os
import re

import torch
from torch import nn

from ..data.images import N_VIEWS, apply_view
from .encoder import DinoEncoder
from .heads import MarioHeads, heads_from_hparams


class EncoderSuffix(nn.Module):
    """Copies of blocks[split:] and the final norm of a DinoEncoder."""

    def __init__(self, encoder: DinoEncoder, split: int):
        super().__init__()
        # ModuleList re-indexes from 0 (slicing a Sequential would keep the original block numbers)
        self.blocks = nn.ModuleList(copy.deepcopy(list(encoder.model.blocks[split:])))
        self.norm = copy.deepcopy(encoder.model.norm)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


def suffix_state_from_encoder(encoder: DinoEncoder, split: int) -> dict[str, torch.Tensor]:
    """State dict in EncoderSuffix layout taken from a (fine-tuned) encoder."""
    out = {}
    for name, t in encoder.model.state_dict().items():
        parts = name.split(".")
        if parts[0] == "blocks" and int(parts[1]) >= split:
            out[".".join(["blocks", str(int(parts[1]) - split), *parts[2:]])] = t.detach().cpu().clone()
        elif parts[0] == "norm":
            out[name] = t.detach().cpu().clone()
    return out


def merge_suffix_into_encoder(encoder: DinoEncoder, split: int, suffix_state: dict) -> dict[str, torch.Tensor]:
    """Full timm state dict of ``encoder`` with blocks[split:] and norm replaced by ``suffix_state``."""
    state = {k: v.detach().cpu().clone() for k, v in encoder.model.state_dict().items()}
    for name, t in suffix_state.items():
        parts = name.split(".")
        if parts[0] == "blocks":
            name = ".".join(["blocks", str(int(parts[1]) + split), *parts[2:]])
        if name not in state:
            raise KeyError(f"suffix key {name!r} not in encoder")
        state[name] = t.detach().cpu().clone()
    return state


class MarioModelDINO(nn.Module):
    def __init__(
        self,
        encoder: DinoEncoder,
        heads: list[MarioHeads],
        suffix_states: list[dict] | None = None,
        split: int | None = None,
    ):
        super().__init__()
        if not heads:
            raise ValueError("need at least one fold's heads")
        self.encoder = encoder
        self.heads = nn.ModuleList(heads)
        self.split = split
        self.suffixes: nn.ModuleList | None = None
        if suffix_states is not None:
            if split is None or len(suffix_states) != len(heads):
                raise ValueError("partial checkpoints need a split and one suffix per fold")
            self.suffixes = nn.ModuleList()
            for st in suffix_states:
                s = EncoderSuffix(encoder, split)
                s.load_state_dict(st, strict=True)
                self.suffixes.append(s)
        self.eval()

    @property
    def n_folds(self) -> int:
        return len(self.heads)

    @property
    def per_fold_features(self) -> bool:
        return self.suffixes is not None

    # ---- encoding -----------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) view-0 images -> features (G, B, N_VIEWS, D) with G = 1 (frozen) or n_folds."""
        per_view = []
        for v in range(N_VIEWS):
            xv = apply_view(x, v)
            if self.suffixes is None:
                per_view.append(self.encoder(xv)[None])
            else:
                tokens = self.encoder.forward_prefix(xv, self.split)
                per_view.append(torch.stack([self.encoder.pool(s(tokens)) for s in self.suffixes]))
        return torch.stack(per_view, dim=2).float()

    def _fold_feats(self, feats: torch.Tensor, f: int) -> torch.Tensor:
        return feats[f if feats.shape[0] > 1 else 0]

    # ---- heads on features --------------------------------------------------------------------
    @torch.no_grad()
    def task1_from_features(self, f_t: torch.Tensor, f_next: torch.Tensor) -> torch.Tensor:
        """(G, B, V, D) x2 -> logits (B, 4), averaged over folds and views."""
        total = 0
        for f, h in enumerate(self.heads):
            a, b = self._fold_feats(f_t, f), self._fold_feats(f_next, f)
            for v in range(N_VIEWS):
                total = total + h.task1_logits(a[:, v], b[:, v])
        return total / (self.n_folds * N_VIEWS)

    @torch.no_grad()
    def task2_from_features(self, f_t: torch.Tensor) -> torch.Tensor:
        """(G, B, V, D) -> logits (B, 3) at each fold's delta_t0, averaged over folds and views."""
        total = 0
        for f, h in enumerate(self.heads):
            a = self._fold_feats(f_t, f)
            for v in range(N_VIEWS):
                total = total + h.task2_logits(a[:, v])
        return total / (self.n_folds * N_VIEWS)

    # ---- end to end on images -----------------------------------------------------------------
    @torch.no_grad()
    def forward_task1(self, x_t: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        return self.task1_from_features(self.encode(x_t), self.encode(x_next))

    @torch.no_grad()
    def forward_task2(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.task2_from_features(self.encode(x_t))

    # ---- loading ------------------------------------------------------------------------------
    @classmethod
    def from_run(cls, run_dir: str, task: int, encoder: DinoEncoder, device="cpu") -> tuple[MarioModelDINO, list]:
        """Load every ``fold*/task{task}.pt`` in a train_heads or train_partial run directory."""
        paths = sorted(
            glob.glob(os.path.join(run_dir, "fold*", f"task{task}.pt")),
            key=lambda p: int(re.search(r"fold(\d+)", p).group(1)),
        )
        if not paths:
            raise FileNotFoundError(f"no fold*/task{task}.pt checkpoints in {run_dir!r}")
        ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        heads = []
        for c in ckpts:
            h = heads_from_hparams(c["hparams"])
            h.load_state_dict(c["heads"], strict=True)
            heads.append(h)
        kinds = {c.get("kind", "heads") for c in ckpts}
        if kinds == {"heads"}:
            model = cls(encoder, heads)
        elif kinds == {"partial"}:
            splits = {c["split"] for c in ckpts}
            if len(splits) != 1:
                raise ValueError(f"fold checkpoints disagree on the encoder split: {splits}")
            model = cls(encoder, heads, [c["suffix"] for c in ckpts], splits.pop())
        else:
            raise ValueError(f"mixed checkpoint kinds in {run_dir!r}: {kinds}")
        return model.to(device).eval(), ckpts
