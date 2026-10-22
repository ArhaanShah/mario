"""Class-weighted CE, the feature-space forecasting loss, and the stage A / stage B objectives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models.heads import MarioHeads


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights N / (C * n_c); absent classes get 0."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes)
    present = counts > 0
    w[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(w, dtype=torch.float32)


def forecast_loss(f_hat: torch.Tensor, f_next: torch.Tensor) -> torch.Tensor:
    """MSE(LN(f_hat), LN(f_next)) with the target detached: the forecaster chases the encoder, not vice versa."""
    d = f_hat.shape[-1:]
    return F.mse_loss(F.layer_norm(f_hat, d), F.layer_norm(f_next.detach(), d))


def stage_a_loss(
    heads: MarioHeads,
    f_t: torch.Tensor,
    f_next: torch.Tensor,
    dt: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    lambda_pred: float,
) -> tuple[torch.Tensor, dict]:
    """CE_w(head(f_t, f_next), y) + lambda_pred * MSE(LN(f_hat), LN(f_next.detach()))."""
    ce = F.cross_entropy(heads.task1_logits(f_t, f_next), y, weight=weights)
    parts = {"ce": ce.item()}
    loss = ce
    if lambda_pred > 0 and heads.has_forecast_loss:
        mse = forecast_loss(heads.forecast(f_t, dt), f_next)
        loss = loss + lambda_pred * mse
        parts["mse"] = mse.item()
    return loss, parts


def stage_b_loss(
    heads: MarioHeads,
    f_t: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    aux: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict]:
    """CE_w(head(f_t, forecaster(f_t, dt0))[:, :3], y), plus the optional Task 1-pair auxiliary term.

    ``aux`` = (f_t, delta_t, y) from Task 1 pairs with y < 3, forecast at each pair's own delta_t.
    """
    ce = F.cross_entropy(heads.task2_logits(f_t), y, weight=weights)
    parts = {"ce": ce.item()}
    loss = ce
    if aux is not None and len(aux[2]) > 0:
        a_ft, a_dt, a_y = aux
        aux_ce = F.cross_entropy(heads.task2_logits(a_ft, a_dt), a_y, weight=weights)
        loss = loss + aux_ce
        parts["aux_ce"] = aux_ce.item()
    return loss, parts
