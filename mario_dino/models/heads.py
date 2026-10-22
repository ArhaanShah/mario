"""Forecaster and fusion head shared by both tasks.

* Task 1 (4 classes): ``head(f_t, f_next)``
* Task 2 (3 classes): ``head(f_t, forecaster(f_t, dt0))[:, :3]``

Ablations (config ``model.*``): ``forecaster: identity`` (f_hat = f_t) and ``head_input: ft_only``
(the head sees f_t alone; no forecaster).
"""

from __future__ import annotations

import torch
from torch import nn


class Forecaster(nn.Module):  # f_hat = f_t + W_out . GRU(dt, W_in . f_t)
    def __init__(self, d, hidden=512, dt_dim=64, dt_scale=42.0):
        super().__init__()
        self.dt_scale = dt_scale  # set to the training median of delta_t
        self.dt_embed = nn.Sequential(nn.Linear(1, dt_dim), nn.GELU())
        self.w_in = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.Tanh())
        self.rnn = nn.GRUCell(dt_dim, hidden)
        self.w_out = nn.Linear(hidden, d)
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)  # f_hat = f_t at init

    def forward(self, f_t, dt, steps=1):
        e = self.dt_embed(dt.float().view(-1, 1) / self.dt_scale)
        h = self.w_in(f_t)
        for _ in range(steps):
            h = self.rnn(e, h)
        return f_t + self.w_out(h)


class IdentityForecaster(nn.Module):  # ablation: f_hat = f_t
    def __init__(self, dt_scale=42.0):
        super().__init__()
        self.dt_scale = dt_scale

    def forward(self, f_t, dt, steps=1):
        return f_t


class FusionHead(nn.Module):  # logits over [a, b, b - a]; Task 2 keeps [:, :3]
    def __init__(self, d, n_classes=4, p=0.2):
        super().__init__()
        self.fc = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(p), nn.Linear(3 * d, n_classes))

    def forward(self, a, b):
        return self.fc(torch.cat([a, b, b - a], dim=1))


class SingleHead(nn.Module):  # ablation: logits from f_t alone
    def __init__(self, d, n_classes=4, p=0.2):
        super().__init__()
        self.fc = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, n_classes))

    def forward(self, a, b=None):
        return self.fc(a)


N_TASK1, N_TASK2 = 4, 3


class MarioHeads(nn.Module):
    """Forecaster + head; ``hparams`` rebuilds it from a checkpoint."""

    def __init__(
        self,
        d: int,
        head_input: str = "pair",
        forecaster: str = "gru",
        hidden: int = 512,
        dt_dim: int = 64,
        dt_scale: float = 42.0,
        dropout: float = 0.2,
        forecast_steps: int = 1,
        delta_t0: float = 42.0,
    ):
        super().__init__()
        self.hparams = dict(
            d=d,
            head_input=head_input,
            forecaster=forecaster,
            hidden=hidden,
            dt_dim=dt_dim,
            dt_scale=float(dt_scale),
            dropout=dropout,
            forecast_steps=forecast_steps,
            delta_t0=float(delta_t0),
        )
        self.head_input = head_input
        self.forecast_steps = forecast_steps
        if forecaster == "gru":
            self.forecaster: nn.Module = Forecaster(d, hidden, dt_dim, float(dt_scale))
        elif forecaster == "identity":
            self.forecaster = IdentityForecaster(float(dt_scale))
        else:
            raise ValueError(f"unknown forecaster {forecaster!r}")
        if head_input == "pair":
            self.head: nn.Module = FusionHead(d, N_TASK1, dropout)
        elif head_input == "ft_only":
            self.head = SingleHead(d, N_TASK1, dropout)
        else:
            raise ValueError(f"unknown head_input {head_input!r}")
        self.register_buffer("delta_t0", torch.tensor(float(delta_t0)))

    @property
    def has_forecast_loss(self) -> bool:
        """The MSE forecasting term only exists for a trainable forecaster that the head consumes."""
        return isinstance(self.forecaster, Forecaster) and self.head_input == "pair"

    def set_delta_t0(self, value: float) -> None:
        self.delta_t0.fill_(float(value))
        self.hparams["delta_t0"] = float(value)

    def forecast(self, f_t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return self.forecaster(f_t, dt, self.forecast_steps)

    def task1_logits(self, f_t: torch.Tensor, f_next: torch.Tensor) -> torch.Tensor:
        return self.head(f_t, f_next)

    def task2_logits(self, f_t: torch.Tensor, dt: torch.Tensor | None = None) -> torch.Tensor:
        if self.head_input == "ft_only":
            return self.head(f_t)[:, :N_TASK2]
        if dt is None:
            dt = self.delta_t0.expand(f_t.shape[0])
        return self.head(f_t, self.forecast(f_t, dt))[:, :N_TASK2]


def heads_from_hparams(hparams: dict) -> MarioHeads:
    return MarioHeads(**hparams)


def load_heads_checkpoint(path: str, map_location="cpu") -> tuple[MarioHeads, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    heads = heads_from_hparams(ckpt["hparams"])
    heads.load_state_dict(ckpt["heads"], strict=True)
    return heads.eval(), ckpt
