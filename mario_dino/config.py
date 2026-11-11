"""YAML -> dataclasses.

A YAML file may name a parent with ``base: other.yaml`` (resolved relative to the file); the child is
deep-merged over the parent. ``--set a.b=value`` overrides are applied last and parsed as YAML scalars.
Unknown keys are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import os
import types
import typing
from dataclasses import dataclass, field
from typing import Any

import yaml

# octip: DF41's OCTIP (FPN localizers, weights on request) | classic: weight-free localizers + OCTIP
# PreProcessor | raw: original images
MODES = ("octip", "classic", "raw")


@dataclass
class DataCfg:
    image_root: str = "data/images"
    image_fallback_roots: list[str] = field(default_factory=list)
    include_image_dirs: list[str] = field(default_factory=list)
    task1_train_csvs: list[str] = field(default_factory=list)
    task2_train_csvs: list[str] = field(default_factory=list)
    # val/test CSVs whose images should also be cached and encoded
    extra_csvs: list[str] = field(default_factory=list)
    splits: str = "outputs/splits.json"
    n_folds: int = 4

    @property
    def image_roots(self) -> list[str]:
        return [self.image_root, *self.image_fallback_roots]


@dataclass
class PreprocessCfg:
    mode: str = "classic"  # octip | classic | raw
    cache_dir: str = "cache"
    octip_weights_dir: str = "weights/octip"
    octip_batch_size: int = 16


@dataclass
class EncoderCfg:
    name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    weights: str | None = "weights/vit_base_patch14_reg4_dinov2.lvd142m.pth"
    img_size: list[int] = field(default_factory=lambda: [196, 518])  # H, W; both multiples of 14
    kwargs: dict[str, Any] = field(default_factory=dict)  # extra timm kwargs, e.g. {depth: 2} for smoke
    kind: str = "dinov2"
    checkpoint_key: str | None = None
    preprocessing: str = "legacy_dino"


@dataclass
class FeaturesCfg:
    store_dir: str = "features/store"
    dtype: str = "float16"  # float16 | float32
    batch_size: int = 64
    chunk_size: int = 1024  # images per resumable chunk
    num_workers: int = 4
    amp: bool = True  # bf16 (or fp16) autocast on CUDA; always fp32 on CPU


@dataclass
class ModelCfg:
    head_input: str = "pair"  # pair | ft_only  (ablation: head on f_t alone)
    forecaster: str = "gru"  # gru | identity  (ablation: f_hat = f_t)
    hidden: int = 512
    dt_dim: int = 64
    dropout: float = 0.2
    forecast_steps: int = 1
    fusion: str = "legacy"  # legacy | delta_mlp | other_gate (Task 1 only)
    fusion_hidden: int = 256


@dataclass
class TrainCfg:
    out_dir: str = "outputs/heads"
    folds: list[int] | None = None  # None = all folds
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_pred: float = 1.0
    select_metric: str = "mcc"  # mcc | f1 | specificity | qwk | loss
    task2_aux_from_task1_pairs: bool = False
    delta_t0: float | None = None  # Task 2 horizon; None = training-fold median of delta_t


@dataclass
class PartialCfg:
    n_blocks: int = 2  # number of last ViT blocks to fine-tune
    lr_blocks: float = 1e-5
    lr_heads: float = 1e-4
    weight_decay: float = 1e-4
    token_store_dir: str = "features/tokens"
    token_views: list[int] = field(default_factory=lambda: [0])
    token_chunk_size: int = 256
    batch_size: int = 32
    max_epochs: int = 10
    patience: int = 3
    max_steps: int | None = None  # cap optimiser steps per fold (smoke tests)
    out_dir: str = "outputs/partial"
    init_heads_from: str | None = None  # train_heads run dir; warm-start heads per fold


@dataclass
class FinetuneCfg:
    """Full ConvNeXt/VisionFM Task 1 training; independent of partial.init_heads_from."""
    out_dir: str = "outputs/finetune"
    batch_size: int = 8
    lr_encoder: float = 1e-5
    lr_heads: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 10
    patience: int = 3
    max_steps: int | None = None
    init_heads_from: str | None = None


@dataclass
class InferenceCfg:
    run_dir: str = "outputs/heads"  # train_heads, train_partial or train_finetune output
    store_dir: str | None = None  # frozen runs only; None = features.store_dir; full/partial use memory
    out_dir: str = "outputs/infer"
    team: str = "dino"
    method: int = 1
    task1_csv: str | None = None
    task2_csv: str | None = None
    batch_size: int = 64


@dataclass
class ResearchCfg:
    """Opt-in patient-held-out selection/refit; legacy training stays the default."""

    protocol: str = "legacy"  # legacy | grouped_holdout
    inner_splits: str | None = None
    cohort_manifest: str | None = None


@dataclass
class Config:
    seed: int = 0
    device: str = "auto"
    class_names: dict[str, list[str]] = field(
        default_factory=lambda: {
            "task1": ["class_0", "class_1", "class_2", "class_3"],
            "task2": ["class_0", "class_1", "class_2"],
        }
    )
    data: DataCfg = field(default_factory=DataCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    partial: PartialCfg = field(default_factory=PartialCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    finetune: FinetuneCfg = field(default_factory=FinetuneCfg)
    research: ResearchCfg = field(default_factory=ResearchCfg)


class ConfigError(ValueError):
    pass


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], where)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return _from_dict(tp, value, where)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_coerce(v, args[0], f"{where}[{i}]") if args else v for i, v in enumerate(value)]
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return dict(value)
    if tp is float:
        # PyYAML reads "1e-3" as a string; accept it
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{where}: expected a float, got {value!r}") from e
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an int, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    return value


def _from_dict(cls: type, d: dict[str, Any], where: str = "config") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; valid keys are {sorted(names)}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}") for k, v in d.items()}
    return cls(**kwargs)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_tree(path: str, seen: tuple[str, ...] = ()) -> dict:
    path = os.path.abspath(path)
    if path in seen:
        raise ConfigError(f"circular 'base' chain: {' -> '.join(seen + (path,))}")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = d.pop("base", None)
    if parent is None:
        return d
    parent_path = os.path.join(os.path.dirname(path), parent)
    return _deep_merge(_load_yaml_tree(parent_path, seen + (path,)), d)


def apply_overrides(d: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` strings; values are parsed as YAML (so ``null``, ``[0, 1]``, ``1e-3`` work)."""
    d = dict(d)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        parts = key.strip().split(".")
        node = d
        for p in parts[:-1]:
            nxt = node.get(p)
            nxt = dict(nxt) if isinstance(nxt, dict) else {}
            node[p] = nxt
            node = nxt
        node[parts[-1]] = value
    return d


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    d = _load_yaml_tree(path) if path else {}
    d = apply_overrides(d, overrides)
    cfg = _from_dict(Config, d)
    validate(cfg)
    return cfg


def config_from_dict(d: dict) -> Config:
    cfg = _from_dict(Config, d)
    validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def validate(cfg: Config) -> None:
    if cfg.model.fusion not in ("legacy", "delta_mlp", "other_gate") or cfg.model.fusion_hidden < 1:
        raise ConfigError("invalid model.fusion or model.fusion_hidden")
    if cfg.model.fusion != "legacy":
        if cfg.model.head_input != "pair" or cfg.data.task2_train_csvs:
            raise ConfigError("experimental fusion requires pair inputs and no Task 2 training CSVs")
    rc = cfg.research
    if rc.protocol not in ("legacy", "grouped_holdout"):
        raise ConfigError("research.protocol must be legacy or grouped_holdout")
    if rc.protocol == "grouped_holdout":
        if not rc.inner_splits or not rc.cohort_manifest:
            raise ConfigError("grouped_holdout requires saved inner_splits and cohort_manifest")
        if cfg.encoder.kind != "convnext" or cfg.data.n_folds != 4 or cfg.data.task2_train_csvs:
            raise ConfigError("the research protocol requires four-fold ConvNeXt Task 1 training")
        if not cfg.encoder.weights:
            raise ConfigError("the research protocol requires a local pretrained ConvNeXt source")
        if cfg.train.select_metric != "mcc" or cfg.finetune.init_heads_from or cfg.finetune.max_steps:
            raise ConfigError("the research protocol requires MCC, fresh heads and uncapped epochs")
    if cfg.preprocess.mode not in MODES:
        raise ConfigError(f"preprocess.mode must be one of {MODES}, got {cfg.preprocess.mode!r}")
    ec = cfg.encoder
    if ec.kind not in ("dinov2", "convnext", "visionfm"):
        raise ConfigError(f"unknown encoder.kind {ec.kind!r}")
    if ec.preprocessing not in ("legacy_dino", "imagenet", "visionfm_oct"):
        raise ConfigError(f"unknown encoder.preprocessing {ec.preprocessing!r}")
    if len(ec.img_size) != 2 or any(s <= 0 for s in ec.img_size):
        raise ConfigError("encoder.img_size must contain two positive integers")
    stride = {"dinov2": 14, "convnext": 32, "visionfm": 16}[ec.kind]
    if any(s % stride for s in ec.img_size):
        raise ConfigError(f"encoder.img_size {ec.img_size} must be multiples of {stride}")
    if ec.kind == "dinov2" and "dinov2" not in ec.name:
        raise ConfigError("dinov2 requires a DINOv2 timm model name")
    if ec.kind == "convnext" and ec.name != "convnext_tiny.in12k_ft_in1k":
        raise ConfigError("convnext supports convnext_tiny.in12k_ft_in1k")
    if ec.kind == "visionfm" and ec.name != "vit_base_patch16":
        raise ConfigError("visionfm supports vit_base_patch16")
    if ec.kind != "visionfm" and ec.checkpoint_key is not None:
        raise ConfigError("encoder.checkpoint_key is only supported for VisionFM")
    if ec.checkpoint_key not in (None, "teacher", "student", "visionfm_state_dict"):
        raise ConfigError("unsupported VisionFM checkpoint_key")
    fc = cfg.finetune
    if min(fc.batch_size, fc.max_epochs, fc.patience) < 1 or (fc.max_steps is not None and fc.max_steps < 1):
        raise ConfigError("finetune batch_size, max_epochs, patience and max_steps must be positive")
    if min(fc.lr_encoder, fc.lr_heads) <= 0 or fc.weight_decay < 0:
        raise ConfigError("finetune learning rates must be positive; weight_decay must be nonnegative")
    if cfg.features.dtype not in ("float16", "float32"):
        raise ConfigError(f"features.dtype must be float16 or float32, got {cfg.features.dtype!r}")
    if cfg.model.head_input not in ("pair", "ft_only"):
        raise ConfigError(f"model.head_input must be pair or ft_only, got {cfg.model.head_input!r}")
    if cfg.model.forecaster not in ("gru", "identity"):
        raise ConfigError(f"model.forecaster must be gru or identity, got {cfg.model.forecaster!r}")
    if cfg.train.select_metric not in ("mcc", "f1", "specificity", "qwk", "loss"):
        raise ConfigError(f"train.select_metric not recognised: {cfg.train.select_metric!r}")
    for task, n in (("task1", 4), ("task2", 3)):
        names = cfg.class_names.get(task)
        if names is None or len(names) != n:
            raise ConfigError(f"class_names.{task} must list {n} names")
    if cfg.partial.n_blocks < 1:
        raise ConfigError("partial.n_blocks must be >= 1")
