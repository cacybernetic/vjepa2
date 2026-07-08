# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Typed configuration objects for the training / evaluation programs.
# Each section is a small dataclass with a ``from_dict`` builder so the YAML
# files stay simple and every value has one clear owner (Single Responsibility).

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

__all__ = [
    "TransformConfig",
    "AugmentConfig",
    "DatasetConfig",
    "MaskingConfig",
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "SchedulerConfig",
    "TrainConfig",
    "EvalConfig",
    "Config",
    "load_yaml",
    "resolve_device",
]


def load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file and return a plain dictionary.

    :param path: file path to a ``.yaml`` file.
    :returns: the parsed mapping, or an empty dict for an empty file.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)}")
    return data


def _get(source: Dict[str, Any], key: str, default: Any) -> Any:
    """Return ``source[key]`` when present and not None, else ``default``."""
    value = source.get(key, default)
    if value is None and default is not None:
        return default
    return value


def resolve_device(name: str) -> str:
    """Map a user device name to a torch device string.

    ROCm (AMD) uses the same ``cuda`` runtime string in PyTorch, so we map it
    to ``cuda`` here. ``auto`` picks ``cuda`` when a GPU is visible.
    """
    name = (name or "cpu").lower()
    if name in ("rocm", "amd", "hip"):
        return "cuda"
    if name == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return name


@dataclass
class TransformConfig:
    """Deterministic pre-processing applied to every clip."""

    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    random_resized_crop: bool = True
    scale: List[float] = field(default_factory=lambda: [0.3, 1.0])
    aspect_ratio: List[float] = field(default_factory=lambda: [0.75, 1.35])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformConfig":
        data = data or {}
        return cls(
            normalize=bool(_get(data, "normalize", True)),
            mean=list(_get(data, "mean", [0.485, 0.456, 0.406])),
            std=list(_get(data, "std", [0.229, 0.224, 0.225])),
            random_resized_crop=bool(_get(data, "random_resized_crop", True)),
            scale=list(_get(data, "scale", [0.3, 1.0])),
            aspect_ratio=list(_get(data, "aspect_ratio", [0.75, 1.35])),
        )


@dataclass
class AugmentConfig:
    """Random photometric / geometric augmentation on the train split."""

    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter: float = 0.4
    grayscale_prob: float = 0.2
    gaussian_blur_prob: float = 0.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AugmentConfig":
        data = data or {}
        return cls(
            enabled=bool(_get(data, "enabled", True)),
            horizontal_flip_prob=float(_get(data, "horizontal_flip_prob", 0.5)),
            color_jitter=float(_get(data, "color_jitter", 0.4)),
            grayscale_prob=float(_get(data, "grayscale_prob", 0.2)),
            gaussian_blur_prob=float(_get(data, "gaussian_blur_prob", 0.5)),
        )


@dataclass
class DatasetConfig:
    """Where the videos live and how many of them to use."""

    use_hdf5: bool = False
    validate: bool = True
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    train_h5: str = "data/train.h5"
    test_h5: str = "data/test.h5"
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    val_prob: float = 0.5
    num_frames: int = 16
    frames_per_second: float = 4.0
    crop_size: int = 256
    transform: TransformConfig = field(default_factory=TransformConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        data = data or {}
        return cls(
            use_hdf5=bool(_get(data, "use_hdf5", False)),
            validate=bool(_get(data, "validate", True)),
            train_path=data.get("train_path"),
            test_path=data.get("test_path"),
            train_h5=str(_get(data, "train_h5", "data/train.h5")),
            test_h5=str(_get(data, "test_h5", "data/test.h5")),
            max_train_samples=data.get("max_train_samples"),
            max_test_samples=data.get("max_test_samples"),
            val_prob=float(_get(data, "val_prob", 0.5)),
            num_frames=int(_get(data, "num_frames", 16)),
            frames_per_second=float(_get(data, "frames_per_second", 4.0)),
            crop_size=int(_get(data, "crop_size", 256)),
            transform=TransformConfig.from_dict(data.get("transform", {})),
            augment=AugmentConfig.from_dict(data.get("augment", {})),
        )


@dataclass
class MaskingConfig:
    """Tube-mask sampling parameters (V-JEPA multi-block masking)."""

    spatial_scale: List[float] = field(default_factory=lambda: [0.15, 0.7])
    aspect_ratio: List[float] = field(default_factory=lambda: [0.75, 1.5])
    num_pred_masks: int = 1
    min_keep: int = 4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskingConfig":
        data = data or {}
        return cls(
            spatial_scale=list(_get(data, "spatial_scale", [0.15, 0.7])),
            aspect_ratio=list(_get(data, "aspect_ratio", [0.75, 1.5])),
            num_pred_masks=int(_get(data, "num_pred_masks", 1)),
            min_keep=int(_get(data, "min_keep", 4)),
        )


@dataclass
class ModelConfig:
    """Encoder / predictor architecture selection."""

    name: str = "vit_tiny"
    patch_size: int = 16
    tubelet_size: int = 2
    pred_depth: int = 12
    pred_embed_dim: int = 384
    num_mask_tokens: int = 8
    n_output_distillation: int = 4
    modality_embedding: bool = True
    use_rope: bool = True
    use_sdpa: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        data = data or {}
        return cls(
            name=str(_get(data, "name", "vit_tiny")),
            patch_size=int(_get(data, "patch_size", 16)),
            tubelet_size=int(_get(data, "tubelet_size", 2)),
            pred_depth=int(_get(data, "pred_depth", 12)),
            pred_embed_dim=int(_get(data, "pred_embed_dim", 384)),
            num_mask_tokens=int(_get(data, "num_mask_tokens", 8)),
            n_output_distillation=int(_get(data, "n_output_distillation", 4)),
            modality_embedding=bool(_get(data, "modality_embedding", True)),
            use_rope=bool(_get(data, "use_rope", True)),
            use_sdpa=bool(_get(data, "use_sdpa", True)),
        )


@dataclass
class LossConfig:
    """Dense predictive loss weighting (predict + weighted context)."""

    loss_exp: float = 1.0
    context_lambda: float = 0.5
    lambda_warmup_start: int = 15000
    lambda_warmup_end: int = 30000
    offset_context_loss: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LossConfig":
        data = data or {}
        return cls(
            loss_exp=float(_get(data, "loss_exp", 1.0)),
            context_lambda=float(_get(data, "context_lambda", 0.5)),
            lambda_warmup_start=int(_get(data, "lambda_warmup_start", 15000)),
            lambda_warmup_end=int(_get(data, "lambda_warmup_end", 30000)),
            offset_context_loss=bool(_get(data, "offset_context_loss", False)),
        )


@dataclass
class OptimConfig:
    """Optimizer and EMA settings."""

    name: str = "adamw"
    lr: float = 1.0e-4
    weight_decay: float = 0.04
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    momentum: float = 0.9
    ema: float = 0.99925

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimConfig":
        data = data or {}
        return cls(
            name=str(_get(data, "name", "adamw")).lower(),
            lr=float(_get(data, "lr", 1.0e-4)),
            weight_decay=float(_get(data, "weight_decay", 0.04)),
            betas=list(_get(data, "betas", [0.9, 0.95])),
            momentum=float(_get(data, "momentum", 0.9)),
            ema=float(_get(data, "ema", 0.99925)),
        )


@dataclass
class SchedulerConfig:
    """Learning-rate schedule settings."""

    name: str = "warmup_hold"
    warmup_steps: int = 12000
    start_lr: float = 1.0e-4
    ref_lr: float = 5.25e-4
    final_lr: float = 1.0e-6

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfig":
        data = data or {}
        return cls(
            name=str(_get(data, "name", "warmup_hold")).lower(),
            warmup_steps=int(_get(data, "warmup_steps", 12000)),
            start_lr=float(_get(data, "start_lr", 1.0e-4)),
            ref_lr=float(_get(data, "ref_lr", 5.25e-4)),
            final_lr=float(_get(data, "final_lr", 1.0e-6)),
        )


@dataclass
class TrainConfig:
    """Training loop control (epochs, batching, checkpointing)."""

    epochs: int = 10
    batch_size: int = 2
    grad_accum: int = 1
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    log_every: int = 16
    ckpt_step: int = 500
    max_checkpoint: int = 5
    runs_dir: str = "runs"
    checkpoints_dirname: str = "checkpoints"
    resume: bool = True
    init_weights: Optional[str] = None
    best_metric: str = "loss"
    best_mode: str = "min"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        data = data or {}
        return cls(
            epochs=int(_get(data, "epochs", 10)),
            batch_size=int(_get(data, "batch_size", 2)),
            grad_accum=int(_get(data, "grad_accum", 1)),
            grad_clip_norm=float(_get(data, "grad_clip_norm", 1.0)),
            num_workers=int(_get(data, "num_workers", 2)),
            log_every=int(_get(data, "log_every", 16)),
            ckpt_step=int(_get(data, "ckpt_step", 500)),
            max_checkpoint=int(_get(data, "max_checkpoint", 5)),
            runs_dir=str(_get(data, "runs_dir", "runs")),
            checkpoints_dirname=str(_get(data, "checkpoints_dirname", "checkpoints")),
            resume=bool(_get(data, "resume", True)),
            init_weights=data.get("init_weights"),
            best_metric=str(_get(data, "best_metric", "loss")),
            best_mode=str(_get(data, "best_mode", "min")).lower(),
        )


@dataclass
class EvalConfig:
    """Evaluation program control."""

    weights: Optional[str] = None
    batch_size: int = 2
    num_workers: int = 2
    num_render: int = 8
    runs_dir: str = "runs"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        data = data or {}
        return cls(
            weights=data.get("weights"),
            batch_size=int(_get(data, "batch_size", 2)),
            num_workers=int(_get(data, "num_workers", 2)),
            num_render=int(_get(data, "num_render", 8)),
            runs_dir=str(_get(data, "runs_dir", "runs")),
        )


@dataclass
class Config:
    """Top-level configuration aggregating every section."""

    run_name: str = "vjepa2_1"
    seed: int = 42
    device: str = "cpu"
    amp: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        data = data or {}
        return cls(
            run_name=str(_get(data, "run_name", "vjepa2_1")),
            seed=int(_get(data, "seed", 42)),
            device=str(_get(data, "device", "cpu")),
            amp=bool(_get(data, "amp", False)),
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
            masking=MaskingConfig.from_dict(data.get("masking", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            loss=LossConfig.from_dict(data.get("loss", {})),
            optim=OptimConfig.from_dict(data.get("optim", {})),
            scheduler=SchedulerConfig.from_dict(data.get("scheduler", {})),
            train=TrainConfig.from_dict(data.get("train", {})),
            eval=EvalConfig.from_dict(data.get("eval", {})),
            raw=data,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a config directly from a YAML file path."""
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict of the resolved config (without ``raw``)."""
        data = asdict(self)
        data.pop("raw", None)
        return data
