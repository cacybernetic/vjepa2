# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Small shared helpers for training: seeding, moving batches to a device, and
# formatting metric dictionaries for the log lines.

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import torch

__all__ = [
    "set_seed",
    "configure_backend",
    "rng_state",
    "set_rng_state",
    "move_clips",
    "move_masks",
    "format_metrics",
]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(device: str) -> None:
    """Enable the fast GPU math paths for fixed-shape training.

    TF32 matmuls and cuDNN autotuning are safe for this workload (all input
    shapes are constant across steps) and leave measurable speed on the table
    when off. No-op on CPU.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def rng_state() -> Dict[str, Any]:
    """Capture the current random number generator states."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore random number generator states from a checkpoint."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except RuntimeError:
            # Device count changed between save and resume; skip rather than
            # abort the whole resume for a reproducibility nicety.
            pass


def move_clips(clips: List[torch.Tensor], device: str) -> List[torch.Tensor]:
    """Move a list of clip tensors to the target device."""
    return [clip.to(device, non_blocking=True) for clip in clips]


def move_masks(masks: List[List[torch.Tensor]], device: str
               ) -> List[List[torch.Tensor]]:
    """Move a nested ``[fpc][mask]`` list of index tensors to the device."""
    return [[m.to(device, non_blocking=True) for m in group] for group in masks]


def format_metrics(values: Dict[str, float]) -> str:
    """Turn a metric dict into a compact ``key=value`` string for logs."""
    parts = [f"{key}={value:.8f}" for key, value in values.items()]
    return " ".join(parts)
