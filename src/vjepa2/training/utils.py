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


def rng_state() -> Dict[str, Any]:
    """Capture the current random number generator states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore random number generator states from a checkpoint."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


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
