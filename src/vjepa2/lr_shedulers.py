# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Learning-rate schedules driven by the optimizer step count. Two options match
# the paper recipe: "warmup_hold" (warm up then hold, the primary phase) and
# "warmup_cosine" (warm up then decay to a small value). Each scheduler sets the
# learning rate on every optimizer group and can save / restore its step count.

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from vjepa2.config import SchedulerConfig, resolve_steps

__all__ = ["WarmupHold", "WarmupCosine", "build_scheduler"]


class _BaseSchedule:
    """Shared step counter and learning-rate application logic."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.step_count = 0
        self.last_lr = 0.0

    def _apply(self, lr: float) -> float:
        """Write ``lr`` to every optimizer parameter group."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def apply(self) -> float:
        """Set the LR for the current step on the optimizer (no advance).

        Call this before the optimizer update so the update uses the LR of the
        current step; it does not move the schedule forward.
        """
        return self._apply(self.lr_at(self.step_count))

    def advance(self) -> None:
        """Move the schedule one step forward.

        Call this only after a real optimizer step actually happened, so the
        learning rate never changes on a skipped (e.g. AMP inf/nan) step.
        """
        self.step_count += 1

    def step(self) -> float:
        """Apply the current LR then advance (convenience for simple loops)."""
        lr = self.apply()
        self.advance()
        return lr

    def lr_at(self, step: int) -> float:
        """Return the learning rate for a given step (subclasses override)."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler state for checkpointing."""
        return {"step_count": self.step_count, "last_lr": self.last_lr}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the scheduler state from a checkpoint."""
        self.step_count = int(state.get("step_count", 0))
        self.last_lr = float(state.get("last_lr", 0.0))


class WarmupHold(_BaseSchedule):
    """Linear warmup from ``start_lr`` to ``ref_lr``, then hold ``ref_lr``."""

    def __init__(self, optimizer, start_lr: float, ref_lr: float,
                 warmup_steps: int):
        super().__init__(optimizer)
        self.start_lr = float(start_lr)
        self.ref_lr = float(ref_lr)
        self.warmup_steps = max(1, int(warmup_steps))

    def lr_at(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.ref_lr
        alpha = step / self.warmup_steps
        return self.start_lr + (self.ref_lr - self.start_lr) * alpha


class WarmupCosine(_BaseSchedule):
    """Linear warmup, then cosine decay from ``ref_lr`` to ``final_lr``."""

    def __init__(self, optimizer, start_lr: float, ref_lr: float,
                 final_lr: float, warmup_steps: int, total_steps: int):
        super().__init__(optimizer)
        self.start_lr = float(start_lr)
        self.ref_lr = float(ref_lr)
        self.final_lr = float(final_lr)
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            return self.start_lr + (self.ref_lr - self.start_lr) * alpha
        return self._cosine(step)

    def _cosine(self, step: int) -> float:
        """Cosine interpolation between ref_lr and final_lr after warmup."""
        span = self.total_steps - self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / span)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr + (self.ref_lr - self.final_lr) * scale


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: SchedulerConfig,
                    total_steps: int) -> _BaseSchedule:
    """Build a scheduler from config and the planned total step count."""
    # Resolve the (possibly fractional) warmup against the real run length and
    # cap it so the warmup always completes within the run.
    warmup = resolve_steps(cfg.warmup_steps, total_steps)
    warmup = min(max(1, warmup), max(1, int(total_steps)))
    name = cfg.name.lower()
    if name in ("warmup_hold", "hold", "constant"):
        return WarmupHold(optimizer, cfg.start_lr, cfg.ref_lr, warmup)
    if name in ("warmup_cosine", "cosine"):
        return WarmupCosine(
            optimizer, cfg.start_lr, cfg.ref_lr, cfg.final_lr,
            warmup, total_steps,
        )
    raise ValueError(f"Unknown scheduler name: {cfg.name}")
