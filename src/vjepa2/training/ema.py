# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Update the target encoder as an exponential moving average (EMA) of the
# online encoder. The target is never trained by gradients; it slowly follows
# the online encoder. This is what stops the model from collapsing.

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["EmaUpdater"]


class EmaUpdater:
    """Move the target encoder weights toward the online encoder weights.

    The momentum is (optionally) ramped from ``momentum`` to ``momentum_end``
    over ``total_steps`` optimizer steps. A rising momentum -- the reference
    V-JEPA recipe -- lets the target track the online encoder quickly early on
    and then freeze progressively toward the end, which stabilises the moving
    prediction target. When ``momentum_end`` (or ``total_steps``) is not given,
    the momentum stays constant at ``momentum``.
    """

    def __init__(self, momentum: float = 0.99925,
                 momentum_end: Optional[float] = None,
                 total_steps: Optional[int] = None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be between 0 and 1")
        if momentum_end is not None and not 0.0 <= momentum_end <= 1.0:
            raise ValueError("EMA momentum_end must be between 0 and 1")
        self.momentum = float(momentum)
        self.momentum_end = (float(momentum_end)
                             if momentum_end is not None else self.momentum)
        self.total_steps = int(total_steps) if total_steps else None

    def momentum_at(self, step: Optional[int]) -> float:
        """Return the momentum for a given optimizer step."""
        if (step is None or self.total_steps is None
                or self.momentum_end == self.momentum):
            return self.momentum
        progress = min(1.0, max(0, step) / max(1, self.total_steps))
        return self.momentum + (self.momentum_end - self.momentum) * progress

    @torch.no_grad()
    def update(self, online: torch.nn.Module, target: torch.nn.Module,
               step: Optional[int] = None) -> None:
        """Blend target params: ``target = m * target + (1 - m) * online``."""
        m = self.momentum_at(step)
        online_params = list(online.parameters())
        target_params = list(target.parameters())
        if len(online_params) != len(target_params):
            raise ValueError("online and target must have the same parameters")
        for online_p, target_p in zip(online_params, target_params):
            target_p.data.mul_(m).add_(online_p.data, alpha=1.0 - m)
