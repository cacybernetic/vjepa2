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

import torch

__all__ = ["EmaUpdater"]


class EmaUpdater:
    """Move the target encoder weights toward the online encoder weights."""

    def __init__(self, momentum: float = 0.99925):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be between 0 and 1")
        self.momentum = float(momentum)

    @torch.no_grad()
    def update(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
        """Blend target params: ``target = m * target + (1 - m) * online``."""
        m = self.momentum
        online_params = list(online.parameters())
        target_params = list(target.parameters())
        if len(online_params) != len(target_params):
            raise ValueError("online and target must have the same parameters")
        for online_p, target_p in zip(online_params, target_params):
            target_p.data.mul_(m).add_(online_p.data, alpha=1.0 - m)
