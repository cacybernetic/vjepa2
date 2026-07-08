# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Small metric helpers. AverageMeter keeps a running mean of a scalar and can
# save / restore its state, which lets a validation or test pass resume from a
# checkpoint without losing the partial averages already computed.

from __future__ import annotations

from typing import Any, Dict, Iterable

__all__ = ["AverageMeter", "MetricTracker"]


class AverageMeter:
    """Track the running average of a single scalar value."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated values."""
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``value`` observed ``n`` times to the running average."""
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        """Return the current mean, or 0.0 when nothing was seen yet."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def state_dict(self) -> Dict[str, Any]:
        """Return the meter state for checkpointing."""
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the meter state from a checkpoint."""
        self.total = float(state.get("total", 0.0))
        self.count = int(state.get("count", 0))


class MetricTracker:
    """Track several named scalar averages together (loss, parts, ...)."""

    def __init__(self, names: Iterable[str]):
        self.meters: Dict[str, AverageMeter] = {n: AverageMeter() for n in names}

    def reset(self) -> None:
        """Reset every tracked meter."""
        for meter in self.meters.values():
            meter.reset()

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        """Update each meter whose name appears in ``values``."""
        for name, value in values.items():
            if name in self.meters:
                self.meters[name].update(value, n)

    def averages(self) -> Dict[str, float]:
        """Return the current average of every meter."""
        return {name: meter.average for name, meter in self.meters.items()}

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of every meter for checkpointing."""
        return {name: meter.state_dict() for name, meter in self.meters.items()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore every meter from a checkpoint."""
        for name, meter in self.meters.items():
            if name in state:
                meter.load_state_dict(state[name])
