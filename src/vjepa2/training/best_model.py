# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Decide when a new validation score is the best so far. The user chooses which
# metric to watch and whether higher or lower is better. When the score is the
# best, the trainer saves the model weights as ``best.pt``.

from __future__ import annotations

import math
from typing import Any, Dict

__all__ = ["BestModelTracker"]


class BestModelTracker:
    """Track the best value of one validation metric."""

    def __init__(self, metric: str, mode: str = "min"):
        mode = mode.lower()
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.metric = metric
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf

    def is_better(self, value: float) -> bool:
        """Return True when ``value`` improves on the best seen so far."""
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def consider(self, value: float) -> bool:
        """Update the best value and return True when it improved."""
        if self.is_better(value):
            self.best = float(value)
            return True
        return False

    def has_best(self) -> bool:
        """Return True once at least one real score has been recorded."""
        return math.isfinite(self.best)

    def state_dict(self) -> Dict[str, Any]:
        """Return the tracker state for checkpointing."""
        return {"metric": self.metric, "mode": self.mode, "best": self.best}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the tracker state from a checkpoint."""
        self.metric = state.get("metric", self.metric)
        self.mode = state.get("mode", self.mode)
        self.best = float(state.get("best", self.best))
