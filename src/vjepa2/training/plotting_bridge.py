# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Thin bridge so the trainer can draw history plots without importing
# matplotlib directly. This keeps the heavy plotting import in one place.

from __future__ import annotations

from typing import Dict, List

__all__ = ["plot_history"]


def plot_history(out_dir: str, rows: List[Dict[str, float]]) -> None:
    """Draw the train vs validation curves for the given history rows."""
    from vjepa2.plotting import HistoryPlotter

    HistoryPlotter(out_dir).plot(rows)
