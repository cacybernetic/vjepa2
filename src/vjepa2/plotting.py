# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Draw the training vs validation curves so a user can see, at a glance, if the
# model is learning or overfitting. We use a non-interactive backend so this
# works on a headless server and simply writes JPG files to disk.

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["HistoryPlotter"]


def _metric_names(history: List[Dict[str, float]]) -> List[str]:
    """Find base metric names that have both a train and a val column."""
    if not history:
        return []
    keys = set(history[0].keys())
    names = []
    for key in keys:
        if key.startswith("train_"):
            base = key[len("train_"):]
            names.append(base)
    return sorted(names)


class HistoryPlotter:
    """Write train vs validation curves for each tracked metric."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def plot(self, history: List[Dict[str, float]]) -> List[str]:
        """Write one combined figure and one figure per metric.

        :param history: list of per-epoch dicts with ``epoch``, ``train_*`` and
            optional ``val_*`` keys.
        :returns: the list of written file paths.
        """
        names = _metric_names(history)
        if not history or not names:
            return []
        written = [self._plot_combined(history, names)]
        for name in names:
            written.append(self._plot_single(history, name))
        return written

    def _epochs(self, history: List[Dict[str, float]]) -> List[float]:
        """Return the x-axis epoch values."""
        return [row.get("epoch", i + 1) for i, row in enumerate(history)]

    def _series(self, history: List[Dict[str, float]], key: str):
        """Return the y-values for a column, keeping only present points."""
        xs, ys = [], []
        for i, row in enumerate(history):
            if key in row and row[key] is not None:
                xs.append(row.get("epoch", i + 1))
                ys.append(row[key])
        return xs, ys

    def _plot_single(self, history: List[Dict[str, float]], name: str) -> str:
        """Draw and save the train/val curve of one metric."""
        fig, axis = plt.subplots(figsize=(7, 4))
        self._draw_axis(axis, history, name)
        axis.set_title(f"{name}: train vs validation")
        path = os.path.join(self.out_dir, f"{name}.jpg")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def _plot_combined(self, history: List[Dict[str, float]],
                       names: List[str]) -> str:
        """Draw every metric on a grid and save one overview figure."""
        cols = 2
        rows = (len(names) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
        flat = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
        for axis, name in zip(flat, names):
            self._draw_axis(axis, history, name)
            axis.set_title(name)
        for axis in flat[len(names):]:
            axis.axis("off")
        path = os.path.join(self.out_dir, "training_history.jpg")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def _draw_axis(self, axis, history: List[Dict[str, float]], name: str) -> None:
        """Plot train and val series for one metric onto an axis."""
        tx, ty = self._series(history, f"train_{name}")
        axis.plot(tx, ty, marker="o", label="train", color="#1f77b4")
        vx, vy = self._series(history, f"val_{name}")
        if vy:
            axis.plot(vx, vy, marker="s", label="val", color="#d62728")
        axis.set_xlabel("epoch")
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)
        axis.legend()
