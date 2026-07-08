# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Keep the per-epoch history (train and validation metrics) in memory and on
# disk as a CSV file. The plotting code reads these rows to draw the curves.

from __future__ import annotations

import csv
import os
from typing import Dict, List

__all__ = ["HistoryWriter"]


class HistoryWriter:
    """Store epoch rows and mirror them to a CSV file."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.rows: List[Dict[str, float]] = []

    def load(self, rows: List[Dict[str, float]]) -> None:
        """Replace the in-memory rows (used when resuming from a checkpoint)."""
        self.rows = [dict(row) for row in rows]

    def append(self, row: Dict[str, float]) -> None:
        """Add one epoch row and rewrite the CSV file."""
        self.rows.append(dict(row))
        self._flush()

    def _columns(self) -> List[str]:
        """Return the union of all column names, with ``epoch`` first."""
        keys = []
        for row in self.rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        if "epoch" in keys:
            keys.remove("epoch")
            keys.insert(0, "epoch")
        return keys

    def _flush(self) -> None:
        """Write all rows to the CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        columns = self._columns()
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
