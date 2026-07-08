# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Manage the ``runs/<run_name>/`` folders. Each training run gets its own
# folder named train, train2, train3, ... and each evaluation run gets eval,
# eval2, ... The first run has no number. When resuming, we reuse the folder
# with the highest number instead of making a new one.

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

__all__ = ["RunPaths", "RunDirManager"]


@dataclass
class RunPaths:
    """Standard sub-paths inside one run folder."""

    root: str
    weights_dir: str
    checkpoints_dir: str
    plotes_dir: str
    renders_dir: str
    logs_dir: str
    history_csv: str
    results_csv: str
    config_used: str
    best_pt: str
    last_pt: str


class RunDirManager:
    """Create or find the run folder for a training or evaluation run."""

    def __init__(self, runs_dir: str, run_name: str):
        self.run_root = os.path.join(runs_dir, run_name)

    def _name_for(self, kind: str, index: int) -> str:
        """Return the folder name for a given kind and 1-based index."""
        return kind if index == 1 else f"{kind}{index}"

    def _existing_indices(self, kind: str) -> List[int]:
        """Return the sorted indices of existing folders of this kind."""
        if not os.path.isdir(self.run_root):
            return []
        pattern = re.compile(rf"^{re.escape(kind)}(\d*)$")
        indices: List[int] = []
        for name in os.listdir(self.run_root):
            match = pattern.match(name)
            if match and os.path.isdir(os.path.join(self.run_root, name)):
                indices.append(1 if match.group(1) == "" else int(match.group(1)))
        return sorted(indices)

    def latest(self, kind: str) -> Optional[str]:
        """Return the path of the highest-numbered existing folder, or None."""
        indices = self._existing_indices(kind)
        if not indices:
            return None
        name = self._name_for(kind, indices[-1])
        return os.path.join(self.run_root, name)

    def _next_index(self, kind: str) -> int:
        """Return the next free index for a new folder of this kind."""
        indices = self._existing_indices(kind)
        return 1 if not indices else indices[-1] + 1

    def resolve(self, kind: str, resume: bool) -> Tuple[str, bool]:
        """Return ``(run_path, is_reused)`` for a training/eval folder.

        When ``resume`` is True and a folder already exists, we reuse the latest
        one. Otherwise a new numbered folder is created.
        """
        if resume:
            latest = self.latest(kind)
            if latest is not None:
                return latest, True
        index = self._next_index(kind)
        path = os.path.join(self.run_root, self._name_for(kind, index))
        return path, False

    def make_paths(self, run_path: str, kind: str = "train") -> RunPaths:
        """Create the sub-folders that a run needs and return their paths.

        A training run needs weights and checkpoints; an evaluation run needs a
        renders folder. Both need plots and logs.
        """
        weights = os.path.join(run_path, "weights")
        checkpoints = os.path.join(run_path, "checkpoints")
        plotes = os.path.join(run_path, "plotes")
        renders = os.path.join(run_path, "renders")
        logs = os.path.join(run_path, "logs")
        wanted = [run_path, plotes, logs]
        wanted += [weights, checkpoints] if kind == "train" else [renders]
        for folder in wanted:
            os.makedirs(folder, exist_ok=True)
        return RunPaths(
            root=run_path,
            weights_dir=weights,
            checkpoints_dir=checkpoints,
            plotes_dir=plotes,
            renders_dir=renders,
            logs_dir=logs,
            history_csv=os.path.join(run_path, "history.csv"),
            results_csv=os.path.join(run_path, "results.csv"),
            config_used=os.path.join(run_path, "config_used.yaml"),
            best_pt=os.path.join(weights, "best.pt"),
            last_pt=os.path.join(weights, "last.pt"),
        )
