# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Save and load full training checkpoints. A checkpoint holds everything needed
# to continue: model, optimizer, scheduler, data loader positions, meters, and
# the training state. Every save goes to its own dedicated file named
# ``checkpoint_<phase>_e<epoch>c<counter>.pth`` (e.g. ``checkpoint_train_
# e0001c0012.pth``): no two checkpoints ever share a file, not even two saves of
# the same epoch, and different epochs never land in the same file. We keep only
# the newest ``max_checkpoint`` files to save disk space, and we write to a
# temporary file first so a crash mid-write can never leave a broken checkpoint.

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = ["CheckpointManager"]

# checkpoint_<phase>_e<epoch>c<counter>.pth -- one file per individual save.
_CKPT_RE = re.compile(r"^checkpoint_(train|val|test)_e(\d+)c(\d+)\.pth$")
# Within one epoch, train checkpoints are written before validation ones, and
# the final-test checkpoints come last. Ranking the phases this way lets the
# (epoch, phase, counter) tuple reproduce the real write order, so "latest" and
# rotation stay correct even when many checkpoints share an epoch.
_PHASE_RANK = {"train": 0, "val": 1, "test": 2}


class CheckpointManager:
    """Write, rotate and read the training checkpoints of a run."""

    def __init__(self, checkpoints_dir: str, max_checkpoint: int = 5):
        self.dir = checkpoints_dir
        self.max_checkpoint = max(1, int(max_checkpoint))
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, epoch: int, counter: int, phase: str) -> str:
        """Return the dedicated file path for one checkpoint save."""
        return os.path.join(
            self.dir, f"checkpoint_{phase}_e{epoch:04d}c{counter:04d}.pth")

    def save(self, state: Dict[str, Any], epoch: int, counter: int,
             phase: str) -> str:
        """Write one checkpoint to its own file, then rotate old ones.

        ``epoch`` and ``counter`` (a within-epoch save index) keep every save in
        a distinct file. The file is written to a temporary path and atomically
        renamed, so an interrupted write cannot corrupt a good checkpoint.
        """
        target = self._path(epoch, counter, phase)
        tmp = target + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, target)
        self._rotate()
        return target

    @staticmethod
    def _sort_key(name: str) -> Tuple[int, int, int]:
        """Order key reproducing the write order: (epoch, phase, counter)."""
        match = _CKPT_RE.match(name)
        phase, epoch, counter = match.group(1), match.group(2), match.group(3)
        return (int(epoch), _PHASE_RANK[phase], int(counter))

    def _all_files(self) -> List[str]:
        """Return existing checkpoint file names sorted oldest to newest."""
        files = [n for n in os.listdir(self.dir) if _CKPT_RE.match(n)]
        files.sort(key=self._sort_key)
        return files

    def _rotate(self) -> None:
        """Delete the oldest checkpoints beyond the maximum count."""
        files = self._all_files()
        excess = len(files) - self.max_checkpoint
        for name in files[:max(0, excess)]:
            os.remove(os.path.join(self.dir, name))

    def latest_path(self) -> Optional[str]:
        """Return the newest checkpoint file path, or None when there is none."""
        files = self._all_files()
        if not files:
            return None
        return os.path.join(self.dir, files[-1])

    def load_latest(self, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        """Load the newest checkpoint, or return None when there is none."""
        path = self.latest_path()
        if path is None:
            return None
        return torch.load(path, map_location=map_location, weights_only=False)

    def has_checkpoint(self) -> bool:
        """Return True when at least one checkpoint file exists."""
        return self.latest_path() is not None
