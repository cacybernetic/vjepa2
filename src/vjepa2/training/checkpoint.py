# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Save and load full training checkpoints. A checkpoint holds everything needed
# to continue: model, optimizer, scheduler, data loader positions, meters, and
# the training state. We keep only the newest ``max_checkpoint`` files to save
# disk space, and we write to a temporary file first so a crash mid-write can
# never leave a broken checkpoint.

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import torch

__all__ = ["CheckpointManager"]

_EPOCH_RE = re.compile(r"^epoch_(\d+)\.pth$")


class CheckpointManager:
    """Write, rotate and read the training checkpoints of a run."""

    def __init__(self, checkpoints_dir: str, max_checkpoint: int = 5):
        self.dir = checkpoints_dir
        self.max_checkpoint = max(1, int(max_checkpoint))
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, epoch: int) -> str:
        """Return the checkpoint file path for an epoch."""
        return os.path.join(self.dir, f"epoch_{epoch:03d}.pth")

    def save(self, state: Dict[str, Any], epoch: int) -> str:
        """Write the checkpoint for an epoch, then rotate old ones.

        The file is written to a temporary path and atomically renamed, so an
        interrupted write cannot corrupt a good checkpoint.
        """
        target = self._path(epoch)
        tmp = target + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, target)
        self._rotate()
        return target

    def _all_files(self) -> List[str]:
        """Return existing checkpoint file names sorted by epoch number."""
        files = []
        for name in os.listdir(self.dir):
            if _EPOCH_RE.match(name):
                files.append(name)
        files.sort(key=lambda n: int(_EPOCH_RE.match(n).group(1)))
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
