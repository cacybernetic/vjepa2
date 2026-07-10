# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# A torch Dataset that reads video clips on the fly from a folder or zip. Each
# item is one *clip window* (a video plus a start frame), so a long video yields
# many overlapping clips instead of a single sub-sampled one. It decodes the
# window and runs the preprocessing pipeline. It yields plain clip tensors;
# there are no labels because V-JEPA learns without labels.

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from vjepa2.dataset.clip_index import ClipWindow
from vjepa2.dataset.transforms import ClipPipeline
from vjepa2.dataset.video_io import ClipReadError, VideoReader, VideoSource

__all__ = ["VideoClipDataset"]


class VideoClipDataset(Dataset):
    """Yield preprocessed clip tensors ``(3, T, H, W)`` from clip windows."""

    def __init__(self, root: str, is_zip: bool, windows: List[ClipWindow],
                 pipeline: ClipPipeline, reader: VideoReader,
                 crop_size: int, num_frames: int, train: bool = True):
        self.root = root
        self.is_zip = is_zip
        self.windows = list(windows)
        self.pipeline = pipeline
        self.reader = reader
        self.crop_size = int(crop_size)
        self.num_frames = int(num_frames)
        self.train = bool(train)
        self._source: Optional[VideoSource] = None

    def __len__(self) -> int:
        return len(self.windows)

    def _get_source(self) -> VideoSource:
        """Open the byte provider lazily, once per worker process."""
        if self._source is None:
            self._source = VideoSource(self.root, self.is_zip)
        return self._source

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read, decode and preprocess the clip window at ``index``."""
        window = self.windows[index]
        rng = np.random.default_rng() if self.train else np.random.default_rng(index)
        clip = self._safe_read(window)
        return self.pipeline(clip, train=self.train, rng=rng)

    def _safe_read(self, window: ClipWindow) -> np.ndarray:
        """Decode one window; on failure return a black clip of the right shape."""
        try:
            return self.reader.read_window(
                self._get_source(), window.entry, window.start_frame, window.step
            )
        except (ClipReadError, Exception):
            return self._black_clip()

    def _black_clip(self) -> np.ndarray:
        """Build an all-zero clip used as a safe fallback for unreadable files."""
        return np.zeros(
            (self.num_frames, self.crop_size, self.crop_size, 3), dtype=np.uint8
        )
