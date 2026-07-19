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
from vjepa2.dataset.video_io import VideoReader, VideoSource
from vjepa2.logging import logger

__all__ = ["VideoClipDataset"]

# How many alternative clip windows to try when one cannot be decoded.
_MAX_READ_ATTEMPTS = 8


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
        self._warned_entries: set = set()

    def __len__(self) -> int:
        return len(self.windows)

    def _get_source(self) -> VideoSource:
        """Open the byte provider lazily, once per worker process."""
        if self._source is None:
            self._source = VideoSource(self.root, self.is_zip)
        return self._source

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read, decode and preprocess the clip window at ``index``.

        An unreadable window is *never* replaced by fabricated data (a constant
        clip is exactly the degenerate input that pushes a JEPA objective
        toward collapse). Instead we log the failure once per file and fall
        back to a neighbouring window; if nothing can be decoded we fail loudly.
        """
        clip = None
        probe = index
        for _ in range(min(_MAX_READ_ATTEMPTS, len(self.windows))):
            clip = self._try_read(self.windows[probe])
            if clip is not None:
                break
            probe = (probe + 1) % len(self.windows)
        if clip is None:
            raise RuntimeError(
                f"could not decode any clip window after "
                f"{_MAX_READ_ATTEMPTS} attempts starting at index {index} "
                f"({self.windows[index].entry}); the dataset looks unreadable"
            )
        rng = self._item_rng(index)
        return self.pipeline(clip, train=self.train, rng=rng)

    def _item_rng(self, index: int) -> np.random.Generator:
        """Return the augmentation RNG for one item.

        Training draws the seed from the torch RNG stream, which the DataLoader
        seeds per worker and per epoch: augmentations vary across epochs yet
        stay reproducible under a fixed global seed. Evaluation stays
        deterministic per index.
        """
        if not self.train:
            return np.random.default_rng(index)
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        return np.random.default_rng(seed)

    def _try_read(self, window: ClipWindow) -> Optional[np.ndarray]:
        """Decode one window, or return None (with one warning per file)."""
        try:
            return self.reader.read_window(
                self._get_source(), window.entry, window.start_frame, window.step
            )
        except Exception as error:
            if window.entry not in self._warned_entries:
                self._warned_entries.add(window.entry)
                logger.warning(
                    "unreadable clip in {} ({}); substituting another window",
                    window.entry, error)
            return None
