# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# A torch Dataset that reads still images on the fly from a folder or zip.
# Each item is one image decoded into a single-frame clip and run through the
# same preprocessing pipeline as video clips, so the rest of the program
# (collator, trainer, model) sees the usual ``(3, T, H, W)`` layout with
# ``T == 1``. It yields plain tensors; V-JEPA learns without labels.

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from vjepa2.dataset.image_io import ImageReader
from vjepa2.dataset.transforms import ClipPipeline
from vjepa2.dataset.video_io import VideoSource
from vjepa2.logging import logger

__all__ = ["ImageClipDataset"]

# How many alternative entries to try when one cannot be decoded.
_MAX_READ_ATTEMPTS = 8


class ImageClipDataset(Dataset):
    """Yield preprocessed single-frame clips ``(3, 1, H, W)`` from image files."""

    def __init__(self, root: str, is_zip: bool, entries: List[str],
                 pipeline: ClipPipeline, reader: ImageReader,
                 crop_size: int, train: bool = True):
        self.root = root
        self.is_zip = is_zip
        self.entries = list(entries)
        self.pipeline = pipeline
        self.reader = reader
        self.crop_size = int(crop_size)
        self.num_frames = 1
        self.train = bool(train)
        self._source: Optional[VideoSource] = None
        self._warned_entries: set = set()

    def __len__(self) -> int:
        return len(self.entries)

    def _get_source(self) -> VideoSource:
        """Open the byte provider lazily, once per worker process."""
        if self._source is None:
            self._source = VideoSource(self.root, self.is_zip)
        return self._source

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read, decode and preprocess the image at ``index``.

        An unreadable image is *never* replaced by fabricated data (a constant
        input is exactly the degenerate case that pushes a JEPA objective
        toward collapse). Instead we log the failure once per file and fall
        back to a neighbouring entry; if nothing can be decoded we fail loudly.
        """
        clip = None
        probe = index
        for _ in range(min(_MAX_READ_ATTEMPTS, len(self.entries))):
            clip = self._try_read(self.entries[probe])
            if clip is not None:
                break
            probe = (probe + 1) % len(self.entries)
        if clip is None:
            raise RuntimeError(
                f"could not decode any image after {_MAX_READ_ATTEMPTS} "
                f"attempts starting at index {index} ({self.entries[index]}); "
                "the dataset looks unreadable"
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

    def _try_read(self, entry: str) -> Optional[np.ndarray]:
        """Decode one image, or return None (with one warning per file)."""
        try:
            return self.reader.read(self._get_source(), entry)
        except Exception as error:
            if entry not in self._warned_entries:
                self._warned_entries.add(entry)
                logger.warning(
                    "unreadable image {} ({}); substituting another entry",
                    entry, error)
            return None
