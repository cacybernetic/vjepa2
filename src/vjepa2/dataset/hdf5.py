# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Build and read HDF5 files that hold already-preprocessed clips. Doing the
# decode + transform once and storing the result lets training and evaluation
# skip that work and run faster. Optionally we also store augmented copies.

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from vjepa2.dataset.transforms import ClipPipeline
from vjepa2.dataset.video_io import ClipReadError, VideoReader, VideoSource

__all__ = ["HDF5Builder", "HDF5ClipDataset"]

CLIPS_KEY = "clips"


class HDF5Builder:
    """Decode, transform and store clips into a single HDF5 file."""

    def __init__(self, pipeline: ClipPipeline, reader: VideoReader):
        self.pipeline = pipeline
        self.reader = reader

    def build(self, out_path: str, root: str, is_zip: bool, entries: List[str],
              clip_shape: Tuple[int, int, int, int],
              augment_copies: int = 0) -> int:
        """Write all clips (plus optional augmented copies) to ``out_path``.

        :param clip_shape: expected ``(C, T, H, W)`` of one preprocessed clip.
        :param augment_copies: extra augmented versions stored per source clip.
        :returns: the number of clips written.
        """
        import h5py

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        source = VideoSource(root, is_zip)
        total = len(entries) * (1 + max(0, augment_copies))
        with h5py.File(out_path, "w") as handle:
            dataset = self._make_dataset(handle, total, clip_shape)
            written = self._fill(dataset, source, entries, augment_copies)
        source.close()
        return written

    def _make_dataset(self, handle, total: int,
                      clip_shape: Tuple[int, int, int, int]):
        """Create the compressed clips dataset inside the HDF5 file."""
        return handle.create_dataset(
            CLIPS_KEY,
            shape=(total, *clip_shape),
            dtype="float32",
            chunks=(1, *clip_shape),
            compression="gzip",
            compression_opts=4,
        )

    def _fill(self, dataset, source: VideoSource, entries: List[str],
              augment_copies: int) -> int:
        """Loop over entries and write each preprocessed clip in place."""
        cursor = 0
        rng = np.random.default_rng(0)
        bar = tqdm(entries, desc="building hdf5", leave=True, ascii="░█",
                   dynamic_ncols=True)
        for entry in bar:
            clip = self._read_clip(source, entry, rng)
            cursor = self._write_variants(dataset, cursor, clip, augment_copies, rng)
        return cursor

    def _read_clip(self, source: VideoSource, entry: str,
                   rng: np.random.Generator) -> np.ndarray:
        """Decode one raw clip, returning zeros when the file cannot be read."""
        try:
            return self.reader.read(source, entry, random_start=False, rng=rng)
        except (ClipReadError, Exception):
            return None

    def _write_variants(self, dataset, cursor: int, clip: Optional[np.ndarray],
                        augment_copies: int, rng: np.random.Generator) -> int:
        """Write the clean clip and any augmented copies, returning new cursor."""
        if clip is None:
            dataset[cursor] = np.zeros(dataset.shape[1:], dtype="float32")
            return cursor + 1
        dataset[cursor] = self.pipeline(clip, train=False, rng=rng).numpy()
        cursor += 1
        for _ in range(max(0, augment_copies)):
            dataset[cursor] = self.pipeline(clip, train=True, rng=rng).numpy()
            cursor += 1
        return cursor


class HDF5ClipDataset(Dataset):
    """Read preprocessed clips ``(C, T, H, W)`` from an HDF5 file."""

    def __init__(self, path: str):
        self.path = path
        self._handle = None
        self._length = self._read_length(path)

    def _read_length(self, path: str) -> int:
        """Open the file once to read the number of stored clips."""
        import h5py

        with h5py.File(path, "r") as handle:
            return int(handle[CLIPS_KEY].shape[0])

    def _file(self):
        """Open the HDF5 file lazily, once per worker process."""
        import h5py

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        clip = self._file()[CLIPS_KEY][index]
        return torch.from_numpy(np.asarray(clip, dtype=np.float32))
