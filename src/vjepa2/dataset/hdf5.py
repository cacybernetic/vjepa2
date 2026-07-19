# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Build and read HDF5 files that hold already-preprocessed clips. Doing the
# decode + transform once and storing the result lets training and evaluation
# skip that work and run faster. Optionally we also store augmented copies.
#
# Clips are stored as uint8 in [0, 255] *before* normalization (4x smaller than
# float32 and far more compressible); the normalization parameters are kept in
# the file attributes and applied at read time. Files written by older versions
# (float32, already normalized) are still readable.

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from vjepa2.dataset.clip_index import ClipWindow
from vjepa2.dataset.transforms import ClipPipeline
from vjepa2.dataset.video_io import VideoSource
from vjepa2.logging import logger

__all__ = ["HDF5Builder", "HDF5ClipDataset"]

CLIPS_KEY = "clips"


class HDF5Builder:
    """Decode, transform and store clips into a single HDF5 file.

    Video datasets are expanded into clip windows (``build``); image datasets
    store one single-frame clip per file (``build_images``). Both write the
    same uint8 ``(N, C, T, H, W)`` layout, so ``HDF5ClipDataset`` reads either.

    :param reader: a ``VideoReader`` for ``build``, or any object with a
        ``read(source, entry)`` method for ``build_images``.
    """

    def __init__(self, pipeline: ClipPipeline, reader):
        self.pipeline = pipeline
        self.reader = reader

    def build(self, out_path: str, root: str, is_zip: bool,
              windows: List[ClipWindow],
              clip_shape: Tuple[int, int, int, int],
              augment_copies: int = 0) -> int:
        """Write all clip windows (plus optional augmented copies) to ``out_path``.

        Windows are grouped by video, so each source file is decoded only once
        even when it contributes many overlapping clips. Windows that cannot be
        decoded are skipped (and logged), never written as fabricated data; the
        dataset is resized down to the number of clips actually written.

        :param clip_shape: expected ``(C, T, H, W)`` of one preprocessed clip.
        :param augment_copies: extra augmented versions stored per source clip.
        :returns: the number of clips written.
        """
        import h5py

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        source = VideoSource(root, is_zip)
        total = len(windows) * (1 + max(0, augment_copies))
        with h5py.File(out_path, "w") as handle:
            dataset = self._make_dataset(handle, total, clip_shape)
            self._write_norm_attrs(dataset)
            written = self._fill(dataset, source, windows, augment_copies)
            if written < total:
                dataset.resize((written, *clip_shape))
        source.close()
        if written < total:
            logger.warning("hdf5 build: {}/{} clips written ({} skipped as "
                           "unreadable)", written, total, total - written)
        return written

    def build_images(self, out_path: str, root: str, is_zip: bool,
                     entries: List[str],
                     clip_shape: Tuple[int, int, int, int],
                     augment_copies: int = 0) -> int:
        """Write every image (plus optional augmented copies) to ``out_path``.

        Each image becomes one preprocessed single-frame clip ``(C, 1, H, W)``.
        Unreadable files are skipped (and logged), never written as fabricated
        data; the dataset is resized down to the clips actually written.

        :param clip_shape: expected ``(C, 1, H, W)`` of one preprocessed clip.
        :param augment_copies: extra augmented versions stored per image.
        :returns: the number of clips written.
        """
        import h5py

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        source = VideoSource(root, is_zip)
        total = len(entries) * (1 + max(0, augment_copies))
        with h5py.File(out_path, "w") as handle:
            dataset = self._make_dataset(handle, total, clip_shape)
            self._write_norm_attrs(dataset)
            written = self._fill_images(dataset, source, entries, augment_copies)
            if written < total:
                dataset.resize((written, *clip_shape))
        source.close()
        if written < total:
            logger.warning("hdf5 build: {}/{} clips written ({} skipped as "
                           "unreadable)", written, total, total - written)
        return written

    def _fill_images(self, dataset, source: VideoSource, entries: List[str],
                     augment_copies: int) -> int:
        """Loop over image entries and write each preprocessed clip in place."""
        cursor = 0
        rng = np.random.default_rng(0)
        bar = tqdm(entries, desc="building hdf5", leave=True, ascii="░█",
                   dynamic_ncols=True)
        for entry in bar:
            clip = self._read_image(source, entry)
            cursor = self._write_variants(dataset, cursor, clip, augment_copies, rng)
        return cursor

    def _read_image(self, source: VideoSource, entry: str) -> Optional[np.ndarray]:
        """Decode one image into a ``(1, H, W, 3)`` clip, or None on failure."""
        try:
            return self.reader.read(source, entry)
        except Exception as error:
            logger.warning("hdf5 build: cannot decode {} ({}); skipped",
                           entry, error)
            return None

    def _make_dataset(self, handle, total: int,
                      clip_shape: Tuple[int, int, int, int]):
        """Create the clips dataset (uint8, resizable, fast lzf compression)."""
        return handle.create_dataset(
            CLIPS_KEY,
            shape=(total, *clip_shape),
            maxshape=(None, *clip_shape),
            dtype="uint8",
            chunks=(1, *clip_shape),
            compression="lzf",
        )

    def _write_norm_attrs(self, dataset) -> None:
        """Store the normalization recipe so readers can apply it lazily."""
        norm = self.pipeline.normalize
        dataset.attrs["normalize"] = bool(norm.enabled)
        dataset.attrs["mean"] = np.asarray(norm.mean, dtype="float32")
        dataset.attrs["std"] = np.asarray(norm.std, dtype="float32")

    def _fill(self, dataset, source: VideoSource, windows: List[ClipWindow],
              augment_copies: int) -> int:
        """Loop over clip windows and write each preprocessed clip in place."""
        cursor = 0
        rng = np.random.default_rng(0)
        cache_entry: Optional[str] = None
        cache_frames = None
        bar = tqdm(windows, desc="building hdf5", leave=True, ascii="░█",
                   dynamic_ncols=True)
        for window in bar:
            if window.entry != cache_entry:
                cache_entry = window.entry
                cache_frames = self._decode(source, window.entry)
            clip = self._read_clip(cache_frames, window)
            cursor = self._write_variants(dataset, cursor, clip, augment_copies, rng)
        return cursor

    def _decode(self, source: VideoSource, entry: str):
        """Decode a whole video once, or None when it cannot be read."""
        try:
            return self.reader.decode_all(source, entry)
        except Exception as error:
            logger.warning("hdf5 build: cannot decode {} ({}); its clips will "
                           "be skipped", entry, error)
            return None

    def _read_clip(self, frames, window: ClipWindow) -> Optional[np.ndarray]:
        """Slice one clip window from decoded frames, or None on failure."""
        if not frames:
            return None
        try:
            return self.reader.window_clip(frames, window.start_frame, window.step)
        except Exception as error:
            logger.warning("hdf5 build: cannot slice window {}@{} ({}); "
                           "skipped", window.entry, window.start_frame, error)
            return None

    def _write_variants(self, dataset, cursor: int, clip: Optional[np.ndarray],
                        augment_copies: int, rng: np.random.Generator) -> int:
        """Write the clean clip and any augmented copies, returning new cursor."""
        if clip is None:
            return cursor
        dataset[cursor] = self._to_uint8(
            self.pipeline(clip, train=False, rng=rng, apply_normalize=False)
        )
        cursor += 1
        for _ in range(max(0, augment_copies)):
            dataset[cursor] = self._to_uint8(
                self.pipeline(clip, train=True, rng=rng, apply_normalize=False)
            )
            cursor += 1
        return cursor

    @staticmethod
    def _to_uint8(clip: torch.Tensor) -> np.ndarray:
        """Quantize a ``[0, 1]`` float clip to uint8 for compact storage."""
        return (
            clip.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8).numpy()
        )


class HDF5ClipDataset(Dataset):
    """Read preprocessed clips ``(C, T, H, W)`` from an HDF5 file.

    New files hold uint8 clips plus normalization attributes applied at read
    time; legacy float32 files (already normalized) are returned as-is.
    """

    def __init__(self, path: str):
        self.path = path
        self._handle = None
        self._length, self._norm = self._read_meta(path)

    def _read_meta(self, path: str):
        """Open the file once to read the clip count and normalization attrs."""
        import h5py

        with h5py.File(path, "r") as handle:
            dataset = handle[CLIPS_KEY]
            length = int(dataset.shape[0])
            norm = None
            if dataset.dtype == np.uint8 and bool(
                dataset.attrs.get("normalize", False)
            ):
                mean = np.asarray(dataset.attrs["mean"], dtype="float32")
                std = np.asarray(dataset.attrs["std"], dtype="float32")
                norm = (
                    torch.from_numpy(mean).view(-1, 1, 1, 1),
                    torch.from_numpy(std).view(-1, 1, 1, 1),
                )
        return length, norm

    def _file(self):
        """Open the HDF5 file lazily, once per worker process."""
        import h5py

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self):
        # h5py handles are not picklable; each worker reopens the file lazily.
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        clip = np.asarray(self._file()[CLIPS_KEY][index])
        if clip.dtype == np.uint8:
            tensor = torch.from_numpy(clip).float().div_(255.0)
            if self._norm is not None:
                mean, std = self._norm
                tensor = (tensor - mean) / std
            return tensor
        return torch.from_numpy(clip.astype(np.float32, copy=False))
