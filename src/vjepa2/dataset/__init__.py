# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Dataset package: file discovery, cleaning / caching, transforms, tube
# masking, HDF5 storage, and the resumable data loader.

from vjepa2.dataset.cache import CacheStore, cache_path
from vjepa2.dataset.cleaning import DatasetCleaner, ScanResult
from vjepa2.dataset.dataloader import ResumableDataLoader, ResumableSampler
from vjepa2.dataset.discovery import VIDEO_EXTENSIONS, VideoFileFinder
from vjepa2.dataset.factory import (
    DataBundle,
    build_collator,
    build_data_bundle,
    build_pipeline,
    build_reader,
)
from vjepa2.dataset.hdf5 import HDF5Builder, HDF5ClipDataset
from vjepa2.dataset.masking import TubeMaskCollator, grid_dims
from vjepa2.dataset.splits import cap_entries, split_val_test
from vjepa2.dataset.transforms import ClipPipeline
from vjepa2.dataset.video_dataset import VideoClipDataset
from vjepa2.dataset.video_io import ClipReadError, VideoReader, VideoSource

__all__ = [
    "CacheStore",
    "cache_path",
    "DatasetCleaner",
    "ScanResult",
    "ResumableDataLoader",
    "ResumableSampler",
    "VIDEO_EXTENSIONS",
    "VideoFileFinder",
    "DataBundle",
    "build_collator",
    "build_data_bundle",
    "build_pipeline",
    "build_reader",
    "HDF5Builder",
    "HDF5ClipDataset",
    "TubeMaskCollator",
    "grid_dims",
    "cap_entries",
    "split_val_test",
    "ClipPipeline",
    "VideoClipDataset",
    "ClipReadError",
    "VideoReader",
    "VideoSource",
]
