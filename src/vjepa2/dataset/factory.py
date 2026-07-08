# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Assemble datasets and resumable data loaders from a Config. This is the one
# place that knows how the small dataset pieces fit together, so the training
# and evaluation programs only ask for ready-to-use loaders.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from vjepa2.config import Config
from vjepa2.dataset.cleaning import DatasetCleaner
from vjepa2.dataset.dataloader import ResumableDataLoader
from vjepa2.dataset.hdf5 import HDF5ClipDataset
from vjepa2.dataset.masking import TubeMaskCollator, grid_dims
from vjepa2.dataset.splits import cap_entries, split_val_test
from vjepa2.dataset.transforms import ClipPipeline
from vjepa2.dataset.video_dataset import VideoClipDataset
from vjepa2.dataset.video_io import VideoReader

__all__ = ["DataBundle", "build_pipeline", "build_reader", "build_collator",
           "build_data_bundle", "build_eval_loader"]


@dataclass
class DataBundle:
    """Group the three loaders and their sizes for one run."""

    train_loader: ResumableDataLoader
    val_loader: Optional[ResumableDataLoader]
    test_loader: Optional[ResumableDataLoader]
    num_train: int
    num_val: int
    num_test: int


def _seed_worker(worker_id: int) -> None:
    """Give each data-loading worker its own numpy seed."""
    import torch

    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


def build_pipeline(cfg: Config, train: bool) -> ClipPipeline:
    """Build the preprocessing pipeline for a split."""
    augment = cfg.dataset.augment
    return ClipPipeline(cfg.dataset.transform, augment, cfg.dataset.crop_size)


def build_reader(cfg: Config) -> VideoReader:
    """Build the video reader shared by every split."""
    return VideoReader(
        num_frames=cfg.dataset.num_frames,
        target_fps=cfg.dataset.frames_per_second,
    )


def build_collator(cfg: Config, seed: Optional[int] = None) -> TubeMaskCollator:
    """Build the tube-mask collator from the model / masking config."""
    grid_size, grid_depth = grid_dims(
        cfg.dataset.crop_size,
        cfg.model.patch_size,
        cfg.dataset.num_frames,
        cfg.model.tubelet_size,
    )
    return TubeMaskCollator(cfg.masking, grid_size, grid_depth, seed=seed)


def _make_video_dataset(cfg: Config, root: str, is_zip: bool,
                        entries: List[str], train: bool) -> VideoClipDataset:
    """Create an on-the-fly video dataset for one split."""
    return VideoClipDataset(
        root=root,
        is_zip=is_zip,
        entries=entries,
        pipeline=build_pipeline(cfg, train),
        reader=build_reader(cfg),
        crop_size=cfg.dataset.crop_size,
        num_frames=cfg.dataset.num_frames,
        train=train,
    )


def _scan(cfg: Config, source: str) -> Tuple[str, bool, List[str]]:
    """Validate (or load cached) entries for a dataset source."""
    cleaner = DatasetCleaner(reader=build_reader(cfg))
    result = cleaner.prepare(source, validate=cfg.dataset.validate, use_cache=True)
    return result.source, result.is_zip, result.entries


def _build_onthefly(cfg: Config) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """Build train, val and test datasets by reading videos on the fly."""
    _, train_zip, train_entries = _scan(cfg, cfg.dataset.train_path)
    train_entries = cap_entries(train_entries, cfg.dataset.max_train_samples)
    train_ds = _make_video_dataset(
        cfg, cfg.dataset.train_path, train_zip, train_entries, train=True
    )
    _, test_zip, test_entries = _scan(cfg, cfg.dataset.test_path)
    test_entries = cap_entries(test_entries, cfg.dataset.max_test_samples)
    val_entries, final_entries = split_val_test(
        test_entries, cfg.dataset.val_prob, seed=cfg.seed
    )
    val_ds = _make_video_dataset(
        cfg, cfg.dataset.test_path, test_zip, val_entries, train=False
    )
    test_ds = _make_video_dataset(
        cfg, cfg.dataset.test_path, test_zip, final_entries, train=False
    )
    return train_ds, val_ds, test_ds


def _build_hdf5(cfg: Config) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """Build datasets from pre-built HDF5 files, splitting test into val/test."""
    train_ds = HDF5ClipDataset(cfg.dataset.train_h5)
    full_test = HDF5ClipDataset(cfg.dataset.test_h5)
    order = np.random.default_rng(cfg.seed).permutation(len(full_test))
    n_val = int(round(len(full_test) * cfg.dataset.val_prob))
    val_idx = [int(i) for i in order[:n_val]]
    test_idx = [int(i) for i in order[n_val:]]
    val_ds = Subset(full_test, val_idx) if val_idx else None
    test_ds = Subset(full_test, test_idx) if test_idx else None
    return train_ds, val_ds, test_ds


def _make_loader(cfg: Config, dataset: Dataset, collate, shuffle: bool,
                 batch_size: int, num_workers: int) -> ResumableDataLoader:
    """Wrap a dataset in the resumable loader used across the program."""
    return ResumableDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        seed=cfg.seed,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(cfg.device == "cuda"),
        worker_init_fn=_seed_worker,
    )


def build_data_bundle(cfg: Config, batch_size: int,
                      num_workers: int) -> DataBundle:
    """Build the train/val/test loaders for a run."""
    if cfg.dataset.use_hdf5:
        train_ds, val_ds, test_ds = _build_hdf5(cfg)
    else:
        train_ds, val_ds, test_ds = _build_onthefly(cfg)
    collate = build_collator(cfg)
    train_loader = _make_loader(cfg, train_ds, collate, True, batch_size, num_workers)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = _make_loader(cfg, val_ds, collate, False, batch_size, num_workers)
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = _make_loader(cfg, test_ds, collate, False, batch_size, num_workers)
    return DataBundle(
        train_loader, val_loader, test_loader,
        len(train_ds), len(val_ds) if val_ds else 0, len(test_ds) if test_ds else 0,
    )


def build_eval_loader(cfg: Config, batch_size: int, num_workers: int
                      ) -> Tuple[ResumableDataLoader, int]:
    """Build one loader over the whole test set (no validation split).

    :returns: ``(loader, num_samples)`` for the full test set.
    """
    if cfg.dataset.use_hdf5:
        dataset = HDF5ClipDataset(cfg.dataset.test_h5)
    else:
        _, is_zip, entries = _scan(cfg, cfg.dataset.test_path)
        entries = cap_entries(entries, cfg.dataset.max_test_samples)
        dataset = _make_video_dataset(
            cfg, cfg.dataset.test_path, is_zip, entries, train=False
        )
    collate = build_collator(cfg)
    loader = _make_loader(cfg, dataset, collate, False, batch_size, num_workers)
    return loader, len(dataset)
