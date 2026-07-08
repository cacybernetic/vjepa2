# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests for dataset discovery, cache, splits, masking and the resumable loader.

import os
import zipfile

import numpy as np
import torch

from vjepa2.config import MaskingConfig
from vjepa2.dataset.cache import CacheStore, cache_path
from vjepa2.dataset.dataloader import ResumableDataLoader, ResumableSampler
from vjepa2.dataset.discovery import VideoFileFinder
from vjepa2.dataset.masking import TubeMaskCollator, grid_dims
from vjepa2.dataset.splits import cap_entries, split_val_test


def test_finder_walks_folder_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "sub" / "b.avi").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")
    entries = VideoFileFinder().find(str(tmp_path))
    assert sorted(entries) == ["a.mp4", os.path.join("sub", "b.avi")]


def test_finder_reads_zip(tmp_path):
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("clip1.mp4", b"x")
        zf.writestr("nested/clip2.mkv", b"x")
        zf.writestr("readme.txt", b"x")
    finder = VideoFileFinder()
    assert finder.is_zip(str(archive)) is True
    assert finder.find(str(archive)) == ["clip1.mp4", "nested/clip2.mkv"]


def test_cache_path_next_to_source(tmp_path):
    src = str(tmp_path / "train.zip")
    assert cache_path(src).endswith("train.cache.json")


def test_cache_store_roundtrip(tmp_path):
    src = str(tmp_path / "train.zip")
    (tmp_path / "train.zip").write_bytes(b"x")
    store = CacheStore()
    store.save(src, ["a.mp4", "b.mp4"], is_zip=True)
    loaded = store.load(src)
    assert loaded["entries"] == ["a.mp4", "b.mp4"]
    assert loaded["is_zip"] is True


def test_cap_entries_limits_count():
    assert cap_entries(["a", "b", "c"], 2) == ["a", "b"]
    assert cap_entries(["a", "b"], None) == ["a", "b"]


def test_split_val_test_is_disjoint_and_stable():
    entries = [f"v{i}" for i in range(10)]
    val, test = split_val_test(entries, 0.4, seed=7)
    assert len(val) == 4 and len(test) == 6
    assert set(val).isdisjoint(set(test))
    assert set(val) | set(test) == set(entries)
    # Deterministic given the seed.
    val2, _ = split_val_test(entries, 0.4, seed=7)
    assert val == val2


def test_mask_collator_partitions_all_tokens():
    grid_size, grid_depth = grid_dims(64, 16, 4, 2)
    collator = TubeMaskCollator(MaskingConfig(), grid_size, grid_depth, seed=0)
    clips, enc, pred = collator([torch.zeros(3, 4, 64, 64) for _ in range(3)])
    total = grid_size * grid_size * grid_depth
    enc_idx = enc[0][0][0].tolist()
    pred_idx = pred[0][0][0].tolist()
    assert clips[0].shape[0] == 3
    assert set(enc_idx).isdisjoint(set(pred_idx))
    assert len(enc_idx) + len(pred_idx) == total


def test_resumable_sampler_order_is_stable_and_skippable():
    sampler = ResumableSampler(6, shuffle=True, seed=1)
    sampler.set_epoch(0)
    full = list(sampler)
    assert sorted(full) == list(range(6))
    sampler.resume(0, position=2)
    assert list(sampler) == full[2:]


def test_resumable_sampler_state_roundtrip():
    sampler = ResumableSampler(8, shuffle=True, seed=3)
    sampler.set_epoch(2)
    sampler.position = 3
    other = ResumableSampler(8, shuffle=True, seed=0)
    other.load_state_dict(sampler.state_dict())
    assert other.order == sampler.order
    assert other.position == 3


def _collate(samples):
    clips = torch.stack(samples, 0)
    return [clips], [[torch.zeros(len(samples), 1, dtype=torch.long)]], \
        [[torch.zeros(len(samples), 1, dtype=torch.long)]]


class _Clips(torch.utils.data.Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, i):
        return torch.zeros(3, 4, 8, 8)


def test_resumable_loader_position_tracks_samples():
    loader = ResumableDataLoader(_Clips(), batch_size=2, shuffle=False,
                                 collate_fn=_collate, seed=0)
    loader.set_epoch(0)
    for _ in loader:
        pass
    assert loader.position == 6
    state = loader.state_dict()
    assert state["sampler"]["position"] == 6
