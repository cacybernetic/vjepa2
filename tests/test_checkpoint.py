# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests for checkpoint rotation, run-folder numbering and the best tracker.

from vjepa2.training.best_model import BestModelTracker
from vjepa2.training.checkpoint import CheckpointManager
from vjepa2.training.runs import RunDirManager


def test_checkpoint_manager_rotates_old_files(tmp_path):
    manager = CheckpointManager(str(tmp_path / "ckpts"), max_checkpoint=2)
    for epoch in range(4):
        manager.save({"epoch": epoch}, epoch, 0, "train")
    latest = manager.load_latest()
    assert latest["epoch"] == 3
    # Only the two newest files remain, each in its own dedicated file.
    kept = manager._all_files()
    assert kept == ["checkpoint_train_e0002c0000.pth",
                    "checkpoint_train_e0003c0000.pth"]


def test_checkpoint_manager_distinct_files_within_epoch(tmp_path):
    manager = CheckpointManager(str(tmp_path / "ckpts"), max_checkpoint=10)
    # Several saves of the SAME epoch must never share a file.
    manager.save({"c": 1}, 1, 1, "train")
    manager.save({"c": 2}, 1, 2, "train")
    manager.save({"c": 3}, 1, 1, "val")
    files = manager._all_files()
    assert files == ["checkpoint_train_e0001c0001.pth",
                     "checkpoint_train_e0001c0002.pth",
                     "checkpoint_val_e0001c0001.pth"]
    # Within an epoch the validation checkpoint is written after the train ones,
    # so it is the latest.
    assert manager.load_latest()["c"] == 3


def test_checkpoint_manager_empty(tmp_path):
    manager = CheckpointManager(str(tmp_path / "c"), max_checkpoint=3)
    assert manager.latest_path() is None
    assert manager.has_checkpoint() is False


def test_run_dir_numbering(tmp_path):
    manager = RunDirManager(str(tmp_path), "myrun")
    first, reused = manager.resolve("train", resume=False)
    assert first.endswith("train") and reused is False
    manager.make_paths(first)
    second, _ = manager.resolve("train", resume=False)
    assert second.endswith("train2")
    manager.make_paths(second)
    # With resume we reuse the latest existing folder.
    latest, reused = manager.resolve("train", resume=True)
    assert latest.endswith("train2") and reused is True


def test_run_dir_eval_numbering(tmp_path):
    manager = RunDirManager(str(tmp_path), "myrun")
    path, _ = manager.resolve("eval", resume=False)
    assert path.endswith("eval")
    manager.make_paths(path, kind="eval")
    path2, _ = manager.resolve("eval", resume=False)
    assert path2.endswith("eval2")


def test_best_tracker_min_and_max():
    low = BestModelTracker("loss", "min")
    assert low.consider(1.0) is True
    assert low.consider(0.5) is True
    assert low.consider(0.8) is False
    high = BestModelTracker("acc", "max")
    assert high.consider(0.5) is True
    assert high.consider(0.9) is True
    assert high.consider(0.7) is False


def test_best_tracker_state_roundtrip():
    tracker = BestModelTracker("loss", "min")
    tracker.consider(0.3)
    other = BestModelTracker("x", "max")
    other.load_state_dict(tracker.state_dict())
    assert other.best == 0.3
    assert other.mode == "min"
