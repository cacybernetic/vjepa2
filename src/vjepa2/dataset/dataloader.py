# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# A DataLoader adapter that can save and restore its progress inside an epoch.
# The sampler builds a fixed index order for the epoch (like RandomSampler with
# a seeded generator). We store that order and the current position in the
# state dict, so a run that crashes can resume at the exact same batch instead
# of starting the whole epoch again (in-epoch checkpointing).

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

__all__ = ["ResumableSampler", "ResumableDataLoader"]


class ResumableSampler(Sampler):
    """Yield a fixed, resumable index order for one epoch.

    When ``shuffle`` is True the order is a permutation drawn from a generator
    seeded by ``seed + epoch``, so every epoch has its own stable order that we
    can reproduce after a restart. Iteration starts at ``position`` so a resumed
    epoch skips the batches that were already processed.
    """

    def __init__(self, num_samples: int, shuffle: bool = True, seed: int = 42):
        self.num_samples = int(num_samples)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.position = 0
        self.order: List[int] = list(range(self.num_samples))

    def set_epoch(self, epoch: int) -> None:
        """Start a fresh epoch: rebuild the order and reset the position."""
        self.epoch = int(epoch)
        self.position = 0
        self.order = self._build_order(self.epoch)

    def resume(self, epoch: int, position: int) -> None:
        """Continue an interrupted epoch from a saved position."""
        self.epoch = int(epoch)
        self.order = self._build_order(self.epoch)
        self.position = max(0, int(position))

    def _build_order(self, epoch: int) -> List[int]:
        """Return the index order used for a given epoch."""
        if not self.shuffle:
            return list(range(self.num_samples))
        rng = np.random.default_rng(self.seed + epoch)
        return [int(i) for i in rng.permutation(self.num_samples)]

    def __iter__(self) -> Iterator[int]:
        return iter(self.order[self.position:])

    def __len__(self) -> int:
        return max(0, self.num_samples - self.position)

    def state_dict(self) -> Dict[str, Any]:
        """Return the sampler state, including the full index order."""
        return {
            "epoch": self.epoch,
            "position": self.position,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "num_samples": self.num_samples,
            "order": list(self.order),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the sampler state from a saved dict."""
        self.seed = int(state.get("seed", self.seed))
        self.shuffle = bool(state.get("shuffle", self.shuffle))
        self.epoch = int(state.get("epoch", 0))
        self.position = int(state.get("position", 0))
        order = state.get("order")
        if order and len(order) == self.num_samples:
            self.order = [int(i) for i in order]
        else:
            self.order = self._build_order(self.epoch)


class ResumableDataLoader:
    """Wrap a DataLoader with a resumable sampler and progress tracking.

    This is the single loader type used everywhere in the program so that each
    sample is seen exactly once per epoch, and so the train, validation and
    test passes can each save and restore their own mid-epoch progress.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool,
                 collate_fn: Callable, seed: int = 42, num_workers: int = 0,
                 drop_last: bool = False, pin_memory: bool = False,
                 worker_init_fn: Optional[Callable] = None):
        self.batch_size = int(batch_size)
        self.sampler = ResumableSampler(len(dataset), shuffle, seed)
        self._position = 0
        self._loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )

    @property
    def epoch(self) -> int:
        return self.sampler.epoch

    @property
    def position(self) -> int:
        return self._position

    def set_epoch(self, epoch: int) -> None:
        """Begin a new epoch from the start."""
        self.sampler.set_epoch(epoch)
        self._position = 0

    def resume(self, epoch: int, position: int) -> None:
        """Resume an epoch from a previously saved position."""
        self.sampler.resume(epoch, position)
        self._position = max(0, int(position))

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        """Iterate batches and count how many samples have been consumed."""
        for batch in self._loader:
            self._position += self._batch_count(batch)
            yield batch

    def _batch_count(self, batch: Any) -> int:
        """Read the real batch size from a collated ``([clips], ...)`` batch."""
        clips = batch[0][0]
        return int(clips.shape[0])

    def state_dict(self) -> Dict[str, Any]:
        """Return the loader progress and the sampler index state."""
        sampler_state = self.sampler.state_dict()
        sampler_state["position"] = self._position
        return {"sampler": sampler_state, "batch_size": self.batch_size}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore loader progress from a saved dict."""
        self.sampler.load_state_dict(state["sampler"])
        self._position = int(state["sampler"].get("position", 0))
