# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Split the test entries into a validation part and a final test part. The
# validation set is a fraction of the test set (``val_prob``). The split is
# deterministic given the seed, so it is stable across runs and resumes.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

__all__ = ["split_val_test", "cap_entries"]


def cap_entries(entries: List[str], max_samples: Optional[int]) -> List[str]:
    """Return at most ``max_samples`` entries (None means keep them all)."""
    if max_samples is None:
        return list(entries)
    if max_samples < 0:
        raise ValueError("max_samples must be zero or positive")
    return list(entries[:max_samples])


def split_val_test(entries: List[str], val_prob: float,
                   seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split test entries into ``(val_entries, test_entries)``.

    :param entries: the validated test entry list.
    :param val_prob: fraction (0..1) of the test set used for validation.
    :param seed: makes the shuffle deterministic.
    :returns: two disjoint lists whose union is ``entries``.
    """
    if not 0.0 <= val_prob <= 1.0:
        raise ValueError("val_prob must be between 0 and 1")
    order = np.random.default_rng(seed).permutation(len(entries))
    n_val = int(round(len(entries) * val_prob))
    val_pos = set(int(i) for i in order[:n_val])
    val_entries = [entries[i] for i in range(len(entries)) if i in val_pos]
    test_entries = [entries[i] for i in range(len(entries)) if i not in val_pos]
    return val_entries, test_entries
