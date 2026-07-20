# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests for the EMA target-encoder updater and its momentum schedule.

import torch
import torch.nn as nn

from vjepa2.training.ema import EmaUpdater


def test_momentum_is_constant_without_a_schedule():
    updater = EmaUpdater(momentum=0.99)
    assert updater.momentum_at(0) == 0.99
    assert updater.momentum_at(10_000) == 0.99
    assert updater.momentum_at(None) == 0.99


def test_momentum_ramps_from_start_to_end():
    updater = EmaUpdater(momentum=0.9, momentum_end=1.0, total_steps=10)
    assert updater.momentum_at(0) == 0.9
    assert updater.momentum_at(10) == 1.0
    assert updater.momentum_at(20) == 1.0            # clamped past the end
    mid = updater.momentum_at(5)
    assert 0.9 < mid < 1.0
    assert abs(mid - 0.95) < 1e-9                    # linear halfway


def test_update_blends_target_toward_online():
    online = nn.Linear(4, 4)
    target = nn.Linear(4, 4)
    with torch.no_grad():
        for p in online.parameters():
            p.fill_(1.0)
        for p in target.parameters():
            p.fill_(0.0)
    # Momentum 0.9 with an end-of-run step (progress 1.0) -> momentum_end 1.0
    # would freeze the target; use a constant momentum here.
    EmaUpdater(momentum=0.9).update(online, target)
    for p in target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1))
