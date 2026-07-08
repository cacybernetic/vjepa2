# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests for the optimizer parameter groups and the learning-rate schedulers.

import torch
import torch.nn as nn

from vjepa2.config import OptimConfig, SchedulerConfig
from vjepa2.lr_shedulers import WarmupCosine, WarmupHold, build_scheduler
from vjepa2.optimizers import build_optimizer, split_param_groups


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, 4))


def test_split_param_groups_excludes_norms_and_1d():
    model = _Tiny()
    groups, counts = split_param_groups(model, weight_decay=0.05)
    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0
    # Only the linear weight matrix gets decay; bias, norm, pos_embed do not.
    assert counts["decay"] == 1
    assert counts["no_decay"] == 4


def test_build_optimizer_adamw():
    model = _Tiny()
    optimizer, _ = build_optimizer(model, OptimConfig(name="adamw", lr=1e-3))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2


def test_warmup_hold_reaches_ref_lr_and_holds():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    sched = WarmupHold(optimizer, start_lr=0.0, ref_lr=1.0, warmup_steps=10)
    assert sched.lr_at(0) == 0.0
    assert abs(sched.lr_at(5) - 0.5) < 1e-9
    assert sched.lr_at(10) == 1.0
    assert sched.lr_at(100) == 1.0


def test_warmup_cosine_decays_to_final():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    sched = WarmupCosine(optimizer, 0.0, 1.0, 0.0, warmup_steps=2, total_steps=12)
    assert sched.lr_at(2) == 1.0            # start of cosine at ref
    assert sched.lr_at(12) < 1e-6           # end of cosine at final
    assert sched.lr_at(1) < 1.0             # still warming up


def test_scheduler_step_applies_lr():
    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([param], lr=0.0)
    sched = build_scheduler(optimizer, SchedulerConfig(name="warmup_hold",
                            warmup_steps=1, start_lr=0.1, ref_lr=0.9), 10)
    sched.step()
    assert optimizer.param_groups[0]["lr"] == 0.1
