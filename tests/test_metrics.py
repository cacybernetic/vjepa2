# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests for the metric meters and the self-supervised quality signals.

import torch

from vjepa2.metrics import AverageMeter, MetricTracker, feature_std, prediction_cosine


def test_average_meter_computes_weighted_mean():
    meter = AverageMeter()
    meter.update(2.0, n=2)   # sum 4, count 2
    meter.update(4.0, n=1)   # sum 8, count 3
    assert abs(meter.average - (8.0 / 3.0)) < 1e-9


def test_average_meter_empty_is_zero():
    assert AverageMeter().average == 0.0


def test_average_meter_state_roundtrip():
    meter = AverageMeter()
    meter.update(3.0, n=4)
    other = AverageMeter()
    other.load_state_dict(meter.state_dict())
    assert other.average == meter.average


def test_metric_tracker_tracks_named_values():
    tracker = MetricTracker(["a", "b"])
    tracker.update({"a": 1.0, "b": 3.0})
    tracker.update({"a": 3.0, "b": 5.0})
    averages = tracker.averages()
    assert averages["a"] == 2.0
    assert averages["b"] == 4.0


def test_feature_std_of_constant_is_zero():
    constant = [torch.ones(2, 5, 8)]
    assert feature_std(constant) < 1e-6


def test_feature_std_of_spread_is_positive():
    spread = [torch.randn(4, 16, 8) * 5.0]
    assert feature_std(spread) > 1.0


def test_prediction_cosine_perfect_match_is_one():
    target = [torch.randn(2, 6, 8)]
    masks = [[torch.tensor([[0, 1, 2], [0, 1, 2]])]]
    picked = target[0][:, :3, :]
    z_pred = [[picked.clone()]]
    assert abs(prediction_cosine(z_pred, target, masks) - 1.0) < 1e-5
