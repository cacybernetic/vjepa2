# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Metrics that describe how the self-supervised model is doing. There are no
# labels, so we look at the prediction error and at how spread out the target
# features are. A very low spread means the features collapsed, which is bad.

from __future__ import annotations

from typing import List

import torch

from vjepa2.modules.tensors import apply_masks

__all__ = [
    "feature_std",
    "feature_correlation",
    "prediction_cosine",
    "METRIC_NAMES",
]

# The scalar names tracked during training and validation.
METRIC_NAMES = ["loss", "predict", "context", "feat_std", "feat_corr", "pred_cos"]


@torch.no_grad()
def feature_std(h_target: List[torch.Tensor]) -> float:
    """Average per-dimension standard deviation of the target features.

    A healthy encoder keeps this well above zero. Values near zero mean the
    representation collapsed to a constant.
    """
    if not h_target:
        return 0.0
    values = []
    for level in h_target:
        flat = level.reshape(-1, level.shape[-1]).float()
        values.append(flat.std(dim=0).mean())
    return float(torch.stack(values).mean())


@torch.no_grad()
def feature_correlation(h_target: List[torch.Tensor], max_dims: int = 512
                        ) -> float:
    """Mean absolute off-diagonal correlation of the target feature dimensions.

    ``feat_std`` alone misses *dimensional* collapse: the per-token LayerNorm of
    the targets keeps the std near 1 even when the encoder has degenerated to a
    low-rank subspace (all dimensions carrying the same signal). This measures
    how correlated the feature dimensions are: ~0 for a healthy, decorrelated
    representation, approaching 1 when the features collapse onto a common
    direction. Dimensions are capped at ``max_dims`` to bound the cost on very
    wide encoders.
    """
    if not h_target:
        return 0.0
    values = []
    for level in h_target:
        flat = level.reshape(-1, level.shape[-1]).float()
        dim = flat.shape[-1]
        if dim > max_dims:
            flat = flat[:, :max_dims]
            dim = max_dims
        if flat.shape[0] < 2 or dim < 2:
            continue
        flat = flat - flat.mean(dim=0, keepdim=True)
        std = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
        flat = flat / std
        corr = (flat.t() @ flat) / flat.shape[0]  # (D, D)
        off_diag_sum = corr.abs().sum() - corr.diagonal().abs().sum()
        values.append(off_diag_sum / (dim * (dim - 1)))
    if not values:
        return 0.0
    return float(torch.stack(values).mean())


def _flatten_nested(nested: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Flatten a ``[fpc][mask]`` nested list into a flat tensor list."""
    flat: List[torch.Tensor] = []
    for group in nested:
        flat.extend(group)
    return flat


@torch.no_grad()
def prediction_cosine(z_pred: List[List[torch.Tensor]],
                      h_target: List[torch.Tensor],
                      masks_pred: List[List[torch.Tensor]]) -> float:
    """Mean cosine similarity between predictions and their targets.

    Closer to 1.0 means the predictor matches the target-encoder features on
    the masked tokens.
    """
    targets = [apply_masks(hi, mi, concat=False)
               for hi, mi in zip(h_target, masks_pred)]
    preds = _flatten_nested(z_pred)
    gts = _flatten_nested(targets)
    if not preds:
        return 0.0
    sims = []
    for pred, gt in zip(preds, gts):
        sims.append(torch.cosine_similarity(pred, gt, dim=-1).mean())
    return float(torch.stack(sims).mean())
