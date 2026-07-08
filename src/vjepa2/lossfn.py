# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The V-JEPA 2.1 dense predictive loss, wrapped as one module:
#   L_dense = L_predict + lambda(iter) * L_ctx
# L_predict is the masked-token L1 loss and L_ctx is the distance-weighted L1
# loss on the context tokens. The context weight grows with a linear warmup.

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from vjepa2.config import Config, LossConfig
from vjepa2.dataset.masking import grid_dims
from vjepa2.modules.losses import (
    Lambda_LinearWarmupHold,
    compute_mask_distance,
    jepa_loss,
)

__all__ = ["DensePredictiveLoss", "build_loss"]

NestedTensor = List[List[torch.Tensor]]


class DensePredictiveLoss(nn.Module):
    """Compute the dense predictive loss and its parts for logging."""

    def __init__(self, grid_size: int, loss_exp: float = 1.0,
                 context_lambda: float = 0.5, warmup_start: int = 15000,
                 warmup_end: int = 30000, offset_context_loss: bool = False):
        super().__init__()
        self.grid_size = int(grid_size)
        self.loss_exp = float(loss_exp)
        self.offset_context_loss = bool(offset_context_loss)
        self.lambda_schedule = Lambda_LinearWarmupHold(
            context_lambda, warmup_start, warmup_end
        )

    def forward(self, z_pred: NestedTensor, z_context: NestedTensor,
                h_target: List[torch.Tensor], masks_enc: NestedTensor,
                masks_pred: NestedTensor, global_iter: int
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the total loss and a dict of its scalar parts.

        :param z_pred: predictor outputs for masked tokens ``[fpc][mask]``.
        :param z_context: predictor outputs for context tokens ``[fpc][mask]``.
        :param h_target: target-encoder outputs ``[fpc]`` (full sequence).
        :param masks_enc: context token indices ``[fpc][mask]``.
        :param masks_pred: masked token indices ``[fpc][mask]``.
        :param global_iter: optimizer step count, drives the lambda warmup.
        """
        predict = jepa_loss(z_pred, h_target, masks_pred, loss_exp=self.loss_exp)
        weights = compute_mask_distance(
            masks_pred, masks_enc, self.grid_size, self.offset_context_loss
        )
        context = jepa_loss(
            z_context, h_target, masks_enc, loss_exp=self.loss_exp, d_weights=weights
        )
        lam = self.lambda_schedule.value(global_iter)
        total = predict + lam * context
        parts = {
            "loss": float(total.detach()),
            "predict": float(predict.detach()),
            "context": float(context.detach()),
            "lambda": float(lam),
        }
        return total, parts


def build_loss(cfg: Config) -> DensePredictiveLoss:
    """Build the dense predictive loss from a Config."""
    grid_size, _ = grid_dims(
        cfg.dataset.crop_size,
        cfg.model.patch_size,
        cfg.dataset.num_frames,
        cfg.model.tubelet_size,
    )
    loss_cfg: LossConfig = cfg.loss
    return DensePredictiveLoss(
        grid_size=grid_size,
        loss_exp=loss_cfg.loss_exp,
        context_lambda=loss_cfg.context_lambda,
        warmup_start=loss_cfg.lambda_warmup_start,
        warmup_end=loss_cfg.lambda_warmup_end,
        offset_context_loss=loss_cfg.offset_context_loss,
    )
