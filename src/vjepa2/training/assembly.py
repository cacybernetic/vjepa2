# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the model and the training pieces from a Config. This keeps the mapping
# from config values to constructor arguments in one place, so the train and
# evaluate programs stay short.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from vjepa2 import logging as vlog
from vjepa2.config import Config
from vjepa2.lr_shedulers import build_scheduler
from vjepa2.model import VJEPA21, init_video_model
from vjepa2.optimizers import build_optimizer
from vjepa2.training.ema import EmaUpdater

__all__ = ["build_model", "build_ema", "build_optimizer_scheduler"]


def build_model(cfg: Config, device: str) -> VJEPA21:
    """Create the full V-JEPA 2.1 model (encoder + predictor + EMA target)."""
    # The predictor only ever consumes one learnable mask token per predict
    # mask (``mask_index`` runs over ``num_pred_masks``). Any extra tokens would
    # never receive a gradient and would ship as dead zero-initialised
    # parameters, so cap the count at ``num_pred_masks``.
    num_pred_masks = max(1, int(cfg.masking.num_pred_masks))
    num_mask_tokens = int(cfg.model.num_mask_tokens)
    if num_mask_tokens > num_pred_masks:
        vlog.logger.warning(
            "num_mask_tokens ({}) exceeds num_pred_masks ({}); only the first "
            "{} are ever trained -- clamping to avoid dead parameters.",
            num_mask_tokens, num_pred_masks, num_pred_masks)
        num_mask_tokens = num_pred_masks
    encoder, predictor = init_video_model(
        device=device,
        model_name=cfg.model.name,
        patch_size=cfg.model.patch_size,
        tubelet_size=cfg.model.tubelet_size,
        max_num_frames=cfg.dataset.num_frames,
        crop_size=cfg.dataset.crop_size,
        pred_depth=cfg.model.pred_depth,
        pred_embed_dim=cfg.model.pred_embed_dim,
        num_mask_tokens=num_mask_tokens,
        use_rope=cfg.model.use_rope,
        use_sdpa=cfg.model.use_sdpa,
        modality_embedding=cfg.model.modality_embedding,
        interpolate_rope=cfg.model.interpolate_rope,
        drop_path_rate=cfg.model.drop_path_rate,
        img_temporal_dim_size=1,
        use_mask_tokens=True,
        return_all_tokens=True,
        n_output_distillation_encoder=cfg.model.n_output_distillation,
        n_output_distillation_predictor=cfg.model.n_output_distillation,
        teacher_embed_dim=None,
    )
    return VJEPA21(encoder, predictor).to(device)


def build_ema(cfg: Config, total_steps: Optional[int] = None) -> EmaUpdater:
    """Create the EMA updater for the target encoder.

    ``total_steps`` enables the momentum ramp (``optim.ema`` -> ``optim.ema_end``);
    when omitted the momentum stays constant at ``optim.ema``.
    """
    return EmaUpdater(
        momentum=cfg.optim.ema,
        momentum_end=cfg.optim.ema_end,
        total_steps=total_steps,
    )


def build_optimizer_scheduler(model: VJEPA21, cfg: Config, total_steps: int
                              ) -> Tuple[torch.optim.Optimizer, object, Dict[str, int]]:
    """Build the optimizer, scheduler and report the parameter group sizes."""
    optimizer, counts = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)
    return optimizer, scheduler, counts
