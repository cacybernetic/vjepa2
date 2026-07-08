# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the optimizer with two parameter groups: one with weight decay for the
# matrix weights, and one without weight decay for biases, norm weights, and
# small embedding vectors (positional, modality and mask tokens). This is the
# standard recipe for training vision transformers.

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from vjepa2.config import OptimConfig

__all__ = ["split_param_groups", "build_optimizer"]

# Name fragments whose parameters should never receive weight decay.
_NO_DECAY_HINTS = ("pos_embed", "mod_embed", "mask_token", "cls_token")


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    """Return True when a parameter must be excluded from weight decay."""
    if param.ndim <= 1:
        return True
    return any(hint in name for hint in _NO_DECAY_HINTS)


def split_param_groups(model: torch.nn.Module, weight_decay: float
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Split trainable parameters into decay and no-decay groups.

    :returns: ``(param_groups, counts)`` where counts holds the number of
        parameters in each group for logging.
    """
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay(name, param):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    counts = {"decay": len(decay), "no_decay": len(no_decay)}
    return groups, counts


def build_optimizer(model: torch.nn.Module, cfg: OptimConfig
                    ) -> Tuple[torch.optim.Optimizer, Dict[str, int]]:
    """Build an optimizer with the correct weight-decay groups.

    Supported names: ``adamw``, ``adam`` and ``sgd``.
    """
    groups, counts = split_param_groups(model, cfg.weight_decay)
    name = cfg.name.lower()
    if name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas))
    elif name == "adam":
        optimizer = torch.optim.Adam(groups, lr=cfg.lr, betas=tuple(cfg.betas))
    elif name == "sgd":
        optimizer = torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError(f"Unknown optimizer name: {cfg.name}")
    return optimizer, counts
