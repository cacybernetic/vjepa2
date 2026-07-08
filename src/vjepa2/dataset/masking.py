# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the context / predict masks used by V-JEPA. We use "tube" masking:
# a 2D spatial block is chosen on the patch grid and repeated across every time
# step. The predict tokens are the masked blocks; the context tokens are the
# rest. This collator also stacks the clips into one batch tensor.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from vjepa2.config import MaskingConfig

__all__ = ["grid_dims", "TubeMaskCollator"]


def grid_dims(crop_size: int, patch_size: int, num_frames: int,
              tubelet_size: int) -> Tuple[int, int]:
    """Return ``(grid_size, grid_depth)`` token counts for a clip.

    ``grid_size`` is the number of patches along height (equals width for a
    square crop); ``grid_depth`` is the number of temporal tokens.
    """
    grid_size = crop_size // patch_size
    grid_depth = max(1, num_frames // tubelet_size)
    return grid_size, grid_depth


class TubeMaskCollator:
    """Collate clips into a batch and generate shared context/predict masks."""

    def __init__(self, cfg: MaskingConfig, grid_size: int, grid_depth: int,
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.grid_size = int(grid_size)
        self.grid_depth = int(grid_depth)
        self._rng = np.random.default_rng(seed)

    def __call__(self, samples: List[torch.Tensor]):
        """Stack samples and attach the nested ``[fpc][mask]`` mask lists."""
        clips = torch.stack(samples, dim=0)
        batch = clips.shape[0]
        enc_idx, pred_idx = self._sample_partition()
        enc = torch.as_tensor(enc_idx, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
        pred = torch.as_tensor(pred_idx, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
        return [clips], [[enc]], [[pred]]

    def _sample_partition(self) -> Tuple[List[int], List[int]]:
        """Return flat (context, predict) token indices over the whole clip."""
        spatial = self._sample_spatial_mask()
        return self._expand_to_tube(spatial)

    def _sample_spatial_mask(self) -> np.ndarray:
        """Return a boolean grid where True marks a token to predict."""
        gh = gw = self.grid_size
        mask = np.zeros((gh, gw), dtype=bool)
        target = self._rng.uniform(self.cfg.spatial_scale[0], self.cfg.spatial_scale[1])
        for _ in range(100):
            if mask.mean() >= target:
                break
            self._place_block(mask)
        return self._enforce_limits(mask)

    def _place_block(self, mask: np.ndarray) -> None:
        """Mark one random rectangular block as predicted (in place)."""
        gh, gw = mask.shape
        area = self._rng.uniform(0.05, 0.2) * gh * gw
        log_ratio = np.log(self.cfg.aspect_ratio)
        ratio = float(np.exp(self._rng.uniform(log_ratio[0], log_ratio[1])))
        block_h = int(np.clip(round(np.sqrt(area * ratio)), 1, gh))
        block_w = int(np.clip(round(np.sqrt(area / ratio)), 1, gw))
        top = int(self._rng.integers(0, gh - block_h + 1))
        left = int(self._rng.integers(0, gw - block_w + 1))
        mask[top:top + block_h, left:left + block_w] = True

    def _enforce_limits(self, mask: np.ndarray) -> np.ndarray:
        """Make sure both context and predict regions are non-trivial."""
        if not mask.any():
            mask[0, 0] = True
        if mask.all():
            mask[-1, -1] = False
        return mask

    def _expand_to_tube(self, spatial: np.ndarray) -> Tuple[List[int], List[int]]:
        """Repeat the 2D spatial mask across every time step to flat indices."""
        gh, gw = spatial.shape
        per_frame = gh * gw
        enc_idx: List[int] = []
        pred_idx: List[int] = []
        flat = spatial.reshape(-1)
        for t in range(self.grid_depth):
            base = t * per_frame
            for cell, is_pred in enumerate(flat):
                if is_pred:
                    pred_idx.append(base + cell)
                else:
                    enc_idx.append(base + cell)
        return enc_idx, pred_idx
