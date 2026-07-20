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
    """Collate clips into a batch and generate per-sample context/predict masks.

    The RNG is deliberately *not* part of the pickled state: a DataLoader
    pickles the collator into every worker (and again at every epoch when
    workers are not persistent), so shipping a generator state would make every
    worker -- and every epoch -- replay the exact same mask sequence. Instead
    each process lazily seeds its own generator: workers derive it from their
    per-worker, per-epoch ``torch.initial_seed()``, the main process uses the
    configured seed (or fresh entropy when none is given).
    """

    def __init__(self, cfg: MaskingConfig, grid_size: int, grid_depth: int,
                 seed: Optional[int] = None, deterministic: bool = False):
        self.cfg = cfg
        self.grid_size = int(grid_size)
        self.grid_depth = int(grid_depth)
        self.seed = seed
        # Deterministic mode (evaluation): every call draws masks from a fresh
        # generator seeded with ``seed``, so the masks -- and therefore the
        # validation / test metric -- are identical across epochs, workers and
        # runs. This makes best-model selection reproducible instead of hostage
        # to a random mask draw. Training keeps the stochastic per-worker RNG.
        self.deterministic = bool(deterministic)
        self._rng: Optional[np.random.Generator] = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_rng"] = None
        return state

    def _ensure_rng(self) -> np.random.Generator:
        if self.deterministic:
            # Reseed on every call for a fully reproducible mask stream.
            self._rng = np.random.default_rng(self.seed)
            return self._rng
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            if worker is not None:
                self._rng = np.random.default_rng(
                    torch.initial_seed() % (2 ** 63)
                )
            else:
                self._rng = np.random.default_rng(self.seed)
        return self._rng

    def __call__(self, samples: List[torch.Tensor]):
        """Stack samples and attach the nested ``[fpc][mask]`` mask lists.

        ``cfg.num_pred_masks`` independent context/predict partitions are drawn
        *per sample*; the encoder and predictor iterate over each of them. The
        per-sample index lists are randomly subsampled to a common length so
        they stack into rectangular ``(B, K)`` tensors (tokens dropped this way
        are simply neither context nor prediction targets for that step).
        """
        self._ensure_rng()
        clips = torch.stack(samples, dim=0)
        batch = clips.shape[0]
        enc_masks: List[torch.Tensor] = []
        pred_masks: List[torch.Tensor] = []
        for _ in range(max(1, int(self.cfg.num_pred_masks))):
            partitions = [self._sample_partition() for _ in range(batch)]
            enc_lists = [p[0] for p in partitions]
            pred_lists = [p[1] for p in partitions]
            min_enc = min(len(e) for e in enc_lists)
            min_pred = min(len(p) for p in pred_lists)
            enc = torch.stack(
                [self._subsample(e, min_enc) for e in enc_lists], dim=0
            )
            pred = torch.stack(
                [self._subsample(p, min_pred) for p in pred_lists], dim=0
            )
            enc_masks.append(enc)
            pred_masks.append(pred)
        return [clips], [enc_masks], [pred_masks]

    def _subsample(self, indices: List[int], count: int) -> torch.Tensor:
        """Randomly keep ``count`` indices, preserving their sorted order."""
        if len(indices) > count:
            keep = self._rng.choice(len(indices), size=count, replace=False)
            indices = [indices[i] for i in sorted(keep)]
        return torch.as_tensor(indices, dtype=torch.long)

    def _sample_partition(self) -> Tuple[List[int], List[int]]:
        """Return flat (context, predict) token indices over the whole clip."""
        spatial = self._sample_spatial_mask()
        spatial = self._enforce_min_keep(spatial)
        return self._expand_to_tube(spatial)

    def _enforce_min_keep(self, mask: np.ndarray) -> np.ndarray:
        """Free enough spatial cells so the context has at least ``min_keep``.

        The mask is tube-expanded across ``grid_depth`` frames, so the number of
        context tokens is ``grid_depth * (#unmasked cells)``. We unmask random
        predicted cells until that product reaches ``min_keep``.
        """
        min_keep = max(1, int(self.cfg.min_keep))
        min_context_cells = -(-min_keep // max(1, self.grid_depth))  # ceil div
        flat = mask.reshape(-1)
        while (~flat).sum() < min_context_cells and flat.any():
            masked_cells = np.flatnonzero(flat)
            flat[self._rng.choice(masked_cells)] = False
        return flat.reshape(mask.shape)

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
