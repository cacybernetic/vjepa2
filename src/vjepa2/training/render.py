# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Draw a PCA feature map of a clip, like the figures in the paper. We take the
# patch features, reduce them to three values with PCA, and map those to red,
# green and blue. Similar parts of the scene get similar colors.

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

__all__ = ["FeatureMapRenderer"]


def _pca_rgb(features: torch.Tensor) -> np.ndarray:
    """Project ``(N, D)`` features to ``(N, 3)`` in the 0..1 range with PCA."""
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:3].T
    projected = projected.cpu().numpy()
    low = np.percentile(projected, 2, axis=0)
    high = np.percentile(projected, 98, axis=0)
    scaled = (projected - low) / np.maximum(high - low, 1e-6)
    return np.clip(scaled, 0.0, 1.0)


class FeatureMapRenderer:
    """Render a montage of per-frame PCA feature maps for one clip."""

    def __init__(self, grid: Tuple[int, int, int]):
        self.grid_t, self.grid_h, self.grid_w = grid

    @torch.no_grad()
    def render(self, model, clip: torch.Tensor, out_path: str) -> bool:
        """Save a PCA feature map montage; return True on success."""
        try:
            features = model.extract_features(clip.unsqueeze(0), use_ema=True)
            rgb = _pca_rgb(features[0].float())
            image = self._montage(rgb)
            self._save(image, out_path)
            return True
        except Exception:
            return False

    def _montage(self, rgb: np.ndarray) -> np.ndarray:
        """Lay out the temporal maps side by side into one image."""
        maps = rgb.reshape(self.grid_t, self.grid_h, self.grid_w, 3)
        frames = [maps[t] for t in range(self.grid_t)]
        return np.concatenate(frames, axis=1)

    def _save(self, image: np.ndarray, out_path: str) -> None:
        """Write the RGB float image to disk as a JPG."""
        from PIL import Image

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        array = (image * 255.0).astype(np.uint8)
        Image.fromarray(array).resize(
            (array.shape[1] * 16, array.shape[0] * 16), Image.NEAREST
        ).save(out_path)
