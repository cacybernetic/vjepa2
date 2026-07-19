# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Turn a raw clip (T, H, W, 3) uint8 array into a normalized tensor
# (C, T, H, W). Geometry, photometric augmentation and normalization each live
# in their own small class, and a pipeline glues them together based on config.

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF

from vjepa2.config import AugmentConfig, TransformConfig

__all__ = [
    "clip_to_tensor",
    "GeometricTransform",
    "PhotometricAugment",
    "Normalizer",
    "ClipPipeline",
]


def clip_to_tensor(clip_thwc: np.ndarray) -> torch.Tensor:
    """Convert a ``(T, H, W, 3)`` uint8 array to ``(T, 3, H, W)`` float in [0,1]."""
    array = np.ascontiguousarray(clip_thwc)
    if not array.flags.writeable:
        # Video decoders often hand out read-only frame buffers;
        # torch.from_numpy needs a writable array.
        array = array.copy()
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return tensor


class GeometricTransform:
    """Resize and crop every frame of a clip the same way."""

    def __init__(self, cfg: TransformConfig, crop_size: int):
        self.cfg = cfg
        self.crop_size = int(crop_size)

    def __call__(self, clip: torch.Tensor, train: bool,
                 rng: np.random.Generator) -> torch.Tensor:
        """Apply random-resized-crop for training, center-crop otherwise."""
        if train and self.cfg.random_resized_crop:
            return self._random_resized_crop(clip, rng)
        return self._resize_center_crop(clip)

    def _random_resized_crop(self, clip: torch.Tensor,
                             rng: np.random.Generator) -> torch.Tensor:
        """Pick one crop box from the scale/ratio range and apply it to all frames."""
        _, _, height, width = clip.shape
        top, left, box_h, box_w = self._sample_box(height, width, rng)
        size = [self.crop_size, self.crop_size]
        return TF.resized_crop(clip, top, left, box_h, box_w, size, antialias=True)

    def _sample_box(self, height: int, width: int,
                    rng: np.random.Generator) -> Tuple[int, int, int, int]:
        """Sample a crop box (top, left, h, w) from scale and aspect ranges."""
        area = float(height * width)
        for _ in range(10):
            target = area * rng.uniform(self.cfg.scale[0], self.cfg.scale[1])
            log_ratio = np.log(self.cfg.aspect_ratio)
            ratio = float(np.exp(rng.uniform(log_ratio[0], log_ratio[1])))
            box_w = int(round(np.sqrt(target * ratio)))
            box_h = int(round(np.sqrt(target / ratio)))
            if 0 < box_w <= width and 0 < box_h <= height:
                top = int(rng.integers(0, height - box_h + 1))
                left = int(rng.integers(0, width - box_w + 1))
                return top, left, box_h, box_w
        side = min(height, width)
        return (height - side) // 2, (width - side) // 2, side, side

    def _resize_center_crop(self, clip: torch.Tensor) -> torch.Tensor:
        """Resize the shorter side to the crop size then center-crop a square."""
        clip = TF.resize(clip, [self.crop_size], antialias=True)
        return TF.center_crop(clip, [self.crop_size, self.crop_size])


class PhotometricAugment:
    """Random flip, color jitter, grayscale and blur applied to a whole clip."""

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def __call__(self, clip: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Apply the enabled augmentations in a fixed, simple order."""
        if not self.cfg.enabled:
            return clip
        clip = self._maybe_flip(clip, rng)
        clip = self._maybe_jitter(clip, rng)
        clip = self._maybe_grayscale(clip, rng)
        clip = self._maybe_blur(clip, rng)
        return clip

    def _maybe_flip(self, clip: torch.Tensor, rng) -> torch.Tensor:
        if rng.random() < self.cfg.horizontal_flip_prob:
            return TF.horizontal_flip(clip)
        return clip

    def _maybe_jitter(self, clip: torch.Tensor, rng) -> torch.Tensor:
        strength = self.cfg.color_jitter
        if strength <= 0:
            return clip
        clip = TF.adjust_brightness(clip, self._factor(strength, rng))
        clip = TF.adjust_contrast(clip, self._factor(strength, rng))
        clip = TF.adjust_saturation(clip, self._factor(strength, rng))
        return clip.clamp_(0.0, 1.0)

    @staticmethod
    def _factor(strength: float, rng) -> float:
        """Return one jitter factor centered on 1.0."""
        return float(1.0 + rng.uniform(-strength, strength))

    def _maybe_grayscale(self, clip: torch.Tensor, rng) -> torch.Tensor:
        if rng.random() < self.cfg.grayscale_prob:
            return TF.rgb_to_grayscale(clip, num_output_channels=3)
        return clip

    def _maybe_blur(self, clip: torch.Tensor, rng) -> torch.Tensor:
        if rng.random() < self.cfg.gaussian_blur_prob:
            sigma = float(rng.uniform(0.1, 2.0))
            return TF.gaussian_blur(clip, kernel_size=[5, 5], sigma=[sigma, sigma])
        return clip


class Normalizer:
    """Standardize a clip with per-channel mean and standard deviation."""

    def __init__(self, mean: List[float], std: List[float], enabled: bool = True):
        self.enabled = enabled
        self.mean = mean
        self.std = std

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Normalize a ``(T, 3, H, W)`` clip in place-safe fashion."""
        if not self.enabled:
            return clip
        return TF.normalize(clip, mean=self.mean, std=self.std)


class ClipPipeline:
    """Full preprocessing: geometry, augmentation, normalization, layout swap."""

    def __init__(self, transform_cfg: TransformConfig, augment_cfg: AugmentConfig,
                 crop_size: int):
        self.geometry = GeometricTransform(transform_cfg, crop_size)
        self.augment = PhotometricAugment(augment_cfg)
        self.normalize = Normalizer(
            transform_cfg.mean, transform_cfg.std, transform_cfg.normalize
        )

    def __call__(self, clip_thwc: np.ndarray, train: bool,
                 rng: np.random.Generator,
                 apply_normalize: bool = True) -> torch.Tensor:
        """Return a ``(3, T, H, W)`` tensor ready for the encoder.

        :param apply_normalize: set to False to keep values in ``[0, 1]`` (used
            by the HDF5 builder, which stores uint8 and normalizes at read time).
        """
        clip = clip_to_tensor(clip_thwc)
        clip = self.geometry(clip, train, rng)
        if train:
            clip = self.augment(clip, rng)
        if apply_normalize:
            clip = self.normalize(clip)
        return clip.permute(1, 0, 2, 3).contiguous()
