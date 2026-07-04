# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modality-specific patch embeddings for the V-JEPA 2.1 multi-modal tokenizer.

import torch.nn as nn

__all__ = ["PatchEmbed", "PatchEmbed3D"]


class PatchEmbed(nn.Module):
    """2D image-to-patch embedding (a single 2D convolution).

    Used by the multi-modal tokenizer to process images in their native form,
    avoiding the temporal duplication used by earlier V-JEPA variants.
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H*W, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """3D video-to-patch embedding (a single 3D convolution over tubelets)."""

    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        # x: (B, C, T, H, W) -> (B, T'*H'*W', embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
