# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Multi-sequence wrappers that let the encoder / predictor process a list of
# inputs with different frame-per-clip (temporal) lengths and multiple masks.

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiSeqWrapper", "PredictorMultiSeqWrapper"]


class MultiSeqWrapper(nn.Module):
    """Run the encoder over a list of clips (possibly with multiple masks each)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    def forward(self, x, masks=None, gram_mode=False, training_mode=False):
        """
        :param x: list of tensors of different sequence lengths.
        :param masks: list (per input) of lists of masks.
        """
        if masks is None:
            outputs = []
            for x_fpc in x:
                if gram_mode:
                    outputs.append(self._gram_forward(x_fpc))
                else:
                    outputs.append(self.backbone(x_fpc, training=training_mode))
            return outputs

        outs = [[] for _ in x]
        for i, (x_fpc, m_fpc) in enumerate(zip(x, masks)):
            for m in m_fpc:
                outs[i] += [self.backbone(x_fpc, masks=m, training=training_mode)]
        return outs

    def _gram_forward(self, x_fpc):
        # Upsample input 2x, run backbone, downsample tokens back to native
        # resolution. Used for high-resolution Gram-anchoring features.
        B, C, T, H, W = x_fpc.shape
        x_2d = x_fpc.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_up = F.interpolate(x_2d, scale_factor=2, mode="bicubic", align_corners=False)
        _, _, H_up, W_up = x_up.shape
        x_up = x_up.view(B, T, C, H_up, W_up).permute(0, 2, 1, 3, 4)

        out = self.backbone(x_up)
        B, N, D = out.shape
        patch_size = self.backbone.patch_size
        tubelet_size = self.backbone.tubelet_size
        H_up_patches = W_up_patches = int(H_up // patch_size)
        T_up_patches = 1 if T == 1 else int(T // tubelet_size)
        out_3d = out.view(B, T_up_patches, H_up_patches, W_up_patches, D)
        out_3d = out_3d.permute(0, 4, 1, 2, 3)

        out_2d = out_3d.permute(0, 2, 1, 3, 4).reshape(
            B * T_up_patches, D, H_up_patches, W_up_patches
        )
        out = F.interpolate(
            out_2d,
            size=(int(H_up_patches // 2), int(W_up_patches // 2)),
            mode="bicubic",
            align_corners=False,
        )
        out = out.view(
            B, T_up_patches, D, int(H_up_patches // 2), int(W_up_patches // 2)
        ).permute(0, 2, 1, 3, 4)
        out = out.permute(0, 2, 3, 4, 1).reshape(
            B, T_up_patches * int(H_up_patches // 2) * int(W_up_patches // 2), D
        )
        return out


class PredictorMultiSeqWrapper(nn.Module):
    """Run the predictor over the encoder outputs / masks of each input."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, mod="video"):
        """
        :param x: list of encoder outputs for different sequence lengths.
        :param masks_x: list of encoder (context) masks.
        :param masks_y: list of predictor (target) masks.
        """
        outs_pred = [[] for _ in x]
        outs_context = [[] for _ in x]
        for i, (x_fpc, mx_fpc, my_fpc) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mx, my in zip(x_fpc, mx_fpc, my_fpc):
                x_pred, x_context = self.backbone(xij, mx, my, mask_index=i, mod=mod)
                outs_pred[i] += [x_pred]
                outs_context[i] += [x_context]
        return outs_pred, outs_context
