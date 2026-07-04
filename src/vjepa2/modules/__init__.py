# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Building blocks of the V-JEPA 2.1 model.

from vjepa2.modules import vision_transformer as video_vit
from vjepa2.modules.attention import (
    Attention,
    CrossAttention,
    RoPEAttention,
    rotate_queries_or_keys,
)
from vjepa2.modules.blocks import Block, CrossAttentionBlock
from vjepa2.modules.losses import (
    Lambda_LinearWarmupHold,
    compute_mask_distance,
    jepa_loss,
    separate_positions,
)
from vjepa2.modules.mlp import MLP, DropPath, SwiGLUFFN, drop_path
from vjepa2.modules.patch_embed import PatchEmbed, PatchEmbed3D
from vjepa2.modules.pos_embs import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from vjepa2.modules.predictor import VisionTransformerPredictor, vit_predictor
from vjepa2.modules.tensors import apply_masks, repeat_interleave_batch, trunc_normal_
from vjepa2.modules.vision_transformer import VIT_EMBED_DIMS, VisionTransformer
from vjepa2.modules.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

__all__ = [
    "video_vit",
    "Attention",
    "RoPEAttention",
    "CrossAttention",
    "rotate_queries_or_keys",
    "Block",
    "CrossAttentionBlock",
    "MLP",
    "SwiGLUFFN",
    "DropPath",
    "drop_path",
    "PatchEmbed",
    "PatchEmbed3D",
    "get_1d_sincos_pos_embed",
    "get_2d_sincos_pos_embed",
    "get_3d_sincos_pos_embed",
    "VisionTransformer",
    "VIT_EMBED_DIMS",
    "VisionTransformerPredictor",
    "vit_predictor",
    "MultiSeqWrapper",
    "PredictorMultiSeqWrapper",
    "apply_masks",
    "repeat_interleave_batch",
    "trunc_normal_",
    "jepa_loss",
    "compute_mask_distance",
    "separate_positions",
    "Lambda_LinearWarmupHold",
]
