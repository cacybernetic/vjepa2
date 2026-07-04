# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Batch inference: encode image / video files into V-JEPA 2.1 dense features
# (or a pooled embedding) and persist them in pickle / numpy / HDF5 form.
#
#   runs -m weights/vjepa2_1_vitb_dist_vitG_384.pt -i clip.mp4 -o clip.pkl
#   runs -m encoder.onnx -d videos/ --output-dir embeddings/ -f npy
#
# This script is intentionally SELF-CONTAINED: it re-implements the V-JEPA 2.1
# ViT-B encoder (3D-RoPE, multi-modal tokenizer, hierarchical norms) inline so
# that it can run a ``.pt`` checkpoint without importing anything else from the
# ``vjepa2`` package. Copy this single file (plus a checkpoint or an exported
# ``.onnx``) anywhere and it will run.

from typing import Optional, List
from functools import partial
from enum import Enum
from argparse import ArgumentParser, Namespace
import logging
import math
import pickle
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ===========================================================================
# Self-contained V-JEPA 2.1 encoder
# ---------------------------------------------------------------------------
# Everything from here down to ``build_encoder`` is a faithful, encoder-only
# inline copy of ``vjepa2.model`` / ``vjepa2.modules`` (the predictor, losses
# and training helpers are not needed for feature extraction). Keeping it here
# frees this script from any intra-repo import.
# ===========================================================================


# -- tensor / init helpers (from modules/tensors.py) ------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """In-place truncated-normal initialization."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_masks(x, masks, concat=True):
    """Keep only the patch tokens indexed by ``masks`` (list of ``[B, K]``)."""
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


# -- fixed sin-cos positional embeddings (from modules/pos_embs.py) ----------
# Only used on the non-RoPE code path; the distilled ViT-B uses 3D-RoPE, but we
# keep these so the encoder class is a faithful drop-in for any config.
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed(
    embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False
):
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)
    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)
    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# -- feed-forward blocks (from modules/mlp.py) ------------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """Standard two-layer feed-forward network."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (used when ``use_silu`` is enabled)."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        wide_silu=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (
                (swiglu_hidden_features + align_as - 1) // align_as * align_as
            )
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


# -- patch embeddings (from modules/patch_embed.py) -------------------------
class PatchEmbed(nn.Module):
    """2D image-to-patch embedding (a single 2D convolution)."""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


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
        return self.proj(x).flatten(2).transpose(1, 2)


# -- attention (from modules/attention.py) ----------------------------------
def rotate_queries_or_keys(x, pos, n_registers=0, has_cls_first=False):
    """Apply rotary positional embedding along one axis to ``x``."""
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dim must be a multiple of 2 for block rotation"

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)

    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with factorized 3D rotary embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        if patch_size == 14:
            self.pretrained_grid_size = int(252 / patch_size)
        elif patch_size == 16:
            self.pretrained_grid_size = int(256 / patch_size)
        else:
            self.pretrained_grid_size = None

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(
        self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False
    ):
        B, N, C = x.size()
        N_ctx = N - self.n_registers
        grid_depth = int(N_ctx // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(
                    int(grid_depth * self.grid_size * self.grid_size), device=x.device
                )
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        if self.interpolate_rope:
            if H_patches is None:
                H_patches = int(self.grid_size)
            if W_patches is None:
                W_patches = int(self.grid_size)
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)

        s = 0
        qd = rotate_queries_or_keys(
            q[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first
        )
        kd = rotate_queries_or_keys(
            k[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first
        )
        s += self.d_dim
        qh = rotate_queries_or_keys(
            q[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first
        )
        kh = rotate_queries_or_keys(
            k[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first
        )
        s += self.h_dim
        qw = rotate_queries_or_keys(
            q[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first
        )
        kw = rotate_queries_or_keys(
            k[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first
        )
        s += self.w_dim

        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x, None


class Attention(nn.Module):
    """Standard multi-head self-attention (no positional rotation)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -- transformer block (from modules/blocks.py) -----------------------------
class Block(nn.Module):
    """Pre-norm transformer block with optional RoPE attention and SwiGLU MLP."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

    def forward(
        self,
        x,
        mask=None,
        T=None,
        H_patches=None,
        W_patches=None,
        return_attn=False,
        mode="video",
    ):
        if self.use_rope:
            y, attn = self.attn(
                self.norm1(x),
                mask=mask,
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                return_attn=return_attn,
            )
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        return x, None


# -- vision transformer encoder (from modules/vision_transformer.py) --------
# Encoder blocks (0-indexed) whose normalized outputs form the multi-level
# representation, keyed by network depth.
HIERARCHICAL_LAYERS = {
    12: [2, 5, 8, 11],
    24: [5, 11, 17, 23],
    40: [9, 19, 29, 39],
    48: [11, 23, 35, 47],
}


class VisionTransformer(nn.Module):
    """Vision Transformer encoder."""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=False,
        init_type: str = "default",
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # -- multi-modal tokenizer (native 3D conv for video, 2D conv for image)
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size[0] // patch_size)
                * (img_size[1] // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
            )
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        if self.img_temporal_dim_size is not None:
            if not isinstance(self.img_temporal_dim_size, int):
                raise ValueError(
                    "img_temporal_dim_size must be an int, got "
                    f"{self.img_temporal_dim_size}"
                )
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=1,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed_img = None

        self.uniform_power = uniform_power
        self.use_rope = use_rope

        # -- fixed sin-cos positional embedding (only used when RoPE is off)
        if not self.use_rope:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
            )
            self._init_pos_embed(self.pos_embed.data)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        self.attn_out = False
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # -- multi-level (deep) supervision layers
        if depth not in HIERARCHICAL_LAYERS:
            raise ValueError(f"Unsupported depth {depth} for hierarchical layers")
        self.hierarchical_layers = HIERARCHICAL_LAYERS[depth]
        if n_output_distillation == 4:
            self.out_layers_distillation = list(self.hierarchical_layers)
        elif n_output_distillation == 1:
            self.out_layers_distillation = [self.hierarchical_layers[-1]]
        else:
            raise ValueError(
                f"Unsupported n_output_distillation {n_output_distillation}"
            )

        self.norms_block = nn.ModuleList(
            [norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))]
        )

        self.cls_token = None
        self.return_hierarchical = False

        # -- learnable modality embeddings (image vs video pathway)
        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_pos_embed(self, pos_embed):
        grid_h = self.img_height // self.patch_size
        grid_w = self.img_width // self.patch_size
        if self.is_video:
            grid_d = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                self.embed_dim,
                grid_h,
                grid_d,
                cls_token=False,
                uniform_power=self.uniform_power,
            )
        else:
            sincos = get_2d_sincos_pos_embed(self.embed_dim, grid_h, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type in ("xavier_uniform", "xavier_normal"):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                if self.init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError(f"Unknown init type {self.init_type}")

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def check_temporal_dim(self, shape) -> bool:
        if self.img_temporal_dim_size is not None:
            if shape[2] == self.img_temporal_dim_size:
                return True
        return False

    def forward(self, x, masks=None, training=False):
        """
        :param x: input image/video, ``(B, C, H, W)`` or ``(B, C, T, H, W)``.
        :param masks: indices of patch tokens to keep (context tokens).
        :param training: if True, return the concatenated multi-level output.
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            if self.check_temporal_dim(x.shape):
                T = T // 1
            else:
                T = T // self.tubelet_size

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)

        if self.check_temporal_dim(x.shape):
            assert self.patch_embed_img is not None
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x = x + self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x = x + self.video_mod_embed.repeat(x.shape[0], 1, 1)

        if not self.use_rope:
            x = x + pos_embed

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        outs = []
        hier = []
        for i, blk in enumerate(self.blocks):
            x, attn = blk(
                x,
                mask=masks,
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                return_attn=self.attn_out,
                mode=mode,
            )

            if self.out_layers is not None and i in self.out_layers:
                out_idx = self.hierarchical_layers.index(i)
                outs.append(self.norms_block[out_idx](x))

            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if self.out_layers is not None:
            return outs

        if training or self.return_hierarchical:
            hier = torch.cat(hier, dim=2)
            return hier
        else:
            x = self.norms_block[-1](x)
            return x

    def interpolate_pos_encoding(self, x, pos_embed):
        _, N, dim = pos_embed.shape

        if self.is_video:
            _, _, T, H, W = x.shape
            if H == self.img_height and W == self.img_width and T == self.num_frames:
                return pos_embed
            elif H == self.img_height and W == self.img_width and T < self.num_frames:
                new_N = int(
                    (T // self.tubelet_size)
                    * (H // self.patch_size)
                    * (W // self.patch_size)
                )
                return pos_embed[:, :new_N, :]

            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            scale_factor = (T / N_t, H / N_h, W / N_w)
            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed
        else:
            _, _, H, W = x.shape
            if H == self.img_height and W == self.img_width:
                return pos_embed

            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)
            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2
                ),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vit_base(patch_size=16, **kwargs):
    """ViT-B/16 encoder (768-d, depth 12, 12 heads)."""
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


class _EncoderWrapper(nn.Module):
    """Holds the ViT backbone under ``.backbone`` to match checkpoint keys.

    The reference model wraps the encoder in a ``MultiSeqWrapper`` whose only
    child is ``backbone``; saved encoder state dicts are therefore keyed
    ``backbone.*`` (after stripping the DDP ``module.`` prefix).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    @torch.no_grad()
    def extract_features(self, clip):
        """Encode a full (unmasked) clip ``(B, C, T, H, W)`` -> ``(B, N, D)``."""
        return self.backbone(clip, masks=None, training=False)


def _strip_prefix(state_dict, prefix="module."):
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def build_encoder(
    checkpoint: Optional[str] = None,
    weights: str = "ema",
    crop_size: int = 256,
    num_frames: int = 16,
    use_sdpa: bool = True,
    device: str = "cpu",
) -> _EncoderWrapper:
    """Build the distilled V-JEPA 2.1 ViT-B encoder and load its weights.

    Mirrors ``build_vjepa2_1_vitb`` from the package, restricted to the encoder:
    ViT-B, 3D-RoPE, multi-modal tokenizer, single-level (distillation) output.
    """
    backbone = vit_base(
        img_size=crop_size,
        patch_size=16,
        num_frames=num_frames,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=use_sdpa,
        use_silu=False,
        wide_silu=False,
        is_causal=False,
        use_rope=True,
        init_type="default",
        img_temporal_dim_size=1,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=True,
        modality_embedding=True,
        n_output_distillation=1,
    )
    encoder = _EncoderWrapper(backbone)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if weights == "ema":
            key = "ema_encoder" if "ema_encoder" in ckpt else "target_encoder"
        else:
            key = "encoder"
        if key not in ckpt:
            raise KeyError(
                f"checkpoint has no '{key}' state dict (keys: {list(ckpt.keys())})"
            )
        sd = _strip_prefix(ckpt[key])
        msg = encoder.load_state_dict(sd, strict=False)
        logger.info(
            "Loaded '%s' encoder weights: %d missing, %d unexpected keys",
            key,
            len(msg.missing_keys),
            len(msg.unexpected_keys),
        )

    encoder.eval().to(device)
    return encoder


# ===========================================================================
# Input / output pipeline
# ===========================================================================

# ImageNet statistics used to normalize pixels in [0, 1] (V-JEPA 2.1 recipe).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


class Preprocess:
    """Turn an image / video file into a model-ready clip ``(1, C, T, H, W)``.

    Frames are resized so the shorter side equals ``crop_size``, center-cropped
    to a square and normalized with the ImageNet statistics. Videos are sampled
    to exactly ``num_frames`` uniformly-spaced frames; images become a single
    frame (``T == 1``), which selects the encoder's image pathway.
    """

    def __init__(self, crop_size: int = 256, num_frames: int = 16):
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)

    def _load_image(self, path: str) -> torch.Tensor:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            frame = torch.from_numpy(np.array(im)).permute(2, 0, 1)  # C H W
        return frame.unsqueeze(1).float()  # C T=1 H W

    def _load_video(self, path: str) -> torch.Tensor:
        from torchvision.io import read_video

        vframes, _, _ = read_video(path, output_format="TCHW", pts_unit="sec")
        if vframes.numel() == 0 or vframes.shape[0] == 0:
            raise ValueError(f"no frames decoded from {path}")
        idx = np.linspace(0, vframes.shape[0] - 1, self.num_frames)
        idx = np.round(idx).astype(int)
        frames = vframes[idx]  # T C H W
        return frames.permute(1, 0, 2, 3).float()  # C T H W

    def _resize_crop(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (C, T, H, W) -> (C, T, crop, crop) via shorter-side resize + center crop
        clip = clip.permute(1, 0, 2, 3)  # T C H W (torchvision ops act on last 2 dims)
        clip = TF.resize(clip, self.crop_size, antialias=True)
        clip = TF.center_crop(clip, [self.crop_size, self.crop_size])
        return clip.permute(1, 0, 2, 3)  # C T H W

    def __call__(self, path: str) -> torch.Tensor:
        if is_image_file(path):
            clip = self._load_image(path)
        elif is_video_file(path):
            clip = self._load_video(path)
        else:
            raise ValueError(f"unsupported input type: {path}")

        clip = self._resize_crop(clip)
        clip = clip.div(255.0).sub(self.mean).div(self.std)
        return clip.unsqueeze(0)  # (1, C, T, H, W)


class Postprocess:
    """Reduce encoder features ``(1, N, D)`` to the saved embedding array.

    ``pooling='mean'`` (default) averages the patch tokens into a single
    ``(D,)`` clip/image embedding; ``pooling='none'`` keeps the dense
    ``(N, D)`` token features.
    """

    def __init__(self, pooling: str = "mean"):
        self.pooling = pooling

    def __call__(self, feats: torch.Tensor) -> np.ndarray:
        feats = feats.squeeze(0)  # (N, D)
        if self.pooling == "mean":
            feats = feats.mean(dim=0)  # (D,)
        elif self.pooling != "none":
            raise ValueError(f"unknown pooling: {self.pooling}")
        return feats.cpu().numpy()


class Model:
    """Feature extractor backed by a PyTorch ``.pt`` or an ONNX ``.onnx`` file."""

    def __init__(self, backend: str, runner, device: str = "cpu"):
        self.backend = backend
        self.runner = runner
        self.device = device

    @classmethod
    def load(
        cls,
        model_filepath: str,
        device: str = "cpu",
        weights: str = "ema",
        crop_size: int = 256,
        num_frames: int = 16,
    ) -> "Model":
        ext = os.path.splitext(model_filepath)[1].lower()
        if ext == ".onnx":
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if device.startswith("cuda"):
                providers = ["CUDAExecutionProvider"] + providers
            sess = ort.InferenceSession(model_filepath, providers=providers)
            logger.info("Loaded ONNX encoder: %s", model_filepath)
            return cls("onnx", sess, device)

        encoder = build_encoder(
            checkpoint=model_filepath,
            weights=weights,
            crop_size=crop_size,
            num_frames=num_frames,
            use_sdpa=True,
            device=device,
        )
        logger.info(
            "Loaded PyTorch checkpoint: %s ('%s' encoder)", model_filepath, weights
        )
        return cls("torch", encoder, device)

    @torch.no_grad()
    def embed(self, clip: torch.Tensor) -> torch.Tensor:
        if self.backend == "onnx":
            (out,) = self.runner.run(None, {"clip": clip.cpu().numpy()})
            return torch.from_numpy(out)
        clip = clip.to(self.device)
        return self.runner.extract_features(clip)


class OutputFormat(Enum):
    PICKLE = 'pkle'
    NUMPY = 'npy'
    HDF5 = 'h5'


# File extension written for each output format.
FORMAT_EXTS = {
    OutputFormat.PICKLE: ".pkl",
    OutputFormat.NUMPY: ".npy",
    OutputFormat.HDF5: ".h5",
}


def get_output_format(format_name: str) -> OutputFormat:
    if format_name == OutputFormat.PICKLE.value:
        return OutputFormat.PICKLE
    elif format_name == OutputFormat.NUMPY.value:
        return OutputFormat.NUMPY
    elif format_name == OutputFormat.HDF5.value:
        return OutputFormat.HDF5
    else:
        raise ValueError("Unsupported output format: {}".format(format_name))


def save_embedding(embedding: np.ndarray, path: str, fmt: OutputFormat) -> None:
    if fmt is OutputFormat.PICKLE:
        with open(path, "wb") as f:
            pickle.dump(embedding, f)
    elif fmt is OutputFormat.NUMPY:
        np.save(path, embedding)
    elif fmt is OutputFormat.HDF5:
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("embedding", data=embedding)


class FileFilter:
    """Collect the image / video files inside a directory."""

    FILE_TYPES = sorted(IMAGE_EXTS | VIDEO_EXTS)

    def __init__(self, recursive: bool = True):
        self.recursive = recursive

    def filter(self, dir_path: str) -> List[str]:
        found: List[str] = []
        if self.recursive:
            for root, _, files in os.walk(dir_path):
                for name in files:
                    if os.path.splitext(name)[1].lower() in self.FILE_TYPES:
                        found.append(os.path.join(root, name))
        else:
            for name in os.listdir(dir_path):
                p = os.path.join(dir_path, name)
                if os.path.isfile(p) and \
                        os.path.splitext(name)[1].lower() in self.FILE_TYPES:
                    found.append(p)
        return sorted(found)


class App:

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.preprocess = Preprocess(
            crop_size=args.crop_size, num_frames=args.num_frames
        )
        self.model: Optional[Model] = None
        self.postprocess = Postprocess(pooling=args.pooling)
        self.file_filter = FileFilter(recursive=args.recursive)

    def init(self) -> None:
        self.model = Model.load(
            self.args.model,
            device=self.args.device,
            weights=self.args.weights,
            crop_size=self.args.crop_size,
            num_frames=self.args.num_frames,
        )

    def _output_path(self, input_path: str, fmt: OutputFormat) -> str:
        """Resolve where the embedding for ``input_path`` should be written."""
        ext = FORMAT_EXTS[fmt]
        if self.args.output_dir is not None:
            stem = os.path.splitext(os.path.basename(input_path))[0]
            return os.path.join(self.args.output_dir, stem + ext)
        if self.args.output_file is not None:
            return self.args.output_file
        # Fall back to a sibling of the input file.
        return os.path.splitext(input_path)[0] + ext

    def run(self) -> int:
        """Running main program and return an integer"""
        input_file: str = self.args.input_file  # file path to video or image.
        input_dir: str = self.args.input_dir    # dir path containing image file and or video files.
        output_dir: str = self.args.output_dir
        #: dir path that will be contain the list of embedding files.
        output_format: OutputFormat = get_output_format(self.args.output_format)
        #: output format in what the embedding computed will be saved.

        # When the input file is provided we take it into account.
        # When a directory path is provided we take into account too.
        # If a directory path is provided at input, we create also the output directory.
        # Whataver the input provided (one file or directory of files) we take all files
        # into account.
        listoffiles: List[str] = []
        if input_file is not None:
            if os.path.isfile(input_file):
                listoffiles.append(input_file)
            else:
                logger.error("input file not found: %s", input_file)
                return 1
        if input_dir is not None:
            if not os.path.isdir(input_dir):
                logger.error("input directory not found: %s", input_dir)
                return 1
            listoffiles.extend(self.file_filter.filter(input_dir))
            if output_dir is not None and not os.path.isdir(output_dir):
                os.makedirs(output_dir)

        if not listoffiles:
            logger.error("no input files to process")
            return 1

        # We run the inference of the model and display the progress bar on the terminal.
        # We compute the embeddings and save them into the output format specified in CLI argument.
        # By default, the output format is `pkle`.
        errors = 0
        for path in tqdm(listoffiles, desc="Encoding", unit="file"):
            try:
                clip = self.preprocess(path)
                feats = self.model.embed(clip)
                embedding = self.postprocess(feats)
                out_path = self._output_path(path, output_format)
                save_embedding(embedding, out_path, output_format)
            except Exception as exc:  # noqa: BLE001 - report and continue
                errors += 1
                logger.error("failed on %s: %s", path, exc)

        logger.info(
            "Done: %d/%d files encoded", len(listoffiles) - errors, len(listoffiles)
        )
        return 1 if errors == len(listoffiles) else 0


def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog="runs",
        description="Encode images / videos into V-JEPA 2.1 embeddings.",
    )
    p.add_argument(
        "-m", "--model", required=True,
        help="path to the encoder: a .pt checkpoint or an exported .onnx",
    )
    p.add_argument("-i", "--input-file", default=None, help="single image/video file")
    p.add_argument("-d", "--input-dir", default=None,
                   help="directory of image/video files")
    p.add_argument("-o", "--output-file", default=None,
                   help="output file for single-input mode")
    p.add_argument("--output-dir", default=None,
                   help="output directory (one embedding file per input)")
    p.add_argument(
        "-f", "--output-format", default=OutputFormat.PICKLE.value,
        choices=[f.value for f in OutputFormat],
        help="embedding serialization format (default: pkle)",
    )
    p.add_argument("--pooling", default="mean", choices=["mean", "none"],
                   help="'mean' pools tokens to a vector, 'none' keeps dense features")
    p.add_argument("--weights", default="ema", choices=["ema", "online"],
                   help="which encoder to use from a .pt checkpoint")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--recursive", action="store_true", default=True,
                   help="recurse into subdirectories (default)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.input_file is None and args.input_dir is None:
        parser.error("provide at least one of --input-file / --input-dir")

    app = App(args)
    app.init()
    code = app.run()
    sys.exit(code)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Canceled by user!")
        sys.exit(125)
