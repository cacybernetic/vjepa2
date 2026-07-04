# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Fixed sin-cos positional embeddings. V-JEPA 2.1 uses 3D-RoPE by default, but
# these helpers are kept for the non-RoPE code paths and for completeness.

import numpy as np

__all__ = [
    "get_1d_sincos_pos_embed",
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_3d_sincos_pos_embed",
]


def get_3d_sincos_pos_embed(
    embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False
):
    """
    3D (depth, height, width) sin-cos positional embedding.

    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    # order of meshgrid is very important for indexing as [d, h, w]
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


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """2D (height, width) sin-cos positional embedding."""
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    # order of meshgrid is very important for indexing as [h, w]
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """1D sin-cos positional embedding."""
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded, of size (M,)
    returns: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
