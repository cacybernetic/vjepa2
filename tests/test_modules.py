"""Unit tests for the low-level V-JEPA 2.1 building blocks."""

import numpy as np
import torch
import torch.nn as nn

from vjepa2.modules.attention import Attention, RoPEAttention, rotate_queries_or_keys
from vjepa2.modules.blocks import Block
from vjepa2.modules.mlp import MLP, SwiGLUFFN, drop_path
from vjepa2.modules.patch_embed import PatchEmbed, PatchEmbed3D
from vjepa2.modules.pos_embs import (
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from vjepa2.modules.tensors import apply_masks, repeat_interleave_batch, trunc_normal_


# --------------------------------------------------------------------------- #
# Patch embeddings (multi-modal tokenizer)
# --------------------------------------------------------------------------- #
def test_patch_embed_2d_shapes():
    pe = PatchEmbed(patch_size=16, in_chans=3, embed_dim=768)
    x = torch.randn(2, 3, 64, 48)
    out = pe(x)
    assert out.shape == (2, (64 // 16) * (48 // 16), 768)


def test_patch_embed_3d_shapes():
    pe = PatchEmbed3D(patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768)
    x = torch.randn(2, 3, 4, 64, 64)
    out = pe(x)
    assert out.shape == (2, (4 // 2) * (64 // 16) * (64 // 16), 768)


def test_patch_embed_3d_tubelet1_for_images():
    # image pathway uses tubelet_size == 1 (no temporal aggregation)
    pe = PatchEmbed3D(patch_size=16, tubelet_size=1, embed_dim=32)
    x = torch.randn(1, 3, 1, 32, 32)
    out = pe(x)
    assert out.shape == (1, 4, 32)


# --------------------------------------------------------------------------- #
# Positional embeddings
# --------------------------------------------------------------------------- #
def test_3d_sincos_shape_and_uniform_power():
    emb = get_3d_sincos_pos_embed(768, grid_size=4, grid_depth=2)
    assert emb.shape == (2 * 4 * 4, 768)
    emb_u = get_3d_sincos_pos_embed(768, 4, 2, uniform_power=True)
    assert emb_u.shape == (2 * 4 * 4, 768)


def test_2d_sincos_shape_and_range():
    emb = get_2d_sincos_pos_embed(64, grid_size=8)
    assert emb.shape == (64, 64)
    assert np.all(np.abs(emb) <= 1.0 + 1e-6)


# --------------------------------------------------------------------------- #
# MLP / SwiGLU / drop_path
# --------------------------------------------------------------------------- #
def test_mlp_shape():
    mlp = MLP(32, hidden_features=128, out_features=16)
    assert mlp(torch.randn(2, 5, 32)).shape == (2, 5, 16)


def test_swiglu_alignment_and_shape():
    ffn = SwiGLUFFN(48, hidden_features=192, wide_silu=True)
    # wide_silu hidden = round_up(2*192/3=128, 8) = 128
    assert ffn.fc1.out_features % 8 == 0
    assert ffn(torch.randn(2, 3, 48)).shape == (2, 3, 48)


def test_drop_path_identity_in_eval():
    x = torch.randn(4, 10)
    out = drop_path(x, drop_prob=0.5, training=False)
    assert torch.equal(out, x)


# --------------------------------------------------------------------------- #
# Tensor / masking helpers
# --------------------------------------------------------------------------- #
def test_trunc_normal_bounds():
    t = torch.empty(5000)
    trunc_normal_(t, std=0.02, a=-2, b=2)
    assert t.max() <= 2 and t.min() >= -2


def test_apply_masks_selects_indices():
    x = torch.arange(2 * 6 * 3).float().reshape(2, 6, 3)
    idx = torch.tensor([[0, 2, 4], [1, 3, 5]])
    out = apply_masks(x, [idx])
    assert out.shape == (2, 3, 3)
    assert torch.equal(out[0, 1], x[0, 2])
    assert torch.equal(out[1, 2], x[1, 5])


def test_repeat_interleave_batch():
    x = torch.arange(4).reshape(4, 1).float()  # B=2, N=2 groups
    out = repeat_interleave_batch(x, B=2, repeat=2)
    assert out.shape == (8, 1)


# --------------------------------------------------------------------------- #
# RoPE properties
# --------------------------------------------------------------------------- #
def test_rope_preserves_norm():
    x = torch.randn(1, 2, 5, 8)
    pos = torch.arange(5).float().view(1, 1, 5).repeat(1, 2, 1)
    out = rotate_queries_or_keys(x, pos)
    # a rotation preserves the L2 norm of each token vector
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


def test_rope_relative_inner_product():
    # RoPE encodes relative position: <R(q,m), R(k,n)> depends only on (m-n).
    torch.manual_seed(1)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def rotated_dot(m, n):
        qm = rotate_queries_or_keys(q, torch.tensor([[[float(m)]]]))
        kn = rotate_queries_or_keys(k, torch.tensor([[[float(n)]]]))
        return (qm * kn).sum().item()

    assert abs(rotated_dot(3, 1) - rotated_dot(5, 3)) < 1e-4
    assert abs(rotated_dot(7, 2) - rotated_dot(10, 5)) < 1e-4


def test_rope_registers_passthrough():
    x = torch.randn(1, 1, 6, 8)
    pos = torch.arange(4).float().view(1, 1, 4)  # 6 - 2 registers = 4 rotated
    out = rotate_queries_or_keys(x, pos, n_registers=2)
    # last two register tokens are left untouched
    assert torch.equal(out[..., -2:, :], x[..., -2:, :])


# --------------------------------------------------------------------------- #
# Attention & Block
# --------------------------------------------------------------------------- #
def test_standard_attention_shape():
    attn = Attention(32, num_heads=4, use_sdpa=False)
    x = torch.randn(2, 7, 32)
    assert attn(x).shape == x.shape


def test_rope_attention_shape():
    attn = RoPEAttention(32, num_heads=4, use_sdpa=False, grid_size=4)
    x = torch.randn(1, 2 * 4 * 4, 32)  # depth 2, 4x4 grid
    out, _ = attn(x)
    assert out.shape == x.shape


def test_block_residual_shape_rope_and_plain():
    for use_rope in (True, False):
        blk = Block(
            dim=32,
            num_heads=4,
            use_rope=use_rope,
            use_sdpa=False,
            grid_size=4,
            norm_layer=nn.LayerNorm,
        )
        x = torch.randn(1, 2 * 4 * 4, 32)
        out, _ = blk(x)
        assert out.shape == x.shape
