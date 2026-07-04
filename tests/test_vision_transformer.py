"""Tests for the V-JEPA 2.1 encoder (VisionTransformer)."""

import pytest
import torch

from vjepa2.modules import vision_transformer as vit
from vjepa2.modules.vision_transformer import VIT_EMBED_DIMS, VisionTransformer


def make_encoder(**kw):
    cfg = dict(
        img_size=64,
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        use_rope=True,
        use_sdpa=False,
        uniform_power=True,
        interpolate_rope=True,
        modality_embedding=True,
        img_temporal_dim_size=1,
        n_output_distillation=1,
    )
    cfg.update(kw)
    return vit.vit_base(**cfg).eval()


def test_factory_dims_match_paper():
    # Embedding widths of the standard ViT variants.
    # Only depths present in HIERARCHICAL_LAYERS (12/24/40/48) are constructible;
    # ViT-H (depth 32) is intentionally unsupported by the reference recipe.
    assert vit.vit_base().embed_dim == 768
    assert vit.vit_large().embed_dim == 1024
    assert vit.vit_giant().embed_dim == 1408
    assert vit.vit_gigantic().embed_dim == 1664
    assert VIT_EMBED_DIMS["vit_gigantic"] == 1664
    assert VIT_EMBED_DIMS["vit_huge"] == 1280


def test_depths_and_heads():
    assert vit.vit_base().get_num_layers() == 12
    assert vit.vit_large().get_num_layers() == 24
    assert vit.vit_giant().get_num_layers() == 40
    assert vit.vit_base().num_heads == 12
    assert vit.vit_large().num_heads == 16


def test_hierarchical_norms_always_four():
    # Deep self-supervision uses four intermediate levels -> four norm heads.
    enc = make_encoder(n_output_distillation=4)
    assert len(enc.norms_block) == 4
    assert enc.hierarchical_layers == [2, 5, 8, 11]
    assert enc.out_layers_distillation == [2, 5, 8, 11]


def test_single_level_distillation():
    enc = make_encoder(n_output_distillation=1)
    assert len(enc.norms_block) == 4  # structural, independent of distillation
    assert enc.out_layers_distillation == [11]


def test_forward_video_inference_shape():
    enc = make_encoder()
    x = torch.randn(1, 3, 4, 64, 64)
    out = enc(x, training=False)
    assert out.shape == (1, 2 * 4 * 4, 768)


def test_forward_image_pathway_shape():
    enc = make_encoder()
    x = torch.randn(1, 3, 1, 64, 64)  # T == img_temporal_dim_size -> image conv
    out = enc(x, training=False)
    assert out.shape == (1, 1 * 4 * 4, 768)


def test_training_output_concats_levels():
    enc = make_encoder(n_output_distillation=4)
    x = torch.randn(1, 3, 4, 64, 64)
    out = enc(x, training=True)
    assert out.shape == (1, 2 * 4 * 4, 768 * 4)


def test_masking_keeps_only_context_tokens():
    enc = make_encoder()
    x = torch.randn(1, 3, 4, 64, 64)
    keep = torch.arange(10).unsqueeze(0)
    out = enc(x, masks=[keep], training=False)
    assert out.shape == (1, 10, 768)


def test_image_vs_video_use_different_conv():
    enc = make_encoder()
    assert enc.patch_embed_img is not None
    # 3D video conv has a temporal kernel of tubelet_size, image conv of 1
    assert enc.patch_embed.proj.kernel_size[0] == 2
    assert enc.patch_embed_img.proj.kernel_size[0] == 1


def test_non_rope_has_pos_embed():
    enc = VisionTransformer(
        img_size=32,
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        use_rope=False,
        use_sdpa=False,
        n_output_distillation=1,
    ).eval()
    assert enc.pos_embed is not None
    assert enc.pos_embed.shape == (1, 2 * 2 * 2, 768)
    out = enc(torch.randn(1, 3, 4, 32, 32), training=False)
    assert out.shape == (1, 8, 768)


def test_interpolate_rope_handles_larger_resolution():
    # A model configured for 64px should still run on 96px inputs thanks to
    # RoPE interpolation to the pretrained grid.
    enc = make_encoder()
    out = enc(torch.randn(1, 3, 4, 96, 96), training=False)
    assert out.shape == (1, 2 * 6 * 6, 768)


def test_unsupported_depth_raises():
    with pytest.raises(ValueError):
        VisionTransformer(depth=7, num_frames=4, use_sdpa=False)
