"""Tests for the V-JEPA 2.1 multi-level predictor."""

import torch

from vjepa2.modules.predictor import VisionTransformerPredictor, vit_predictor


def make_predictor(**kw):
    cfg = dict(
        img_size=64,
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        num_heads=6,
        use_mask_tokens=True,
        num_mask_tokens=8,
        use_rope=True,
        use_sdpa=False,
        interpolate_rope=True,
        modality_embedding=True,
        img_temporal_dim_size=1,
        return_all_tokens=True,
        n_output_distillation=1,
        teacher_embed_dim=1664,
    )
    cfg.update(kw)
    return vit_predictor(**cfg).eval()


def test_single_level_predictor_embed_is_linear():
    pred = make_predictor(n_output_distillation=1)
    assert isinstance(pred.predictor_embed, torch.nn.Linear)
    assert pred.predictor_embed.in_features == 768
    assert pred.predictor_embed.out_features == 384


def test_multi_level_predictor_embed_is_mlp():
    pred = make_predictor(n_output_distillation=4)
    assert isinstance(pred.predictor_embed, torch.nn.Sequential)
    # first linear ingests concat of 4 encoder levels
    assert pred.predictor_embed[0].in_features == 768 * 4


def test_teacher_projection_dim():
    pred = make_predictor(n_output_distillation=1, teacher_embed_dim=1664)
    # single level -> out_embed_dim == teacher dim
    assert pred.predictor_proj.out_features == 1664
    assert pred.predictor_proj_context.out_features == 1664


def test_num_mask_tokens():
    pred = make_predictor()
    assert len(pred.mask_tokens) == 8


def test_forward_shapes_masked_and_context():
    pred = make_predictor()
    N = 2 * 4 * 4  # 32 tokens
    perm = torch.randperm(N)
    ctx = perm[:20].sort().values.unsqueeze(0)
    tgt = perm[20:].sort().values.unsqueeze(0)
    x = torch.randn(1, 20, 768)  # encoder context output
    z_pred, z_ctx = pred(x, ctx, tgt, mod="video")
    assert z_pred.shape == (1, 12, 1664)
    assert z_ctx.shape == (1, 20, 1664)


def test_forward_returns_none_context_when_not_return_all():
    pred = make_predictor(return_all_tokens=False)
    N = 2 * 4 * 4
    ctx = torch.arange(20).unsqueeze(0)
    tgt = torch.arange(20, N).unsqueeze(0)
    x = torch.randn(1, 20, 768)
    z_pred, z_ctx = pred(x, ctx, tgt)
    assert z_pred.shape == (1, N - 20, 1664)
    assert z_ctx is None


def test_predictor_num_patches_matches_grid():
    pred = make_predictor()
    assert pred.num_patches == 2 * 4 * 4


def test_default_out_dim_falls_back_to_embed_dim():
    pred = VisionTransformerPredictor(
        img_size=64,
        num_frames=4,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        num_heads=6,
        use_mask_tokens=True,
        use_rope=True,
        use_sdpa=False,
        n_output_distillation=1,
    )
    assert pred.predictor_proj.out_features == 768
