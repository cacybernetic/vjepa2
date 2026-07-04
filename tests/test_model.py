"""Tests for the assembled V-JEPA 2.1 model and checkpoint conformity."""

import torch

from vjepa2 import build_vjepa2_1_vitb
from vjepa2.model import VJEPA21, init_video_model


def small_model():
    return build_vjepa2_1_vitb(use_sdpa=False, crop_size=64, max_num_frames=4).eval()


def test_build_returns_vjepa21():
    m = small_model()
    assert isinstance(m, VJEPA21)
    assert m.embed_dim == 768


def test_target_encoder_frozen():
    m = small_model()
    assert all(not p.requires_grad for p in m.target_encoder.parameters())
    assert any(p.requires_grad for p in m.encoder.parameters())


def test_init_video_model_component_dims():
    enc, pred = init_video_model(
        crop_size=64, max_num_frames=4, use_sdpa=False, teacher_embed_dim=1664
    )
    assert enc.backbone.embed_dim == 768
    assert pred.backbone.predictor_proj.out_features == 1664
    assert len(pred.backbone.mask_tokens) == 8
    assert len(pred.backbone.predictor_blocks) == 12


def test_feature_extraction_video_and_image():
    m = small_model()
    vid = torch.randn(1, 3, 4, 64, 64)
    img = torch.randn(1, 3, 1, 64, 64)
    assert m.extract_features(vid).shape == (1, 2 * 4 * 4, 768)
    assert m.extract_features(img).shape == (1, 1 * 4 * 4, 768)


def test_full_jepa_forward_shapes():
    m = small_model()
    N = 2 * 4 * 4
    perm = torch.randperm(N)
    ctx = perm[:20].sort().values.unsqueeze(0)
    tgt = perm[20:].sort().values.unsqueeze(0)
    z_pred, z_ctx, h = m(
        [torch.randn(1, 3, 4, 64, 64)], [[ctx]], [[tgt]], mod="video"
    )
    assert z_pred[0][0].shape == (1, tgt.shape[1], 1664)
    assert z_ctx[0][0].shape == (1, ctx.shape[1], 1664)
    assert h[0].shape == (1, N, 768)


def test_dense_loss_end_to_end():
    from vjepa2.modules.losses import compute_mask_distance, jepa_loss

    m = small_model()
    N = 2 * 4 * 4
    perm = torch.randperm(N)
    ctx = perm[:20].sort().values.unsqueeze(0)
    tgt = perm[20:].sort().values.unsqueeze(0)
    masks_enc, masks_pred = [[ctx]], [[tgt]]
    z_pred, z_ctx, h = m(
        [torch.randn(1, 3, 4, 64, 64)], masks_enc, masks_pred, mod="video"
    )
    # predictor targets live in the 1664-d teacher space; use a matching target
    h_tgt = [torch.randn(1, N, 1664)]
    loss_pred = jepa_loss(z_pred, h_tgt, masks_pred)
    d = compute_mask_distance(masks_pred, masks_enc, grid_size=4)
    loss_ctx = jepa_loss(z_ctx, h_tgt, masks_enc, d_weights=d)
    total = loss_pred + 0.5 * loss_ctx
    assert torch.isfinite(total)


# --------------------------------------------------------------------------- #
# Checkpoint conformity (skipped if the weights file is absent)
# --------------------------------------------------------------------------- #
def test_checkpoint_loads_exactly(weights_path):
    m = build_vjepa2_1_vitb()
    msgs = m.load_pretrained(weights_path, strict=False)
    for name, msg in msgs.items():
        assert list(msg.missing_keys) == [], f"{name} missing {msg.missing_keys}"
        assert list(msg.unexpected_keys) == [], f"{name} unexpected {msg.unexpected_keys}"


def test_checkpoint_structural_dims(weights_path):
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    enc = ck["encoder"]
    pred = ck["predictor"]
    # multi-modal tokenizer: distinct 3D (video) and image patch embeds
    assert enc["module.backbone.patch_embed.proj.weight"].shape == (768, 3, 2, 16, 16)
    assert enc["module.backbone.patch_embed_img.proj.weight"].shape == (768, 3, 1, 16, 16)
    # four hierarchical norm heads
    assert "module.backbone.norms_block.3.weight" in enc
    # eight predictor mask tokens
    assert "module.backbone.mask_tokens.7" in pred
    # predictor projects to the ViT-G teacher embedding (1664)
    assert pred["module.backbone.predictor_proj.weight"].shape == (1664, 384)
    assert pred["module.backbone.predictor_proj_context.weight"].shape == (1664, 384)


def test_loaded_model_runs_forward(weights_path):
    m = build_vjepa2_1_vitb(use_sdpa=False)
    m.load_pretrained(weights_path)
    m.eval()
    vid = torch.randn(1, 3, 4, 64, 64)
    with torch.no_grad():
        feat = m.extract_features(vid)
    assert feat.shape[-1] == 768
    assert torch.isfinite(feat).all()
