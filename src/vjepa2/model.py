# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Assembly of the full V-JEPA 2.1 model: a multi-modal RoPE ViT encoder and a
# multi-level predictor, wired together following the reference training recipe
# (``app/vjepa_2_1``). Includes helpers to instantiate the distilled ViT-B/G
# checkpoint shipped in ``weights/``.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa2.modules import vision_transformer as video_vit
from vjepa2.modules.predictor import vit_predictor
from vjepa2.modules.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logger = logging.getLogger(__name__)

__all__ = ["VJEPA21", "init_video_model", "build_vjepa2_1_vitb"]


def init_video_model(
    device="cpu",
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=256,
    pred_depth=12,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=True,
    use_mask_tokens=True,
    num_mask_tokens=8,
    zero_init_mask_tokens=True,
    use_sdpa=True,
    use_rope=True,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=True,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=1,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=True,
    modality_embedding=True,
    n_output_distillation_encoder=1,
    n_output_distillation_predictor=1,
    teacher_embed_dim=None,
):
    """Build the ``(encoder, predictor)`` pair used by V-JEPA 2.1.

    Defaults reproduce the distilled ViT-B (teacher ViT-G) configuration, which
    does not use deep self-supervision (single-level output).

    The predictor fuses exactly the levels produced by the encoder, so
    ``n_output_distillation_encoder`` and ``n_output_distillation_predictor``
    must match: use ``1`` for distillation / single-level, ``4`` for the full
    deep-self-supervision recipe.
    """
    if n_output_distillation_encoder != n_output_distillation_predictor:
        raise ValueError(
            "n_output_distillation_encoder and n_output_distillation_predictor "
            f"must match (got {n_output_distillation_encoder} and "
            f"{n_output_distillation_predictor}); the predictor's fusion MLP "
            "ingests exactly the levels concatenated by the encoder."
        )
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        n_output_distillation=n_output_distillation_encoder,
    )
    encoder = MultiSeqWrapper(encoder)

    predictor = vit_predictor(
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=(
            encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads
        ),
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
        n_output_distillation=n_output_distillation_predictor,
        teacher_embed_dim=teacher_embed_dim,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    return encoder, predictor


def _strip_prefix(state_dict, prefix="module."):
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


class VJEPA21(nn.Module):
    """The full V-JEPA 2.1 self-supervised model.

    Holds the online ``encoder`` (context encoder), the ``predictor`` and an EMA
    ``target_encoder``. During training the context encoder + predictor are
    optimized to match the (stop-grad) target-encoder representations of the
    unmasked input, with the dense L1 + weighted-context loss.
    """

    def __init__(self, encoder, predictor, target_encoder=None,
                 distillation_teacher=None):
        super().__init__()
        import copy

        self.encoder = encoder
        self.predictor = predictor
        self.is_distillation = distillation_teacher is not None
        if self.is_distillation:
            # Distillation recipe: the loss target is a *frozen* teacher encoder
            # (e.g. ViT-G, producing ``teacher_embed_dim`` features). We keep an
            # EMA copy of the online student as the final exported model, but it
            # never contributes to the loss.
            self.target_encoder = distillation_teacher
            self.ema_encoder = copy.deepcopy(encoder)
            for p in self.ema_encoder.parameters():
                p.requires_grad = False
        else:
            # Pretraining: the target is an EMA of the online encoder, so it
            # shares the student's embedding dimension.
            if target_encoder is None:
                target_encoder = copy.deepcopy(encoder)
            self.target_encoder = target_encoder
            self.ema_encoder = None
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        # Dimension of *one* level of the target: drives the per-level LayerNorm
        # and must match the predictor's per-level output width.
        self.target_embed_dim = self.target_encoder.embed_dim

    @property
    def embed_dim(self):
        return self.encoder.embed_dim

    def ema_target(self):
        """Return the module the EMA updater should write into.

        In distillation the frozen teacher must not move, so the EMA is applied
        to the student copy instead.
        """
        return self.ema_encoder if self.is_distillation else self.target_encoder

    def forward(self, clips, masks_enc, masks_pred, mod="video", training_mode=True):
        """Run the JEPA forward pass.

        :param clips: list of input tensors (per frames-per-clip group).
        :param masks_enc: nested context masks ``[fpc][mask]``.
        :param masks_pred: nested target masks ``[fpc][mask]``.
        :returns: ``(z_pred, z_context, h_target)`` where ``h_target`` is the
            stop-grad target-encoder representation of the full (unmasked) clips.
        """
        z = self.encoder(clips, masks_enc, training_mode=training_mode)
        z_pred, z_context = self.predictor(z, masks_enc, masks_pred, mod=mod)
        with torch.no_grad():
            h_target = self.target_encoder(clips, training_mode=training_mode)
            # Per-level LayerNorm of the targets before the loss, matching the
            # reference ``forward_target``: each ``target_embed_dim`` chunk of
            # the concatenated multi-level output is normalized independently.
            h_target = [
                self._normalize_target(h, self.target_embed_dim) for h in h_target
            ]
        # Fail loudly (and early) on a predictor/target width mismatch instead of
        # letting the loss raise a cryptic broadcasting error further down.
        pred_dim = z_pred[0][0].shape[-1]
        tgt_dim = h_target[0].shape[-1]
        if pred_dim != tgt_dim:
            raise ValueError(
                "Predictor output width does not match the target encoder: "
                f"predictor produces {pred_dim}-d, target produces {tgt_dim}-d. "
                "For distillation pass a `distillation_teacher` whose embedding "
                "dimension equals the predictor's `teacher_embed_dim`."
            )
        return z_pred, z_context, h_target

    @staticmethod
    def _normalize_target(h, embed_dim):
        """LayerNorm each ``embed_dim``-wide level of a multi-level target."""
        n_levels = h.shape[-1] // embed_dim
        if n_levels <= 1:
            return F.layer_norm(h, (h.shape[-1],))
        chunks = [
            F.layer_norm(c, (embed_dim,)) for c in h.split(embed_dim, dim=-1)
        ]
        return torch.cat(chunks, dim=-1)

    @torch.no_grad()
    def extract_features(self, clips, use_ema=True):
        """Encode full (unmasked) clips into representations for downstream use.

        :param clips: single tensor or list of tensors.
        :param use_ema: use the target (EMA) encoder rather than the online one.
        """
        if use_ema:
            # In distillation the EMA student is the deliverable, not the teacher.
            enc = self.ema_encoder if self.is_distillation else self.target_encoder
        else:
            enc = self.encoder
        single = not isinstance(clips, (list, tuple))
        if single:
            clips = [clips]
        out = enc(clips, training_mode=False)
        return out[0] if single else out

    # -- checkpoint loading -------------------------------------------------
    def load_pretrained(self, checkpoint, strict=False):
        """Load an ``app/vjepa_2_1``-style checkpoint dict (or path)."""
        if isinstance(checkpoint, str):
            checkpoint = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        msgs = {}
        enc_sd = _strip_prefix(checkpoint["encoder"])
        msgs["encoder"] = self.encoder.load_state_dict(enc_sd, strict=strict)
        pred_sd = _strip_prefix(checkpoint["predictor"])
        msgs["predictor"] = self.predictor.load_state_dict(pred_sd, strict=strict)
        tgt_key = "ema_encoder" if "ema_encoder" in checkpoint else "target_encoder"
        if tgt_key in checkpoint:
            tgt_sd = _strip_prefix(checkpoint[tgt_key])
            # The EMA-student weights of a distilled checkpoint belong in
            # ``ema_encoder``; otherwise they are the pretraining EMA target.
            dest = (
                self.ema_encoder
                if (self.is_distillation and tgt_key == "ema_encoder")
                else self.target_encoder
            )
            msgs["target_encoder"] = dest.load_state_dict(tgt_sd, strict=strict)
        return msgs


def build_vjepa2_1_vitb(checkpoint=None, device="cpu", **overrides):
    """Instantiate the distilled V-JEPA 2.1 ViT-B (teacher ViT-G) model.

    Matches ``weights/vjepa2_1_vitb_dist_vitG_384.pt``:
      * encoder: ViT-B, 3D-RoPE, multi-modal tokenizer, 4 hierarchical norms;
      * predictor: 12 blocks, 8 mask tokens, single-level output projecting to
        the ViT-G (1664-d) teacher embedding, ``return_all_tokens``.
    """
    cfg = dict(
        model_name="vit_base",
        patch_size=16,
        tubelet_size=2,
        max_num_frames=16,
        crop_size=256,
        pred_depth=12,
        pred_num_heads=12,
        pred_embed_dim=384,
        num_mask_tokens=8,
        use_rope=True,
        use_sdpa=True,
        uniform_power=True,
        interpolate_rope=True,
        modality_embedding=True,
        img_temporal_dim_size=1,
        use_mask_tokens=True,
        return_all_tokens=True,
        n_output_distillation_encoder=1,
        n_output_distillation_predictor=1,
        teacher_embed_dim=1664,
        device=device,
    )
    cfg.update(overrides)
    encoder, predictor = init_video_model(**cfg)
    model = VJEPA21(encoder, predictor).to(device)
    if checkpoint is not None:
        msgs = model.load_pretrained(checkpoint)
        logger.info("Loaded V-JEPA 2.1 ViT-B checkpoint: %s", msgs)
    return model
