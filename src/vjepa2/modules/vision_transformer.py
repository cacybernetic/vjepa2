# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# V-JEPA 2.1 Vision Transformer encoder with a multi-modal tokenizer, 3D-RoPE,
# learnable modality embeddings and multi-level (deep) supervision outputs.

import math
from functools import partial

import torch
import torch.nn as nn

from vjepa2.modules.blocks import Block
from vjepa2.modules.patch_embed import PatchEmbed, PatchEmbed3D
from vjepa2.modules.pos_embs import (
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from vjepa2.modules.tensors import apply_masks, trunc_normal_

__all__ = [
    "VisionTransformer",
    "vit_synthetic",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_large_rope",
    "vit_huge",
    "vit_huge_rope",
    "vit_giant",
    "vit_giant_rope",
    "vit_giant_xformers",
    "vit_giant_xformers_rope",
    "vit_gigantic",
    "vit_gigantic_xformers",
    "VIT_EMBED_DIMS",
]

# Encoder blocks (0-indexed) whose normalized outputs are concatenated to form
# the multi-level representation, keyed by network depth.
HIERARCHICAL_LAYERS = {
    12: [2, 5, 8, 11],
    24: [5, 11, 17, 23],
    40: [9, 19, 29, 39],
    # ViT-G: equally spaced indices matching the paper's [12, 24, 36, 48]
    # (0-indexed, uniform spacing of 12). The reference code shipped a
    # non-uniform [11, 23, 37, 47]; we follow the paper here.
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

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

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
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    use_reentrant=False,
                    return_attn=self.attn_out,
                    mode=mode,
                )
            else:
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


# ---------------------------------------------------------------------------
# Model factory functions (name -> configured VisionTransformer)
# ---------------------------------------------------------------------------
def _vit(embed_dim, depth, num_heads, mlp_ratio=4, patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_synthetic(patch_size=16, **kwargs):
    return _vit(1, 1, 1, 4, patch_size, **kwargs)


def vit_tiny(patch_size=16, **kwargs):
    return _vit(192, 12, 3, 4, patch_size, **kwargs)


def vit_small(patch_size=16, **kwargs):
    return _vit(384, 12, 6, 4, patch_size, **kwargs)


def vit_base(patch_size=16, **kwargs):
    return _vit(768, 12, 12, 4, patch_size, **kwargs)


def vit_large(patch_size=16, **kwargs):
    return _vit(1024, 24, 16, 4, patch_size, **kwargs)


def vit_large_rope(patch_size=16, **kwargs):
    return _vit(1024, 24, 16, 4, patch_size, use_rope=True, **kwargs)


def vit_huge(patch_size=16, **kwargs):
    return _vit(1280, 32, 16, 4, patch_size, **kwargs)


def vit_huge_rope(patch_size=16, **kwargs):
    return _vit(1280, 32, 16, 4, patch_size, use_rope=True, **kwargs)


def vit_giant(patch_size=16, **kwargs):
    return _vit(1408, 40, 16, 48 / 11, patch_size, **kwargs)


def vit_giant_rope(patch_size=16, **kwargs):
    return _vit(1408, 40, 16, 48 / 11, patch_size, use_rope=True, **kwargs)


def vit_giant_xformers(patch_size=16, **kwargs):
    return _vit(1408, 40, 22, 48 / 11, patch_size, **kwargs)


def vit_giant_xformers_rope(patch_size=16, **kwargs):
    return _vit(1408, 40, 22, 48 / 11, patch_size, use_rope=True, **kwargs)


def vit_gigantic(patch_size=16, **kwargs):
    return _vit(1664, 48, 16, 64 / 13, patch_size, **kwargs)


def vit_gigantic_xformers(patch_size=16, **kwargs):
    return _vit(1664, 48, 26, 64 / 13, patch_size, **kwargs)


VIT_EMBED_DIMS = {
    "vit_synthetic": 1,
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
    "vit_gigantic": 1664,
}
