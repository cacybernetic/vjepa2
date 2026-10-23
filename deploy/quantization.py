#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quantization.py - step 1, program 1 of 2: V-JEPA 2.1 checkpoint (.pth) -> FP16 or INT4 Q4_K_M ONNX.

What is this program for?
=========================

V-JEPA 2.1 is a video/image encoder trained by Meta (paper in
resources/vjepa2_1_arXiv, code in resources/vjepa2-main). Its weights are
published as PyTorch checkpoints (.pt / .pth). This program turns the encoder
of such a checkpoint into compressed ONNX models, so that embed.py can compute
embeddings with ONNX Runtime, without PyTorch. The format is chosen with
--precision (both by default):

    fp16     all the weights and computations in float16: half the size of float32, almost no loss
    q4_k_m   the weights of the linear layers in 4 bits (6 bits for the most sensitive ones) with the
             Q4_K_M recipe of llama.cpp, the rest in float32: 3 to 4 times smaller than float32
    fp32     the float32 model itself (no compression, the reference)

V-JEPA 2.1 has a "multi-modal tokenizer" (paper, section 3.3.4): an image is
cut into 16x16 patches by a 2D convolution, a video into 2x16x16 "tubelets" by
a 3D convolution, and a learned "modality embedding" tells the transformer
which kind of input it reads. The choice is made in Python from the input
shape, so one ONNX graph can only hold one of the two paths. That is why two
models are exported for each precision:

    encoder_image_<precision>.onnx   pixel_values [B, 3, H, W]     -> patch_tokens [B, h*w, D]
    encoder_video_<precision>.onnx   pixel_values [B, 3, T, H, W]  -> patch_tokens [B, T/2*h*w, D]

with h = H/16, w = W/16. patch_tokens are the features of the last encoder
block after its LayerNorm: the default output of the encoder in evaluation
mode (the "frozen features" used for all the evaluations of the paper).

The two programs of step 1
--------------------------

    1. quantization.py  (this program) checkpoint -> <output-dir>/{*.onnx, config.json}
    2. embed.py         ONNX Runtime only: image/video files -> one .pkl per image/frame

The programs never import each other. They only share files: the .onnx
models and config.json (file names, input shapes, normalization). This is
the "file contract" of step 1. A run in a folder that already holds models of
the same checkpoint and shapes adds its models to config.json, so fp16 and
q4_k_m can be made by two separate runs.

Important words
===============

Checkpoint key
    A V-JEPA checkpoint is a dict that holds several networks. The weights to
    use are, in this order of preference (same choice as src/hub/backbones.py):
    "ema_encoder" (distilled ViT-B/L: the EMA of the student is the released
    model), "target_encoder" (ViT-g/G), then "encoder".

FP16 conversion
    The model is first exported in float32 (so the rotary position tables,
    which are constants, are computed in float32 and folded into the graph),
    then every float32 weight and operation is converted to float16 with
    onnxconverter-common. The input and output stay float32 (keep_io_types):
    the user of the model does not need to know about float16.

Q4_K_M
    A 4-bit quantization recipe of llama.cpp (the "K-quants" of GGUF files).
    Each row of a weight matrix is cut into super-blocks of 256 values, cut
    again into 8 blocks of 32 values. In a block, value = scale * q - min with
    q a 4-bit integer (0..15); the scale and min of each block are stored in 6
    bits, relative to two float16 numbers of the super-block ("Q4_K"). The
    "_M" (medium) mix keeps the most sensitive matrices in 6 bits ("Q6_K": 16
    blocks of 16 values, value = scale * (q - 32)): as in llama.cpp
    (src/llama-quant.cpp), the attention values and the second MLP layer
    ("ffn_down") of half of the blocks: the first and last eighths, then one
    block out of three. V-JEPA computes the attention q, k and v with a single
    matrix (qkv): its last third, v, follows the rule of llama.cpp's attn_v.
    The quantized values are those of llama.cpp's reference code (ggml
    quantize_row_q4_K_ref / quantize_row_q6_K_ref, no importance matrix),
    ported to NumPy. A row whose length is not a multiple of 256 (ViT-g:
    1408, ViT-G: 1664) ends with a shorter super-block.

MatMulNBits
    llama.cpp cannot run V-JEPA, so the Q4_K_M weights are stored in the 4-bit
    (and 8-bit) matrix product of ONNX Runtime: the MatMulNBits operator
    (com.microsoft domain), which reads blocks of 32 (or 16) integers with one
    float scale and one float zero point per block. (q - zero point) * scale
    is the Q4_K value scale * q - min, so the model computes with exactly the
    values chosen by llama.cpp. The patch embedding, LayerNorm, biases and
    attention stay in float32. The file is bigger than a GGUF file would be
    (the block scales are stored in float32, not in 6 bits) but the weights
    are the same.

Validation
    The ONNX models and the float32 PyTorch model read the same random input;
    we print the largest absolute difference and the cosine similarity of the
    patch tokens. fp16 is a good conversion if the minimum cosine is above
    0.99. Q4_K_M loses more: on ViT-B, the mean cosine is 0.994 to 0.996 for
    the patch tokens and 0.998 for their mean (the "embedding" of embed.py).

On CPU
    ONNX Runtime has few float16 kernels on CPU: it inserts casts and computes
    in float32, so the float16 model is smaller but slower than the float32
    one. On a CUDA GPU, float16 is faster. The Q4_K_M model expands its 4-bit
    weights to float32 just before each product: about the speed of the
    float32 model, 3 to 4 times smaller on disk and in memory (ViT-B: 84 MB
    for images, 133 MB for videos, whose rotary position tables stay in
    float32).

Usage
=====

    python quantization.py --checkpoint weights/vjepa2_1_vitl_dist_vitG_384.pt
    python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt --precision q4_k_m \\
        --output-dir models/vjepa2_1_vitb_384 --img-size 384 --num-frames 16

Requirements (PyTorch is only needed by this program):

    torch, onnx, onnxscript, onnxconverter-common, onnxruntime, numpy, loguru, tqdm

The encoder itself is defined in this file (a port of app/vjepa_2_1/models of the official
vjepa2 code): the official code is not needed to run it.
"""

import argparse
import ctypes
import gc
import hashlib
import json
import logging
import pickle
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript.optimizer
import torch
from loguru import logger
from onnx import helper, numpy_helper
from onnxconverter_common import float16
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent

# Same preference as src/hub/backbones.py: the distilled ViT-B/L release their EMA student
# as "ema_encoder" (their "target_encoder", if any, may be the ViT-G teacher).
CHECKPOINT_KEYS = ("ema_encoder", "target_encoder", "encoder")

# (embed_dim, depth) -> architecture name (see ARCHITECTURE_SIZES).
# Names and head counts follow ARCH_NAME_MAP of src/hub/backbones.py.
ARCHITECTURES = {
    (768, 12): "vit_base",
    (1024, 24): "vit_large",
    (1408, 40): "vit_giant_xformers",
    (1664, 48): "vit_gigantic_xformers",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUT_NAME = "pixel_values"
OUTPUT_NAME = "patch_tokens"
MODALITIES = ("image", "video")
PRECISIONS = ("fp16", "q4_k_m", "fp32")

# Validation: (statistic of compare(), lowest value) below which an ONNX model is reported as too far from PyTorch.
EXPECTED_COSINE = {"fp32": ("min_cosine", 0.99), "fp16": ("min_cosine", 0.99), "q4_k_m": ("mean_cosine", 0.98)}

# A protobuf file cannot exceed 2 GiB: bigger models keep their weights in a ".data" file.
PROTOBUF_LIMIT_BYTES = 1_800_000_000


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show the defaults, except None (required values, or values computed later)."""

    def _get_help_string(self, action):
        return action.help if action.default is None else super()._get_help_string(action)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the encoder of a V-JEPA 2.1 checkpoint to FP16 or INT4 Q4_K_M ONNX (image and video).",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="V-JEPA 2.1 checkpoint (.pth / .pt)")
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=PRECISIONS,
        default=["fp16", "q4_k_m"],
        help="model formats to write: fp16 (float16), q4_k_m (INT4 Q4_K_M weights), fp32 (no compression)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="where to write the models (default: models/<checkpoint name>)"
    )
    parser.add_argument(
        "--checkpoint-key", default=None, help=f"dict entry holding the encoder (default: first of {CHECKPOINT_KEYS})"
    )
    parser.add_argument(
        "--arch",
        choices=sorted(ARCHITECTURES.values()),
        default=None,
        help="encoder architecture (default: detected from the weights)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs="+",
        default=[384],
        metavar="PIXELS",
        help="input size: one value (square) or two values H W; multiples of the patch size (16)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16, help="frames per clip of the video model (even number)"
    )
    parser.add_argument(
        "--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES), help="graphs to export"
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset")
    parser.add_argument("--no-validate", action="store_true", help="skip the PyTorch vs ONNX Runtime comparison")
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="allow arbitrary pickled objects in the checkpoint (only for files you trust)",
    )
    return parser.parse_args()


def check_args(args):
    if not args.checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    if len(args.img_size) not in (1, 2):
        sys.exit("--img-size takes one value (square) or two values (H W)")
    if args.num_frames < 2 or args.num_frames % 2:
        sys.exit("--num-frames must be an even number >= 2 (the video tokenizer reads frames by pairs)")
    if args.output_dir is None:
        args.output_dir = PROJECT_DIR / "models" / args.checkpoint.stem
    args.img_size = (args.img_size[0], args.img_size[-1])
    args.precision = [precision for precision in PRECISIONS if precision in args.precision]


def quiet_libraries():
    """Hide the deprecation warnings of the exporter (they are not about our model)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for name in ("torch.onnx", "torch.export", "torch._dynamo", "onnxscript"):
        logging.getLogger(name).setLevel(logging.ERROR)


def release_memory():
    """Free the unused Python objects and give the freed memory back to the system.

    glibc keeps the memory freed by the exporter and ONNX for later use (about 400 MB after the image
    export of a ViT-B), so the next export would start with that much less free RAM. malloc_trim
    returns it. On a system without glibc only the garbage collector runs.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Checkpoint -> PyTorch encoder
# ---------------------------------------------------------------------------


def load_checkpoint(path, unsafe):
    try:
        # mmap: the tensors stay on disk until they are copied into the model (low RAM use).
        return torch.load(path, map_location="cpu", weights_only=not unsafe, mmap=True)
    except pickle.UnpicklingError as error:
        sys.exit(
            f"{path} holds objects that are not plain tensors ({error}).\n"
            "If you trust this file, run again with --unsafe-load."
        )
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        # Old (non-zip) checkpoint format: no memory mapping.
        return torch.load(path, map_location="cpu", weights_only=not unsafe)


def select_encoder_state_dict(checkpoint, key):
    if not isinstance(checkpoint, dict):
        sys.exit(f"The checkpoint is a {type(checkpoint).__name__}, not a dict of weights")
    if key is not None:
        if key not in checkpoint:
            sys.exit(f"No entry '{key}' in the checkpoint; entries: {sorted(checkpoint)}")
        state_dict = checkpoint[key]
    else:
        key = next((k for k in CHECKPOINT_KEYS if k in checkpoint), None)
        if key is not None:
            state_dict = checkpoint[key]
        elif any(str(k).endswith("patch_embed.proj.weight") for k in checkpoint):
            key, state_dict = "<root>", checkpoint
        else:
            sys.exit(f"No encoder found in the checkpoint; entries: {sorted(checkpoint)} (see --checkpoint-key)")
    logger.info(f"Checkpoint entries: {sorted(map(str, checkpoint))} -> encoder read from '{key}'")
    # Same cleaning as src/hub/backbones.py::_clean_backbone_key (DDP and wrapper prefixes).
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    return key, state_dict


def detect_architecture(state_dict):
    if "patch_embed.proj.weight" not in state_dict or "patch_embed_img.proj.weight" not in state_dict:
        sys.exit("These weights have no image and video patch embeddings: is it really a V-JEPA 2.1 encoder?")
    # Video tokenizer: Conv3d weight [D, 3, tubelet, patch, patch].
    embed_dim, _, tubelet_size, patch_size, _ = state_dict["patch_embed.proj.weight"].shape
    depth = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks."))
    return int(embed_dim), depth, int(patch_size), int(tubelet_size)


def build_encoder(state_dict, arch, patch_size, tubelet_size):
    embed_dim, depth, num_heads, mlp_ratio = ARCHITECTURE_SIZES[arch]
    # Same settings as src/hub/backbones.py::_make_vjepa2_1_model (GELU MLP: the hub passes use_SiLU,
    # which the official constructor ignores, so its default use_silu=False applies).
    use_swiglu = any(".mlp.fc3." in k for k in state_dict)
    encoder = VisionTransformer(embed_dim, depth, num_heads, mlp_ratio, patch_size, tubelet_size, use_swiglu)
    encoder.load_state_dict(state_dict, strict=True)
    return encoder.eval().requires_grad_(False)


# ---------------------------------------------------------------------------
# V-JEPA 2.1 encoder (port of app/vjepa_2_1/models of the official vjepa2 code)
# ---------------------------------------------------------------------------
#
# Only the evaluation path of the official VisionTransformer is kept: RoPE attention, no masks, no registers,
# no class token, interpolate_rope=True, img_temporal_dim_size=1, modality embeddings, output of the last block
# after its LayerNorm. The parameter names are those of the official model, so a checkpoint loads with
# strict=True, and the operations are done in the same order, so the outputs are the same.

# arch -> (embed_dim, depth, num_heads, mlp_ratio), as in app/vjepa_2_1/models/vision_transformer.py.
ARCHITECTURE_SIZES = {
    "vit_base": (768, 12, 12, 4.0),
    "vit_large": (1024, 24, 16, 4.0),
    "vit_giant_xformers": (1408, 40, 22, 48 / 11),
    "vit_gigantic_xformers": (1664, 48, 26, 64 / 13),
}

# Grid size of the pre-training images (256 px / 16): interpolate_rope maps every grid onto this one.
PRETRAINED_GRID_SIZE = 16


class PatchEmbed3D(torch.nn.Module):
    """[B, 3, T, H, W] -> [B, T/tubelet * H/patch * W/patch, D] with a Conv3d (tubelet 1: the image tokenizer)."""

    def __init__(self, patch_size, tubelet_size, embed_dim, in_chans=3):
        super().__init__()
        self.proj = torch.nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFN(torch.nn.Module):
    """The use_silu=True variant of the official code (wide_silu=True): fc3(silu(fc1(x)) * fc2(x))."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = (int(2 * hidden_dim / 3) + 7) // 8 * 8
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.fc2 = torch.nn.Linear(dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc3(torch.nn.functional.silu(self.fc1(x)) * self.fc2(x))


def rotate_queries_or_keys(x, pos):
    """Rotary position embedding of x [B, heads, N, D] at the positions pos [N] (D even)."""
    D = x.size(-1)
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)
    y1, y2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class RoPEAttention(torch.nn.Module):
    """Multi-head self-attention whose queries and keys are rotated along the time, height and width axes."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)
        # Each axis rotates the same even number of channels; the remaining ones are not rotated.
        self.axis_dim = 2 * ((self.head_dim // 3) // 2)

    def forward(self, x, T, H_patches, W_patches):
        B, N, C = x.size()
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Token index -> (frame, row, column), in float, as separate_positions() of the official code.
        ids = torch.arange(int(T * H_patches * W_patches), device=x.device)
        tokens_per_frame = int(H_patches * W_patches)
        frame_ids = ids // tokens_per_frame
        height_ids = (ids - tokens_per_frame * frame_ids) // W_patches
        width_ids = (ids - tokens_per_frame * frame_ids) - W_patches * height_ids
        d_pos = 1.0 * frame_ids
        # interpolate_rope: the rows and columns are spread over the pre-training grid.
        h_pos = 1.0 * height_ids * (PRETRAINED_GRID_SIZE - 1) / (H_patches - 1)
        w_pos = 1.0 * width_ids * (PRETRAINED_GRID_SIZE - 1) / (W_patches - 1)

        q_parts, k_parts, s = [], [], 0
        for pos in (d_pos, h_pos, w_pos):
            q_parts.append(rotate_queries_or_keys(q[..., s : s + self.axis_dim], pos))
            k_parts.append(rotate_queries_or_keys(k[..., s : s + self.axis_dim], pos))
            s += self.axis_dim
        if s < self.head_dim:
            q_parts.append(q[..., s:])
            k_parts.append(k[..., s:])
        q = torch.cat(q_parts, dim=-1)
        k = torch.cat(k_parts, dim=-1)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, use_swiglu):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.attn = RoPEAttention(dim, num_heads)
        self.norm2 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.mlp = (SwiGLUFFN if use_swiglu else MLP)(dim, int(dim * mlp_ratio))

    def forward(self, x, T, H_patches, W_patches):
        x = x + self.attn(self.norm1(x), T, H_patches, W_patches)
        return x + self.mlp(self.norm2(x))


class VisionTransformer(torch.nn.Module):
    """V-JEPA 2.1 encoder: [B, 3, 1, H, W] (image) or [B, 3, T, H, W] (video) -> patch tokens [B, N, D]."""

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, patch_size, tubelet_size, use_swiglu):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, embed_dim)
        self.patch_embed_img = PatchEmbed3D(patch_size, 1, embed_dim)
        self.blocks = torch.nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, use_swiglu) for _ in range(depth)]
        )
        # One LayerNorm per hierarchical output (used for distillation); evaluation only uses the last one.
        self.norms_block = torch.nn.ModuleList([torch.nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(4)])
        self.img_mod_embed = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.video_mod_embed = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        _, _, T, H, W = x.shape
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        # A temporal size of 1 selects the image tokenizer and the image modality embedding.
        if T == 1:
            x = self.patch_embed_img(x) + self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            T = T // self.tubelet_size
            x = self.patch_embed(x) + self.video_mod_embed.repeat(x.shape[0], 1, 1)
        for block in self.blocks:
            x = block(x, T, H_patches, W_patches)
        return self.norms_block[-1](x)


class ImageEncoder(torch.nn.Module):
    """[B, 3, H, W] -> patch_tokens [B, h*w, D] through the 2D (image) tokenizer."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values):
        # A temporal size of 1 (== img_temporal_dim_size) selects the 2D tokenizer
        # (patch_embed_img) and the image modality embedding in VisionTransformer.forward.
        return self.encoder(pixel_values.unsqueeze(2))


class VideoEncoder(torch.nn.Module):
    """[B, 3, T, H, W] -> patch_tokens [B, T/2*h*w, D] through the 3D (video) tokenizer."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values)


# ---------------------------------------------------------------------------
# PyTorch -> ONNX float32 -> ONNX float16
# ---------------------------------------------------------------------------


def export_fp32(module, example, path, opset, num_tokens):
    # The batch is the only dynamic axis: the rotary position tables depend on (T, H, W),
    # which are fixed, so they can be folded into constants computed in float32.
    batch = torch.export.Dim("batch", min=1, max=4096)
    program = torch.onnx.export(
        module,
        (example,),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"pixel_values": {0: batch}},
        opset_version=opset,
        dynamo=True,
        optimize=False,
        verbose=False,
    )
    # The rotary tables hold num_tokens x 10 values per axis. With the default folding limits (8192 values)
    # those of a video clip stay as Sin/Cos nodes, and the float16 conversion would then compute the
    # rotation angles in float16 (min cosine 0.9993 instead of 0.99998 on a 16-frame 384 px clip).
    limit = max(262_144, 64 * num_tokens)
    onnxscript.optimizer.optimize_ir(program.model, input_size_limit=limit, output_size_limit=limit)
    unfolded = sum(node.op_type in ("Sin", "Cos") for node in program.model.graph)
    if unfolded:
        logger.warning(f"{unfolded} rotary Sin/Cos nodes were not folded: they will run in float16")
    program.save(str(path), external_data=True)


def save_model(model, dst, num_bytes):
    """Write an ONNX model, with its weights in a ".data" file next to it if they are too big for protobuf."""
    for name in (dst.name, dst.name + ".data"):
        (dst.parent / name).unlink(missing_ok=True)
    if num_bytes > PROTOBUF_LIMIT_BYTES:
        onnx.save_model(
            model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=dst.name + ".data"
        )
    else:
        onnx.save_model(model, str(dst))


def convert_fp16(src, dst, num_params):
    model = onnx.load(str(src))
    # The exporter already wrote the type of every intermediate tensor (value_info), so the
    # converter does not need onnx.shape_inference, which fails above 2 GiB.
    model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    save_model(model, dst, num_params * 2)
    del model
    release_memory()


# ---------------------------------------------------------------------------
# Q4_K_M: the K-quants of llama.cpp in NumPy
# ---------------------------------------------------------------------------
#
# Port of the reference code of ggml (ggml/src/ggml-quants.c: make_qkx2_quants, quantize_row_q4_K_ref,
# make_qx_quants, quantize_row_q6_K_ref). Each block is a column of an array x [values per block, blocks],
# so that all the blocks of a matrix are quantized at once. The computations are in float32, in the order of
# the C code (a sum over axis 0 adds the rows one after the other, like the C loops): the quantized values
# are the same as those of llama-quantize without importance matrix.

QK_K = 256  # values per super-block
F32 = np.float32


def nearest_int(values):
    """ggml nearest_int(): round half to even, from float32."""
    return np.rint(values).astype(np.int32)


def column_sum(values):
    return np.add.reduce(values, axis=0, dtype=F32)


def make_qkx2_quants(x, weights, nmax, rmin, rdelta, nstep):
    """ggml make_qkx2_quants() on each column of x [n, blocks]: x ~ scale * L - min with L in 0..nmax.

    Tries nstep + 1 scales around nmax / (max - min), fits scale and min by weighted least squares for
    each, and keeps the smallest weighted squared error. Returns L (float32) [n, blocks], scale and min [blocks].
    """
    low = np.minimum(x.min(axis=0), F32(0))
    high = x.max(axis=0)
    flat = high == low
    sum_w = column_sum(weights)
    sum_x = column_sum(weights * x)
    with np.errstate(all="ignore"):  # flat blocks divide by 0; their result is replaced at the end
        iscale = F32(nmax) / (high - low)
        scale = F32(1) / iscale
        L = np.clip(nearest_int(iscale * (x - low)), 0, nmax).astype(F32)
        diff = scale * L + low - x
        best_error = column_sum(weights * (diff * diff))
        for step in range(nstep + 1):
            iscale = (F32(rmin) + F32(rdelta) * F32(step) + F32(nmax)) / (high - low)
            laux = np.clip(nearest_int(iscale * (x - low)), 0, nmax).astype(F32)
            wl = weights * laux
            sum_l = column_sum(wl)
            sum_l2 = column_sum(wl * laux)
            sum_xl = column_sum(wl * x)
            det = sum_w * sum_l2 - sum_l * sum_l
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / det
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det
            positive = this_min > 0
            this_scale = np.where(positive, sum_xl / sum_l2, this_scale)
            this_min = np.where(positive, F32(0), this_min)
            diff = this_scale * laux + this_min - x
            error = column_sum(weights * (diff * diff))
            better = (det > 0) & (error < best_error) & ~flat
            L = np.where(better, laux, L)
            best_error = np.where(better, error, best_error)
            scale = np.where(better, this_scale, scale)
            low = np.where(better, this_min, low)
    return np.where(flat, F32(0), L), np.where(flat, F32(0), scale), -low


def quantize_q4_k(w):
    """ggml quantize_row_q4_K_ref() on each row of w [rows, K], K multiple of 256.

    Returns codes [rows, K] (uint8, 0..15), scale and minimum [rows, K/32] (float32):
    value = scale * code - minimum.
    """
    rows, k = w.shape
    x = np.ascontiguousarray(w.astype(F32).reshape(-1, 32).T)  # one block of 32 values per column
    av_x = np.sqrt(column_sum(x * x) / F32(32))
    L, scales, mins = make_qkx2_quants(x, av_x + np.abs(x), 15, -1.0, 0.1, 20)
    # The scales and mins of the 8 blocks of a super-block are stored on 6 bits, relative to their maximum.
    scales, mins = scales.reshape(-1, 8), mins.reshape(-1, 8)
    max_scale = np.maximum(scales.max(axis=1), F32(0))
    max_min = np.maximum(mins.max(axis=1), F32(0))
    with np.errstate(divide="ignore"):
        inv_scale = np.where(max_scale > 0, F32(63) / max_scale, F32(0))
        inv_min = np.where(max_min > 0, F32(63) / max_min, F32(0))
    sc = np.minimum(nearest_int(inv_scale[:, None] * scales) & 0xFF, 63)  # uint8_t ls = ...; MIN(63, ls)
    m = np.minimum(nearest_int(inv_min[:, None] * mins) & 0xFF, 63)
    d = (max_scale / F32(63)).astype(np.float16).astype(F32)
    dmin = (max_min / F32(63)).astype(np.float16).astype(F32)
    scale = (d[:, None] * sc.astype(F32)).reshape(-1)
    minimum = (dmin[:, None] * m.astype(F32)).reshape(-1)
    # Quantize again with the stored scales; a block whose scale is 0 keeps its first codes (they are unused).
    with np.errstate(all="ignore"):
        requantized = np.clip(nearest_int((x + minimum) / scale), 0, 15).astype(F32)
    L = np.where(scale != 0, requantized, L)
    return L.T.reshape(rows, k).astype(np.uint8), scale.reshape(rows, -1), minimum.reshape(rows, -1)


def make_qx_quants(x, nmax):
    """ggml make_qx_quants(rmse_type=1) on each column of x [n, blocks]: x ~ scale * (L - nmax), L in 0..2 nmax - 1.

    Tries 19 scales around -nmax / (value of largest magnitude), with weights x^2, and keeps the best
    least squares fit. Returns L (float32) [n, blocks] and scale [blocks].
    """
    peak = np.take_along_axis(x, np.abs(x).argmax(axis=0)[None], axis=0)[0]
    zero = np.abs(peak) < F32(1e-15)
    w = x * x
    wx = w * x

    def fit(iscale):
        l = np.clip(nearest_int(iscale * x), -nmax, nmax - 1).astype(F32)
        return l, column_sum(wx * l), column_sum(w * l * l)

    with np.errstate(all="ignore"):  # all-zero blocks divide by 0; their result is replaced at the end
        l, sumlx, suml2 = fit(F32(-nmax) / peak)
        L = l + F32(nmax)
        scale = np.where(suml2 != 0, sumlx / suml2, F32(0))
        best = scale * sumlx
        for step in (*range(-9, 0), *range(1, 10)):
            l, sumlx, suml2 = fit(-(F32(nmax) + F32(0.1) * F32(step)) / peak)
            better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
            L = np.where(better, l + F32(nmax), L)
            scale = np.where(better, sumlx / suml2, scale)
            best = np.where(better, scale * sumlx, best)
    return np.where(zero, F32(0), L), np.where(zero, F32(0), scale)


def quantize_q6_k(w):
    """ggml quantize_row_q6_K_ref() on each row of w [rows, K], K multiple of 256.

    Returns codes [rows, K] (uint8, 0..63) and scale [rows, K/16] (float32): value = scale * (code - 32).
    """
    rows, k = w.shape
    x = np.ascontiguousarray(w.astype(F32).reshape(-1, 16).T)  # one block of 16 values per column
    L, scales = make_qx_quants(x, 32)
    # The scales of the 16 blocks of a super-block are stored on 8 bits (signed), relative to the largest one.
    scales = scales.reshape(-1, 16)
    max_scale = np.take_along_axis(scales, np.abs(scales).argmax(axis=1)[:, None], axis=1)[:, 0]
    zero = np.abs(max_scale) < F32(1e-15)
    with np.errstate(all="ignore"):
        iscale = F32(-128) / max_scale
        d = np.where(zero, F32(0), (F32(1) / iscale).astype(np.float16).astype(F32))
        sc = np.where(zero[:, None], 0, np.minimum(nearest_int(iscale[:, None] * scales), 127))
        scale = (d[:, None] * sc.astype(F32)).reshape(-1)
        requantized = np.clip(nearest_int(x / scale), -32, 31).astype(F32) + F32(32)
    L = np.where(scale != 0, requantized, L)
    L = np.where(np.repeat(zero, 16), F32(0), L)  # an all-zero super-block is stored as zeros
    return L.T.reshape(rows, k).astype(np.uint8), scale.reshape(rows, -1)


# ---------------------------------------------------------------------------
# ONNX float32 -> ONNX Q4_K_M (MatMulNBits)
# ---------------------------------------------------------------------------


def use_more_bits(layer, num_layers):
    """llama.cpp (src/llama-quant.cpp): the layers whose attn_v and ffn_down are in Q6_K in a Q4_K_M model."""
    return layer < num_layers // 8 or layer >= 7 * num_layers // 8 or (layer - num_layers // 8) % 3 == 2


def q4_k_m_plan(name, rows, depth, ffn_down):
    """Q4_K_M types of the output rows of a Linear layer: [(first row, end row, "q4_k" or "q6_k"), ...]."""
    block = re.search(r"blocks\.(\d+)\.", name)
    more_bits = block is not None and use_more_bits(int(block.group(1)), depth)
    if more_bits and name.endswith("attn.qkv"):
        # Output rows [q; k; v]: v is llama.cpp's attn_v.
        return [(0, 2 * rows // 3, "q4_k"), (2 * rows // 3, rows, "q6_k")]
    if more_bits and name.endswith(ffn_down):
        return [(0, rows, "q6_k")]
    return [(0, rows, "q4_k")]


def matmul_nbits_weights(w, qtype):
    """K-quantize w [rows, K] and lay it out as the inputs and attributes of a MatMulNBits node."""
    rows, k = w.shape
    # Zeros up to a multiple of 256: all-zero blocks change no scale, so the last super-block is a shorter one.
    padded = np.pad(w, ((0, 0), (0, -k % QK_K)))
    if qtype == "q4_k":
        codes, scale, minimum = quantize_q4_k(padded)
        codes = codes[:, :k].reshape(rows, k // 32, 32)
        scale, minimum = scale[:, : k // 32], minimum[:, : k // 32]
        # scale * q - minimum == (q - minimum / scale) * scale: a float zero point per block. A block whose
        # scale is 0 holds the constant -minimum: q = 0, zero point 1, scale minimum.
        empty = scale == 0
        codes[empty] = 0
        zero_points = np.divide(minimum, scale, out=np.ones_like(scale), where=~empty)
        scale = np.where(empty, minimum, scale)
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)  # two codes per byte, the first one in the low bits
        zero_points = zero_points.reshape(-1)
        return {"B": packed, "scales": scale.reshape(-1), "zero_points": zero_points, "bits": 4, "block_size": 32}
    codes, scale = quantize_q6_k(padded)
    # scale * (q - 32) == scale * ((q + 96) - 128): 8-bit codes q + 96 with the default zero point of 8 bits.
    codes = codes[:, :k].reshape(rows, k // 16, 16) + np.uint8(96)
    scale = np.ascontiguousarray(scale[:, : k // 16]).reshape(-1)
    return {"B": codes, "scales": scale, "zero_points": None, "bits": 8, "block_size": 16}


def weight_key(array):
    return hashlib.sha1(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def quantize_linear_layers(encoder, depth):
    """Q4_K_M weights of every Linear layer, computed once for the image and the video graphs.

    The exporter gives the weights anonymous names, so they are found by content: returns
    {weight_key of the weight as stored in the graph ([in, out]): (layer name, MatMulNBits parts)}
    and the number of weights in each type.
    """
    linears = [(name, module) for name, module in encoder.named_modules() if isinstance(module, torch.nn.Linear)]
    ffn_down = "mlp.fc3" if any(name.endswith("mlp.fc3") for name, _ in linears) else "mlp.fc2"  # SwiGLU or MLP
    quantized, counts = {}, {"q4_k": 0, "q6_k": 0}
    for name, module in tqdm(linears, desc="Q4_K_M", unit="layer", leave=False):
        w = module.weight.detach().float().numpy()  # [out, in]: one row per output, as in ggml
        if w.shape[1] % 32:
            logger.warning(f"{name}: {w.shape[1]} inputs is not a multiple of 32, the layer stays in float32")
            continue
        parts = []
        for start, stop, qtype in q4_k_m_plan(name, w.shape[0], depth, ffn_down):
            parts.append(matmul_nbits_weights(w[start:stop], qtype))
            counts[qtype] += (stop - start) * w.shape[1]
        quantized[weight_key(w.T)] = (name, parts)
    return quantized, counts


def convert_q4_k_m(src, dst, quantized):
    """Float32 ONNX model -> Q4_K_M model: each MatMul by a Linear weight becomes MatMulNBits node(s)."""
    model = onnx.load(str(src))
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    nodes, new_initializers, replaced, inputs_of = [], [], set(), {}
    for node in graph.node:
        weight = initializers.get(node.input[1]) if node.op_type == "MatMul" else None
        key = weight_key(numpy_helper.to_array(weight)) if weight is not None else None
        if key not in quantized:
            nodes.append(node)
            continue
        replaced.add(weight.name)
        name, parts = quantized[key]
        if key not in inputs_of:
            inputs_of[key] = []
            for index, part in enumerate(parts):
                names = []
                for role in ("B", "scales", "zero_points"):
                    if part[role] is not None:
                        names.append(f"{name}.part{index}.{role}")
                        new_initializers.append(numpy_helper.from_array(part[role], names[-1]))
                inputs_of[key].append(names)
        # Several parts (qkv: q and k in Q4_K, v in Q6_K) are concatenated along the output features.
        outputs = [node.output[0]] if len(parts) == 1 else [f"{node.output[0]}.part{i}" for i in range(len(parts))]
        for index, (part, part_inputs, output) in enumerate(zip(parts, inputs_of[key], outputs)):
            rows, num_blocks = part["B"].shape[:2]
            nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    [node.input[0], *part_inputs],
                    [output],
                    name=f"{node.name}.nbits{index}",
                    domain="com.microsoft",
                    K=num_blocks * part["block_size"],
                    N=rows,
                    bits=part["bits"],
                    block_size=part["block_size"],
                )
            )
        if len(parts) > 1:
            nodes.append(helper.make_node("Concat", outputs, [node.output[0]], name=f"{node.name}.concat", axis=-1))

    used = {name for node in nodes for name in node.input}
    kept = [tensor for tensor in graph.initializer if tensor.name not in replaced or tensor.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept + new_initializers)
    del graph.node[:]
    graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    save_model(model, dst, sum(len(tensor.raw_data) for tensor in graph.initializer))
    del model
    release_memory()
    return len(inputs_of)


def cosine_similarity(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def compare(reference, output):
    """Patch tokens [B, N, D] of PyTorch and ONNX Runtime -> differences and cosine similarities."""
    reference = reference.astype(np.float64)
    output = output.astype(np.float64)
    cosine = cosine_similarity(reference, output)
    return {
        "finite": bool(np.isfinite(output).all()),
        "max_abs_diff": float(np.abs(reference - output).max()),
        "mean_abs_diff": float(np.abs(reference - output).mean()),
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
        # The "embedding" of embed.py: the mean of the patch tokens.
        "embedding_min_cosine": float(cosine_similarity(reference.mean(1), output.mean(1)).min()),
    }


def run_onnx(path, inputs):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run([OUTPUT_NAME], {INPUT_NAME: inputs})[0]


def export_modality(modality, encoder, args, info, quantized):
    height, width = args.img_size
    grid = (height // info["patch_size"], width // info["patch_size"])
    if modality == "image":
        module = ImageEncoder(encoder).eval()
        sample_shape = (3, height, width)
        token_grid = (1, *grid)
    else:
        module = VideoEncoder(encoder).eval()
        sample_shape = (3, args.num_frames, height, width)
        token_grid = (args.num_frames // info["tubelet_size"], *grid)

    out_dir = args.output_dir
    # The float32 model is written next to the others if it is kept, else in a temporary folder.
    tmp_dir = out_dir / f".tmp_{modality}_fp32"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    fp32_path = (out_dir if "fp32" in args.precision else tmp_dir) / f"encoder_{modality}_fp32.onnx"
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    for name in (fp32_path.name, fp32_path.name + ".data"):
        (fp32_path.parent / name).unlink(missing_ok=True)

    generator = torch.Generator().manual_seed(0)
    # Batch of 2 for the export: an example batch of 1 would be specialized to a constant.
    example = torch.randn(2, *sample_shape, generator=generator)
    # The video check uses one clip only (memory); the image check uses 2 images (dynamic batch).
    check = example if modality == "image" else example[:1]

    reference = None
    if not args.no_validate:
        start = time.time()
        with torch.inference_mode():
            reference = module(check).float().numpy()
        logger.info(f"[{modality}] PyTorch float32 reference {list(reference.shape)} in {time.time() - start:.1f} s")

    release_memory()
    start = time.time()
    logger.info(f"[{modality}] Export to ONNX float32, input [batch, {', '.join(map(str, sample_shape))}] ...")
    export_fp32(module, example, fp32_path, args.opset, int(np.prod(token_grid)))
    logger.info(f"[{modality}] Exported in {time.time() - start:.1f} s")

    files = {}
    for precision in args.precision:
        path = out_dir / f"encoder_{modality}_{precision}.onnx"
        start = time.time()
        if precision == "fp16":
            convert_fp16(fp32_path, path, info["num_params"])
        elif precision == "q4_k_m":
            found = convert_q4_k_m(fp32_path, path, quantized)
            if found < len(quantized):
                missing = len(quantized) - found
                logger.warning(f"[{modality}] {missing} linear layer(s) not found in the graph stay in float32")
        files[precision] = path.name
        size = sum(p.stat().st_size for p in out_dir.glob(path.name + "*"))
        logger.info(f"[{modality}] {precision} model: {path} ({size / 1e6:,.0f} MB, {time.time() - start:.1f} s)")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    release_memory()

    validation = {}
    if reference is not None:
        for precision, name in files.items():
            start = time.time()
            output = run_onnx(out_dir / name, check.numpy())
            validation[precision] = v = compare(reference, output)
            logger.info(
                f"[{modality}] ONNX {precision} vs PyTorch: max |diff| {v['max_abs_diff']:.2e}, patch cosine "
                f"mean {v['mean_cosine']:.6f} / min {v['min_cosine']:.6f}, embedding cosine "
                f"{v['embedding_min_cosine']:.6f} ({time.time() - start:.1f} s)"
            )
            statistic, lowest = EXPECTED_COSINE[precision]
            if not v["finite"] or v[statistic] < lowest:
                logger.warning(f"[{modality}] ONNX {precision} output differs from PyTorch: check the conversion")

    return {
        "files": files,
        "input": INPUT_NAME,
        "output": OUTPUT_NAME,
        "input_shape": ["batch", *sample_shape],
        "output_shape": ["batch", int(np.prod(token_grid)), info["embed_dim"]],
        "token_grid": list(token_grid),
        "validation": validation,
    }


def merge_previous_run(config, config_path):
    """Keep in config.json the models of an earlier run in the same folder (other precisions or modalities),
    if they were made from the same checkpoint with the same shapes and their files are still there."""
    try:
        previous = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return
    same = ("format", "checkpoint", "checkpoint_key", "arch", "image_size", "num_frames")
    if any(previous.get(key) != config[key] for key in same):
        return
    for modality, old in previous.get("models", {}).items():
        new = config["models"].setdefault(modality, {**old, "files": {}, "validation": {}, "quantization": {}})
        for precision, name in old.get("files", {}).items():
            if precision not in new["files"] and (config_path.parent / name).is_file():
                new["files"][precision] = name
                for key in ("validation", "quantization"):
                    if precision in old.get(key, {}):
                        new.setdefault(key, {})[precision] = old[key][precision]
        if not new["files"]:
            del config["models"][modality]


def main():
    args = parse_args()
    check_args(args)
    quiet_libraries()
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
    torch.set_grad_enabled(False)

    logger.info(f"Loading {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, args.unsafe_load)
    key, state_dict = select_encoder_state_dict(checkpoint, args.checkpoint_key)
    embed_dim, depth, patch_size, tubelet_size = detect_architecture(state_dict)
    arch = args.arch or ARCHITECTURES.get((embed_dim, depth))
    if arch is None:
        sys.exit(f"Unknown encoder (embed_dim={embed_dim}, depth={depth}); give it with --arch")
    if any(s % patch_size for s in args.img_size):
        sys.exit(f"--img-size {args.img_size} must be a multiple of the patch size {patch_size}")
    if any(s < 2 * patch_size for s in args.img_size):
        sys.exit(f"--img-size {args.img_size} must hold at least 2 patches per side (RoPE interpolation)")

    encoder = build_encoder(state_dict, arch, patch_size, tubelet_size)
    del checkpoint, state_dict
    release_memory()
    num_params = sum(p.numel() for p in encoder.parameters())
    logger.info(
        f"Encoder {arch}: {num_params / 1e6:,.1f} M parameters, embed_dim {embed_dim}, depth {depth}, "
        f"{encoder.num_heads} heads, patch {patch_size}, tubelet {tubelet_size}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "arch": arch,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": encoder.num_heads,
        "num_params": num_params,
        "patch_size": patch_size,
        "tubelet_size": tubelet_size,
    }

    quantized, quantization = None, {}
    if "q4_k_m" in args.precision:
        start = time.time()
        quantized, counts = quantize_linear_layers(encoder, depth)
        quantization["q4_k_m"] = {
            "recipe": "llama.cpp Q4_K_M (ggml reference quantizers, no importance matrix)",
            "q4_k_weights": counts["q4_k"],
            "q6_k_weights": counts["q6_k"],
            "float32_weights": num_params - counts["q4_k"] - counts["q6_k"],
        }
        logger.info(
            f"Q4_K_M: {len(quantized)} linear layers, {counts['q4_k'] / 1e6:,.1f} M weights in Q4_K (4 bits) and "
            f"{counts['q6_k'] / 1e6:,.1f} M in Q6_K (6 bits), {time.time() - start:.1f} s"
        )

    models = {}
    for modality in args.modalities:
        models[modality] = export_modality(modality, encoder, args, info, quantized)
        models[modality]["quantization"] = dict(quantization)
        release_memory()

    config = {
        "format": "vjepa2.1-onnx",
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_key": key,
        **info,
        "image_size": list(args.img_size),
        "num_frames": args.num_frames,
        "opset": args.opset,
        "io_dtype": "float32",
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "models": models,
    }
    config_path = args.output_dir / "config.json"
    merge_previous_run(config, config_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    available = ", ".join(f"{modality} {sorted(spec['files'])}" for modality, spec in config["models"].items())
    logger.info(f"Done: {config_path} ({available})")


if __name__ == "__main__":
    main()
