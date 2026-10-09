#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quantize.py - step 1, program 1 of 2: V-JEPA 2.1 checkpoint (.pth) -> FP16 ONNX.

What is this program for?
=========================

V-JEPA 2.1 is a video/image encoder trained by Meta (paper in
resources/vjepa2_1_arXiv, code in resources/vjepa2-main). Its weights are
published as PyTorch checkpoints (.pt / .pth). This program turns the encoder
of such a checkpoint into ONNX models whose weights and computations are in
float16 (half the size of float32), so that embed.py can compute embeddings
with ONNX Runtime, without PyTorch.

V-JEPA 2.1 has a "multi-modal tokenizer" (paper, section 3.3.4): an image is
cut into 16x16 patches by a 2D convolution, a video into 2x16x16 "tubelets" by
a 3D convolution, and a learned "modality embedding" tells the transformer
which kind of input it reads. The choice is made in Python from the input
shape, so one ONNX graph can only hold one of the two paths. That is why two
models are exported:

    encoder_image_fp16.onnx   pixel_values [B, 3, H, W]     -> patch_tokens [B, h*w, D]
    encoder_video_fp16.onnx   pixel_values [B, 3, T, H, W]  -> patch_tokens [B, T/2*h*w, D]

with h = H/16, w = W/16. patch_tokens are the features of the last encoder
block after its LayerNorm: the default output of the encoder in evaluation
mode (the "frozen features" used for all the evaluations of the paper).

The two programs of step 1
--------------------------

    1. quantize.py  (this program) checkpoint -> <output-dir>/{*.onnx, config.json}
    2. embed.py     ONNX Runtime only: image/video files -> one .pkl per image/frame

The programs never import each other. They only share files: the .onnx
models and config.json (file names, input shapes, normalization). This is
the "file contract" of step 1.

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

Validation
    The float16 ONNX model and the float32 PyTorch model read the same random
    input; we print the largest absolute difference and the cosine similarity
    of the patch tokens. A minimum cosine above 0.99 is a good conversion.

FP16 on CPU
    ONNX Runtime has few float16 kernels on CPU: it inserts casts and computes
    in float32, so the float16 model is smaller but slower than the float32
    one. On a CUDA GPU, float16 is faster. Use --keep-fp32 to keep the float32
    model as well (embed.py --precision fp32).

Usage
=====

    python quantize.py --checkpoint weights/vjepa2_1_vitl_dist_vitG_384.pt
    python quantize.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt \\
        --output-dir models/vjepa2_1_vitb_384 --img-size 384 --num-frames 16 --keep-fp32

Requirements (PyTorch is only needed by this program):

    torch, timm, einops (for the official V-JEPA code), onnx, onnxscript,
    onnxconverter-common, onnxruntime, numpy, loguru
"""

import argparse
import gc
import json
import logging
import pickle
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
from onnxconverter_common import float16

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VJEPA_ROOT = PROJECT_DIR / "resources" / "vjepa2-main"

# Same preference as src/hub/backbones.py: the distilled ViT-B/L release their EMA student
# as "ema_encoder" (their "target_encoder", if any, may be the ViT-G teacher).
CHECKPOINT_KEYS = ("ema_encoder", "target_encoder", "encoder")

# (embed_dim, depth) -> constructor of app/vjepa_2_1/models/vision_transformer.py.
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

# A protobuf file cannot exceed 2 GiB: bigger models keep their weights in a ".data" file.
PROTOBUF_LIMIT_BYTES = 1_800_000_000


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show the defaults, except None (required values, or values computed later)."""

    def _get_help_string(self, action):
        return action.help if action.default is None else super()._get_help_string(action)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the encoder of a V-JEPA 2.1 checkpoint to FP16 ONNX (image and video graphs).",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="V-JEPA 2.1 checkpoint (.pth / .pt)")
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
    parser.add_argument("--keep-fp32", action="store_true", help="also keep the float32 ONNX models")
    parser.add_argument("--no-validate", action="store_true", help="skip the PyTorch vs ONNX Runtime comparison")
    parser.add_argument(
        "--vjepa-root", type=Path, default=DEFAULT_VJEPA_ROOT, help="root of the official vjepa2 code"
    )
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
    if not (args.vjepa_root / "app" / "vjepa_2_1").is_dir():
        sys.exit(f"Official V-JEPA code not found in {args.vjepa_root} (see --vjepa-root)")
    if args.output_dir is None:
        args.output_dir = PROJECT_DIR / "models" / args.checkpoint.stem
    args.img_size = (args.img_size[0], args.img_size[-1])


def quiet_libraries():
    """Hide the deprecation warnings of the official code and the exporter (they are not about our model)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for name in ("torch.onnx", "torch.export", "torch._dynamo", "onnxscript"):
        logging.getLogger(name).setLevel(logging.ERROR)


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


def build_encoder(state_dict, arch, img_size, patch_size, tubelet_size, vjepa_root):
    sys.path.insert(0, str(vjepa_root))
    from app.vjepa_2_1.models import vision_transformer

    # Same settings as src/hub/backbones.py::_make_vjepa2_1_model. num_frames only has to be > 1
    # so that the video tokenizer is a Conv3d; the real number of frames comes from the input.
    encoder = vision_transformer.__dict__[arch](
        img_size=img_size,
        patch_size=patch_size,
        num_frames=64,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    encoder.load_state_dict(state_dict, strict=True)
    return encoder.eval().requires_grad_(False)


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


def convert_fp16(src, dst, num_params):
    model = onnx.load(str(src))
    # The exporter already wrote the type of every intermediate tensor (value_info), so the
    # converter does not need onnx.shape_inference, which fails above 2 GiB.
    model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    for name in (dst.name, dst.name + ".data"):
        (dst.parent / name).unlink(missing_ok=True)
    if num_params * 2 > PROTOBUF_LIMIT_BYTES:
        onnx.save_model(
            model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=dst.name + ".data"
        )
    else:
        onnx.save_model(model, str(dst))
    del model
    gc.collect()


def compare(reference, output):
    reference = reference.astype(np.float64)
    output = output.astype(np.float64)
    cosine = (reference * output).sum(-1) / (
        np.linalg.norm(reference, axis=-1) * np.linalg.norm(output, axis=-1) + 1e-12
    )
    return {
        "finite": bool(np.isfinite(output).all()),
        "max_abs_diff": float(np.abs(reference - output).max()),
        "mean_abs_diff": float(np.abs(reference - output).mean()),
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
    }


def run_onnx(path, inputs):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run([OUTPUT_NAME], {INPUT_NAME: inputs})[0]


def export_modality(modality, encoder, args, info):
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
    fp16_path = out_dir / f"encoder_{modality}_fp16.onnx"
    # The float32 model is written next to the float16 one if it is kept, else in a temporary folder.
    tmp_dir = out_dir / f".tmp_{modality}_fp32"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    fp32_path = (out_dir if args.keep_fp32 else tmp_dir) / f"encoder_{modality}_fp32.onnx"
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

    start = time.time()
    logger.info(f"[{modality}] Export to ONNX float32, input [batch, {', '.join(map(str, sample_shape))}] ...")
    export_fp32(module, example, fp32_path, args.opset, int(np.prod(token_grid)))
    logger.info(f"[{modality}] Exported in {time.time() - start:.1f} s")

    start = time.time()
    convert_fp16(fp32_path, fp16_path, info["num_params"])
    size = sum(p.stat().st_size for p in out_dir.glob(fp16_path.name + "*"))
    logger.info(f"[{modality}] Float16 model: {fp16_path} ({size / 1e6:,.0f} MB, {time.time() - start:.1f} s)")

    files = {"fp16": fp16_path.name}
    if args.keep_fp32:
        files["fp32"] = fp32_path.name
    shutil.rmtree(tmp_dir, ignore_errors=True)
    gc.collect()

    validation = {}
    if reference is not None:
        for precision, name in files.items():
            start = time.time()
            output = run_onnx(out_dir / name, check.numpy())
            validation[precision] = compare(reference, output)
            v = validation[precision]
            logger.info(
                f"[{modality}] ONNX {precision} vs PyTorch: max |diff| {v['max_abs_diff']:.2e}, "
                f"cosine mean {v['mean_cosine']:.6f} / min {v['min_cosine']:.6f} ({time.time() - start:.1f} s)"
            )
            if not v["finite"] or v["min_cosine"] < 0.99:
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

    encoder = build_encoder(state_dict, arch, args.img_size, patch_size, tubelet_size, args.vjepa_root)
    del checkpoint, state_dict
    gc.collect()
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
    models = {modality: export_modality(modality, encoder, args, info) for modality in args.modalities}

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
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"Done: {config_path}")


if __name__ == "__main__":
    main()
