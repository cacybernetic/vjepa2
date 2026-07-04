# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Export the V-JEPA 2.1 encoder to ONNX.
#
# Takes a training-style ``.pt`` checkpoint (e.g.
# ``weights/vjepa2_1_vitb_dist_vitG_384.pt``) and writes a self-contained
# ``.onnx`` graph of the ViT encoder that maps a clip tensor to dense features.
#
#   exportencoder weights/vjepa2_1_vitb_dist_vitG_384.pt -o encoder.onnx
#
# Production notes (read before deploying the exported graph):
#   * Fixed geometry. The number of frames (T) and the spatial resolution
#     (H, W) are BAKED IN at export time. Feeding a different clip length or
#     resolution raises a shape error in onnxruntime. Re-export per geometry.
#     Only the batch dimension can be made dynamic (``--dynamic-batch``).
#   * One modality per export. ``T == 1`` traces the image pathway (2D-style
#     tokenizer + image modality token); ``T > 1`` traces the video pathway.
#     A single ``.onnx`` therefore serves images OR videos, not both.
#   * Preprocessing. By default the graph expects an already-normalized clip
#     (ImageNet mean/std over pixels in [0, 1], layout ``NCTHW``). Pass
#     ``--bake-normalization`` to fold the ImageNet normalization into the graph
#     so it accepts raw RGB clips in [0, 255] instead.
#   * Weights are stored in a sidecar ``<output>.data`` file unless
#     ``--single-file`` is given; ship both files together.
#   * The exported model embeds provenance/config in its ``metadata_props``
#     (checkpoint hash, geometry, normalization, layout, ...).

import argparse
import hashlib
import json
import logging
import os
import sys

import torch
import torch.nn as nn

from vjepa2.model import build_vjepa2_1_vitb
from vjepa2.modules.vision_transformer import VIT_EMBED_DIMS

logger = logging.getLogger(__name__)

# ImageNet statistics (pixels in [0, 1]); matches ``inference.Preprocess``.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _EncoderForExport(nn.Module):
    """Trace-friendly wrapper: a single tensor in, dense features out.

    Pins ``masks=None`` and ``training=False`` so the exported graph reproduces
    :meth:`VJEPA21.extract_features` — the last-layer, LayerNorm-ed patch
    features ``(B, num_tokens, embed_dim)``.

    When ``normalize`` is set, the ImageNet normalization is applied inside the
    graph: the input is then a raw RGB clip in ``[0, 255]`` rather than an
    already-normalized tensor.
    """

    def __init__(self, backbone, normalize=False):
        super().__init__()
        self.backbone = backbone
        self.normalize = normalize
        if normalize:
            self.register_buffer(
                "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
            )

    def forward(self, clip):
        if self.normalize:
            clip = clip / 255.0
            clip = (clip - self.mean) / self.std
        return self.backbone(clip, masks=None, training=False)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find(state_dict, suffix):
    """First tensor whose key ends with ``suffix`` (prefix-agnostic)."""
    for k, v in state_dict.items():
        if k.endswith(suffix):
            return v
    return None


def _encoder_state_dict(ckpt):
    """The encoder sub-state-dict to validate against (shapes are shared)."""
    for key in ("encoder", "ema_encoder", "target_encoder"):
        if key in ckpt:
            return ckpt[key]
    return None


def validate_geometry(ckpt, args):
    """Fail fast if CLI geometry contradicts the checkpoint tensor shapes.

    ``load_pretrained`` uses ``strict=False``, so a wrong ``--patch-size`` /
    ``--tubelet-size`` / ``--model-name`` would otherwise load partially and
    silently produce a wrong graph. We check the tokenizer conv and embed dim.
    """
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint is not a state-dict container")
    sd = _encoder_state_dict(ckpt)
    if sd is None:
        raise ValueError(
            "checkpoint has no 'encoder'/'ema_encoder' state dict "
            f"(keys: {list(ckpt.keys())})"
        )

    errors = []
    w = _find(sd, "patch_embed.proj.weight")  # video conv3d: (E, C, tub, ph, pw)
    if w is not None and w.ndim == 5:
        embed_dim, _, tub, ph, pw = w.shape
        if ph != args.patch_size or pw != args.patch_size:
            errors.append(
                f"--patch-size {args.patch_size} != checkpoint patch {ph}x{pw}"
            )
        if tub != args.tubelet_size:
            errors.append(
                f"--tubelet-size {args.tubelet_size} != checkpoint tubelet {tub}"
            )
        expected_embed = VIT_EMBED_DIMS.get(args.model_name)
        if expected_embed is not None and embed_dim != expected_embed:
            errors.append(
                f"--model-name {args.model_name} (embed {expected_embed}) != "
                f"checkpoint embed {embed_dim}"
            )
    else:
        logger.warning(
            "could not locate patch_embed weights; skipping geometry validation"
        )

    if errors:
        raise ValueError("checkpoint/CLI geometry mismatch: " + "; ".join(errors))


def _build_wrapper(ckpt, args):
    """Build the encoder, load the requested weights and wrap it for export."""
    overrides = dict(
        model_name=args.model_name,
        crop_size=args.crop_size,
        max_num_frames=args.num_frames if args.num_frames > 1 else 16,
        tubelet_size=args.tubelet_size,
        patch_size=args.patch_size,
        use_sdpa=args.sdpa,
    )
    model = build_vjepa2_1_vitb(checkpoint=ckpt, device="cpu", **overrides)
    model.eval()

    wrapper = model.target_encoder if args.weights == "ema" else model.encoder
    return _EncoderForExport(
        wrapper.backbone, normalize=args.bake_normalization
    ).eval()


def _dummy_input(args, batch=None):
    """Reference clip: ``(B, C, T, H, W)``; ``T == 1`` selects the image path."""
    b = args.batch_size if batch is None else batch
    t = args.num_frames
    clip = torch.randn(b, 3, t, args.crop_size, args.crop_size)
    if args.bake_normalization:
        # Graph expects raw [0, 255] RGB when normalization is baked in.
        clip = clip.mul(64).add(128).clamp(0, 255)
    return clip


def sanity_forward(wrapper, dummy):
    """Run the PyTorch wrapper once; catches geometry errors before export."""
    with torch.no_grad():
        out = wrapper(dummy)
    if not torch.isfinite(out).all():
        raise RuntimeError("PyTorch encoder produced non-finite features")
    logger.info("PyTorch sanity forward OK: output %s", tuple(out.shape))
    return out


def _export(wrapper, dummy, args):
    # The RoPE encoder trips the legacy TorchScript exporter (opset9 ``cat``
    # assertion), so we drive the torch.export-based ("dynamo") exporter, which
    # becomes the PyTorch default in 2.9 and lowers these ops cleanly.
    dynamic_shapes = None
    if args.dynamic_batch:
        from torch.export import Dim

        dynamic_shapes = {"clip": {0: Dim("batch")}}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            args.output,
            input_names=["clip"],
            output_names=["features"],
            opset_version=args.opset,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            optimize=True,
            external_data=not args.single_file,
        )
    logger.info("Wrote ONNX graph to %s", args.output)
    data = args.output + ".data"
    if os.path.exists(data):
        logger.info("Weights stored as external data in %s", data)


def _metadata(args, dummy):
    """Provenance / config to embed in the ONNX ``metadata_props``."""
    return {
        "producer": "vjepa2.entrypoints.exportencoder",
        "model_name": args.model_name,
        "weights": args.weights,
        "modality": "image" if args.num_frames == 1 else "video",
        "input_layout": "NCTHW",
        "input_shape": json.dumps(list(dummy.shape)),
        "dynamic_batch": str(args.dynamic_batch),
        "crop_size": str(args.crop_size),
        "num_frames": str(args.num_frames),
        "patch_size": str(args.patch_size),
        "tubelet_size": str(args.tubelet_size),
        "output": "last-layer LayerNorm dense features (B, num_tokens, embed_dim)",
        "normalization": (
            "baked:imagenet,input_range=[0,255]"
            if args.bake_normalization
            else "external:imagenet,input_range=[0,1]"
        ),
        "imagenet_mean": json.dumps(IMAGENET_MEAN),
        "imagenet_std": json.dumps(IMAGENET_STD),
        "precision": "fp16" if args.fp16 else "fp32",
        "opset": str(args.opset),
        "checkpoint": os.path.basename(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "torch_version": torch.__version__,
    }


def _save(model, args):
    import onnx

    if args.single_file:
        onnx.save_model(model, args.output, save_as_external_data=False)
    else:
        onnx.save_model(
            model,
            args.output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(args.output) + ".data",
            size_threshold=1024,
        )


def add_metadata(args, dummy):
    """Load the exported graph, attach ``metadata_props`` and re-save."""
    import onnx

    model = onnx.load(args.output)
    meta = _metadata(args, dummy)
    del model.metadata_props[:]
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, str(v)
    _save(model, args)
    logger.info("Embedded %d metadata_props (checkpoint_sha256, geometry, ...)", len(meta))


def to_fp16(args):
    """Convert the exported graph to fp16 (keeping fp32 I/O for the client)."""
    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            "fp16 conversion needs 'onnxconverter-common' "
            "(pip install onnxconverter-common)"
        ) from exc
    import onnx

    model = onnx.load(args.output)
    model = float16.convert_float_to_float16(model, keep_io_types=True)
    _save(model, args)
    logger.info("Converted weights to fp16 (I/O kept fp32)")


def _check(args):
    import onnx

    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    logger.info("onnx.checker: model is well-formed")


def _verify(wrapper, dummy, args):
    """Compare PyTorch vs onnxruntime outputs on the reference input."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping numerical verification")
        return

    import numpy as np

    with torch.no_grad():
        ref = wrapper(dummy).cpu().numpy()

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"clip": dummy.cpu().numpy()})

    if out.shape != ref.shape:
        raise RuntimeError(f"shape mismatch: torch {ref.shape} vs onnx {out.shape}")

    max_abs = float(np.abs(out - ref).max())
    # fp16 internals diverge more from fp32 torch; relax the tolerance.
    tol = max(args.tol, 1e-1) if args.fp16 else args.tol
    logger.info("onnxruntime match: shape %s, max |Δ| = %.3e (tol %.1e)", out.shape, max_abs, tol)
    if max_abs > tol:
        raise RuntimeError(f"outputs differ by {max_abs:.3e} > tol {tol:.3e}")

    # Under a dynamic batch, prove the graph actually accepts B != traced size.
    if args.dynamic_batch:
        dummy2 = _dummy_input(args, batch=args.batch_size + 1)
        (out2,) = sess.run(None, {"clip": dummy2.cpu().numpy()})
        if out2.shape[0] != args.batch_size + 1:
            raise RuntimeError(
                f"dynamic batch broken: fed {args.batch_size + 1}, got {out2.shape[0]}"
            )
        logger.info("dynamic-batch check OK: ran B=%d -> %s", args.batch_size + 1, out2.shape)


def build_parser():
    p = argparse.ArgumentParser(
        prog="exportencoder",
        description="Export the V-JEPA 2.1 ViT encoder to ONNX.",
    )
    p.add_argument("checkpoint", help="path to the .pt checkpoint to convert")
    p.add_argument(
        "-o",
        "--output",
        help="output .onnx path (default: <checkpoint>.onnx)",
    )
    p.add_argument(
        "--weights",
        choices=["ema", "online"],
        default="ema",
        help="which encoder to export: the EMA/target encoder (default) or the "
        "online context encoder",
    )
    # -- model geometry (defaults match vjepa2_1_vitb_dist_vitG_384) ------------
    p.add_argument("--model-name", default="vit_base")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="frames per clip in the reference input (BAKED IN); use 1 to export "
        "the image pathway. One export serves images OR videos, not both.",
    )
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    # -- export knobs ----------------------------------------------------------
    p.add_argument("--opset", type=int, default=18)
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        default=False,
        help="mark the batch dimension as dynamic so one graph serves any batch "
        "size. Automatically enables --sdpa, which is required for the batched "
        "attention to lower to ONNX. (Frames and resolution stay fixed.)",
    )
    p.add_argument(
        "--sdpa",
        action="store_true",
        help="trace attention with fused scaled-dot-product-attention. Required "
        "for --dynamic-batch (and auto-enabled by it); the non-fused matmul path "
        "fails to decompose for ONNX under a symbolic batch dimension.",
    )
    p.add_argument(
        "--bake-normalization",
        action="store_true",
        help="fold ImageNet normalization into the graph; the exported model "
        "then takes raw RGB clips in [0, 255] instead of pre-normalized tensors.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="convert weights to fp16 for cheaper inference (I/O stays fp32); "
        "requires the 'onnxconverter-common' package.",
    )
    p.add_argument(
        "--single-file",
        action="store_true",
        help="embed weights in the .onnx file instead of a sidecar .onnx.data "
        "(recommended for deployment: one artifact to ship)",
    )
    p.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the onnxruntime numerical cross-check",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="max allowed |torch - onnx| during verification (deep fp32 "
        "transformers differ ~1e-3 across backends)",
    )
    return p


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.checkpoint):
        logger.error("checkpoint not found: %s", args.checkpoint)
        return 1
    if args.output is None:
        base = os.path.splitext(args.checkpoint)[0]
        args.output = base + ".onnx"

    # Dynamic batch only lowers to ONNX with fused SDPA attention.
    if args.dynamic_batch and not args.sdpa:
        logger.warning(
            "enabling --sdpa: required for --dynamic-batch to decompose to ONNX"
        )
        args.sdpa = True

    logger.info("Loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    try:
        validate_geometry(ckpt, args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Building encoder (%s) and loading '%s' weights",
        args.model_name,
        args.weights,
    )
    wrapper = _build_wrapper(ckpt, args)
    dummy = _dummy_input(args)

    sanity_forward(wrapper, dummy)

    logger.info("Tracing input %s -> ONNX (opset %d)", tuple(dummy.shape), args.opset)
    _export(wrapper, dummy, args)
    if args.fp16:
        try:
            to_fp16(args)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
    add_metadata(args, dummy)
    _check(args)
    if args.verify:
        _verify(wrapper, dummy, args)

    logger.info("Done: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
