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

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

from vjepa2.model import build_vjepa2_1_vitb

logger = logging.getLogger(__name__)


class _EncoderForExport(nn.Module):
    """Trace-friendly wrapper: a single tensor in, dense features out.

    Pins ``masks=None`` and ``training=False`` so the exported graph reproduces
    :meth:`VJEPA21.extract_features` — the last-layer, LayerNorm-ed patch
    features ``(B, num_tokens, embed_dim)``.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, clip):
        return self.backbone(clip, masks=None, training=False)


def _build_wrapper(args):
    """Build the encoder, load the requested weights and wrap it for export."""
    overrides = dict(
        model_name=args.model_name,
        crop_size=args.crop_size,
        max_num_frames=args.num_frames if args.num_frames > 1 else 16,
        tubelet_size=args.tubelet_size,
        patch_size=args.patch_size,
        use_sdpa=args.sdpa,
    )
    model = build_vjepa2_1_vitb(checkpoint=args.checkpoint, device="cpu", **overrides)
    model.eval()

    wrapper = model.target_encoder if args.weights == "ema" else model.encoder
    return _EncoderForExport(wrapper.backbone).eval()


def _dummy_input(args):
    """Reference clip: ``(B, C, T, H, W)``; ``T == 1`` selects the image path."""
    t = args.num_frames
    return torch.randn(args.batch_size, 3, t, args.crop_size, args.crop_size)


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

    with torch.no_grad():
        ref = wrapper(dummy).cpu().numpy()

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"clip": dummy.cpu().numpy()})

    if out.shape != ref.shape:
        raise RuntimeError(f"shape mismatch: torch {ref.shape} vs onnx {out.shape}")
    import numpy as np

    max_abs = float(np.abs(out - ref).max())
    logger.info("onnxruntime match: shape %s, max |Δ| = %.3e", out.shape, max_abs)
    if max_abs > args.tol:
        raise RuntimeError(f"outputs differ by {max_abs:.3e} > tol {args.tol:.3e}")


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
        help="frames per clip in the reference input; use 1 to export the image "
        "pathway",
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
        help="mark the batch dimension as dynamic (experimental: the batched "
        "attention matmul may fail to decompose for ONNX; re-export per batch "
        "size if so)",
    )
    p.add_argument(
        "--sdpa",
        action="store_true",
        help="trace with fused scaled-dot-product-attention; off by default "
        "because its decomposition breaks ONNX export under a dynamic batch",
    )
    p.add_argument(
        "--single-file",
        action="store_true",
        help="embed weights in the .onnx file instead of a sidecar .onnx.data",
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

    logger.info(
        "Building encoder (%s) and loading '%s' weights",
        args.model_name,
        args.weights,
    )
    wrapper = _build_wrapper(args)
    dummy = _dummy_input(args)

    logger.info("Tracing input %s -> ONNX (opset %d)", tuple(dummy.shape), args.opset)
    _export(wrapper, dummy, args)
    _check(args)
    if args.verify:
        _verify(wrapper, dummy, args)

    logger.info("Done: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
