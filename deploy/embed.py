#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""embed.py - step 1, program 2 of 2: a folder of images -> V-JEPA 2.1 embeddings (.pkl), ONNX Runtime only.

What is this program for?
=========================

This program looks for every image file inside a folder (and inside all of
its sub-folders), and computes the V-JEPA 2.1 embedding of each image with
the ONNX models made by quantization.py, in any of its formats
(--precision): q4_k_m (INT4 Q4_K_M weights), fp16 or fp32. It does not use
PyTorch: only ONNX Runtime, NumPy and OpenCV.

Each embedding is saved next to its image, with the same name and the
.pkl extension. The image file itself is kept:

    /path/to/photos/myimage.png        ->  /path/to/photos/myimage.pkl
    /path/to/photos/cats/cat_01.jpg    ->  /path/to/photos/cats/cat_01.pkl

Only images are read (.jpg .jpeg .png .bmp .webp .tif .tiff); other files,
videos included, are ignored. Each .pkl holds a dict, readable with pickle
and NumPy only:

    "embedding"      float32 [D]        mean of the patch tokens: one global vector for the image
    "patch_tokens"   float16 [h, w, D]  dense features, one vector per 16x16 patch (not saved with --no-patch-tokens)
    "source", "modality", "encoder", "arch", "precision", "image_size"

"precision" tells which model format computed the embedding (q4_k_m, fp16 or fp32).

D is 768 (ViT-B), 1024 (ViT-L), 1408 (ViT-g) or 1664 (ViT-G); h = H/16 and w = W/16 for a model input of
H x W pixels (24 x 24 at 384 x 384).

An image that already has its .pkl is not computed again when this file
loads fine with pickle and holds an "embedding" of shape [D] (D = embed_dim
of config.json). Otherwise (missing, broken or truncated file, wrong shape,
for example made by another model size), the embedding is computed again
and the .pkl is replaced. This check is always done before any computation,
so an interrupted run can simply be started again.

Two images of the same folder with the same name and different extensions
(photo.jpg and photo.png) would both give photo.pkl. In that case only the
first one, in alphabetical order, is embedded, and the others are skipped
with a warning.

The two programs of step 1
--------------------------

    1. quantization.py  checkpoint -> <model-dir>/{*.onnx, config.json}
    2. embed.py         (this program) folder of images -> one .pkl next to each image

The programs never import each other: embed.py only reads the files of
<model-dir> (config.json gives the model files of each format, input shapes
and normalization). Only the image model is needed (quantization.py
--modalities image).

Model formats (--precision)
===========================

All are ONNX models read by ONNX Runtime, with float32 input and output:

q4_k_m
    The weights of the linear layers in 4 bits (6 bits for the most
    sensitive ones), quantized with the Q4_K_M recipe of llama.cpp and stored
    in ONNX Runtime's MatMulNBits operator. 3 to 4 times smaller than
    float32; the embeddings lose a little precision (ViT-B: cosine similarity
    about 0.998 with those of PyTorch).
fp16
    All the weights and computations in float16: half the size of float32,
    the same embeddings to about 1e-5.
fp32
    No compression (quantization.py --precision fp32).
auto (default)
    The first of q4_k_m, fp16, fp32 found in config.json.

Important words
===============

Patch tokens
    The encoder cuts the image into patches of 16 x 16 pixels and returns one
    vector per patch: the output of its last block after LayerNorm. V-JEPA
    2.1 is trained so that these dense features are good (segmentation,
    depth, tracking). The "embedding" is their mean.

Preprocessing
    The same as the evaluation transform of the official code
    (evals/video_classification_frozen/utils.py, VideoTransform): resize so
    that the image covers int(size * 256 / 224) pixels (bilinear, OpenCV),
    crop the center to the model size, scale to [0, 1], normalize with the
    ImageNet mean and std. The center crop cuts the borders of non-square
    images; --crop-ratio 1.0 keeps as much of them as possible.

Execution provider
    The ONNX Runtime backend: CUDA if onnxruntime-gpu and a GPU are there
    (float16 is fast), else CPU. On CPU, ONNX Runtime computes the float16
    model in float32 with extra casts, and expands the 4-bit weights of the
    q4_k_m model to float32 before each product: q4_k_m runs at about the
    speed of fp32, fp16 is slower.

Usage
=====

    python embed.py --model-dir models/vjepa2_1_vitl_dist_vitG_384 /path/to/photos
    python embed.py --model-dir models/vjepa2_1_vitl_dist_vitG_384 --precision fp16 /path/to/photos

    # read a result
    import pickle
    record = pickle.load(open("/path/to/photos/myimage.pkl", "rb"))
    record["embedding"].shape, record["patch_tokens"].shape   # (1024,), (24, 24, 1024)

Requirements (no PyTorch): onnxruntime (or onnxruntime-gpu), numpy, opencv-python, loguru, tqdm
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Model formats of quantization.py, in the order --precision auto looks for them.
PRECISIONS = ("q4_k_m", "fp16", "fp32")

# Official evaluation transform: resize to int(size * 256 / 224), then center crop to size.
EVAL_CROP_RATIO = 224 / 256


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show the defaults, except None (required values, or values computed later)."""

    def _get_help_string(self, action):
        return action.help if action.default is None else super()._get_help_string(action)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the V-JEPA 2.1 embedding of every image of a folder with ONNX Runtime (no PyTorch). "
        "Each embedding is saved next to its image: /path/to/myimage.png -> /path/to/myimage.pkl.",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "image_dir", type=Path, help="folder of images, searched with all its sub-folders (the images are kept)"
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="output folder of quantization.py (config.json + .onnx files)"
    )
    parser.add_argument(
        "--precision",
        choices=("auto", *PRECISIONS),
        default="auto",
        help=f"model format to run (auto: the first of {', '.join(PRECISIONS)} found in config.json)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="images per run")
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=EVAL_CROP_RATIO,
        help="model size / resized size before the center crop (1.0 = keep as much border as possible)",
    )
    parser.add_argument("--no-patch-tokens", action="store_true", help="save only the mean embedding (much smaller)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="ONNX Runtime backend")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads of ONNX Runtime (0 = its default)")
    args = parser.parse_args()
    if not 0 < args.crop_ratio <= 1:
        parser.error("--crop-ratio must be in ]0, 1]")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def load_config(model_dir):
    path = model_dir / "config.json"
    if not path.is_file():
        sys.exit(f"{path} not found: --model-dir must be an output folder of quantization.py")
    config = json.loads(path.read_text())
    if config.get("format") != "vjepa2.1-onnx":
        sys.exit(f"{path} was not written by quantization.py")
    return config


def choose_providers(device):
    available = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        sys.exit(f"--device cuda, but ONNX Runtime has no CUDA backend ({available}); install onnxruntime-gpu")
    return ["CPUExecutionProvider"]


class Encoder:
    """One ONNX model of quantization.py (image or video, in one format) in an ONNX Runtime session."""

    def __init__(self, model_dir, config, modality, precision, providers, threads):
        spec = config["models"].get(modality)
        if spec is None:
            sys.exit(f"{model_dir} has no {modality} model: run quantization.py with --modalities {modality}")
        available = [p for p in PRECISIONS if p in spec["files"]]
        if precision == "auto" and available:
            precision = available[0]
        if precision not in available:
            sys.exit(
                f"{model_dir} has no {precision} {modality} model (available: {', '.join(available) or 'none'}): "
                f"run quantization.py with --precision {precision}"
            )
        file = spec["files"][precision]
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        logger.info(f"Loading {model_dir / file} ({precision}, {', '.join(providers)})")
        self.session = ort.InferenceSession(str(model_dir / file), options, providers=providers)
        self.input = spec["input"]
        self.output = spec["output"]
        self.token_grid = spec["token_grid"]  # [t, h, w]
        self.name = modality
        self.precision = precision

    def __call__(self, batch):
        """float32 [B, 3, H, W] or [B, 3, T, H, W] -> float32 patch tokens [B, t, h, w, D]."""
        tokens = self.session.run([self.output], {self.input: batch})[0]
        return tokens.reshape(len(batch), *self.token_grid, -1)


class Preprocessor:
    """uint8 RGB [h, w, 3] -> float32 [3, H, W], as the official evaluation transform."""

    def __init__(self, image_size, crop_ratio, mean, std):
        self.height, self.width = image_size
        self.cover_height = int(self.height / crop_ratio)
        self.cover_width = int(self.width / crop_ratio)
        # (x / 255 - mean) / std == (x - 255 mean) / (255 std)
        self.mean = np.asarray(mean, np.float32) * 255
        self.std = np.asarray(std, np.float32) * 255

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        # Smallest size that covers cover_height x cover_width with the same aspect ratio. For a square
        # model input this is "short side = int(size * 256 / 224)" of the official Resize.
        if w * self.cover_height >= h * self.cover_width:
            new_h, new_w = self.cover_height, self.cover_height * w // h
        else:
            new_h, new_w = self.cover_width * h // w, self.cover_width
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = int(round((new_h - self.height) / 2.0))
        left = int(round((new_w - self.width) / 2.0))
        crop = rgb[top : top + self.height, left : left + self.width].astype(np.float32)
        return ((crop - self.mean) / self.std).transpose(2, 0, 1)


def read_image(path):
    # imdecode instead of imread: works with any file name (non-ASCII paths).
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode {path} as an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_batches(items, encoder, batch_size):
    """Encode (array, meta) items by batches; yield (patch tokens [t, h, w, D], meta) in input order."""
    arrays, metas = [], []
    for array, meta in items:
        arrays.append(array)
        metas.append(meta)
        if len(arrays) == batch_size:
            yield from zip(encoder(np.stack(arrays)), metas)
            arrays, metas = [], []
    if arrays:
        yield from zip(encoder(np.stack(arrays)), metas)


class Writer:
    """Write one .pkl per image."""

    def __init__(self, config, encoder, keep_patch_tokens):
        self.keep_patch_tokens = keep_patch_tokens
        self.model_info = {
            "encoder": encoder.name,
            "arch": config["arch"],
            "precision": encoder.precision,
            "image_size": config["image_size"],
        }
        self.count = 0

    def __call__(self, path, tokens, info):
        """tokens: float32 patch tokens [h, w, D] of one image."""
        record = {**info, "embedding": tokens.mean(axis=(0, 1), dtype=np.float32)}
        if self.keep_patch_tokens:
            record["patch_tokens"] = tokens.astype(np.float16)
        record.update(self.model_info)
        with open(path, "wb") as file:
            pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.count += 1


def find_images(image_dir):
    """Full paths of the image files of image_dir and of all its sub-folders, sorted."""
    paths = sorted(
        path.resolve()
        for path in image_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )
    # photo.jpg and photo.png of one folder would both write photo.pkl: keep the first one.
    kept, outputs = [], {}
    for path in paths:
        output = path.with_suffix(".pkl")
        if output in outputs:
            logger.warning(f"{path} skipped: {outputs[output].name} already gives {output.name}")
            continue
        outputs[output] = path
        kept.append(path)
    return kept


def has_embedding(path, embed_dim):
    """True if path is a .pkl that loads fine and holds an "embedding" of shape [embed_dim]."""
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as file:
            record = pickle.load(file)
    except Exception as error:  # truncated or broken file: any unpickling error means "compute it again"
        logger.warning(f"{path} does not load ({type(error).__name__}): computed again")
        return False
    shape = getattr(record.get("embedding") if isinstance(record, dict) else None, "shape", None)
    if shape != (embed_dim,):
        logger.warning(f"{path} has an embedding of shape {shape}, not ({embed_dim},): computed again")
        return False
    return True


def embed_images(paths, encoder, preprocess, writer, batch_size):
    def items():
        for path in paths:
            try:
                yield preprocess(read_image(path)), path
            except ValueError as error:
                logger.warning(f"{error}: skipped")

    for tokens, path in tqdm(run_batches(items(), encoder, batch_size), total=len(paths), unit="image"):
        info = {"source": str(path), "modality": "image"}
        # The embedding goes next to the image, with the same name: /path/to/myimage.png -> /path/to/myimage.pkl.
        writer(path.with_suffix(".pkl"), tokens[0], info)


def main():
    args = parse_args()
    logger.remove()
    logger.add(
        lambda message: tqdm.write(str(message), file=sys.stderr, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )

    if not args.image_dir.is_dir():
        sys.exit(f"Image folder not found: {args.image_dir}")
    images = find_images(args.image_dir)
    if not images:
        sys.exit(f"No image file ({' '.join(sorted(IMAGE_EXTENSIONS))}) in {args.image_dir}")
    logger.info(f"{len(images)} image(s) found in {args.image_dir.resolve()}")

    config = load_config(args.model_dir)
    # An image whose .pkl already loads fine with an embedding of the good shape is not computed again.
    todo = [path for path in images if not has_embedding(path.with_suffix(".pkl"), config["embed_dim"])]
    logger.info(f"{len(images) - len(todo)} image(s) already have a valid embedding, {len(todo)} to compute")
    if not todo:
        logger.info("Done: nothing to compute")
        return
    images = todo

    providers = choose_providers(args.device)
    norm = config["normalization"]
    preprocess = Preprocessor(config["image_size"], args.crop_ratio, norm["mean"], norm["std"])

    encoder = Encoder(args.model_dir, config, "image", args.precision, providers, args.threads)
    writer = Writer(config, encoder, not args.no_patch_tokens)
    embed_images(images, encoder, preprocess, writer, args.batch_size)
    logger.info(f"Done: {writer.count} embedding file(s), each next to its image in {args.image_dir.resolve()}")


if __name__ == "__main__":
    main()
