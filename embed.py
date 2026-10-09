#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""embed.py - step 1, program 2 of 2: images / videos -> V-JEPA 2.1 embeddings (.pkl), ONNX Runtime only.

What is this program for?
=========================

This program reads image and video files and computes their V-JEPA 2.1
embeddings with the ONNX models made by quantize.py. It does not use
PyTorch: only ONNX Runtime, NumPy and OpenCV.

    image  photo.jpg  -> <output-dir>/photo.pkl
    video  clip.mp4   -> <output-dir>/clip/frame_000000.pkl ... frame_001289.pkl

A video gives one file per decoded frame: a video of 1290 frames gives 1290
.pkl files. Each .pkl holds a dict, readable with pickle and NumPy only:

    "embedding"      float32 [D]        mean of the patch tokens: one global vector for the image/frame
    "patch_tokens"   float16 [h, w, D]  dense features, one vector per 16x16 patch (not saved with --no-patch-tokens)
    "source", "modality", "frame_index", "timestamp_sec", "encoder", "arch", "precision", "image_size", ...

D is 768 (ViT-B), 1024 (ViT-L), 1408 (ViT-g) or 1664 (ViT-G); h = H/16 and w = W/16 for a model input of
H x W pixels (24 x 24 at 384 x 384).

The two programs of step 1
--------------------------

    1. quantize.py  checkpoint -> <model-dir>/{*.onnx, config.json}
    2. embed.py     (this program) image/video files -> one .pkl per image/frame

The programs never import each other: embed.py only reads the files of
<model-dir> (config.json gives the model files, input shapes and
normalization).

Video modes (--video-mode)
==========================

frame (default)
    Every frame is encoded alone by the image model (2D tokenizer), like a
    photo. N frames give N different embeddings.

clip
    The frames are grouped in clips of T frames (T = num_frames of
    config.json, 16 by default), read by the video model (3D tokenizer). The
    features of a frame then see the other frames of its clip (motion,
    temporal consistency). The video tokenizer reads "tubelets" of 2 frames,
    so frames 2k and 2k+1 of a clip share the same features: they get the
    same embedding. The last clip is completed by repeating its last frame.
    This mode needs more memory: attention over T/2 * h * w tokens.

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
    model in float32 with extra casts: the model made with
    quantize.py --keep-fp32 and --precision fp32 is faster.

Usage
=====

    python embed.py --model-dir models/vjepa2_1_vitl_dist_vitG_384 photo.jpg clip.mp4
    python embed.py --model-dir models/vjepa2_1_vitl_dist_vitG_384 --video-mode clip --output-dir embeddings clip.mp4

    # read a result
    import pickle
    record = pickle.load(open("embeddings/clip/frame_000017.pkl", "rb"))
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

# Official evaluation transform: resize to int(size * 256 / 224), then center crop to size.
EVAL_CROP_RATIO = 224 / 256


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show the defaults, except None (required values, or values computed later)."""

    def _get_help_string(self, action):
        return action.help if action.default is None else super()._get_help_string(action)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute V-JEPA 2.1 embeddings of images and videos with ONNX Runtime (no PyTorch).",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="image and/or video files")
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="output folder of quantize.py (config.json + .onnx files)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"), help="where to write the .pkl files")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16", help="which ONNX model to run")
    parser.add_argument(
        "--video-mode",
        choices=("frame", "clip"),
        default="frame",
        help="frame: each frame alone (image model); clip: clips of frames (video model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="images or frames per run (default 8); in clip mode, clips per run (default 1)",
    )
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
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def load_config(model_dir):
    path = model_dir / "config.json"
    if not path.is_file():
        sys.exit(f"{path} not found: --model-dir must be an output folder of quantize.py")
    config = json.loads(path.read_text())
    if config.get("format") != "vjepa2.1-onnx":
        sys.exit(f"{path} was not written by quantize.py")
    return config


def choose_providers(device):
    available = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        sys.exit(f"--device cuda, but ONNX Runtime has no CUDA backend ({available}); install onnxruntime-gpu")
    return ["CPUExecutionProvider"]


class Encoder:
    """One ONNX model of quantize.py (image or video) in an ONNX Runtime session."""

    def __init__(self, model_dir, config, modality, precision, providers, threads):
        spec = config["models"].get(modality)
        if spec is None:
            sys.exit(f"{model_dir} has no {modality} model: run quantize.py with --modalities {modality}")
        file = spec["files"].get(precision)
        if file is None:
            sys.exit(f"{model_dir} has no {precision} {modality} model (float32 needs quantize.py --keep-fp32)")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        logger.info(f"Loading {model_dir / file} ({', '.join(providers)})")
        self.session = ort.InferenceSession(str(model_dir / file), options, providers=providers)
        self.input = spec["input"]
        self.output = spec["output"]
        self.token_grid = spec["token_grid"]  # [t, h, w]
        self.name = modality

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


def iter_frames(capture):
    """Yield (frame index, uint8 RGB frame) for every frame of an opened cv2.VideoCapture."""
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            return
        yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index += 1


def iter_clips(frames, num_frames):
    """Group ([3, H, W] frame, index) items into ([3, T, H, W] clip, frame indices); the last clip is padded."""
    clip, indices = [], []
    for frame, index in frames:
        clip.append(frame)
        indices.append(index)
        if len(clip) == num_frames:
            yield np.stack(clip, axis=1), indices
            clip, indices = [], []
    if clip:
        clip += [clip[-1]] * (num_frames - len(clip))
        yield np.stack(clip, axis=1), indices


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
    """Write one .pkl per image/frame."""

    def __init__(self, config, encoder, precision, keep_patch_tokens):
        self.keep_patch_tokens = keep_patch_tokens
        self.model_info = {
            "encoder": encoder.name,
            "arch": config["arch"],
            "precision": precision,
            "image_size": config["image_size"],
        }
        self.count = 0

    def __call__(self, path, tokens, info):
        """tokens: float32 patch tokens [h, w, D] of one image/frame."""
        record = {**info, "embedding": tokens.mean(axis=(0, 1), dtype=np.float32)}
        if self.keep_patch_tokens:
            record["patch_tokens"] = tokens.astype(np.float16)
        record.update(self.model_info)
        with open(path, "wb") as file:
            pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.count += 1


def embed_images(paths, encoder, preprocess, writer, output_dir, batch_size):
    def items():
        for path in paths:
            try:
                yield preprocess(read_image(path)), path
            except ValueError as error:
                logger.warning(f"{error}: skipped")

    for tokens, path in tqdm(run_batches(items(), encoder, batch_size), total=len(paths), unit="image"):
        info = {"source": str(path.resolve()), "modality": "image"}
        writer(output_dir / f"{path.stem}.pkl", tokens[0], info)


def embed_video(path, encoder, preprocess, writer, output_dir, batch_size, mode, tubelet_size):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        logger.warning(f"cannot open {path} as a video: skipped")
        return
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dir = output_dir / path.stem
    video_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{path}: ~{total} frames at {fps:.2f} fps -> {video_dir}/ ({mode} mode)")

    progress = tqdm(total=total if total > 0 else None, unit="frame")
    frames = ((preprocess(rgb), index) for index, rgb in iter_frames(capture))

    def save(index, tokens, extra):
        info = {
            "source": str(path.resolve()),
            "modality": "video",
            "frame_index": index,
            "timestamp_sec": index / fps if fps > 0 else None,
            "fps": fps,
            **extra,
        }
        writer(video_dir / f"frame_{index:06d}.pkl", tokens, info)
        progress.update(1)

    if mode == "frame":
        for tokens, index in run_batches(frames, encoder, batch_size):
            save(index, tokens[0], {"video_mode": "frame"})
    else:
        clips = iter_clips(frames, encoder.token_grid[0] * tubelet_size)
        for tokens, indices in run_batches(clips, encoder, batch_size):
            for position, index in enumerate(indices):
                tubelet = position // tubelet_size
                extra = {"video_mode": "clip", "clip_start_frame": indices[0], "tubelet_index": tubelet}
                save(index, tokens[tubelet], extra)
    progress.close()
    capture.release()


def main():
    args = parse_args()
    logger.remove()
    logger.add(
        lambda message: tqdm.write(str(message), file=sys.stderr, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )

    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        sys.exit(f"Input file(s) not found: {', '.join(missing)}")
    images = [path for path in args.inputs if path.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [path for path in args.inputs if path.suffix.lower() not in IMAGE_EXTENSIONS]

    config = load_config(args.model_dir)
    providers = choose_providers(args.device)
    norm = config["normalization"]
    preprocess = Preprocessor(config["image_size"], args.crop_ratio, norm["mean"], norm["std"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    # Images and videos in frame mode use the image model; videos in clip mode use the video model.
    if images or (videos and args.video_mode == "frame"):
        encoder = Encoder(args.model_dir, config, "image", args.precision, providers, args.threads)
        writer = Writer(config, encoder, args.precision, not args.no_patch_tokens)
        batch_size = args.batch_size or 8
        if images:
            embed_images(images, encoder, preprocess, writer, args.output_dir, batch_size)
        if args.video_mode == "frame":
            for path in videos:
                embed_video(
                    path, encoder, preprocess, writer, args.output_dir, batch_size, "frame", config["tubelet_size"]
                )
        total += writer.count
    if videos and args.video_mode == "clip":
        encoder = Encoder(args.model_dir, config, "video", args.precision, providers, args.threads)
        writer = Writer(config, encoder, args.precision, not args.no_patch_tokens)
        for path in videos:
            embed_video(
                path, encoder, preprocess, writer, args.output_dir, args.batch_size or 1, "clip", config["tubelet_size"]
            )
        total += writer.count
    logger.info(f"Done: {total} embedding file(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
