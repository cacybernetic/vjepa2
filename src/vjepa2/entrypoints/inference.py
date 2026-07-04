# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Batch inference: encode image / video files into V-JEPA 2.1 dense features
# (or a pooled embedding) and persist them in pickle / numpy / HDF5 form.
#
#   runs -m encoder.onnx -i clip.mp4 -o clip.pkl
#   runs -m encoder.onnx -d videos/ --output-dir embeddings/ -f npy
#
# This script runs an EXPORTED ONNX encoder only (produced by ``exportencoder``)
# and is intentionally torch-free: preprocessing uses NumPy + Pillow (images) and
# PyAV (videos), and inference runs on onnxruntime. Geometry (crop size, number
# of frames) and the normalization convention are read from the ONNX
# ``metadata_props`` when present, so the client matches how the model was
# exported without extra flags.

from typing import Optional, List
from enum import Enum
from argparse import ArgumentParser, Namespace
import logging
import pickle
import sys
import os

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ImageNet statistics used to normalize pixels in [0, 1] (V-JEPA 2.1 recipe).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _resize_shorter_side(im, size: int):
    """Resize a PIL image so its shorter side equals ``size`` (aspect kept)."""
    from PIL import Image

    w, h = im.size
    if w <= h:
        new_w, new_h = size, max(1, round(h * size / w))
    else:
        new_h, new_w = size, max(1, round(w * size / h))
    return im.resize((new_w, new_h), Image.BILINEAR)


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a ``(H, W, C)`` array to ``(size, size, C)``."""
    h, w = arr.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[top:top + size, left:left + size]


class Preprocess:
    """Turn an image / video file into a model-ready clip ``(1, C, T, H, W)``.

    Frames are resized so the shorter side equals ``crop_size``, center-cropped
    to a square and (optionally) normalized with the ImageNet statistics. Videos
    are sampled to exactly ``num_frames`` uniformly-spaced frames; images become
    a single frame (``T == 1``), which selects the encoder's image pathway.

    ``normalize=False`` is used when the ONNX graph bakes the normalization in
    and therefore expects raw RGB pixels in ``[0, 255]``.
    """

    def __init__(self, crop_size: int = 256, num_frames: int = 16,
                 normalize: bool = True):
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.normalize = normalize
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1, 1)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1, 1)

    def _load_image(self, path: str) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as im:
            im = _resize_shorter_side(im.convert("RGB"), self.crop_size)
            frame = _center_crop(np.asarray(im), self.crop_size)  # H W C
        return frame[None, ...]  # T=1 H W C

    def _load_video(self, path: str) -> np.ndarray:
        import av

        frames = []
        with av.open(path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))  # H W C uint8
        if not frames:
            raise ValueError(f"no frames decoded from {path}")

        idx = np.linspace(0, len(frames) - 1, self.num_frames)
        idx = np.round(idx).astype(int)

        from PIL import Image

        out = []
        for i in idx:
            im = _resize_shorter_side(Image.fromarray(frames[i]), self.crop_size)
            out.append(_center_crop(np.asarray(im), self.crop_size))
        return np.stack(out)  # T H W C

    def __call__(self, path: str) -> np.ndarray:
        if is_image_file(path):
            clip = self._load_image(path)
        elif is_video_file(path):
            clip = self._load_video(path)
        else:
            raise ValueError(f"unsupported input type: {path}")

        # (T, H, W, C) -> (C, T, H, W)
        clip = clip.astype(np.float32).transpose(3, 0, 1, 2)
        if self.normalize:
            clip = clip / 255.0
            clip = (clip - self.mean) / self.std
        return clip[None, ...]  # (1, C, T, H, W)


class Postprocess:
    """Reduce encoder features ``(1, N, D)`` to the saved embedding array.

    ``pooling='mean'`` (default) averages the patch tokens into a single
    ``(D,)`` clip/image embedding; ``pooling='none'`` keeps the dense
    ``(N, D)`` token features.
    """

    def __init__(self, pooling: str = "mean"):
        self.pooling = pooling

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        feats = np.squeeze(feats, axis=0)  # (N, D)
        if self.pooling == "mean":
            feats = feats.mean(axis=0)  # (D,)
        elif self.pooling != "none":
            raise ValueError(f"unknown pooling: {self.pooling}")
        return np.ascontiguousarray(feats)


class Model:
    """Feature extractor backed by an exported ONNX encoder."""

    def __init__(self, session, input_name: str, meta: dict):
        self.session = session
        self.input_name = input_name
        self.meta = meta

    @classmethod
    def load(cls, model_filepath: str, device: str = "cpu") -> "Model":
        ext = os.path.splitext(model_filepath)[1].lower()
        if ext != ".onnx":
            raise ValueError(
                f"expected an exported .onnx encoder, got '{ext or model_filepath}'. "
                "Export a checkpoint first with 'exportencoder'."
            )
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider"] + providers
        sess = ort.InferenceSession(model_filepath, providers=providers)
        input_name = sess.get_inputs()[0].name
        meta = dict(sess.get_modelmeta().custom_metadata_map or {})
        logger.info("Loaded ONNX encoder: %s", model_filepath)
        if meta:
            logger.info(
                "Encoder metadata: modality=%s, geometry=%sx%s @ %s frames, %s",
                meta.get("modality", "?"),
                meta.get("crop_size", "?"),
                meta.get("crop_size", "?"),
                meta.get("num_frames", "?"),
                meta.get("normalization", "?"),
            )
        return cls(sess, input_name, meta)

    # -- preprocessing hints read from the exported graph -------------------
    def crop_size(self, fallback: int) -> int:
        return int(self.meta.get("crop_size", fallback))

    def num_frames(self, fallback: int) -> int:
        return int(self.meta.get("num_frames", fallback))

    def normalize(self) -> bool:
        # If the graph bakes normalization in, feed raw [0, 255] pixels.
        return not self.meta.get("normalization", "").startswith("baked")

    def embed(self, clip: np.ndarray) -> np.ndarray:
        (out,) = self.session.run(None, {self.input_name: clip})
        return out


class OutputFormat(Enum):
    PICKLE = 'pkle'
    NUMPY = 'npy'
    HDF5 = 'h5'


# File extension written for each output format.
FORMAT_EXTS = {
    OutputFormat.PICKLE: ".pkl",
    OutputFormat.NUMPY: ".npy",
    OutputFormat.HDF5: ".h5",
}


def get_output_format(format_name: str) -> OutputFormat:
    if format_name == OutputFormat.PICKLE.value:
        return OutputFormat.PICKLE
    elif format_name == OutputFormat.NUMPY.value:
        return OutputFormat.NUMPY
    elif format_name == OutputFormat.HDF5.value:
        return OutputFormat.HDF5
    else:
        raise ValueError("Unsupported output format: {}".format(format_name))


def save_embedding(embedding: np.ndarray, path: str, fmt: OutputFormat) -> None:
    if fmt is OutputFormat.PICKLE:
        with open(path, "wb") as f:
            pickle.dump(embedding, f)
    elif fmt is OutputFormat.NUMPY:
        np.save(path, embedding)
    elif fmt is OutputFormat.HDF5:
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("embedding", data=embedding)


class FileFilter:
    """Collect the image / video files inside a directory."""

    FILE_TYPES = sorted(IMAGE_EXTS | VIDEO_EXTS)

    def __init__(self, recursive: bool = True):
        self.recursive = recursive

    def filter(self, dir_path: str) -> List[str]:
        found: List[str] = []
        if self.recursive:
            for root, _, files in os.walk(dir_path):
                for name in files:
                    if os.path.splitext(name)[1].lower() in self.FILE_TYPES:
                        found.append(os.path.join(root, name))
        else:
            for name in os.listdir(dir_path):
                p = os.path.join(dir_path, name)
                if os.path.isfile(p) and \
                        os.path.splitext(name)[1].lower() in self.FILE_TYPES:
                    found.append(p)
        return sorted(found)


class App:

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.model: Optional[Model] = None
        self.preprocess: Optional[Preprocess] = None
        self.postprocess = Postprocess(pooling=args.pooling)
        self.file_filter = FileFilter(recursive=args.recursive)

    def init(self) -> None:
        self.model = Model.load(self.args.model, device=self.args.device)
        # Configure preprocessing from the graph metadata; CLI args override.
        crop = self.args.crop_size or self.model.crop_size(256)
        num_frames = self.args.num_frames or self.model.num_frames(16)
        self.preprocess = Preprocess(
            crop_size=crop, num_frames=num_frames, normalize=self.model.normalize()
        )

    def _output_path(self, input_path: str, fmt: OutputFormat) -> str:
        """Resolve where the embedding for ``input_path`` should be written."""
        ext = FORMAT_EXTS[fmt]
        if self.args.output_dir is not None:
            stem = os.path.splitext(os.path.basename(input_path))[0]
            return os.path.join(self.args.output_dir, stem + ext)
        if self.args.output_file is not None:
            return self.args.output_file
        # Fall back to a sibling of the input file.
        return os.path.splitext(input_path)[0] + ext

    def run(self) -> int:
        """Running main program and return an integer"""
        input_file: str = self.args.input_file  # file path to video or image.
        input_dir: str = self.args.input_dir    # dir path containing image file and or video files.
        output_dir: str = self.args.output_dir
        #: dir path that will be contain the list of embedding files.
        output_format: OutputFormat = get_output_format(self.args.output_format)
        #: output format in what the embedding computed will be saved.

        # When the input file is provided we take it into account.
        # When a directory path is provided we take into account too.
        # If a directory path is provided at input, we create also the output directory.
        # Whataver the input provided (one file or directory of files) we take all files
        # into account.
        listoffiles: List[str] = []
        if input_file is not None:
            if os.path.isfile(input_file):
                listoffiles.append(input_file)
            else:
                logger.error("input file not found: %s", input_file)
                return 1
        if input_dir is not None:
            if not os.path.isdir(input_dir):
                logger.error("input directory not found: %s", input_dir)
                return 1
            listoffiles.extend(self.file_filter.filter(input_dir))
            if output_dir is not None and not os.path.isdir(output_dir):
                os.makedirs(output_dir)

        if not listoffiles:
            logger.error("no input files to process")
            return 1

        # We run the inference of the model and display the progress bar on the terminal.
        # We compute the embeddings and save them into the output format specified in CLI argument.
        # By default, the output format is `pkle`.
        errors = 0
        for path in tqdm(listoffiles, desc="Encoding", unit="file"):
            try:
                clip = self.preprocess(path)
                feats = self.model.embed(clip)
                embedding = self.postprocess(feats)
                out_path = self._output_path(path, output_format)
                save_embedding(embedding, out_path, output_format)
            except Exception as exc:  # noqa: BLE001 - report and continue
                errors += 1
                logger.error("failed on %s: %s", path, exc)

        logger.info(
            "Done: %d/%d files encoded", len(listoffiles) - errors, len(listoffiles)
        )
        return 1 if errors == len(listoffiles) else 0


def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog="runs",
        description="Encode images / videos into V-JEPA 2.1 embeddings (ONNX).",
    )
    p.add_argument(
        "-m", "--model", required=True,
        help="path to an exported .onnx encoder (see 'exportencoder')",
    )
    p.add_argument("-i", "--input-file", default=None, help="single image/video file")
    p.add_argument("-d", "--input-dir", default=None,
                   help="directory of image/video files")
    p.add_argument("-o", "--output-file", default=None,
                   help="output file for single-input mode")
    p.add_argument("--output-dir", default=None,
                   help="output directory (one embedding file per input)")
    p.add_argument(
        "-f", "--output-format", default=OutputFormat.PICKLE.value,
        choices=[f.value for f in OutputFormat],
        help="embedding serialization format (default: pkle)",
    )
    p.add_argument("--pooling", default="mean", choices=["mean", "none"],
                   help="'mean' pools tokens to a vector, 'none' keeps dense features")
    p.add_argument("--crop-size", type=int, default=None,
                   help="override crop size (default: read from ONNX metadata)")
    p.add_argument("--num-frames", type=int, default=None,
                   help="override frame count (default: read from ONNX metadata)")
    p.add_argument("--recursive", action="store_true", default=True,
                   help="recurse into subdirectories (default)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.input_file is None and args.input_dir is None:
        parser.error("provide at least one of --input-file / --input-dir")

    app = App(args)
    app.init()
    code = app.run()
    sys.exit(code)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Canceled by user!")
        sys.exit(125)
