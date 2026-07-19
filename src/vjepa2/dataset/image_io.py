# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Read a still image (from a folder file or a zip member) and turn it into a
# single-frame clip. The reader mirrors the ``VideoReader`` interface used by
# the dataset cleaner (``inspect``) and the dataset (``read``), so the same
# scanning / caching machinery serves both modalities.

from __future__ import annotations

from typing import Tuple

import numpy as np

from vjepa2.dataset.video_io import ClipReadError, VideoSource

__all__ = ["ImageReader"]


class ImageReader:
    """Decode one image entry into a ``(1, H, W, 3)`` uint8 clip.

    Images have no temporal axis: every clip is exactly one frame, which the
    encoder tokenizes through its image pathway (2D-style tokenizer + image
    modality embedding) per the V-JEPA 2.1 multi-modal recipe.
    """

    num_frames = 1

    def inspect(self, source: VideoSource, entry: str) -> Tuple[int, float]:
        """Return ``(num_frames, fps)`` for the cleaner / cache.

        Fully decodes the image so this doubles as the corruption check; an
        unreadable file raises and is dropped by the cleaner. Images always
        report one frame and a null frame rate.
        """
        self.read(source, entry)
        return 1, 0.0

    def read(self, source: VideoSource, entry: str) -> np.ndarray:
        """Return one image as a ``(1, H, W, 3)`` uint8 clip array."""
        from PIL import Image

        stream = source.open(entry)
        try:
            with Image.open(stream) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except ClipReadError:
            raise
        except Exception as error:
            raise ClipReadError(f"cannot decode image {entry}: {error}") from error
        if array.ndim != 3 or array.shape[2] != 3:
            raise ClipReadError(f"unexpected image shape {array.shape} in {entry}")
        return array[None, ...]
