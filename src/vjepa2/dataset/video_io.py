# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Read a video (from a folder file or a zip member) and turn it into a fixed
# length clip of RGB frames. We use PyAV for decoding, which handles almost any
# container. One class opens the byte stream, one class samples the frames.

from __future__ import annotations

import io
import zipfile
from typing import List, Optional

import numpy as np

__all__ = ["ClipReadError", "VideoSource", "VideoReader"]


class ClipReadError(Exception):
    """Raised when a video cannot be opened or holds no usable frames."""


class VideoSource:
    """Give raw bytes of a video entry from a folder or a zip archive."""

    def __init__(self, root: str, is_zip: bool):
        self.root = root
        self.is_zip = is_zip
        self._archive: Optional[zipfile.ZipFile] = None

    def _zip(self) -> zipfile.ZipFile:
        """Open the zip archive once and keep it for later reads."""
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.root, "r")
        return self._archive

    def open(self, entry: str) -> io.BytesIO:
        """Return a seekable byte stream for a single video entry."""
        if self.is_zip:
            data = self._zip().read(entry)
            return io.BytesIO(data)
        import os

        path = os.path.join(self.root, entry)
        with open(path, "rb") as handle:
            return io.BytesIO(handle.read())

    def close(self) -> None:
        """Release the zip handle if one was opened."""
        if self._archive is not None:
            self._archive.close()
            self._archive = None


def _decode_all_frames(stream: io.BytesIO, max_frames: int):
    """Decode up to ``max_frames`` RGB frames and read the source fps.

    :returns: ``(frames, fps)`` where fps is the average frame rate (float).
    """
    import av

    frames: List[np.ndarray] = []
    with av.open(stream) as container:
        if not container.streams.video:
            raise ClipReadError("no video stream")
        video = container.streams.video[0]
        fps = float(video.average_rate) if video.average_rate else 30.0
        for frame in container.decode(video):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ClipReadError("no frames decoded")
    return frames, fps


class VideoReader:
    """Sample a fixed number of frames from a decoded video.

    :param num_frames: number of frames returned per clip.
    :param target_fps: sampling rate; the stride is derived from source fps.
    :param max_decode: safety cap on how many frames we decode per file.
    """

    def __init__(self, num_frames: int = 16, target_fps: float = 4.0,
                 max_decode: int = 4096):
        self.num_frames = int(num_frames)
        self.target_fps = float(target_fps)
        self.max_decode = int(max_decode)

    def probe(self, source: VideoSource, entry: str) -> bool:
        """Try to decode a couple of frames to confirm the file is usable."""
        stream = source.open(entry)
        frames, _ = _decode_all_frames(stream, max_frames=2)
        return len(frames) > 0

    def read(self, source: VideoSource, entry: str,
             random_start: bool = False, rng: Optional[np.random.Generator] = None,
             ) -> np.ndarray:
        """Return one clip as a ``(T, H, W, 3)`` uint8 array.

        The temporal stride comes from the source fps versus ``target_fps``.
        When there are not enough frames we repeat the last frame so the clip
        always has ``num_frames``.
        """
        stream = source.open(entry)
        frames, fps = _decode_all_frames(stream, self.max_decode)
        step = self._frame_step(fps)
        indices = self._pick_indices(len(frames), step, random_start, rng)
        clip = np.stack([frames[i] for i in indices], axis=0)
        return clip

    def _frame_step(self, fps: float) -> int:
        """Convert source fps and target fps into an integer frame stride."""
        if self.target_fps <= 0:
            return 1
        return max(1, int(round(fps / self.target_fps)))

    def _pick_indices(self, total: int, step: int, random_start: bool,
                      rng: Optional[np.random.Generator]) -> List[int]:
        """Choose ``num_frames`` frame indices with the given step."""
        span = step * self.num_frames
        if total <= 0:
            raise ClipReadError("empty frame list")
        start = self._start_offset(total, span, random_start, rng)
        raw = [start + i * step for i in range(self.num_frames)]
        return [min(idx, total - 1) for idx in raw]

    def _start_offset(self, total: int, span: int, random_start: bool,
                      rng: Optional[np.random.Generator]) -> int:
        """Pick the first frame index of the sampling window."""
        slack = total - span
        if slack <= 0:
            return 0
        if random_start:
            generator = rng if rng is not None else np.random.default_rng()
            return int(generator.integers(0, slack + 1))
        return slack // 2
