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
    # metadata_errors="ignore": some files carry non-UTF-8 tags on side data
    # streams; without this PyAV raises UnicodeDecodeError and a good video is
    # dropped.
    with av.open(stream, metadata_errors="ignore") as container:
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


def _probe_metadata(stream: io.BytesIO, max_decode: int):
    """Read ``(num_frames, fps)`` from a video, decoding only when needed.

    We first trust the container header (``stream.frames`` or the duration times
    the frame rate), which is cheap. When the header is missing or zero -- common
    for streamed / variable-frame-rate files -- we fall back to counting decoded
    frames up to ``max_decode``.
    """
    import av

    with av.open(stream, metadata_errors="ignore") as container:
        if not container.streams.video:
            raise ClipReadError("no video stream")
        video = container.streams.video[0]
        fps = float(video.average_rate) if video.average_rate else 30.0
        count = int(video.frames or 0)
        if count <= 0:
            count = _estimate_from_duration(container, video, fps)
        if count <= 0:
            count = sum(1 for _ in container.decode(video))
            count = min(count, max_decode)
    if count <= 0:
        raise ClipReadError("no frames")
    return count, fps


def _estimate_from_duration(container, video, fps: float) -> int:
    """Estimate a frame count from the stream / container duration."""
    if video.duration and video.time_base:
        seconds = float(video.duration * video.time_base)
        return int(round(seconds * fps))
    if container.duration:
        import av

        seconds = float(container.duration) / float(av.time_base)
        return int(round(seconds * fps))
    return 0


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

    def inspect(self, source: VideoSource, entry: str):
        """Return ``(num_frames, fps)`` of a video without loading every frame.

        Used at scan time to plan how many clips a video can yield; the values
        are cached so later runs never re-open the file.
        """
        stream = source.open(entry)
        return _probe_metadata(stream, self.max_decode)

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
        step = self.frame_step(fps)
        indices = self._pick_indices(len(frames), step, random_start, rng)
        return self._gather(frames, indices)

    def read_window(self, source: VideoSource, entry: str, start_frame: int,
                    step: int) -> np.ndarray:
        """Read one clip that starts at raw frame ``start_frame``.

        Frames are taken every ``step`` raw frames (the temporal stride). We only
        decode up to the last frame the window needs, so early windows of a long
        video are cheap. Missing tail frames are padded by repeating the last one.
        """
        step = max(1, int(step))
        start_frame = max(0, int(start_frame))
        need = start_frame + (self.num_frames - 1) * step + 1
        stream = source.open(entry)
        frames, _ = _decode_all_frames(stream, min(self.max_decode, need))
        return self.window_clip(frames, start_frame, step)

    def decode_all(self, source: VideoSource, entry: str) -> List[np.ndarray]:
        """Decode the whole video once (capped at ``max_decode`` frames).

        Handy for the HDF5 builder, which reads every clip window of a video and
        can slice them all from a single decode.
        """
        stream = source.open(entry)
        frames, _ = _decode_all_frames(stream, self.max_decode)
        return frames

    def window_clip(self, frames: List[np.ndarray], start_frame: int,
                    step: int) -> np.ndarray:
        """Slice one clip out of already-decoded ``frames``."""
        step = max(1, int(step))
        start_frame = max(0, int(start_frame))
        total = len(frames)
        indices = [min(start_frame + i * step, total - 1)
                   for i in range(self.num_frames)]
        return self._gather(frames, indices)

    def _gather(self, frames: List[np.ndarray], indices: List[int]) -> np.ndarray:
        """Stack the chosen frames into a ``(T, H, W, 3)`` clip."""
        return np.stack([frames[i] for i in indices], axis=0)

    def frame_step(self, fps: float) -> int:
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
