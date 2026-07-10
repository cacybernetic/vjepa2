# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Turn a list of videos into a flat list of clip windows. Instead of taking a
# single clip per video, we tile every video into overlapping clips of
# ``num_frames`` sampled frames, hopping ``clip_stride`` frames each step. A long
# video therefore yields many clips, so the model sees all of its content. Each
# window is a light record ``(entry, start_frame, step)`` that the dataset reads
# on demand.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["ClipWindow", "frame_step", "sampled_length", "num_windows",
           "build_clip_windows"]


@dataclass(frozen=True)
class ClipWindow:
    """One clip to read: where it starts in the video and at what stride.

    :param entry: the video path (relative to the dataset root / zip).
    :param start_frame: raw frame index of the window's first sampled frame.
    :param step: raw frames between two sampled frames (the temporal stride).
    """

    entry: str
    start_frame: int
    step: int


def frame_step(fps: float, target_fps: float) -> int:
    """Raw frames between two sampled frames, from source vs. target fps."""
    if target_fps <= 0:
        return 1
    return max(1, int(round(float(fps) / float(target_fps))))


def sampled_length(frames: int, step: int) -> int:
    """Number of frames available after sub-sampling at ``step``."""
    if frames <= 0:
        return 0
    return (int(frames) - 1) // max(1, int(step)) + 1


def num_windows(length: int, num_frames: int, stride: int) -> int:
    """How many clips of ``num_frames`` tile ``length`` at hop ``stride``.

    Mirrors the inference ``--chunk`` count: ``ceil((L - n) / s) + 1``, and at
    least one clip even when the video is shorter than one clip.
    """
    if length <= num_frames:
        return 1
    stride = max(1, int(stride))
    return (length - num_frames + stride - 1) // stride + 1


def _entry_windows(entry: str, frames: int, fps: float, num_frames: int,
                   stride: int, target_fps: float, chunk: bool) -> List[ClipWindow]:
    """Build every clip window for a single video."""
    step = frame_step(fps, target_fps)
    length = sampled_length(frames, step)
    if length <= 0:
        return [ClipWindow(entry, 0, step)]
    last_start = max(0, length - num_frames)
    if not chunk:
        return [ClipWindow(entry, (last_start // 2) * step, step)]
    count = num_windows(length, num_frames, stride)
    windows = []
    for i in range(count):
        start = min(i * stride, last_start)
        windows.append(ClipWindow(entry, start * step, step))
    return windows


def _subsample_windows(windows: List[ClipWindow], cap: int) -> List[ClipWindow]:
    """Keep at most ``cap`` evenly-spaced windows of one video."""
    if cap <= 0 or len(windows) <= cap:
        return windows
    picks = np.linspace(0, len(windows) - 1, cap).round().astype(int)
    seen = sorted(dict.fromkeys(int(p) for p in picks))
    return [windows[i] for i in seen]


def build_clip_windows(entries: List[str],
                       meta: Dict[str, Tuple[int, float]],
                       num_frames: int, stride: int, target_fps: float,
                       sampling: str = "chunk",
                       max_clips_per_video: int = 0) -> List[ClipWindow]:
    """Expand a list of videos into a flat list of clip windows.

    :param meta: ``{entry: (num_frames, fps)}`` from the dataset scan/cache.
    :param sampling: ``"chunk"`` tiles the whole video into overlapping clips;
        anything else keeps a single centered clip per video.
    :param max_clips_per_video: cap on clips from one video (0 = unlimited).
    """
    chunk = str(sampling).lower() == "chunk"
    windows: List[ClipWindow] = []
    for entry in entries:
        frames, fps = meta.get(entry, (0, 30.0))
        entry_windows = _entry_windows(
            entry, int(frames), float(fps), num_frames, stride, target_fps, chunk
        )
        windows.extend(_subsample_windows(entry_windows, max_clips_per_video))
    return windows
