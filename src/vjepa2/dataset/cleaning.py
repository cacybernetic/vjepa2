# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Scan a dataset source, drop broken or unreadable videos, and keep only the
# good ones. The result is cached so we do not repeat this slow step. One class
# checks the files, the cache store owns the disk format.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from vjepa2.dataset.cache import CacheStore, cache_path
from vjepa2.dataset.discovery import VideoFileFinder
from vjepa2.dataset.video_io import VideoReader, VideoSource
from vjepa2.logging import logger

__all__ = ["ScanResult", "DatasetCleaner"]


def _describe_error(error: Optional[BaseException]) -> str:
    """One-line reason for a rejected video (exception type + message)."""
    if error is None:
        return "unreadable"
    message = str(error).strip()
    name = type(error).__name__
    return f"{name}: {message}" if message else name


@dataclass
class ScanResult:
    """Outcome of a dataset scan: which entries survived validation."""

    source: str
    is_zip: bool
    entries: List[str]
    num_found: int
    num_dropped: int
    # {entry: (num_frames, fps)} used to plan clip windows.
    meta: Dict[str, Tuple[int, float]] = field(default_factory=dict)


class DatasetCleaner:
    """Validate every video and cache the entries that decode correctly."""

    def __init__(self, reader: Optional[VideoReader] = None,
                 finder: Optional[VideoFileFinder] = None,
                 store: Optional[CacheStore] = None):
        self.reader = reader or VideoReader()
        self.finder = finder or VideoFileFinder()
        self.store = store or CacheStore()

    def prepare(self, source: str, validate: bool = True,
                use_cache: bool = True) -> ScanResult:
        """Return the validated entry list, using the cache when possible.

        :param source: folder or zip path.
        :param validate: when False, keep every found file without decoding it.
        :param use_cache: when True, reuse an existing ``*.cache.json``.
        """
        if use_cache:
            cached = self.store.load(source)
            if cached is not None:
                return self._from_cache(source, cached)
        return self.scan(source, validate=validate)

    def _from_cache(self, source: str, cached: dict) -> ScanResult:
        """Build a scan result from a loaded cache payload."""
        entries = list(cached["entries"])
        raw_meta = cached.get("meta", {}) or {}
        meta = {e: (int(v[0]), float(v[1]))
                for e, v in raw_meta.items() if e in set(entries)}
        logger.info("dataset cache: reusing {} videos from {} "
                    "(delete the .cache.json to re-scan)",
                    len(entries), cache_path(source))
        return ScanResult(
            source=source,
            is_zip=bool(cached.get("is_zip", False)),
            entries=entries,
            num_found=len(entries),
            num_dropped=0,
            meta=meta,
        )

    def scan(self, source: str, validate: bool = True) -> ScanResult:
        """Find candidate videos, optionally validate them, and cache the good ones."""
        is_zip = self.finder.is_zip(source)
        candidates = self.finder.find(source)
        good, meta = self._inspect_all(source, is_zip, candidates, validate)
        self.store.save(source, good, is_zip, meta)
        dropped = len(candidates) - len(good)
        return ScanResult(source, is_zip, good, len(candidates), dropped, meta)

    def _inspect_all(self, source: str, is_zip: bool, candidates: List[str],
                     validate: bool) -> Tuple[List[str], Dict[str, Tuple[int, float]]]:
        """Read ``(frames, fps)`` for each video; drop the ones that fail.

        With ``validate`` on, a video is kept only when its metadata reads back;
        this doubles as the corruption check. With ``validate`` off we still try
        to read the (cheap) header so clip planning has a frame count, but a
        failure only drops that file from the plan, not from the dataset.

        Files that fail to read are collected with their reason and logged, so a
        video silently vanishing from the dataset always leaves a trace.
        """
        provider = VideoSource(source, is_zip)
        desc = "validating dataset" if validate else "scanning dataset"
        good: List[str] = []
        meta: Dict[str, Tuple[int, float]] = {}
        failures: List[Tuple[str, str]] = []
        bar = tqdm(candidates, desc=desc, leave=True, ascii="░█",
                   dynamic_ncols=True)
        for entry in bar:
            info, error = self._inspect(provider, entry)
            if info is not None:
                meta[entry] = info
                good.append(entry)
            else:
                failures.append((entry, _describe_error(error)))
                if not validate:
                    good.append(entry)
        provider.close()
        self._log_scan(len(candidates), len(good), failures, validate)
        return good, meta

    def _log_scan(self, num_found: int, num_good: int,
                  failures: List[Tuple[str, str]], validate: bool) -> None:
        """Log a ``found / usable / dropped`` recap and every failing file."""
        num_dropped = len(failures) if validate else 0
        logger.info("dataset scan: found {} videos | usable {} | dropped {}",
                    num_found, num_good, num_dropped)
        if not failures:
            return
        if validate:
            for entry, reason in failures:
                logger.warning("  dropped (unreadable): {} -- {}", entry, reason)
        else:
            for entry, reason in failures:
                logger.info("  no metadata, kept as 1 clip: {} -- {}",
                            entry, reason)

    def _inspect(self, provider: VideoSource, entry: str
                 ) -> Tuple[Optional[Tuple[int, float]], Optional[BaseException]]:
        """Return ``((num_frames, fps), None)`` or ``(None, error)`` on failure."""
        try:
            return self.reader.inspect(provider, entry), None
        except Exception as exc:  # noqa: BLE001 - reason is reported, not hidden
            return None, exc
