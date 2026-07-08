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

from dataclasses import dataclass
from typing import List, Optional

from tqdm import tqdm

from vjepa2.dataset.cache import CacheStore
from vjepa2.dataset.discovery import VideoFileFinder
from vjepa2.dataset.video_io import VideoReader, VideoSource

__all__ = ["ScanResult", "DatasetCleaner"]


@dataclass
class ScanResult:
    """Outcome of a dataset scan: which entries survived validation."""

    source: str
    is_zip: bool
    entries: List[str]
    num_found: int
    num_dropped: int


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
        return ScanResult(
            source=source,
            is_zip=bool(cached.get("is_zip", False)),
            entries=entries,
            num_found=len(entries),
            num_dropped=0,
        )

    def scan(self, source: str, validate: bool = True) -> ScanResult:
        """Find candidate videos, optionally validate them, and cache the good ones."""
        is_zip = self.finder.is_zip(source)
        candidates = self.finder.find(source)
        if not validate:
            self.store.save(source, candidates, is_zip)
            return ScanResult(source, is_zip, candidates, len(candidates), 0)
        good = self._validate_all(source, is_zip, candidates)
        self.store.save(source, good, is_zip)
        dropped = len(candidates) - len(good)
        return ScanResult(source, is_zip, good, len(candidates), dropped)

    def _validate_all(self, source: str, is_zip: bool,
                      candidates: List[str]) -> List[str]:
        """Try to decode each candidate; keep only the ones that work."""
        provider = VideoSource(source, is_zip)
        good: List[str] = []
        bar = tqdm(candidates, desc="validating dataset", leave=True,
                   ascii="░█", dynamic_ncols=True)
        for entry in bar:
            if self._is_readable(provider, entry):
                good.append(entry)
        provider.close()
        return good

    def _is_readable(self, provider: VideoSource, entry: str) -> bool:
        """Return True when a video entry decodes at least one frame."""
        try:
            return self.reader.probe(provider, entry)
        except Exception:
            return False
