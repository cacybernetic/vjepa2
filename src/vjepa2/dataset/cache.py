# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Read and write the ``*.cache.json`` files that store the list of valid video
# entries for a dataset. This lets the next run skip the slow scan-and-validate
# step. One class, one job: cache file input / output.

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

__all__ = ["cache_path", "CacheStore"]

CACHE_VERSION = 1


def cache_path(source: str) -> str:
    """Return the cache file path that sits next to a dataset source.

    Example: ``/data/train.zip`` -> ``/data/train.cache.json`` and a folder
    ``/data/train`` -> ``/data/train.cache.json``.
    """
    source = os.path.abspath(source)
    parent = os.path.dirname(source)
    base = os.path.basename(source.rstrip(os.sep))
    stem = os.path.splitext(base)[0]
    return os.path.join(parent, f"{stem}.cache.json")


class CacheStore:
    """Persist and restore the validated entry list of a dataset."""

    def path_for(self, source: str) -> str:
        """Return the cache path used for a given dataset source."""
        return cache_path(source)

    def exists(self, source: str) -> bool:
        """Tell whether a cache file already exists for the source."""
        return os.path.isfile(self.path_for(source))

    def save(self, source: str, entries: List[str], is_zip: bool) -> str:
        """Write the validated entries to the cache file and return its path."""
        payload = {
            "version": CACHE_VERSION,
            "source": os.path.abspath(source),
            "is_zip": bool(is_zip),
            "created": datetime.now().isoformat(timespec="seconds"),
            "num_entries": len(entries),
            "entries": list(entries),
        }
        target = self.path_for(source)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return target

    def load(self, source: str) -> Optional[Dict]:
        """Load the cached payload, or None when it is missing or invalid."""
        target = self.path_for(source)
        if not os.path.isfile(target):
            return None
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != CACHE_VERSION:
            return None
        if "entries" not in payload:
            return None
        return payload
