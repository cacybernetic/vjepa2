# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Find every media file inside a dataset source. The source can be a plain
# folder (searched recursively) or a ``.zip`` archive (also searched at any
# depth). One class, one job: list candidate entries for one modality.

from __future__ import annotations

import os
import zipfile
from typing import List, Set

__all__ = [
    "VIDEO_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "FileFinder",
    "VideoFileFinder",
    "ImageFileFinder",
]

# Common container extensions we accept as video files.
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".wmv", ".flv", ".3gp", ".ts", ".ogv",
}

# Common still-image extensions we accept as image files.
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}


class FileFinder:
    """List all entries with the given extensions inside a folder or a zip.

    The returned entries are paths relative to the source root. For a folder
    they are relative file paths; for a zip they are archive member names.
    """

    def __init__(self, extensions: Set[str]):
        self.extensions = {ext.lower() for ext in extensions}

    def _accepts(self, name: str) -> bool:
        """Return True when the file name ends with an accepted extension."""
        return os.path.splitext(name)[1].lower() in self.extensions

    def is_zip(self, source: str) -> bool:
        """Tell whether the source path points to a zip archive."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"Dataset source not found: {source}")
        return zipfile.is_zipfile(source)

    def find(self, source: str) -> List[str]:
        """Return the sorted list of relative entry paths in the source."""
        if self.is_zip(source):
            entries = self._find_in_zip(source)
        else:
            entries = self._find_in_folder(source)
        entries.sort()
        return entries

    def _find_in_folder(self, root: str) -> List[str]:
        """Walk a folder tree and collect relative file paths."""
        found: List[str] = []
        for current, _dirs, files in os.walk(root):
            for name in files:
                if not self._accepts(name):
                    continue
                full = os.path.join(current, name)
                found.append(os.path.relpath(full, root))
        return found

    def _find_in_zip(self, archive_path: str) -> List[str]:
        """Read a zip index and collect matching member names."""
        found: List[str] = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if self._accepts(info.filename):
                    found.append(info.filename)
        return found


class VideoFileFinder(FileFinder):
    """List all video entries inside a folder or a zip archive."""

    def __init__(self):
        super().__init__(VIDEO_EXTENSIONS)


class ImageFileFinder(FileFinder):
    """List all image entries inside a folder or a zip archive."""

    def __init__(self):
        super().__init__(IMAGE_EXTENSIONS)
