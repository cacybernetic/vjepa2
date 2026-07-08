# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The dataset build program. It scans and cleans a video dataset, then decodes
# and transforms every clip once and stores the result in an HDF5 file. Training
# and evaluation can then read these ready-to-use clips and run faster.

from __future__ import annotations

from typing import List, Tuple

from vjepa2 import logging as vlog
from vjepa2.config import Config
from vjepa2.dataset.cleaning import DatasetCleaner
from vjepa2.dataset.factory import build_pipeline, build_reader
from vjepa2.dataset.hdf5 import HDF5Builder
from vjepa2.dataset.splits import cap_entries
from vjepa2.entrypoints import common

__all__ = ["BuildApp", "main"]


class BuildApp:
    """Build HDF5 files from the train and test video datasets."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.augment_copies = int(cfg.raw.get("hdf5", {}).get("augment_copies", 0))

    def setup(self) -> None:
        """Configure logging in a build logs folder and print the config."""
        vlog.configure_logging("runs/build_logs", program="buildds")
        vlog.logger.info("Starting HDF5 build pipeline")
        common.log_dataset_block(self.cfg)

    def run(self) -> None:
        """Build the train and test HDF5 files."""
        self._build_one(self.cfg.dataset.train_path, self.cfg.dataset.train_h5,
                        self.cfg.dataset.max_train_samples, train=True)
        self._build_one(self.cfg.dataset.test_path, self.cfg.dataset.test_h5,
                        self.cfg.dataset.max_test_samples, train=False)
        vlog.logger.info("HDF5 build finished")

    def _build_one(self, source: str, out_path: str, max_samples,
                   train: bool) -> None:
        """Scan, clean and store one dataset split into an HDF5 file."""
        is_zip, entries = self._scan(source, max_samples)
        vlog.logger.info("Building {} clips from {} into {}",
                         len(entries), source, out_path)
        builder = HDF5Builder(build_pipeline(self.cfg, train), build_reader(self.cfg))
        copies = self.augment_copies if train else 0
        written = builder.build(
            out_path, source, is_zip, entries, self._clip_shape(), copies
        )
        vlog.logger.info("Wrote {} clips to {}", written, out_path)

    def _scan(self, source: str, max_samples) -> Tuple[bool, List[str]]:
        """Validate (or load cached) entries and apply the sample cap."""
        cleaner = DatasetCleaner(reader=build_reader(self.cfg))
        result = cleaner.prepare(source, validate=self.cfg.dataset.validate)
        return result.is_zip, cap_entries(result.entries, max_samples)

    def _clip_shape(self) -> Tuple[int, int, int, int]:
        """Return the expected ``(C, T, H, W)`` of one preprocessed clip."""
        size = self.cfg.dataset.crop_size
        return (3, self.cfg.dataset.num_frames, size, size)


def main() -> None:
    """Entry point for the ``buildh5ds`` command."""
    args = common.parse_config_arg("hdf5 build")
    cfg = common.load_config(args.config)
    app = BuildApp(cfg)
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
