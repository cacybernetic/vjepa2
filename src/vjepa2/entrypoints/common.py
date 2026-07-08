# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Shared setup used by the train and evaluate programs: parse the config path,
# resolve the run folder, configure logging, save the used config, and print
# the dataset configuration block.

from __future__ import annotations

import argparse
import math
from typing import Tuple

import yaml

from vjepa2 import logging as vlog
from vjepa2.config import Config, load_yaml, resolve_device
from vjepa2.training.runs import RunDirManager, RunPaths

__all__ = [
    "parse_config_arg",
    "load_config",
    "resolve_run",
    "save_config_used",
    "log_dataset_block",
    "optimizer_steps",
]


def parse_config_arg(program: str) -> argparse.Namespace:
    """Parse the ``--config`` command line argument."""
    parser = argparse.ArgumentParser(description=f"V-JEPA 2.1 {program} program")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to the YAML configuration file")
    return parser.parse_args()


def load_config(path: str) -> Config:
    """Load and resolve a Config from a YAML file path."""
    cfg = Config.from_dict(load_yaml(path))
    cfg.device = resolve_device(cfg.device)
    return cfg


def resolve_run(cfg: Config, kind: str, runs_dir: str,
                resume: bool) -> Tuple[RunPaths, bool]:
    """Create or reuse the run folder and set up its logging."""
    manager = RunDirManager(runs_dir, cfg.run_name)
    run_path, reused = manager.resolve(kind, resume)
    paths = manager.make_paths(run_path, kind=kind)
    vlog.configure_logging(paths.logs_dir, program=kind)
    return paths, reused


def save_config_used(cfg: Config, path: str) -> None:
    """Write the resolved configuration next to the run outputs."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)


def log_dataset_block(cfg: Config) -> None:
    """Print the dataset configuration so the log stays traceable."""
    vlog.log_config_block(cfg.to_dict().get("dataset", {}), title="dataset config")


def optimizer_steps(num_train: int, batch_size: int, grad_accum: int,
                    epochs: int) -> Tuple[int, int]:
    """Return ``(steps_per_epoch, total_steps)`` for the scheduler and logs."""
    batches = math.ceil(max(1, num_train) / max(1, batch_size))
    per_epoch = max(1, math.ceil(batches / max(1, grad_accum)))
    return per_epoch, per_epoch * max(1, epochs)
