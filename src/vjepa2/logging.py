# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Terminal rendering and journaling. We use loguru for logs (console + file)
# and tqdm for progress bars. The console sink writes through ``tqdm.write`` so
# log lines never break the progress bars. Bars use plain keyboard-typable
# characters only: full block for progress and light block for the background.

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

from loguru import logger
from tqdm import tqdm

__all__ = [
    "logger",
    "configure_logging",
    "log_config_block",
    "log_kv",
    "log_model_summary",
    "count_parameters",
    "make_epoch_bar",
    "make_step_bar",
    "BAR_FILL",
    "BAR_BACKGROUND",
]

# Progress-bar characters. First char = empty cell, last char = full cell.
BAR_FILL = "█"        # full block
BAR_BACKGROUND = "░"  # light shade block
_BAR_ASCII = BAR_BACKGROUND + BAR_FILL

_CONSOLE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} "
    "<level>{level: <7}</level> | "
    "<level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} {level: <7} | {name}:{line} | {message}"
)


def _tqdm_sink(message: str) -> None:
    """Write one log line without breaking active tqdm progress bars."""
    tqdm.write(message, end="")


def configure_logging(
    log_dir: str,
    program: str = "train",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> str:
    """Set up loguru console and file sinks.

    :param log_dir: folder that will hold the ``.log`` files.
    :param program: short name used as the log file prefix (train, eval, ...).
    :param console_level: minimum level printed to the terminal.
    :param file_level: minimum level written to the log file.
    :returns: the path of the created log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{program}_{stamp}.log")
    logger.remove()
    logger.add(
        _tqdm_sink,
        level=console_level,
        format=_CONSOLE_FORMAT,
        colorize=True,
        enqueue=False,
    )
    logger.add(
        log_path,
        level=file_level,
        format=_FILE_FORMAT,
        encoding="utf-8",
        enqueue=True,
    )
    logger.info("Logging to file: {}", log_path)
    return log_path


def _flatten(data: Dict[str, Any], indent: int = 0):
    """Yield ``(indent, key, value_or_None)`` for a nested mapping."""
    for key, value in data.items():
        if isinstance(value, dict):
            yield indent, key, None
            for item in _flatten(value, indent + 1):
                yield item
        else:
            yield indent, key, value


def log_config_block(data: Dict[str, Any], title: str = "config") -> None:
    """Print a nested config dict with clean indentation.

    Each nested level is indented by two spaces, matching the sample logs.
    """
    logger.info("===== {} =====", title)
    for indent, key, value in _flatten(data):
        pad = "  " * (indent + 1)
        if value is None:
            logger.info("{}{}:", pad, key)
        else:
            logger.info("{}{}: {}", pad, key, value)


def log_kv(label: str, value: Any, width: int = 22) -> None:
    """Log a single aligned ``label = value`` summary line."""
    logger.info("  {} = {}", str(label).ljust(width), value)


def count_parameters(model) -> Dict[str, int]:
    """Count total and trainable parameters of a module.

    :returns: dict with keys ``total`` and ``trainable``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _model_size_mb(model) -> float:
    """Approximate parameter memory footprint in megabytes."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024.0 * 1024.0)


def _log_torchinfo(model) -> bool:
    """Log a torchinfo layer table (params only, no forward). Return success.

    We do not pass an input because the JEPA forward takes nested mask lists
    that torchinfo cannot build on its own. A params-only summary still gives
    the full per-layer architecture and parameter state we want to see.
    """
    try:
        from torchinfo import summary
    except Exception:
        return False
    try:
        report = summary(model, depth=3, verbose=0,
                         col_names=("num_params", "trainable"))
    except Exception:
        return False
    for line in str(report).splitlines():
        logger.info("  {}", line)
    return True


def log_model_summary(model, name: str = "model") -> Dict[str, int]:
    """Log a parameter / size summary of a model.

    We first try a torchinfo layer table for the full architecture view. We
    always log per top-level submodule counts, the total, and the memory
    footprint, which stay correct even when torchinfo is not available.
    """
    counts = count_parameters(model)
    logger.info("===== {} summary =====", name)
    _log_torchinfo(model)
    for child_name, child in model.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        logger.info("  {}: {:,} params", child_name, int(child_total))
    logger.info(
        "  total {:,} | trainable {:,} | size {:.1f} MB",
        counts["total"],
        counts["trainable"],
        _model_size_mb(model),
    )
    return counts


def make_epoch_bar(total: int, initial: int = 0, desc: str = "TRAINING") -> tqdm:
    """Build the outer (per-epoch) progress bar.

    The outer bar stays on screen (``leave=True``) so log lines printed with
    ``tqdm.write`` do not corrupt it.
    """
    bar_format = (
        "{desc}: {percentage:3.0f}%|{bar}| "
        "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        leave=True,
        position=0,
        dynamic_ncols=True,
        ascii=_BAR_ASCII,
        bar_format=bar_format,
    )


def make_step_bar(total: int, initial: int = 0, desc: str = "step") -> tqdm:
    """Build an inner (per-step) progress bar that disappears when finished."""
    bar_format = (
        "{desc}: {percentage:3.0f}%|{bar}| "
        "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    )
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        leave=False,
        position=1,
        dynamic_ncols=True,
        ascii=_BAR_ASCII,
        bar_format=bar_format,
    )
