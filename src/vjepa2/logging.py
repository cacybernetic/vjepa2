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
    "log_component_summaries",
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
    """Write one log line without leaving progress-bar residue.

    ``tqdm.write`` moves the active bars out of the way, but a log line shorter
    than the bar can leave leftover bar characters at the end of the line. We
    prepend a carriage return + "erase to end of line" (``\\r\\x1b[K``) so the
    whole line is wiped before the log text is drawn, and strip the trailing
    newline loguru adds so ``tqdm.write`` owns the single line break.
    """
    tqdm.write("\r\x1b[K" + message.rstrip("\n"))


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


def _run_torchinfo(module, input_data=None, forward_kwargs=None,
                   device=None) -> bool:
    """Log a torchinfo layer table for ``module``. Return True on success.

    When ``input_data`` is given, torchinfo runs one real forward pass, so the
    table gains an input-size and an output-size column for every layer. Without
    it we fall back to a params-only table (no forward) that still shows the
    whole per-layer architecture and parameter state.
    """
    try:
        from torchinfo import summary
    except Exception:
        return False
    forward_kwargs = forward_kwargs or {}
    with_shapes = ("input_size", "output_size", "num_params", "trainable")
    try:
        if input_data is None:
            report = summary(module, depth=3, verbose=0,
                             col_names=("num_params", "trainable"))
        else:
            report = summary(module, input_data=input_data, depth=3, verbose=0,
                             col_names=with_shapes, device=device,
                             **forward_kwargs)
    except Exception as exc:
        if input_data is None:
            return False
        logger.debug("torchinfo forward table failed ({}); params only", exc)
        return _run_torchinfo(module)
    for line in str(report).splitlines():
        logger.info("  {}", line)
    return True


def log_model_summary(model, name: str = "model") -> Dict[str, int]:
    """Log a parameter / size summary of a model and of each component.

    We log the per-submodule parameter counts and the total size, then a
    detailed torchinfo table (with input and output shapes) for the encoder,
    the predictor and the target encoder. All counts stay correct even when
    torchinfo is not installed.
    """
    counts = count_parameters(model)
    logger.info("===== {} summary =====", name)
    for child_name, child in model.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        logger.info("  {}: {:,} params", child_name, int(child_total))
    logger.info(
        "  total {:,} | trainable {:,} | size {:.1f} MB",
        counts["total"],
        counts["trainable"],
        _model_size_mb(model),
    )
    _maybe_log_components(model)
    return counts


def _maybe_log_components(model) -> None:
    """Log per-component summaries when the model exposes JEPA components."""
    if not (hasattr(model, "encoder") and hasattr(model, "predictor")):
        return
    try:
        log_component_summaries(model)
    except Exception as exc:
        logger.debug("per-component summaries skipped: {}", exc)


def log_component_summaries(model) -> None:
    """Log a torchinfo summary with input / output shapes per component.

    Covers the online ``encoder``, the ``predictor`` and the EMA
    ``target_encoder``. Each summary runs one dummy forward on the component so
    the table and the logged lines show its real input and output shapes.
    """
    for comp_name in ("encoder", "predictor", "target_encoder"):
        component = getattr(model, comp_name, None)
        if component is None:
            continue
        backbone = getattr(component, "backbone", component)
        _log_one_component(comp_name, backbone)


def _log_one_component(name: str, backbone) -> None:
    """Log the architecture, input / output shapes and size of one component."""
    device = _module_device(backbone)
    logger.info("===== {} architecture =====", name)
    try:
        input_data, forward_kwargs = _component_dummy(backbone, device)
    except Exception as exc:
        logger.debug("could not build dummy input for {}: {}", name, exc)
        input_data, forward_kwargs = None, None
    _log_io_shapes(backbone, input_data, forward_kwargs, device)
    _run_torchinfo(backbone, input_data, forward_kwargs, device)
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info("  params total {:,} | trainable {:,} | size {:.1f} MB",
                int(total), int(trainable), _model_size_mb(backbone))


def _module_device(module):
    """Return the device the module parameters live on (cpu if none)."""
    import torch

    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _component_dummy(backbone, device):
    """Build a dummy ``(input_data, forward_kwargs)`` for one component."""
    if hasattr(backbone, "predictor_embed"):
        return _predictor_dummy(backbone, device)
    return _encoder_dummy(backbone, device)


def _encoder_dummy(backbone, device):
    """Build a dummy clip input for an encoder / target-encoder backbone."""
    import torch

    if getattr(backbone, "is_video", False):
        clip = torch.zeros(1, 3, backbone.num_frames, backbone.img_height,
                           backbone.img_width, device=device)
    else:
        clip = torch.zeros(1, 3, backbone.img_height, backbone.img_width,
                           device=device)
    return clip, {"training": True}


def _predictor_dummy(backbone, device):
    """Build dummy context tokens and masks for a predictor backbone."""
    import torch
    import torch.nn as nn

    embed = backbone.predictor_embed
    linear = embed if isinstance(embed, nn.Linear) else embed[0]
    n_patches = int(backbone.num_patches)
    n_ctx = max(1, n_patches // 2)
    idx = torch.arange(n_patches, device=device)
    tokens = torch.zeros(1, n_ctx, linear.in_features, device=device)
    masks_x = idx[:n_ctx].unsqueeze(0)
    masks_y = idx[n_ctx:].unsqueeze(0)
    mod = "video" if getattr(backbone, "is_video", False) else "image"
    return [tokens, masks_x, masks_y], {"mod": mod, "mask_index": 0}


def _log_io_shapes(backbone, input_data, forward_kwargs, device) -> None:
    """Run one forward pass and log the component input / output shapes."""
    if input_data is None:
        return
    try:
        in_desc, out_desc = _forward_shapes(backbone, input_data,
                                            forward_kwargs, device)
    except Exception as exc:
        logger.debug("input / output shape probe failed: {}", exc)
        return
    logger.info("  input  {}", in_desc)
    logger.info("  output {}", out_desc)


def _forward_shapes(backbone, input_data, forward_kwargs, device):
    """Return ``(input_desc, output_desc)`` from one no-grad forward pass."""
    import torch

    if isinstance(input_data, (list, tuple)):
        args = list(input_data)
    else:
        args = [input_data]
    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            out = backbone(*args, **(forward_kwargs or {}))
    finally:
        backbone.train(was_training)
    in_desc = ", ".join(_describe_shape(a) for a in args)
    return in_desc, _describe_shape(out)


def _describe_shape(obj) -> str:
    """Describe the shape of a tensor or a nested list / tuple of tensors."""
    if hasattr(obj, "shape"):
        return "x".join(str(int(d)) for d in tuple(obj.shape))
    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(_describe_shape(o) for o in obj) + "]"
    return type(obj).__name__


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
