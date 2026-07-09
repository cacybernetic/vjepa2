# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The training loop. It ties together the model, the dense loss, the optimizer,
# the scheduler, the EMA target encoder, the resumable loaders, the meters, the
# checkpoints and the plots. It supports gradient accumulation, per-step
# checkpointing, and resuming an interrupted run at the exact same place.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from vjepa2 import logging as vlog
from vjepa2.config import resolve_steps
from vjepa2.metrics import MetricTracker, feature_std, prediction_cosine
from vjepa2.metrics.ssl_metrics import METRIC_NAMES
from vjepa2.training import utils
from vjepa2.training.best_model import BestModelTracker
from vjepa2.training.checkpoint import CheckpointManager
from vjepa2.training.history import HistoryWriter
from vjepa2.training.plotting_bridge import plot_history

__all__ = ["RunState", "Trainer"]


@dataclass
class RunState:
    """Mutable position inside the whole training process."""

    epoch: int = 0
    phase: str = "train"
    global_step: int = 0


class Trainer:
    """Run training with validation, checkpointing and resume support."""

    def __init__(self, model, loss_fn, optimizer, scheduler, ema, bundle,
                 paths, train_cfg, device: str, amp: bool = False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.bundle = bundle
        self.paths = paths
        self.cfg = train_cfg
        self.device = device
        self.state = RunState()
        self._pending = False
        self._resumed = False
        self._last_grad_norm = 0.0
        self._build_helpers(amp)

    def _build_helpers(self, amp: bool) -> None:
        """Create meters, trackers, checkpoint manager and history writer."""
        self.ckpt = CheckpointManager(self.paths.checkpoints_dir,
                                      self.cfg.max_checkpoint)
        self.history = HistoryWriter(self.paths.history_csv)
        self.best = BestModelTracker(self.cfg.best_metric, self.cfg.best_mode)
        self.train_meter = MetricTracker(METRIC_NAMES)
        self.val_meter = MetricTracker(METRIC_NAMES)
        self.test_meter = MetricTracker(METRIC_NAMES)
        self.use_amp = bool(amp) and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.steps_per_epoch = self._steps_per_epoch()
        self.total_steps = self.steps_per_epoch * self.cfg.epochs
        self._log_every = self._resolve_cadence(self.cfg.log_every, "log_every")
        self._ckpt_step = self._resolve_cadence(self.cfg.ckpt_step, "ckpt_step")

    def _steps_per_epoch(self) -> int:
        """Optimizer steps in a full epoch, given batch size and accumulation."""
        batches = math.ceil(self.bundle.num_train / self.cfg.batch_size)
        return max(1, math.ceil(batches / self.cfg.grad_accum))

    def _resolve_cadence(self, value: float, name: str) -> int:
        """Resolve a periodic cadence (log / checkpoint) into optimizer steps.

        A value in ``(0, 1)`` is a fraction of one epoch; ``>= 1`` is an absolute
        number of steps. The result is clamped to ``steps_per_epoch`` so that a
        periodic log and a periodic checkpoint always happen at least once per
        epoch -- otherwise a too-large value would silently never fire.
        """
        wanted = max(1, resolve_steps(value, self.steps_per_epoch))
        steps = min(wanted, self.steps_per_epoch)
        if steps < wanted:
            vlog.logger.warning(
                "{} ({}) exceeds steps_per_epoch ({}); clamped to {} so it "
                "fires at least once per epoch", name, wanted,
                self.steps_per_epoch, steps)
        return steps

    # -- public entry point --------------------------------------------------
    def run(self) -> None:
        """Resume if possible, then train every epoch and evaluate at the end."""
        self._maybe_resume()
        start_epoch = self.state.epoch
        epoch_bar = vlog.make_epoch_bar(self.cfg.epochs, initial=start_epoch)
        self._epoch_start_time = time.time()
        self._epochs_done = 0
        for epoch in range(start_epoch, self.cfg.epochs):
            self._run_epoch(epoch)
            self._update_epoch_bar(epoch_bar)
        epoch_bar.close()
        self._final_test()
        vlog.logger.info("Training finished. Best {} = {:.4f}",
                         self.best.metric, self.best.best)

    def _run_epoch(self, epoch: int) -> None:
        """Run the train and validation passes for one epoch."""
        vlog.logger.info("===== Starting epoch {}/{} =====",
                         epoch + 1, self.cfg.epochs)
        self.state.epoch = epoch
        if not (self._resumed and self.state.phase == "val"):
            self._train_pass(epoch)
        self._validate_pass(epoch)
        self._record_epoch(epoch)
        self._resumed = False
        # Advance the saved position to the next epoch so a finished run does
        # not repeat this epoch when it is restarted.
        self.state.epoch = epoch + 1
        self.state.phase = "train"
        self._save_checkpoint("train")

    # -- train pass ----------------------------------------------------------
    def _train_pass(self, epoch: int) -> None:
        """One full pass over the training data with gradient accumulation."""
        self.model.train()
        loader = self.bundle.train_loader
        if self._prepare_loader(loader, epoch):
            self.train_meter.reset()
        bar = vlog.make_step_bar(len(loader), desc=f"train e{epoch + 1}")
        for micro, batch in enumerate(loader):
            parts, n = self._train_step(batch, micro)
            self.train_meter.update(parts, n)
            self._render_train(bar)
        self._flush_accumulation()
        bar.close()

    def _train_step(self, batch, micro: int):
        """Forward, scaled backward, and an optimizer step when accumulation is full."""
        loss, parts, n = self._forward_loss(batch)
        self.scaler.scale(loss / self.cfg.grad_accum).backward()
        self._pending = True
        if (micro + 1) % self.cfg.grad_accum == 0:
            self._optimizer_step()
        return parts, n

    def _optimizer_step(self) -> None:
        """Clip gradients and step the optimizer.

        The learning rate, EMA target and step counter only move when a real
        optimizer step happened. Under AMP, ``scaler.step`` skips the update on
        inf/nan gradients (the scale then shrinks), and in that case nothing is
        advanced -- the LR is held for the retry.
        """
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.cfg.grad_clip_norm)
        self._last_grad_norm = float(grad_norm)
        # Apply the LR of the current step before the update (first step uses
        # ``start_lr``), but do not advance the schedule yet.
        self.scheduler.apply()
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending = False
        if self.scaler.get_scale() < scale_before:
            vlog.logger.debug("optimizer step skipped (inf/nan grads); "
                              "lr and ema held")
            return
        self.scheduler.advance()
        self.ema.update(self.model.encoder, self.model.ema_target())
        self.state.global_step += 1
        self._after_step()

    def _after_step(self) -> None:
        """Log periodically and save a checkpoint every ``ckpt_step`` steps."""
        step = self.state.global_step
        if step % self._log_every == 0:
            avg = self.train_meter.averages()
            # INFO so the periodic training progress is visible on the terminal
            # (the console sink defaults to INFO), not only in the log file.
            vlog.logger.info("step {}/{} {} grad_norm={:.3f} lr={:.2e}", step,
                             self.total_steps, utils.format_metrics(avg),
                             self._last_grad_norm, self.scheduler.last_lr)
        if step % self._ckpt_step == 0:
            self._save_checkpoint("train")

    def _flush_accumulation(self) -> None:
        """Apply any leftover gradients that did not fill a full accumulation."""
        if self._pending:
            vlog.logger.debug("flushing leftover gradient accumulation")
            self._optimizer_step()

    # -- validation and test passes -----------------------------------------
    def _validate_pass(self, epoch: int) -> None:
        """Run validation on the held-out fraction of the test set."""
        loader = self.bundle.val_loader
        if loader is None:
            return
        self.state.phase = "val"
        self._eval_pass(loader, self.val_meter, f"val e{epoch + 1}", "val")

    def _eval_pass(self, loader, meter, desc: str, phase: str) -> None:
        """Shared no-grad pass used by validation and final test."""
        self.model.eval()
        if self._prepare_loader(loader, self.state.epoch):
            meter.reset()
        bar = vlog.make_step_bar(len(loader), desc=desc)
        for batch_index, batch in enumerate(loader):
            with torch.no_grad():
                _, parts, n = self._forward_loss(batch)
            meter.update(parts, n)
            self._render_eval(bar, meter)
            self._checkpoint_eval(batch_index, phase)
        bar.close()

    def _checkpoint_eval(self, batch_index: int, phase: str) -> None:
        """Save a checkpoint during an evaluation pass for mid-pass resume."""
        if (batch_index + 1) % self.cfg.ckpt_step == 0:
            self._save_checkpoint(phase)

    # -- shared forward ------------------------------------------------------
    def _forward_loss(self, batch):
        """Run the model and the dense loss, returning loss, parts and count."""
        clips, masks_enc, masks_pred = batch
        clips = utils.move_clips(clips, self.device)
        masks_enc = utils.move_masks(masks_enc, self.device)
        masks_pred = utils.move_masks(masks_pred, self.device)
        # Route the batch through the correct tokenizer / modality embedding:
        # a single temporal step (or a 4D tensor) is an image, otherwise video.
        c0 = clips[0]
        is_image = c0.ndim == 4 or (c0.ndim == 5 and c0.shape[2] == 1)
        mod = "image" if is_image else "video"
        with self._autocast():
            z_pred, z_ctx, target = self.model(
                clips, masks_enc, masks_pred, mod=mod, training_mode=True
            )
            loss, parts = self.loss_fn(
                z_pred, z_ctx, target, masks_enc, masks_pred, self.state.global_step
            )
        parts["feat_std"] = feature_std(target)
        parts["pred_cos"] = prediction_cosine(z_pred, target, masks_pred)
        return loss, parts, int(clips[0].shape[0])

    def _autocast(self):
        """Return an autocast context on CUDA when AMP is on, else a no-op."""
        if self.use_amp:
            return torch.autocast("cuda", enabled=True)
        import contextlib

        return contextlib.nullcontext()

    # -- bars ----------------------------------------------------------------
    def _render_train(self, bar) -> None:
        # Show the accumulated running average over the epoch, not the noisy
        # single-step value -- matching the eval bar and the periodic step log.
        bar.update(1)
        bar.set_postfix_str(utils.format_metrics(self.train_meter.averages()))

    def _render_eval(self, bar, meter: MetricTracker) -> None:
        bar.update(1)
        bar.set_postfix_str(utils.format_metrics(meter.averages()))

    def _update_epoch_bar(self, bar) -> None:
        """Advance the outer bar and show avg epoch time, best score and lr."""
        self._epochs_done += 1
        best = (f"{self.best.best:.4f}" if self.best.has_best() else "n/a")
        bar.update(1)
        bar.set_postfix_str(
            f"avg_epoch={self._avg_epoch_str()} "
            f"best_{self.best.metric}={best} "
            f"lr={self.scheduler.last_lr:.2e}"
        )

    def _avg_epoch_str(self) -> str:
        """Return the mean wall-clock time per finished epoch as ``MM:SS``."""
        if self._epochs_done <= 0:
            return "n/a"
        seconds = (time.time() - self._epoch_start_time) / self._epochs_done
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    # -- epoch bookkeeping ---------------------------------------------------
    def _prepare_loader(self, loader, epoch: int) -> bool:
        """Prepare a loader for ``epoch``; return True if it started fresh.

        On resume we keep the restored position only when the loader was
        genuinely part-way through *this* epoch (same epoch order, some samples
        already consumed but not all). A checkpoint saved at an epoch boundary
        leaves the loader exhausted for the finished epoch, so we must start the
        next epoch fresh -- otherwise the resumed epoch would iterate an empty
        loader and be silently skipped.
        """
        mid_epoch = (
            self._resumed
            and loader.epoch == epoch
            and 0 < loader.position < loader.sampler.num_samples
        )
        if mid_epoch:
            return False
        loader.set_epoch(epoch)
        return True

    def _record_epoch(self, epoch: int) -> None:
        """Write the history row, plots, best and last weights for an epoch."""
        row = self._epoch_row(epoch)
        self.history.append(row)
        plot_history(self.paths.plotes_dir, self.history.rows)
        self._save_best(row)
        self._save_weights(self.paths.last_pt)

    def _epoch_row(self, epoch: int) -> Dict[str, float]:
        """Build the per-epoch metric row from the train and val meters."""
        row: Dict[str, float] = {"epoch": epoch + 1}
        for name, value in self.train_meter.averages().items():
            row[f"train_{name}"] = round(value, 6)
        for name, value in self.val_meter.averages().items():
            row[f"val_{name}"] = round(value, 6)
        return row

    def _save_best(self, row: Dict[str, float]) -> None:
        """Save best.pt when the watched validation metric improves."""
        key = f"val_{self.best.metric}"
        if key not in row:
            key = f"train_{self.best.metric}"
        value = row.get(key)
        if value is not None and self.best.consider(value):
            self._save_weights(self.paths.best_pt)
            vlog.logger.info("New best {} = {:.4f} -> saved best.pt",
                             self.best.metric, value)

    # -- final evaluation ----------------------------------------------------
    def _final_test(self) -> None:
        """Evaluate on the full test set once, after all epochs."""
        loader = self.bundle.test_loader
        if loader is None:
            return
        vlog.logger.info("===== Final evaluation on full test set =====")
        # ``_eval_pass`` resets the meter only when it starts the pass fresh, so
        # resuming mid-test keeps the averages already computed before the crash.
        self.state.phase = "test"
        self._eval_pass(loader, self.test_meter, "test", "test")
        vlog.logger.info("Test results: {}",
                         utils.format_metrics(self.test_meter.averages()))

    # -- checkpoint / weights ------------------------------------------------
    def _save_weights(self, path: str) -> None:
        """Save only the model weights (for inference / export)."""
        torch.save({"model": self.model.state_dict()}, path)

    def _save_checkpoint(self, phase: str) -> None:
        """Write a full training checkpoint for the current epoch."""
        self.state.phase = phase
        self.ckpt.save(self._checkpoint_state(), self.state.epoch)

    def _checkpoint_state(self) -> Dict[str, Any]:
        """Collect everything needed to resume the run."""
        state = {
            "run_state": vars(self.state),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best": self.best.state_dict(),
            "history": self.history.rows,
            "rng": utils.rng_state(),
        }
        self._add_loader_states(state)
        self._add_meter_states(state)
        return state

    def _add_loader_states(self, state: Dict[str, Any]) -> None:
        """Attach the resumable loader positions to the checkpoint state."""
        state["train_loader"] = self.bundle.train_loader.state_dict()
        if self.bundle.val_loader is not None:
            state["val_loader"] = self.bundle.val_loader.state_dict()
        if self.bundle.test_loader is not None:
            state["test_loader"] = self.bundle.test_loader.state_dict()

    def _add_meter_states(self, state: Dict[str, Any]) -> None:
        """Attach the partial meter states to the checkpoint state."""
        state["train_meter"] = self.train_meter.state_dict()
        state["val_meter"] = self.val_meter.state_dict()
        state["test_meter"] = self.test_meter.state_dict()

    # -- resume --------------------------------------------------------------
    def _maybe_resume(self) -> None:
        """Load the newest checkpoint of this run when one exists."""
        state = self.ckpt.load_latest(map_location=self.device)
        if state is None:
            return
        self._restore(state)
        self._resumed = True
        vlog.logger.info("Resumed from checkpoint: epoch {} phase {} step {}",
                         self.state.epoch + 1, self.state.phase,
                         self.state.global_step)

    def _restore(self, state: Dict[str, Any]) -> None:
        """Restore every piece of state from a checkpoint dict."""
        self.state = RunState(**state["run_state"])
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.best.load_state_dict(state["best"])
        self.history.load(state.get("history", []))
        utils.set_rng_state(state.get("rng", {}))
        self._restore_loaders(state)
        self._restore_meters(state)

    def _restore_loaders(self, state: Dict[str, Any]) -> None:
        """Restore the resumable loader positions from a checkpoint."""
        self.bundle.train_loader.load_state_dict(state["train_loader"])
        if self.bundle.val_loader is not None and "val_loader" in state:
            self.bundle.val_loader.load_state_dict(state["val_loader"])
        if self.bundle.test_loader is not None and "test_loader" in state:
            self.bundle.test_loader.load_state_dict(state["test_loader"])

    def _restore_meters(self, state: Dict[str, Any]) -> None:
        """Restore the partial validation / test meters from a checkpoint."""
        self.train_meter.load_state_dict(state.get("train_meter", {}))
        self.val_meter.load_state_dict(state.get("val_meter", {}))
        self.test_meter.load_state_dict(state.get("test_meter", {}))
