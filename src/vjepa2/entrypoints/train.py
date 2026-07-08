# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The training program. It reads a YAML config, builds the data, the model and
# the training pieces, prints a clear run summary, then trains with validation,
# checkpointing and resume. Priority for a warm start is: checkpoint first, then
# a plain init-weights file only when there is no checkpoint yet.

from __future__ import annotations

import torch

from vjepa2 import logging as vlog
from vjepa2.config import Config
from vjepa2.dataset.factory import build_data_bundle
from vjepa2.entrypoints import common
from vjepa2.lossfn import build_loss
from vjepa2.training import (
    Trainer,
    build_ema,
    build_model,
    build_optimizer_scheduler,
)
from vjepa2.training import utils
from vjepa2.training.checkpoint import CheckpointManager

__all__ = ["TrainApp", "main"]


class TrainApp:
    """Wire together every piece needed to train the model."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def setup(self) -> None:
        """Resolve the run folder, logging and random seed."""
        self.paths, self.reused = common.resolve_run(
            self.cfg, "train", self.cfg.train.runs_dir, self.cfg.train.resume
        )
        common.save_config_used(self.cfg, self.paths.config_used)
        utils.set_seed(self.cfg.seed)
        vlog.logger.info("Starting training pipeline")
        common.log_dataset_block(self.cfg)

    def build(self) -> None:
        """Build the data bundle, model, optimizer, scheduler, ema and loss."""
        self.bundle = build_data_bundle(
            self.cfg, self.cfg.train.batch_size, self.cfg.train.num_workers
        )
        self.model = build_model(self.cfg, self.cfg.device)
        self._warm_start()
        self.per_epoch, self.total_steps = common.optimizer_steps(
            self.bundle.num_train, self.cfg.train.batch_size,
            self.cfg.train.grad_accum, self.cfg.train.epochs,
        )
        self.optimizer, self.scheduler, self.counts = build_optimizer_scheduler(
            self.model, self.cfg, self.total_steps
        )
        self.ema = build_ema(self.cfg)
        self.loss_fn = build_loss(self.cfg)

    def _warm_start(self) -> None:
        """Load init weights only when there is no checkpoint to resume from."""
        manager = CheckpointManager(
            self.paths.checkpoints_dir, self.cfg.train.max_checkpoint
        )
        if manager.has_checkpoint():
            vlog.logger.info("Checkpoint found: training will resume from it")
            return
        init_path = self.cfg.train.init_weights
        if init_path:
            state = torch.load(init_path, map_location=self.cfg.device,
                               weights_only=False)
            self.model.load_state_dict(state.get("model", state), strict=False)
            vlog.logger.info("Loaded init weights from {}", init_path)

    def summarize(self) -> None:
        """Print the run summary and the model architecture summary."""
        counts = vlog.log_model_summary(self.model, name="V-JEPA 2.1")
        self._log_run_summary(counts)

    def _log_run_summary(self, counts) -> None:
        """Print the key hyper-parameters of this run."""
        vlog.logger.info("===== Run summary =====")
        vlog.log_kv("device", self.cfg.device)
        vlog.log_kv("epochs", self.cfg.train.epochs)
        vlog.log_kv("batch_size", self._batch_line())
        vlog.log_kv("optimizer_steps/epoch", self.per_epoch)
        vlog.log_kv("total optimizer steps", self.total_steps)
        vlog.log_kv("grad_clip_norm", self.cfg.train.grad_clip_norm)
        vlog.log_kv("optimizer", self._optimizer_line())
        vlog.log_kv("scheduler", self.cfg.scheduler.name)
        vlog.log_kv("best criterion",
                    f"{self.cfg.train.best_metric} (mode={self.cfg.train.best_mode})")
        vlog.log_kv("param groups", self.counts)
        vlog.log_kv("model parameters", counts)
        vlog.log_kv("data", self._data_line())
        vlog.log_kv("checkpoint dir",
                    f"{self.paths.checkpoints_dir} (max={self.cfg.train.max_checkpoint})")
        vlog.log_kv("outputs", self.paths.root)

    def _batch_line(self) -> str:
        """Return the batch-size / accumulation / effective summary."""
        effective = self.cfg.train.batch_size * self.cfg.train.grad_accum
        return (f"{self.cfg.train.batch_size} x grad_accum="
                f"{self.cfg.train.grad_accum} -> effective={effective}")

    def _optimizer_line(self) -> str:
        """Return a short optimizer description."""
        return (f"{self.cfg.optim.name} lr={self.cfg.optim.lr:.3e} "
                f"wd={self.cfg.optim.weight_decay}")

    def _data_line(self) -> str:
        """Return the train / val / test sample counts."""
        return (f"train {self.bundle.num_train} | val {self.bundle.num_val} "
                f"| test {self.bundle.num_test}")

    def train(self) -> None:
        """Run the training loop."""
        trainer = Trainer(
            self.model, self.loss_fn, self.optimizer, self.scheduler, self.ema,
            self.bundle, self.paths, self.cfg.train, self.cfg.device,
            amp=self.cfg.amp,
        )
        trainer.run()


def main() -> None:
    """Entry point for the ``trainvjepa`` command."""
    args = common.parse_config_arg("training")
    cfg = common.load_config(args.config)
    app = TrainApp(cfg)
    app.setup()
    app.build()
    app.summarize()
    app.train()


if __name__ == "__main__":
    main()
