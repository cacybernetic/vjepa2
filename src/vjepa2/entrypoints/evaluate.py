# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The evaluation program. It loads a trained model and measures its quality on
# the whole test set, writing metrics and a few PCA renders into a new eval
# folder.

from __future__ import annotations

import torch

from vjepa2 import logging as vlog
from vjepa2.config import Config
from vjepa2.dataset.factory import build_eval_loader
from vjepa2.dataset.masking import grid_dims
from vjepa2.entrypoints import common
from vjepa2.lossfn import build_loss
from vjepa2.training import Evaluator, build_model
from vjepa2.training import utils

__all__ = ["EvalApp", "main"]


class EvalApp:
    """Wire together every piece needed to evaluate a trained model."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def setup(self) -> None:
        """Resolve a fresh eval folder, logging and seed."""
        self.paths, _ = common.resolve_run(
            self.cfg, "eval", self.cfg.eval.runs_dir, resume=False
        )
        common.save_config_used(self.cfg, self.paths.config_used)
        utils.set_seed(self.cfg.seed)
        utils.configure_backend(self.cfg.device)
        vlog.logger.info("Starting evaluation pipeline")
        common.log_dataset_block(self.cfg)

    def build(self) -> None:
        """Build the test loader, model and loss, then load the weights."""
        self.loader, self.num_test = build_eval_loader(
            self.cfg, self.cfg.eval.batch_size, self.cfg.eval.num_workers
        )
        self.model = build_model(self.cfg, self.cfg.device)
        self._load_weights()
        self.loss_fn = build_loss(self.cfg)
        vlog.log_model_summary(self.model, name="V-JEPA 2.1")
        vlog.log_kv("test samples", self.num_test)

    def _load_weights(self) -> None:
        """Load the model weights named in the eval config."""
        path = self.cfg.eval.weights
        if not path:
            raise ValueError("eval.weights must point to a model .pt file")
        try:
            state = torch.load(path, map_location=self.cfg.device,
                               weights_only=True)
        except Exception:
            vlog.logger.warning(
                "weights at {} are not a plain tensor file; falling back to "
                "weights_only=False (only do this with trusted files)", path)
            state = torch.load(path, map_location=self.cfg.device,
                               weights_only=False)
        result = self.model.load_state_dict(state.get("model", state),
                                            strict=False)
        vlog.logger.info("Loaded model weights from {}", path)
        if result.missing_keys:
            vlog.logger.warning("eval weights: {} missing keys (e.g. {})",
                                len(result.missing_keys),
                                result.missing_keys[:3])
        if result.unexpected_keys:
            vlog.logger.warning("eval weights: {} unexpected keys (e.g. {})",
                                len(result.unexpected_keys),
                                result.unexpected_keys[:3])

    def evaluate(self) -> None:
        """Run the evaluation and save the results."""
        grid_size, grid_depth = grid_dims(
            self.cfg.dataset.crop_size, self.cfg.model.patch_size,
            self.cfg.dataset.num_frames, self.cfg.model.tubelet_size,
        )
        grid = (grid_depth, grid_size, grid_size)
        evaluator = Evaluator(
            self.model, self.loss_fn, self.loader, self.paths,
            self.cfg.device, grid, num_render=self.cfg.eval.num_render,
        )
        evaluator.run()


def main() -> None:
    """Entry point for the ``evalvjepa`` command."""
    args = common.parse_config_arg("evaluation")
    cfg = common.load_config(args.config)
    app = EvalApp(cfg)
    app.setup()
    app.build()
    app.evaluate()


if __name__ == "__main__":
    main()
