# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The evaluation program logic. It runs the frozen model over the whole test
# set, reports the self-supervised metrics, writes them to results.csv, and
# saves a few PCA feature-map renders as visual examples.

from __future__ import annotations

import csv
from typing import Dict, List

import torch

from vjepa2 import logging as vlog
from vjepa2.metrics import MetricTracker, feature_std, prediction_cosine
from vjepa2.metrics.ssl_metrics import METRIC_NAMES
from vjepa2.training import utils
from vjepa2.training.render import FeatureMapRenderer

__all__ = ["Evaluator"]


class Evaluator:
    """Evaluate a frozen model on the full test set and save results."""

    def __init__(self, model, loss_fn, test_loader, paths, device: str,
                 grid, num_render: int = 8):
        self.model = model
        self.loss_fn = loss_fn
        self.loader = test_loader
        self.paths = paths
        self.device = device
        self.meter = MetricTracker(METRIC_NAMES)
        self.renderer = FeatureMapRenderer(grid)
        self.num_render = int(num_render)

    def run(self) -> Dict[str, float]:
        """Run the full evaluation and return the average metrics."""
        vlog.logger.info("===== Evaluating on the full test set =====")
        self.model.eval()
        self._score_pass()
        results = self.meter.averages()
        self._write_results(results)
        self._render_examples()
        vlog.logger.info("Evaluation results: {}", utils.format_metrics(results))
        return results

    def _score_pass(self) -> None:
        """Run the model over every test batch and update the meters."""
        self.loader.set_epoch(0)
        bar = vlog.make_step_bar(len(self.loader), desc="evaluate")
        for batch in self.loader:
            parts, n = self._score_batch(batch)
            self.meter.update(parts, n)
            bar.update(1)
            bar.set_postfix_str(utils.format_metrics(self.meter.averages()))
        bar.close()

    def _score_batch(self, batch):
        """Compute the loss and quality metrics for one test batch."""
        clips, masks_enc, masks_pred = batch
        clips = utils.move_clips(clips, self.device)
        masks_enc = utils.move_masks(masks_enc, self.device)
        masks_pred = utils.move_masks(masks_pred, self.device)
        with torch.no_grad():
            z_pred, z_ctx, target = self.model(
                clips, masks_enc, masks_pred, mod="video", training_mode=True
            )
            _, parts = self.loss_fn(
                z_pred, z_ctx, target, masks_enc, masks_pred, 10 ** 9
            )
        parts["feat_std"] = feature_std(target)
        parts["pred_cos"] = prediction_cosine(z_pred, target, masks_pred)
        return parts, int(clips[0].shape[0])

    def _write_results(self, results: Dict[str, float]) -> None:
        """Write the average metrics to results.csv."""
        with open(self.paths.results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for name, value in results.items():
                writer.writerow([name, round(value, 6)])

    def _render_examples(self) -> None:
        """Save a few PCA feature-map renders for a visual sanity check."""
        saved = 0
        for batch in self.loader:
            saved = self._render_batch(batch, saved)
            if saved >= self.num_render:
                break

    def _render_batch(self, batch, saved: int) -> int:
        """Render each clip in a batch until the render budget is reached."""
        clips = batch[0][0]
        for i in range(clips.shape[0]):
            if saved >= self.num_render:
                break
            path = f"{self.paths.renders_dir}/render_{saved:03d}.jpg"
            if self.renderer.render(self.model, clips[i].to(self.device), path):
                saved += 1
        return saved
