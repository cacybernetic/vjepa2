# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Loss test and a small end-to-end trainer test using random clips (no video
# decoding). This checks that the pieces fit together and that a run produces
# the expected output files and can resume.

import os
import time

import torch

from vjepa2.config import Config
from vjepa2.dataset.factory import DataBundle, build_collator
from vjepa2.dataset.dataloader import ResumableDataLoader
from vjepa2.lossfn import build_loss
from vjepa2.training import (
    Trainer,
    build_ema,
    build_model,
    build_optimizer_scheduler,
)
from vjepa2.training.runs import RunDirManager


def _tiny_cfg(tmp_path) -> Config:
    return Config.from_dict({
        "run_name": "unit",
        "device": "cpu",
        "dataset": {"crop_size": 64, "num_frames": 4},
        "masking": {"spatial_scale": [0.3, 0.6]},
        "model": {"name": "vit_tiny", "patch_size": 16, "tubelet_size": 2,
                  "pred_depth": 12, "pred_embed_dim": 96,
                  "n_output_distillation": 4},
        "loss": {"context_lambda": 0.5, "lambda_warmup_start": 100,
                 "lambda_warmup_end": 200},
        "scheduler": {"name": "warmup_hold", "warmup_steps": 2},
        "train": {"epochs": 1, "batch_size": 2, "grad_accum": 2,
                  "num_workers": 0, "log_every": 1, "ckpt_step": 2,
                  "max_checkpoint": 2, "runs_dir": str(tmp_path / "runs"),
                  "best_metric": "loss", "best_mode": "min"},
    })


class _RandomClips(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        torch.manual_seed(i)
        return torch.randn(3, 4, 64, 64)


def _bundle(cfg):
    collate = build_collator(cfg)
    train = ResumableDataLoader(_RandomClips(4), 2, True, collate, seed=cfg.seed)
    val = ResumableDataLoader(_RandomClips(2), 2, False, collate, seed=cfg.seed)
    test = ResumableDataLoader(_RandomClips(2), 2, False, collate, seed=cfg.seed)
    return DataBundle(train, val, test, 4, 2, 2)


def test_dense_loss_and_forward_shapes():
    cfg = Config.from_dict({
        "dataset": {"crop_size": 64, "num_frames": 4},
        "model": {"name": "vit_tiny", "patch_size": 16, "tubelet_size": 2,
                  "pred_embed_dim": 96, "n_output_distillation": 4},
        "masking": {"spatial_scale": [0.3, 0.6]},
        "loss": {"lambda_warmup_start": 100, "lambda_warmup_end": 200},
    })
    model = build_model(cfg, "cpu")
    loss_fn = build_loss(cfg)
    collate = build_collator(cfg)
    clips, enc, pred = collate([torch.randn(3, 4, 64, 64) for _ in range(2)])
    start = time.time()
    z_pred, z_ctx, target = model(clips, enc, pred, training_mode=True)
    loss, parts = loss_fn(z_pred, z_ctx, target, enc, pred, global_iter=0)
    assert loss.item() > 0.0
    assert parts["lambda"] == 0.0            # before the warmup window
    assert set(parts) >= {"loss", "predict", "context", "lambda"}
    assert time.time() - start < 60.0        # a tiny forward must be quick


def test_trainer_runs_and_writes_outputs(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    model = build_model(cfg, "cpu")
    optimizer, scheduler, _ = build_optimizer_scheduler(model, cfg, 10)
    bundle = _bundle(cfg)
    paths = RunDirManager(cfg.train.runs_dir, cfg.run_name).make_paths(
        os.path.join(cfg.train.runs_dir, cfg.run_name, "train")
    )
    trainer = Trainer(model, build_loss(cfg), optimizer, scheduler,
                      build_ema(cfg), bundle, paths, cfg.train, "cpu")
    trainer.run()
    assert os.path.isfile(paths.best_pt)
    assert os.path.isfile(paths.last_pt)
    assert os.path.isfile(paths.history_csv)
    assert trainer.ckpt.has_checkpoint()


def test_trainer_resumes_from_checkpoint(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    run_path = os.path.join(cfg.train.runs_dir, cfg.run_name, "train")
    paths = RunDirManager(cfg.train.runs_dir, cfg.run_name).make_paths(run_path)

    def make_trainer():
        model = build_model(cfg, "cpu")
        optimizer, scheduler, _ = build_optimizer_scheduler(model, cfg, 10)
        return Trainer(model, build_loss(cfg), optimizer, scheduler,
                       build_ema(cfg), _bundle(cfg), paths, cfg.train, "cpu")

    make_trainer().run()
    # A second trainer over the same folder resumes past the finished epoch.
    resumed = make_trainer()
    resumed._maybe_resume()
    assert resumed.state.epoch >= 1
