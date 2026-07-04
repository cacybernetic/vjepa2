"""Tests for the ONNX encoder exporter (``vjepa2.entrypoints.exportencoder``)."""

import os

import numpy as np
import pytest
import torch

from vjepa2 import build_vjepa2_1_vitb
from vjepa2.entrypoints import exportencoder as ee

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _tiny_checkpoint(path):
    """Save a small, randomly-initialized vitb-shaped checkpoint to ``path``."""
    m = build_vjepa2_1_vitb(crop_size=64, max_num_frames=4, use_sdpa=False)
    torch.save(
        {
            "encoder": m.encoder.state_dict(),
            "predictor": m.predictor.state_dict(),
            "ema_encoder": m.target_encoder.state_dict(),
        },
        path,
    )


def _args(**kw):
    argv = [kw.pop("checkpoint"), "-o", kw.pop("output")]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return ee.build_parser().parse_args(argv)


def test_export_video_encoder_runs_and_matches(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    out = str(tmp_path / "enc.onnx")
    _tiny_checkpoint(ckpt)

    rc = ee.main([ckpt, "-o", out, "--crop-size", "64", "--num-frames", "4"])
    assert rc == 0
    assert os.path.exists(out)

    onnx.checker.check_model(onnx.load(out))

    # 2 temporal * 4 * 4 spatial tokens, 768-d features
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    clip = np.random.randn(1, 3, 4, 64, 64).astype(np.float32)
    (feat,) = sess.run(None, {"clip": clip})
    assert feat.shape == (1, 2 * 4 * 4, 768)
    assert np.isfinite(feat).all()


def test_export_image_pathway(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    out = str(tmp_path / "enc_img.onnx")
    _tiny_checkpoint(ckpt)

    rc = ee.main([ckpt, "-o", out, "--crop-size", "64", "--num-frames", "1"])
    assert rc == 0

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    img = np.random.randn(1, 3, 1, 64, 64).astype(np.float32)
    (feat,) = sess.run(None, {"clip": img})
    assert feat.shape == (1, 1 * 4 * 4, 768)


def test_single_file_has_no_sidecar(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    out = str(tmp_path / "enc_single.onnx")
    _tiny_checkpoint(ckpt)

    rc = ee.main(
        [ckpt, "-o", out, "--crop-size", "64", "--num-frames", "4", "--single-file"]
    )
    assert rc == 0
    assert os.path.exists(out)
    assert not os.path.exists(out + ".data")


def test_missing_checkpoint_returns_error(tmp_path):
    rc = ee.main([str(tmp_path / "nope.pt"), "-o", str(tmp_path / "x.onnx")])
    assert rc == 1


def test_dynamic_batch_serves_multiple_batch_sizes(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    out = str(tmp_path / "enc_dyn.onnx")
    _tiny_checkpoint(ckpt)

    # --dynamic-batch must auto-enable SDPA and produce a genuinely batchable
    # graph (the built-in _verify already exercises B != traced size).
    rc = ee.main(
        [ckpt, "-o", out, "--crop-size", "64", "--num-frames", "4", "--dynamic-batch"]
    )
    assert rc == 0

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].shape[0] in ("batch", "batch_0", -1)
    for b in (1, 2, 5):
        clip = np.random.randn(b, 3, 4, 64, 64).astype(np.float32)
        (feat,) = sess.run(None, {"clip": clip})
        assert feat.shape == (b, 2 * 4 * 4, 768)


def test_metadata_props_embedded(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    out = str(tmp_path / "enc_meta.onnx")
    _tiny_checkpoint(ckpt)

    rc = ee.main([ckpt, "-o", out, "--crop-size", "64", "--num-frames", "4"])
    assert rc == 0

    meta = {p.key: p.value for p in onnx.load(out).metadata_props}
    assert meta["modality"] == "video"
    assert meta["input_layout"] == "NCTHW"
    assert meta["normalization"].startswith("external:")
    assert len(meta["checkpoint_sha256"]) == 64
    assert meta["precision"] == "fp32"


def test_bake_normalization_takes_raw_pixels(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    out = str(tmp_path / "enc_norm.onnx")
    _tiny_checkpoint(ckpt)

    rc = ee.main(
        [ckpt, "-o", out, "--crop-size", "64", "--num-frames", "4",
         "--bake-normalization"]
    )
    assert rc == 0

    meta = {p.key: p.value for p in onnx.load(out).metadata_props}
    assert meta["normalization"].startswith("baked:")

    # Feed a raw [0, 255] clip; the graph normalizes internally.
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    clip = (np.random.rand(1, 3, 4, 64, 64) * 255).astype(np.float32)
    (feat,) = sess.run(None, {"clip": clip})
    assert feat.shape == (1, 2 * 4 * 4, 768)
    assert np.isfinite(feat).all()


def test_geometry_mismatch_returns_error(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    out = str(tmp_path / "enc_bad.onnx")
    _tiny_checkpoint(ckpt)

    # The checkpoint is patch-16 / tubelet-2; a contradicting flag must fail
    # fast rather than silently load a wrong graph (load is strict=False).
    rc = ee.main(
        [ckpt, "-o", out, "--crop-size", "64", "--num-frames", "4",
         "--tubelet-size", "4"]
    )
    assert rc == 1
    assert not os.path.exists(out)
