<div align="center">

<img src="banner.png" width="720" alt="V-JEPA 2.1"/>

![](https://img.shields.io/badge/STATUS-alpha-orange)
![](https://img.shields.io/badge/Python-3.10-blue)
![](https://img.shields.io/badge/PyTorch-2.8.0-orange)
![](https://img.shields.io/badge/LICENSE-MIT-%2300557f)
![](https://img.shields.io/badge/latest-2026--07--05-green)

</div>

A clean reimplementation of **V-JEPA 2.1** (*Unlocking Dense Features in Video
Self-Supervised Learning*). V-JEPA learns image and video representations with
**no labels**: a JEPA-style encoder and predictor are trained so the predictor
matches the representations of a slow **EMA target encoder** at masked
spatio-temporal positions. PyTorch is used to build the model; the encoder is
exported to ONNX for fast, framework-agnostic feature extraction.

**Table of Contents**

- [Description](#description)
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
  - [Quick install](#quick-install-without-cloning)
  - [Python — Linux](#python--linux)
  - [Python — Windows](#python--windows)
  - [ONNX (optional)](#onnx-optional)
- [Model weights](#model-weights)
- [Usage](#usage)
  - [1. Export the encoder to ONNX](#1-export-the-encoder-to-onnx)
  - [2. Choosing the weights: `--weights {ema,online}`](#2-choosing-the-weights---weights-emaonline)
  - [3. Run inference on images or videos](#3-run-inference-on-images-or-videos)
- [ONNX export constraints](#onnx-export-constraints)
- [Roadmap](#roadmap)
- [To contribute](#to-contribute)
- [Licence](#licence)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

---

## Description

Modern video encoders that learn by reconstructing pixels waste capacity on
low-level detail that never helps downstream tasks. JEPA sidesteps this by
predicting in **representation space** instead of pixel space.

V-JEPA 2.1 works as follows:

1. A tokenizer turns an image or a short clip into spatio-temporal tokens
   (a 3D patch embedding, tubelet size 2 for video, plus a modality embedding
   so one model handles both images and videos).
2. A **RoPE ViT encoder** (3D rotary position embeddings) builds token
   representations, with hierarchical LayerNorm outputs.
3. A block of tokens is masked; a lightweight **predictor** fills in the masked
   positions from the visible context and the mask tokens.
4. The training target is the representation of the *same* input produced by a
   **slow EMA copy of the encoder** (stop-gradient). The predictor is trained to
   match it with a dense L1 objective.

After training, only the encoder is kept — the predictor and the online/target
bookkeeping are discarded. The encoder is exported to ONNX and used as a
general-purpose feature extractor: feed it an image or a clip, get dense token
features (or a pooled embedding) to hand to a small task head.

## Features

- **Multi-modal RoPE ViT encoder** — a single architecture for images
  (`T = 1`) and video (`T > 1`), with 3D rotary position embeddings, a modality
  embedding, and hierarchical LayerNorm outputs.
- **Full V-JEPA 2.1 assembly** — online encoder + predictor + frozen EMA target
  encoder, with the dense L1 / per-level LayerNorm target normalization from the
  reference recipe (`VJEPA21` in `model.py`).
- **Checkpoint loader** for the distilled ViT-B / teacher ViT-G weights
  (`build_vjepa2_1_vitb`).
- **ONNX export** of the encoder with numerical cross-check against PyTorch,
  optional **fp16**, **dynamic batch**, **baked-in normalization**, and a
  **single-file** artifact for deployment.
- **Self-describing graphs** — the exporter embeds geometry, the normalization
  convention, and provenance (checkpoint SHA-256, opset, torch version) in the
  ONNX `metadata_props`, so the inference client configures itself.
- **Torch-free inference** — the `runs` CLI depends only on NumPy, Pillow, PyAV,
  and `onnxruntime`; no `torch` at runtime. It encodes a single file or a whole
  directory (recursively) to pickle / NumPy / HDF5.

## Project structure

```
.
├── README.md
├── Makefile                    # install (CPU/CUDA/ROCm), test
├── pyproject.toml              # package metadata + CLI entry points
├── assets/                     # sample clip.mp4 and pic.png
├── weights/                    # pretrained checkpoints (.pt), not tracked
├── src/
│   └── vjepa2/
│       ├── model.py            # VJEPA21, init_video_model, build_vjepa2_1_vitb
│       ├── logging.py          # logging setup
│       ├── modules/
│       │   ├── vision_transformer.py  # the RoPE ViT encoder
│       │   ├── predictor.py           # the multi-level predictor
│       │   ├── attention.py           # RoPE / SDPA attention
│       │   ├── blocks.py              # transformer blocks
│       │   ├── mlp.py                 # MLP / SiLU feed-forward
│       │   ├── patch_embed.py         # 3D tokenizer
│       │   ├── pos_embs.py            # rotary / sincos position embeddings
│       │   ├── losses.py              # dense prediction loss
│       │   ├── tensors.py            # masking / gather helpers
│       │   └── wrappers.py           # multi-sequence encoder/predictor wrappers
│       ├── training/           # training engine (work in progress)
│       └── entrypoints/
│           ├── exportencoder.py       # exportw: export the encoder to ONNX
│           └── inference.py           # runs: standalone ONNX inference
└── tests/                      # unit tests (model, modules, predictor, export, ...)
```

---

## Installation

### Quick install (without cloning)

You can install the package directly from GitHub with either `pip` or `uv`. This
gives you the CLI tools (`exportw`, `runs`) without downloading the full
repository.

**With pip:**

```bash
pip install git+https://github.com/mokira3d48/vjepa2
```

**With uv** (faster, after installing `uv`):

```bash
uv pip install git+https://github.com/mokira3d48/vjepa2
```

> **Note for contributors**: if you plan to modify the code or contribute,
> please follow the full local installation instructions below.

### Python — Linux

**1. Install `uv` (fast Python package manager)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository**

```bash
git clone https://github.com/mokira3d48/vjepa2
cd vjepa2
```

**3. Create a virtual environment with Python 3.10**

```bash
uv venv --python 3.10
source .venv/bin/activate
```

**4. Install PyTorch for your hardware, then the package**

The `Makefile` picks the right PyTorch build for your machine and installs the
project (editable), registering the command-line tools. Each target installs
both `torch` and `torchvision` from the matching index, so the two always agree.

```bash
make install        # CPU only
make cuda_install   # NVIDIA CUDA 12.4
make rocm_install   # AMD ROCm 6.2
```

> **Important — always pick the build that matches your hardware.** `torch` and
> `torchvision` ship per-hardware wheels (CPU, CUDA, ROCm). If you let `pip`
> install them from the default index, you may get a CUDA build on a machine
> with no GPU, and hit a `libcudart.so: cannot open shared object file` error at
> import time. Reinstall both from the right index if that happens:
> ```bash
> pip uninstall -y torch torchvision
> # CPU only (no GPU):
> pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
> # NVIDIA CUDA 12.4 (check your driver with nvidia-smi):
> pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu124
> ```
> The `make` targets already do this for you.

Then run the tests to check everything works:

```bash
make test
```

### Python — Windows

1. Download and install Python 3.10 from [python.org](https://www.python.org/downloads/).
2. Open a command prompt inside the project folder.
3. Install `uv`:
   ```bash
   pip install uv
   ```
4. Create the virtual environment:
   ```bash
   uv venv --python 3.10
   .venv\Scripts\activate
   ```
5. Install PyTorch for your hardware first, then the package:
   ```bash
   # CPU only (no GPU):
   uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
   # or NVIDIA CUDA 12.4 (check your driver with nvidia-smi):
   uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu124
   uv pip install -e .
   ```

### ONNX (optional)

Needed to export the encoder and run the standalone inference script. Skip this
if you only work with the PyTorch model.

```bash
uv pip install -e ".[onnx]"
```

This adds `onnx`, `onnxruntime`, `onnxscript`, and `onnxconverter-common` (the
last one is only required for `--fp16`).

---

## Model weights

The distilled ViT-B checkpoint (teacher ViT-G, 384) is loaded by
`build_vjepa2_1_vitb` and exported by `exportw`. Place it under `weights/`:

```
weights/
  vjepa2_1_vitb_dist_vitG_384.pt
```

A checkpoint is a dict with `encoder`, `predictor`, and one of
`ema_encoder` / `target_encoder` state dicts. Loading is non-strict, and the
exporter validates the checkpoint geometry (patch size, tubelet size, embed dim)
against the requested model before tracing, so a mismatched flag fails fast
instead of producing a silently wrong graph.

---

## Usage

| Command   | Job                                       | Example                                            |
|-----------|-------------------------------------------|----------------------------------------------------|
| `exportw` | Export the encoder to ONNX                | `exportw weights/vjepa2_1_vitb_dist_vitG_384.pt -o encoder.onnx` |
| `runs`    | Standalone ONNX inference on files        | `runs -m encoder.onnx -i assets/clip.mp4 -o clip.pkl` |

### 1. Export the encoder to ONNX

```bash
exportw weights/vjepa2_1_vitb_dist_vitG_384.pt -o encoder.onnx
```

Only the encoder is exported (the predictor and target-encoder bookkeeping are
for training). Useful flags:

```bash
# image pathway (T = 1) instead of video, with a dynamic batch dimension:
exportw weights/vjepa2_1_vitb_dist_vitG_384.pt --num-frames 1 --dynamic-batch -o image_encoder.onnx

# a single deployable artifact, fp16 weights, normalization folded into the graph:
exportw weights/vjepa2_1_vitb_dist_vitG_384.pt --single-file --fp16 --bake-normalization -o encoder.onnx
```

| Flag                  | Effect                                                                             |
|-----------------------|------------------------------------------------------------------------------------|
| `--weights {ema,online}` | Which encoder to export (see below). Default: `ema`.                            |
| `--num-frames N`      | Frames per clip, **baked into the graph**. `1` exports the image pathway.           |
| `--crop-size N`       | Input resolution, baked in (default `256`).                                        |
| `--dynamic-batch`     | Make the batch dimension dynamic (one graph, any batch). Auto-enables `--sdpa`.    |
| `--sdpa`              | Trace attention with fused scaled-dot-product-attention.                           |
| `--bake-normalization`| Fold ImageNet normalization into the graph; the model then takes raw `[0,255]` RGB.|
| `--fp16`              | fp16 weights (I/O stays fp32); needs `onnxconverter-common`.                        |
| `--single-file`       | Embed the weights in the `.onnx` instead of a sidecar `.onnx.data`.                |
| `--no-verify`         | Skip the onnxruntime-vs-PyTorch numerical cross-check.                              |

After tracing, the exporter runs the graph on `onnxruntime` and checks its
output against PyTorch (max `|Δ|` under `--tol`, default `1e-2`), then embeds the
geometry, normalization convention, and provenance in the ONNX `metadata_props`.

### 2. Choosing the weights: `--weights {ema,online}`

A trained V-JEPA 2.1 model carries **two copies of the encoder**, and this flag
selects which one is written to the `.onnx`:

| `--weights` | Internal module        | Role during self-supervised training                                      |
|-------------|------------------------|---------------------------------------------------------------------------|
| `ema` *(default)* | `model.target_encoder` | The **EMA target encoder** — a *frozen* copy (`requires_grad=False`) updated as an exponential moving average of the online encoder. It produces the target representations the predictor is trained to match. |
| `online`    | `model.encoder`        | The **online context encoder** — the one optimized directly by gradient descent; it sees the masked input. |

The whole choice comes down to a single line in the exporter:

```python
wrapper = model.target_encoder if args.weights == "ema" else model.encoder
```

**Why `ema` is the default.** As in BYOL / DINO / data2vec, the EMA encoder
averages out the online encoder's step-to-step noise, so it yields **more stable
and higher-quality representations**. For feature extraction — the intended use
of the exported model — that is the encoder you want.

**When to pick `online`.** Mostly for debugging or comparison: measuring the
online-vs-EMA gap, or exporting from a checkpoint where only the online weights
are trustworthy. For serving the model, keep the default.

> For a fully distilled checkpoint the two weight sets are often nearly
> identical by the end of training, but `ema` remains the canonical choice.

### 3. Run inference on images or videos

`runs` is fully self-contained and **torch-free**: it imports only `numpy`,
`Pillow`, `av` (PyAV), and `onnxruntime`, so you can copy it into another
project.

```bash
# a single file:
runs -m encoder.onnx -i assets/clip.mp4 -o clip.pkl

# a whole directory, one NumPy embedding per input:
runs -m encoder.onnx -d videos/ --output-dir embeddings/ -f npy
```

It reads the crop size, frame count, and normalization convention straight from
the ONNX `metadata_props`, so it matches how the model was exported without extra
flags. Useful options:

| Flag                   | Effect                                                            |
|------------------------|------------------------------------------------------------------|
| `-i` / `-d`            | Single input file / directory of inputs (recursive by default).  |
| `-o` / `--output-dir`  | Output file (single input) / output directory (one file each).   |
| `-f {pkle,npy,h5}`     | Serialization format (default `pkle`).                           |
| `--pooling {mean,none}`| `mean` pools tokens into one vector; `none` keeps dense features.|
| `--crop-size` / `--num-frames` | Override the geometry read from the ONNX metadata.       |

The output features are what you feed to a small downstream head (classification,
detection, segmentation, ...).

---

## ONNX export constraints

The RoPE ViT export has a few hard, non-obvious limits — worth knowing before you
export:

- **Dynamic batch requires SDPA.** `--dynamic-batch` alone fails to lower to
  ONNX (the explicit attention matmul will not decompose under a symbolic batch
  dimension), so `--dynamic-batch` automatically enables `--sdpa`, which lowers
  cleanly.
- **Frames and resolution cannot be made dynamic.** RoPE builds its position
  grid from the traced `T`, `H`, `W`, so those are **baked in** at export time.
  Only the batch dimension can be dynamic — re-export per geometry.
- **One export = one modality.** `--num-frames 1` traces the **image** pathway;
  `--num-frames > 1` traces the **video** pathway. A single `.onnx` serves images
  *or* videos, never both.

---

## Roadmap

The model, the ONNX exporter (`exportw`), and the ONNX inference client (`runs`)
are implemented and tested. The self-supervised **training pipeline is a work in
progress**; the following entry points are declared in `pyproject.toml` but not
yet implemented:

- `buildh5ds` (`entrypoints/buildds.py`) — build a ready-to-train HDF5 dataset.
- `trainvjepa` (`entrypoints/train.py`) — the masked-prediction training loop
  (online encoder + predictor, EMA target, dense L1 loss).
- `evalvjepa` (`entrypoints/evaluate.py`) — downstream evaluation of the learned
  representations.

The `training/` package (trainer, optimizers, schedulers, checkpointing) is
scaffolded and being filled in.

---

## To contribute

Contributions are welcome! Please follow these steps:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add a new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request.

## Licence

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.

## Acknowledgments

This project was built while studying the inner workings of V-JEPA 2. A big
thank-you to the **Meta AI (FAIR)** team behind V-JEPA and V-JEPA 2, and to the
reference implementation [**facebookresearch/vjepa2**](https://github.com/facebookresearch/vjepa2),
which served as the primary reference for the architecture and training recipe.

If you find this project useful, please consider giving the original **vjepa2**
repository a star as a token of appreciation for the work that made it possible.

## References

The implementation is based on the following papers and resources:

- **V-JEPA 2.1** — *Unlocking Dense Features in Video Self-Supervised Learning*.
  The paper this repository reimplements (see `resources/papers/V-JEPA2.1/`).
- **V-JEPA 2** — Assran, M., et al. (2025). *V-JEPA 2: Self-Supervised Video
  Models Enable Understanding, Prediction and Planning*.
  [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
- **V-JEPA** — Bardes, A., et al. (2024). *Revisiting Feature Prediction for
  Learning Visual Representations from Video*.
  [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
- **I-JEPA** — Assran, M., et al. (2023). *Self-Supervised Learning from Images
  with a Joint-Embedding Predictive Architecture*. CVPR 2023.
  [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- **JEPA** — LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*.
  The encoder-predictor pattern with a learned mask token.
- **RoPE** — Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary
  Position Embedding*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **data2vec** — Baevski, A., et al. (2022). *data2vec: A General Framework for
  Self-Supervised Learning*. ICML 2022. The EMA target-encoder recipe.
  [arXiv:2202.03555](https://arxiv.org/abs/2202.03555)

## Contact

For questions or suggestions:

- **Author**: Dr Mokira — arnoldmokira3d48@gmail.com
- **Maintainer**: CONSOLE ART CYBERNETIC — ca.cybernetic@gmail.com
- **GitHub**: [mokira3d48/vjepa2](https://github.com/mokira3d48/vjepa2)
