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
  directory (recursively) to pickle / NumPy / HDF5, and can **chunk long videos**
  into a sliding window of consecutive clips (`--chunk` / `--stride`).

## Project structure

```
.
├── README.md
├── docs/                       # beginner guides (en_concepts.md, fr_concepts.md)
├── Makefile                    # install (CPU/CUDA/ROCm), test
├── pyproject.toml              # package metadata + CLI entry points
├── cpu/configs/                # CPU YAML configs (hdf5, train, eval, export)
├── gpu/configs/                # GPU (NVIDIA/AMD) YAML configs
├── assets/                     # sample clip.mp4 and pic.png
├── weights/                    # pretrained checkpoints (.pt), not tracked
├── src/
│   └── vjepa2/
│       ├── model.py            # VJEPA21, init_video_model, build_vjepa2_1_vitb
│       ├── config.py           # typed configuration objects (YAML -> dataclasses)
│       ├── lossfn.py           # DensePredictiveLoss (predict + weighted context)
│       ├── optimizers.py       # optimizer + weight-decay parameter groups
│       ├── lr_shedulers.py     # warmup-hold / warmup-cosine schedulers
│       ├── logging.py          # loguru logging + tqdm progress bars
│       ├── plotting.py         # train vs validation history curves
│       ├── onnx_export.py      # reusable ONNX export helpers
│       ├── metrics/            # meters + self-supervised quality signals
│       ├── dataset/            # discovery, cleaning/cache, transforms, masking,
│       │                       # HDF5, and the resumable data loader
│       ├── modules/            # the model components (ViT, predictor, ...)
│       ├── training/           # runs, checkpoints, EMA, Trainer, Evaluator
│       └── entrypoints/
│           ├── buildds.py            # buildh5ds: build HDF5 datasets
│           ├── train.py             # trainvjepa: training program
│           ├── evaluate.py          # evalvjepa: evaluation program
│           ├── exportencoder.py     # exportw: export the encoder to ONNX
│           └── inference.py         # runs: standalone ONNX inference
└── tests/                      # unit tests (dataset, metrics, training, ...)
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

| Command     | Job                                          | Example                                            |
|-------------|----------------------------------------------|----------------------------------------------------|
| `buildh5ds` | Build HDF5 datasets from videos              | `buildh5ds --config cpu/configs/hdf5.yaml`         |
| `trainvjepa`| Train the model (train + val + final test)   | `trainvjepa --config cpu/configs/train.yaml`       |
| `evalvjepa` | Evaluate a trained model on the full test set| `evalvjepa --config cpu/configs/eval.yaml`         |
| `exportw`   | Export the encoder to ONNX                   | `exportw weights/vjepa2_1_vitb_dist_vitG_384.pt -o encoder.onnx` |
| `runs`      | Standalone ONNX inference on files           | `runs -m encoder.onnx -i assets/clip.mp4 -o clip.pkl` |

---

## Training and evaluation

New to the project? Read the beginner guides first:
[`docs/en_concepts.md`](docs/en_concepts.md) (English) and
[`docs/fr_concepts.md`](docs/fr_concepts.md) (Français). They explain the model
and the whole pipeline with simple words and analogies.

### Dataset layout

A dataset is a folder or a `.zip` archive that holds video files. Files may sit
at the root or in any sub-folder, in any common format. Point the config at your
train and test sources:

```yaml
dataset:
  train_path: /path/to/train.zip   # or a folder
  test_path:  /path/to/test.zip
  val_prob: 0.5                     # fraction of test used for validation
```

The first run **scans and validates** every file (dropping corrupt ones) and
writes a `train.cache.json` / `test.cache.json` next to the dataset. Later runs
reuse the cache, so the slow scan happens only once.

### 1. (Optional) Build HDF5 files for speed

Decoding video is slow. You can pre-process every clip once and store the result:

```bash
buildh5ds --config cpu/configs/hdf5.yaml   # writes train.h5 and test.h5
```

Then set `dataset.use_hdf5: true` in the train/eval config to read ready clips.

### 2. Train

```bash
trainvjepa --config cpu/configs/train.yaml     # CPU
trainvjepa --config gpu/configs/train.yaml     # GPU (device: cuda or rocm)
```

Each epoch runs **training** then **validation** (on a fraction of the test set);
after the last epoch the model is **evaluated on the full test set**. The program
prints a run summary, a model summary, and two progress bars (epoch + step).

Key training features, all controlled from the config:

- **Gradient accumulation** (`grad_accum`) to simulate a large batch; any
  leftover accumulation at the end of an epoch is still applied.
- **Checkpointing** every `ckpt_step` optimizer steps into `checkpoints/`, keeping
  at most `max_checkpoint` files. Save contents: model, optimizer, scheduler,
  data-loader positions, partial meters and the run state.
- **Resume** (`resume: true`): if a checkpoint exists, the run reuses the latest
  `runs/<name>/train*/` folder and continues at the exact same batch (in-epoch
  checkpointing), for all three passes. Priority is **checkpoint first**, then a
  `init_weights` file only when there is no checkpoint.
- **Best model**: after each validation, `best.pt` is saved when the chosen
  `best_metric` improves (`best_mode: min|max`). `last.pt` is always the latest.
- **History plots**: `plotes/training_history.jpg` shows train vs validation
  curves to spot overfitting.

Outputs land in `runs/<run_name>/train/` (then `train2`, `train3`, ... for new
runs):

```
runs/<run_name>/train/
  history.csv  config_used.yaml
  weights/{best.pt,last.pt}
  checkpoints/epoch_000.pth ...
  plotes/training_history.jpg
  logs/train_YYYY-MM-DD_HH-MM-SS.log
```

#### The context-loss weight `lambda` (important!)

V-JEPA 2.1's key idea is the **dense loss**: on top of predicting the *masked*
tokens, it also checks the *visible* (context) tokens. The total loss is:

```
total = predict_loss  +  lambda * context_loss
        (masked tokens)          (visible/context tokens)
```

`lambda` is **not constant**. It follows a **linear warmup**, measured in
**optimizer steps** (not epochs, not micro-batches):

| step range                                  | value of `lambda`                 |
|---------------------------------------------|-----------------------------------|
| `step < lambda_warmup_start`                | `0`                               |
| `lambda_warmup_start .. lambda_warmup_end`  | ramps `0 -> context_lambda`       |
| `step >= lambda_warmup_end`                 | `context_lambda` (e.g. `0.5`)     |

The warmup exists on purpose: turning the context loss on too strongly, too
early, lets the model cheat (just copy the visible features) and lose global
understanding. Ramping it up slowly keeps training stable (this follows the
paper).

> **Why is my `lambda` stuck at `0.0000` in the logs?**
>
> Because the run has **fewer optimizer steps than `lambda_warmup_start`**. The
> warmup never even begins, so `lambda` stays `0` the whole time. The context
> loss is still *computed* (you see `context=...` in the logs) but it is
> multiplied by `0`, so it has **no effect on the gradients** — you are then
> training a plain "predict-only" V-JEPA and missing the whole 2.1 contribution.
>
> This is common on tiny / demo datasets. Example: `100` train samples,
> `batch_size=2`, `grad_accum=8`, `epochs=10` gives
> `optimizer_steps/epoch = ceil(ceil(100/2)/8) = 7`, so only **70** optimizer
> steps in total. With the default `lambda_warmup_start: 1000`, step 1000 is
> never reached and `lambda` never leaves `0`.

**How to size the warmup.** First read `optimizer_steps/epoch` and
`total optimizer steps` from the run summary the program prints, or compute:

```
optimizer_steps/epoch = ceil(ceil(num_train / batch_size) / grad_accum)
total optimizer steps = optimizer_steps/epoch * epochs
```

Then set `lambda_warmup_start` / `lambda_warmup_end` **well below** the total.
Examples:

```yaml
# Small / demo run (a few dozen steps): turn the context loss on almost at once
loss:
  context_lambda: 0.5
  lambda_warmup_start: 0     # active from the first step
  lambda_warmup_end: 30      # reaches 0.5 by step 30

# Full-scale run (~135k steps, paper-like): a gentle early ramp
loss:
  context_lambda: 0.5
  lambda_warmup_start: 15000
  lambda_warmup_end: 30000
```

Rule of thumb: `lambda_warmup_end` should be a small fraction of
`total optimizer steps` so `lambda` actually reaches `context_lambda` and the
dense loss does its job.

### 3. Evaluate

```bash
evalvjepa --config cpu/configs/eval.yaml
```

This measures the frozen model on the **whole** test set, writes `results.csv`,
and saves a few PCA feature-map renders under `renders/`, into a new
`runs/<run_name>/eval*/` folder.

### 4. Export the encoder to ONNX

```bash
exportw weights/vjepa2_1_vitb_dist_vitG_384.pt -o encoder.onnx
```

See `cpu/configs/export.yaml` / `gpu/configs/export.yaml` for the recommended
flags for a model you trained yourself.

---

## Encoder export details

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
| `--chunk`              | Cut a long video into consecutive clips instead of subsampling it (see below). |
| `--stride N`           | Hop in frames between consecutive clips in `--chunk` mode (default: `num_frames`). |
| `--crop-size` / `--num-frames` | Override the geometry read from the ONNX metadata.       |

The output features are what you feed to a small downstream head (classification,
detection, segmentation, ...).

#### Input and output shapes

The ONNX encoder takes a clip of a **fixed** geometry — `num_frames` frames at
`crop_size × crop_size`, baked in at export time (see
[ONNX export constraints](#onnx-export-constraints)) — laid out as `NCTHW`:

```
input clip:  (B, 3, T, crop, crop)         e.g. (1, 3, 16, 256, 256)
raw output:  (B, N, D)                      dense token features
             N = (T / tubelet) * (crop / patch)^2      D = encoder embed dim
```

For the shipped ViT-B (`T=16`, `crop=256`, `patch=16`, `tubelet=2`): `N =
(16/2) * (256/16)^2 = 8 * 256 = 2048` tokens of `D = 768`, so one clip encodes to
`(1, 2048, 768)`. `--pooling` then reduces the per-clip token axis `N`:

| `--pooling` | Per-clip embedding | Meaning                          |
|-------------|--------------------|----------------------------------|
| `mean` *(default)* | `(D,)` — e.g. `(768,)`      | one vector per clip (token-averaged) |
| `none`      | `(N, D)` — e.g. `(2048, 768)` | dense per-token features         |

The output is always saved as a single array per input file (in the format set by
`-f`). Its shape depends on how many clips the video produced:

| Input | Mode | Clips | Saved shape (`mean` / `none`) |
|-------|------|-------|-------------------------------|
| image | — | 1 (`T=1` pathway) | `(768,)` / `(N, 768)` |
| video | subsample *(default)* | 1 | `(768,)` / `(2048, 768)` |
| video | `--chunk` | `K` | `(K, 768)` / `(K, 2048, 768)` |

A single clip keeps the plain shape (no leading axis); `--chunk` adds a leading
**clip axis** `K` (the video's temporal sequence of embeddings).

#### Long videos: subsample vs. `--chunk`

By default `runs` **subsamples** the whole video to exactly `num_frames`
uniformly-spaced frames, so it always makes **one** clip regardless of length. A
1250-frame video and a 16-frame one both collapse to a single 16-frame clip — you
get one embedding for the entire video and lose all temporal resolution beyond
those 16 samples.

`--chunk` keeps the temporal resolution: the video is cut into **consecutive
windows** of `num_frames` frames, each encoded into its own embedding, stacked
along the leading axis. The **stride** is how many frames the fixed-size window
slides between two clips — exactly like the stride of a convolution.

**`--stride 16` (= `num_frames`) → no overlap (the default).** The window jumps a
full clip length each step, so every frame belongs to exactly one clip. Cheapest.

```
Frames :  0 ......15 16 ......31 32 ......47 ...
Clip 0 : [0 ......15]
Clip 1 :             [16 ......31]
Clip 2 :                         [32 ......47]
```

**`--stride 8` (< `num_frames`) → 50% overlap (a sliding window).** The window
advances only half a clip, so consecutive clips share 8 frames. This gives finer
temporal resolution and smoother clip-to-clip transitions, at the cost of more
clips to encode and some redundancy.

```
Frames :  0 ...... 8 ...... 15 16 ...... 23 ...
Clip 0 : [0 ............... 15]
Clip 1 :          [8 ............... 23]
Clip 2 :                   [16 ............... 31]
```

The number of clips for a `T`-frame video (`n = num_frames`, `s = stride`) is:

```
K = ceil((T - n) / s) + 1          (K = 1 when T <= n)
```

The final window is clamped onto the tail so the last frames are always covered.
For `T = 1250`, `n = 16`:

| `--stride` | Overlap | `K` clips | `--chunk` shape (`mean`) |
|-----------|---------|-----------|--------------------------|
| `16` *(default)* | none | 79 | `(79, 768)` |
| `8`  | 50% | 156 | `(156, 768)` |
| `4`  | 75% | 310 | `(310, 768)` |

```bash
# one embedding per non-overlapping 16-frame clip of a long video:
runs -m encoder.onnx -i long_video.mp4 --chunk -f npy -o feats.npy        # -> (K, 768)

# a 50%-overlapping sliding window, dense per-token features:
runs -m encoder.onnx -i long_video.mp4 --chunk --stride 8 --pooling none -f npy -o feats.npy  # -> (K, 2048, 768)
```

> Videos shorter than `num_frames` still produce one clip; the last frame is
> repeated to pad the window up to `num_frames`.

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
- **GitHub**: [cacybernetic/vjepa2](https://github.com/cacybernetic/vjepa2)
