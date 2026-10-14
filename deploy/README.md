# V-JEPA 2.1: quantization and embeddings without PyTorch

This project turns a pretrained **V-JEPA 2.1** encoder (Meta, paper in
`resources/vjepa2_1_arXiv`, official code in `resources/vjepa2-main`) into
compressed ONNX models. It then uses them to compute **embeddings of images and
videos** with ONNX Runtime only, without PyTorch.

Two programs, used one after the other:

| Program | Needs PyTorch | Input | Output |
|---|---|---|---|
| `quantization.py` | yes | a V-JEPA 2.1 checkpoint (`.pt` / `.pth`) | a model folder: `.onnx` files and `config.json` |
| `embed.py` | **no** | images and/or videos + a model folder | one `.pkl` file per image and per video frame |

The two programs never import each other. They only share the files of the
model folder. `config.json` tells `embed.py` which model files exist, their
input shapes and the image normalization.

---

## Contents

1. [Installation](#1-installation)
2. [Model formats: fp16, q4_k_m, fp32](#2-model-formats-fp16-q4_k_m-fp32)
3. [quantization.py: checkpoint → ONNX](#3-quantizationpy-checkpoint--onnx)
4. [embed.py: images and videos → embeddings](#4-embedpy-images-and-videos--embeddings)
5. [Reading the .pkl files](#5-reading-the-pkl-files)
6. [Accuracy, size and speed](#6-accuracy-size-and-speed)
7. [How Q4_K_M is done](#7-how-q4_k_m-is-done)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Installation

Use **Python 3.10** (every pin of `requirements.txt` has Python 3.10 wheels:
`numpy==2.2.6`, `onnxruntime==1.23.2`, …).

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
# CPU-only PyTorch (smaller than the default CUDA build):
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

The two programs need different packages:

- **`quantization.py`**: torch, timm, einops (for the official V-JEPA code),
  onnx, onnxscript, onnxconverter-common, onnxruntime, numpy, loguru, tqdm.
  It imports the official model code from `resources/vjepa2-main` (see
  `--vjepa-root`).
- **`embed.py`**: onnxruntime (or onnxruntime-gpu), numpy, opencv-python,
  loguru, tqdm. **No PyTorch.** On a machine that only computes embeddings,
  this is enough:

  ```bash
  pip install onnxruntime==1.23.2 numpy==2.2.6 opencv-python loguru tqdm
  ```

  For an NVIDIA GPU, install `onnxruntime-gpu==1.23.2` instead of `onnxruntime`.

Checkpoints go in `weights/`, for example
`weights/vjepa2_1_vitb_dist_vitG_384.pt` (ViT-B distilled from ViT-G, 384 px).

---

## 2. Model formats: fp16, q4_k_m, fp32

`quantization.py` can write three formats, chosen with `--precision`.
`embed.py` can run any of them.

| Format | What is compressed | Size (ViT-B, image / video) | Loss vs PyTorch | When to use it |
|---|---|---|---|---|
| `fp16` | every weight and computation in float16 | 177 MB / 202 MB | none in practice (cosine ≈ 1.000000) | best accuracy with a 2× smaller file; fast on GPU |
| `q4_k_m` | the linear-layer weights in 4 bits (6 bits for the most sensitive ones), llama.cpp's **Q4_K_M** recipe; the rest stays in float32 | 84 MB / 133 MB | small (embedding cosine ≈ 0.998) | smallest model, CPU use, less memory |
| `fp32` | nothing (the reference) | ≈ 350 MB / ≈ 400 MB | none | reference, comparisons |

Every format takes **float32 input and gives float32 output**, so the user of
the model never handles float16 or integers.

### Why two models per format (image and video)?

V-JEPA 2.1 has a *multi-modal tokenizer*:

- an **image** is cut into 16×16 patches by a 2D convolution;
- a **video** is cut into 2×16×16 "tubelets" (2 frames × 16×16 pixels) by a 3D
  convolution;
- a learned "modality embedding" tells the transformer which kind of input
  it reads.

PyTorch picks the path from the input shape, but one ONNX graph can only hold
one path. So each format has two files:

```
encoder_image_<format>.onnx   pixel_values [B, 3, H, W]     -> patch_tokens [B, h*w, D]
encoder_video_<format>.onnx   pixel_values [B, 3, T, H, W]  -> patch_tokens [B, T/2*h*w, D]
```

with `h = H/16`, `w = W/16`, and D = 768 (ViT-B), 1024 (ViT-L), 1408 (ViT-g)
or 1664 (ViT-G). At 384×384 with 16 frames:

- image: `[B, 576, 768]`, a grid of 24×24 tokens;
- video: `[B, 4608, 768]`, a grid of 8×24×24 tokens.

`patch_tokens` are the features of the last encoder block after its
LayerNorm. These are the "frozen features" used in all the evaluations of the
paper.

---

## 3. quantization.py: checkpoint → ONNX

### Basic use

```bash
# Both fp16 and q4_k_m (the default), image and video models,
# written to models/<checkpoint name>/
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt
```

What it does:

1. Loads the checkpoint (memory-mapped, low RAM). It finds the encoder
   weights and detects the architecture (ViT-B/L/g/G, patch size, tubelet
   size).
2. Builds the encoder with the official code (`app/vjepa_2_1`) and loads the
   weights strictly.
3. If `q4_k_m` is asked: quantizes all linear layers once (about 50 s for
   ViT-B).
4. For each modality (image, then video):
   - exports a float32 ONNX graph;
   - converts it to each requested format;
   - compares each model with PyTorch on a random input (validation).
5. Writes `config.json`.

### All options

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | *(required)* | V-JEPA 2.1 checkpoint (`.pt` / `.pth`) |
| `--precision {fp16,q4_k_m,fp32} …` | `fp16 q4_k_m` | formats to write; one or several |
| `--output-dir DIR` | `models/<checkpoint name>` | where to write the models |
| `--modalities {image,video} …` | `image video` | which graphs to export |
| `--img-size PIXELS [PIXELS]` | `384` | model input size: one value (square) or `H W`; multiples of 16 |
| `--num-frames N` | `16` | frames per clip of the video model (even number) |
| `--checkpoint-key KEY` | auto | dict entry holding the encoder; by default the first of `ema_encoder`, `target_encoder`, `encoder` |
| `--arch NAME` | auto | force the architecture: `vit_base`, `vit_large`, `vit_giant_xformers`, `vit_gigantic_xformers` |
| `--opset N` | `18` | ONNX opset |
| `--no-validate` | off | skip the PyTorch vs ONNX Runtime comparison (faster, less memory) |
| `--vjepa-root DIR` | `resources/vjepa2-main` | root of the official vjepa2 code |
| `--unsafe-load` | off | allow arbitrary Python objects in the checkpoint (**only for files you trust**) |

### Use cases

**Only the INT4 Q4_K_M model**

```bash
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt --precision q4_k_m
```

**Only the float16 model**

```bash
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt --precision fp16
```

**Every format, including the float32 reference**

```bash
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt --precision fp16 q4_k_m fp32
```

**Only images (no video model)**

This is enough for photos, and for videos in `frame` mode (see section 4).
The video export is the slowest and most memory-hungry part, so this also
saves time.

```bash
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt --modalities image
```

**Other input size or clip length**

```bash
# 256x256 images, 8-frame clips
python quantization.py --checkpoint weights/vjepa2_1_vitl_dist_vitG_384.pt \
    --img-size 256 --num-frames 8 --output-dir models/vjepa2_1_vitl_256_8f

# non-square input (H W)
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt --img-size 288 512
```

The ONNX models have a **fixed** spatial size and frame count. The rotary
position tables are computed for them and stored in the graph, so a new size
needs a new export. Only the batch size is dynamic.

**Adding a format to an existing model folder**

A run into a folder that already holds models of the **same checkpoint, image
size and number of frames** adds its models to `config.json`. It keeps the
other formats and modalities that are still there. You can therefore build
the folder step by step:

```bash
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt \
    --output-dir models/vjepa2_1_vitb_384 --precision fp16
python quantization.py --checkpoint weights/vjepa2_1_vitb_dist_vitG_384.pt \
    --output-dir models/vjepa2_1_vitb_384 --precision q4_k_m
# config.json now lists fp16 and q4_k_m for image and video
```

If the checkpoint or the shapes differ, the old entries are dropped: the new
`config.json` only describes the new models.

**Checkpoint with an unusual layout**

```bash
python quantization.py --checkpoint my_finetune.pth --checkpoint-key target_encoder --arch vit_large
```

DDP and wrapper prefixes (`module.`, `backbone.`) are removed automatically.
A checkpoint that is a bare state dict (no `encoder` entry) is also
accepted.

### What you get

```
models/vjepa2_1_vitb_dist_vitG_384/
├── config.json
├── encoder_image_fp16.onnx
├── encoder_image_q4_k_m.onnx
├── encoder_video_fp16.onnx
└── encoder_video_q4_k_m.onnx
```

Models above about 1.8 GB (fp32 or fp16 of ViT-g/G) keep their weights in a
`<file>.onnx.data` file next to the `.onnx`. **Keep the two files together.**

`config.json` contains:

- the source: checkpoint path, checkpoint key, architecture, number of
  parameters, patch and tubelet sizes;
- `image_size`, `num_frames` and the normalization (ImageNet mean/std);
- for each modality:
  - `files` (format → file name);
  - input/output names and shapes;
  - `token_grid` (`[t, h, w]`);
  - `validation` (per format: max difference, mean/min cosine of the patch
    tokens, cosine of the mean embedding);
  - `quantization` (for `q4_k_m`: number of weights in Q4_K, in Q6_K and
    left in float32).

### Reading the log

```
Q4_K_M: 48 linear layers, 67.3 M weights in Q4_K (4 bits) and 17.7 M in Q6_K (6 bits), 50.2 s
[image] Exported in 74.1 s
[image] fp16 model: .../encoder_image_fp16.onnx (177 MB, 7.5 s)
[image] q4_k_m model: .../encoder_image_q4_k_m.onnx (84 MB, 2.6 s)
[image] ONNX fp16 vs PyTorch: max |diff| 4.81e-02, patch cosine mean 0.999999 / min 0.999983, embedding cosine 1.000000
[image] ONNX q4_k_m vs PyTorch: max |diff| 1.12e+00, patch cosine mean 0.996441 / min 0.989725, embedding cosine 0.998417
```

The program prints a **warning** when a model drifts too far from PyTorch.
The thresholds are:

- fp16 / fp32: minimum patch cosine below 0.99;
- q4_k_m: mean patch cosine below 0.98;
- any format: non-finite output.

A large `max |diff|` alone is normal for q4_k_m: some token values are large.

### Resources needed

- ViT-B: about 5 minutes on a 4-core CPU for both modalities with
  validation, about 3 GB of RAM.
- ViT-L: also exportable with 7 GB of RAM.
- ViT-g/G: need much more RAM.

The video validation (one 16-frame 384 px clip through each model) is the
slowest step. Use `--no-validate` to skip it.

---

## 4. embed.py: images and videos → embeddings

### Basic use

```bash
python embed.py --model-dir models/vjepa2_1_vitb_dist_vitG_384 photo.jpg clip.mp4
```

```
embeddings/
├── photo.pkl
└── clip/
    ├── frame_000000.pkl
    ├── frame_000001.pkl
    └── ...               # one file per decoded frame: 1290 frames -> 1290 files
```

Inputs are recognized by their extension:

- **images**: `.jpg .jpeg .png .bmp .webp .tif .tiff`;
- **videos**: every other file, read with OpenCV (mp4, avi, mkv, webm, …,
  depending on your OpenCV build).

Several files can be given at once. Images are processed first, then
videos. A file that can't be decoded is skipped with a warning.

### All options

| Option | Default | Meaning |
|---|---|---|
| `inputs …` | *(required)* | image and/or video files |
| `--model-dir DIR` | *(required)* | output folder of `quantization.py` |
| `--output-dir DIR` | `embeddings` | where to write the `.pkl` files |
| `--precision {auto,q4_k_m,fp16,fp32}` | `auto` | model format to run; `auto` = first of q4_k_m, fp16, fp32 present in `config.json` |
| `--video-mode {frame,clip}` | `frame` | how videos are encoded (see below) |
| `--batch-size N` | 8 (frame/image), 1 (clip) | images/frames per run, or clips per run in clip mode |
| `--crop-ratio R` | `0.875` (224/256) | model size ÷ resized size before the center crop; `1.0` keeps as much border as possible |
| `--no-patch-tokens` | off | save only the mean embedding (a few KB instead of ~0.9 MB per frame) |
| `--device {auto,cpu,cuda}` | `auto` | ONNX Runtime backend; `auto` uses CUDA if onnxruntime-gpu and a GPU are present |
| `--threads N` | `0` | CPU threads of ONNX Runtime (0 = its default: all cores) |

### Choosing the model format

```bash
python embed.py --model-dir models/vjepa2_1_vitb_384 photo.jpg                       # auto: q4_k_m if present
python embed.py --model-dir models/vjepa2_1_vitb_384 --precision q4_k_m photo.jpg    # INT4 Q4_K_M
python embed.py --model-dir models/vjepa2_1_vitb_384 --precision fp16 photo.jpg      # float16
python embed.py --model-dir models/vjepa2_1_vitb_384 --precision fp32 photo.jpg      # needs quantization.py --precision fp32
```

If the asked format is missing, the program stops and lists what is
available, for example: `has no fp32 video model (available: q4_k_m, fp16)`.
Each `.pkl` records which format produced it (`"precision"`).

**Do not mix formats in one dataset** if you compare embeddings with each
other. Q4_K_M and fp16 embeddings of the same image have a cosine of about
0.998, not 1.

### Videos: frame mode (default)

Every frame is encoded **alone** by the image model, like a photo. N frames
give N different embeddings. This mode is the closest to "one embedding per
image of the video". It is also the cheapest in memory and only needs the
image model.

```bash
python embed.py --model-dir models/vjepa2_1_vitb_384 --batch-size 4 video.mp4
```

### Videos: clip mode

The frames are grouped into clips of T frames (`num_frames` of `config.json`,
16 by default). The video model reads each clip, so a frame's features also
see the other frames of its clip (motion, temporal consistency).

```bash
python embed.py --model-dir models/vjepa2_1_vitb_384 --video-mode clip video.mp4
```

Details:

- You still get **one `.pkl` per frame**.
- The video tokenizer reads frames by pairs (tubelets of 2), so frames 2k
  and 2k+1 of a clip get the **same** embedding.
- The last clip is completed by repeating its last frame. The padding
  frames are not saved.
- It is much heavier: attention runs over T/2 × 24 × 24 = 4608 tokens per
  clip (about 1 minute per 16-frame clip on a 4-core CPU).
- It needs the video model (`--modalities video`, exported by default).

### Images only, or a whole folder

```bash
python embed.py --model-dir models/vjepa2_1_vitb_384 --output-dir emb_photos photos/*.jpg
```

Each image `name.ext` gives `<output-dir>/name.pkl`. Two images with the same
name in different folders would overwrite each other, so use separate
`--output-dir`s.

### Saving disk space

Patch tokens take about 0.9 MB per frame for ViT-B (1.2 MB for ViT-L). A
1290-frame video therefore needs 1.1–1.5 GB. If you only need one vector per
image/frame:

```bash
python embed.py --model-dir models/vjepa2_1_vitb_384 --no-patch-tokens video.mp4
```

### Running on GPU or with fewer threads

```bash
pip install onnxruntime-gpu==1.23.2
python embed.py --model-dir models/vjepa2_1_vitb_384 --device cuda --precision fp16 video.mp4

# leave CPU cores free for other work
python embed.py --model-dir models/vjepa2_1_vitb_384 --threads 2 video.mp4
```

### Preprocessing

The same as the evaluation transform of the official code
(`VideoTransform`), rewritten with OpenCV and NumPy:

1. Resize (bilinear) so that the image covers `size / crop_ratio` pixels:
   `int(384 × 256/224) = 438` for a square model.
2. Crop the center to the model size.
3. Scale to [0, 1].
4. Normalize with the ImageNet mean `(0.485, 0.456, 0.406)` and std
   `(0.229, 0.224, 0.225)`.

The center crop cuts the borders of non-square images. `--crop-ratio 1.0`
keeps as much of them as possible.

---

## 5. Reading the .pkl files

Each file is a plain dict, readable with `pickle` and NumPy only:

```python
import pickle

record = pickle.load(open("embeddings/clip/frame_000017.pkl", "rb"))
record["embedding"].shape      # (768,)          float32: mean of the patch tokens
record["patch_tokens"].shape   # (24, 24, 768)   float16: one vector per 16x16 patch
```

| Key | Present | Meaning |
|---|---|---|
| `embedding` | always | float32 `[D]`, mean of the patch tokens: one global vector for the image/frame |
| `patch_tokens` | unless `--no-patch-tokens` | float16 `[h, w, D]` dense features (segmentation, depth, tracking, …) |
| `source` | always | absolute path of the input file |
| `modality` | always | `"image"` or `"video"` |
| `encoder` | always | model used: `"image"` or `"video"` |
| `arch`, `image_size` | always | e.g. `"vit_base"`, `[384, 384]` |
| `precision` | always | `"q4_k_m"`, `"fp16"` or `"fp32"` |
| `frame_index`, `timestamp_sec`, `fps` | videos | position of the frame (`timestamp_sec` is `None` if the fps is unknown) |
| `video_mode` | videos | `"frame"` or `"clip"` |
| `clip_start_frame`, `tubelet_index` | clip mode | first frame of the clip, and tubelet (pair of frames) inside it |

Example: load all the frame embeddings of a video into a matrix and compare
them.

```python
import glob, pickle
import numpy as np

files = sorted(glob.glob("embeddings/clip/frame_*.pkl"))
E = np.stack([pickle.load(open(f, "rb"))["embedding"] for f in files])   # [frames, D]
E /= np.linalg.norm(E, axis=1, keepdims=True)
similarity = E @ E.T                                                     # cosine between frames
```

---

## 6. Accuracy, size and speed

Measured with `vjepa2_1_vitb_dist_vitG_384.pt` (ViT-B, 384 px, 16 frames)
against the float32 PyTorch model:

| Model | Patch cosine, mean / min | Embedding cosine | Size |
|---|---|---|---|
| image fp16 | 0.999999 / 0.99998 | 1.000000 | 177 MB |
| image q4_k_m | 0.9964 / 0.990 | 0.9984 | 84 MB |
| video fp16 | 0.99999 / 0.9992 | 1.000000 | 202 MB |
| video q4_k_m | 0.9943 / 0.978 | 0.9982 | 133 MB |

The video q4_k_m file is relatively bigger because its rotary position tables
(about 50 MB) stay in float32.

**Speed on CPU** (Intel i7-6600U, 4 threads, one 384 px image):

| Format | Time per image |
|---|---|
| fp32 | ~1.3 s |
| q4_k_m | ~1.45 s |
| fp16 | ~1.4 s alone, ~2 s in batches |

- **fp16 on CPU:** ONNX Runtime has few float16 kernels on CPU. It inserts
  casts and computes in float32, so fp16 is not faster there. On a CUDA GPU,
  float16 is fast.
- **q4_k_m on CPU:** ONNX Runtime expands the 4-bit weights to float32 just
  before each product. Speed is therefore close to fp32, with 3–4× less
  disk and memory. Q4_K_M brings size and memory gains, not speed gains, on
  this kind of CPU.
- **Clip mode** costs about 1 minute per 16-frame clip on the same CPU.

---

## 7. How Q4_K_M is done

**Q4_K_M** is a quantization recipe of [llama.cpp](https://github.com/ggml-org/llama.cpp)
(the "K-quants" of GGUF files).

**Q4_K (4.5 bits per weight in GGUF):**

- Each row of a weight matrix is cut into super-blocks of 256 values, then
  into 8 blocks of 32.
- In a block, `value = scale × q − min`, with `q` a 4-bit integer (0..15).
- Each block's `scale` and `min` are stored on 6 bits, relative to two
  float16 numbers of the super-block.
- They are chosen by a weighted least-squares search over 21 candidate scales.

**Q6_K:**

- 16 blocks of 16 values; `value = scale × (q − 32)`, with `q` on 6 bits.
- 8-bit block scales, relative to one float16 number per super-block.

**The "_M" (medium) mix**

It follows llama.cpp's rules in `src/llama-quant.cpp`. The attention values
(`attn_v`) and the second MLP layer (`ffn_down`) are stored in **Q6_K** on
"use_more_bits" layers. These are the first and last eighths of the blocks,
then one block in three: blocks 0, 3, 6, 9, 10 and 11 of a 12-block ViT-B.
Everything else is in **Q4_K**.

V-JEPA computes q, k and v with a single `qkv` matrix. Its last third (v)
follows the `attn_v` rule, so on those layers q and k are in Q4_K and v in
Q6_K. For ViT-B, this gives 67.3 M weights in Q4_K, 17.7 M in Q6_K and 1.8 M
left in float32 (patch embedding, LayerNorm, biases).

**Faithfulness to llama.cpp**

The quantizer is a NumPy port of ggml's reference code
(`quantize_row_q4_K_ref`, `quantize_row_q6_K_ref`). Like
`llama-quantize ... Q4_K_M`, it uses no importance matrix. It was checked
against ggml's C code and gives the **same quantized values**, bit for bit,
on random and real ViT weights.

Rows whose length is not a multiple of 256 (ViT-g: 1408, ViT-G: 1664) end
with a shorter super-block.

**Storage: ONNX, not GGUF**

llama.cpp cannot run V-JEPA. So the quantized weights are stored in ONNX
Runtime's 4-bit / 8-bit matrix-product operator, **`MatMulNBits`** (domain
`com.microsoft`):

- **Q4_K**: 4-bit codes in blocks of 32, with a float32 scale and a float32
  zero point per block. `(q − min/scale) × scale = scale × q − min`, so ONNX
  Runtime computes with exactly llama.cpp's values.
- **Q6_K**: 8-bit codes `q + 96` in blocks of 16, with the default 8-bit
  zero point 128. This gives `scale × (q − 32)`.

The file is bigger than a GGUF would be (~6 bits per weight instead of ~4.5),
because the block scales are stored in float32. The weights themselves are
identical. The model needs ONNX Runtime; other ONNX runtimes do not know the
`com.microsoft` operators.

---

## 8. Troubleshooting

| Message / problem | Cause and fix |
|---|---|
| `holds objects that are not plain tensors … --unsafe-load` | The checkpoint pickles Python objects. Use `--unsafe-load` only if you trust the file. |
| `No encoder found in the checkpoint; entries: […]` | Give the right entry with `--checkpoint-key`. |
| `Unknown encoder (embed_dim=…, depth=…)` | Give the architecture with `--arch`. |
| `--img-size … must be a multiple of the patch size` | Use sizes divisible by 16. |
| `--num-frames must be an even number` | The video tokenizer reads frames by pairs. |
| `Official V-JEPA code not found` | Point `--vjepa-root` to the `vjepa2-main` folder. |
| `… has no q4_k_m image model (available: fp16)` | Run `quantization.py --precision q4_k_m` into the same `--output-dir`. |
| `… has no video model` with `--video-mode clip` | Export with `--modalities video` (or both). |
| `--device cuda, but ONNX Runtime has no CUDA backend` | Install `onnxruntime-gpu` instead of `onnxruntime`. |
| `config.json not found` / `was not written by quantization.py` | `--model-dir` must be an output folder of `quantization.py`. |
| Out of memory during export | Export one modality at a time (`--modalities image`), add `--no-validate`, or use a smaller `--img-size` / `--num-frames`. |
| Out of memory / very slow in clip mode | Use `--batch-size 1` (the default), or the frame mode. |
| A video gives fewer `.pkl` than expected | OpenCV stopped decoding (damaged file or unsupported codec). The log shows the frame count announced by the file; re-encode the video, e.g. with `ffmpeg -i in.mkv out.mp4`. |
| `.onnx.data` file missing | Big models keep their weights next to the `.onnx`; copy both files. |
