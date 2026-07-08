# V-JEPA 2.1, explained gently (English)

> This document is written for beginners. The English is simple on purpose.
> If a word looks hard, we explain it. Take your time. Grab a coffee.

## 1. The big question

Imagine you watch a short video: a hand reaches for a cup. Even if we hide a
part of the screen, your brain can still guess what is behind the hidden part.
You do not need someone to label the video for you. You just *understand* it.

**Can a computer learn to understand video the same way, without labels?**

That is the goal of **V-JEPA 2.1**. It learns from raw video, with no labels,
by playing a simple game with itself: *"I will hide part of the video, and I
will try to guess what I hid."*

> **Naive reader:** "Wait, if it hides the pixels and then guesses the pixels,
> is it just drawing the missing part?"
>
> Good question! No. It does **not** guess the pixels. It guesses the *meaning*
> of the missing part. We will see why this is smarter in a moment.

## 2. Words we will use

- **Frame**: one image of the video.
- **Clip**: a small stack of frames (for example 16 frames).
- **Patch / token**: we cut each frame into small squares (16x16 pixels). Each
  square becomes one "token". A token is just a vector of numbers.
- **Encoder**: a network that reads tokens and outputs a *representation* (a
  richer vector that captures meaning).
- **Predictor**: a network that guesses the representation of the hidden tokens.
- **Representation / feature**: the numbers that describe the meaning of a
  patch. Think of it as a short summary.

## 3. Why guess *meaning* instead of *pixels*?

Pixels are noisy. The exact color of one pixel does not matter much. If the
model spends all its effort drawing exact pixels, it wastes time on tiny
details (grain, light flicker) that do not help understanding.

Instead, V-JEPA guesses in **latent space** (the space of representations).
This is the "JEPA" idea: **J**oint **E**mbedding **P**redictive **A**rchitecture.

> **Analogy:** A film critic does not memorize every pixel of a movie. They
> remember *what happens*: "a man opens a door, looks scared, runs away." That
> summary is the representation. V-JEPA learns to predict the summary of the
> hidden part, not the exact pixels.

## 4. The three networks

V-JEPA 2.1 has three parts that work together:

1. **The context encoder** (the *online* encoder). It sees the video **with
   holes** (some patches removed) and encodes the visible patches.
2. **The predictor**. It takes the encoded visible patches, plus a marker for
   each hole, and predicts the representation at the holes.
3. **The target encoder** (the *EMA* encoder). It sees the **full** video (no
   holes) and produces the "correct answer" the predictor must match.

> **Naive reader:** "If the target encoder gives the answer, why doesn't the
> model cheat and make both encoders output zero? Then the answer is always
> zero and the loss is always perfect!"
>
> Excellent! This is called **collapse**, and it is the classic danger. Two
> tricks stop it:
>
> - **Stop-gradient:** we never train the target encoder directly with the
>   loss. So the model cannot pull both sides to zero at once.
> - **EMA (exponential moving average):** the target encoder is a slow copy of
>   the online encoder. After each step, it moves a tiny bit toward the online
>   encoder: `target = 0.99925 * target + 0.00075 * online`. It follows, but
>   slowly. This keeps the target stable and meaningful.

## 5. Masking: hiding the right way

We hide patches using **tube masking**. We pick a rectangle on the frame grid
and hide it in **every** frame of the clip (a "tube" through time). The hidden
tokens are the ones to **predict**; the visible ones are the **context**.

```
Frame grid (each cell is a patch). X = hidden (predict), . = visible (context)
. . . . . .        The X block is hidden in every frame,
. . X X . .        so it forms a tube across time.
. . X X . .
. . . . . .
```

## 6. The key idea of version 2.1: the dense loss

The older V-JEPA only checked the guess on the **hidden** patches. Version 2.1
found a problem: the **visible** patches were never checked, so the model was
lazy about them. It used them as a "notepad" to store global summaries, and it
lost the fine local detail. The feature maps looked noisy.

**The fix (the heart of V-JEPA 2.1):** also check the visible patches. This is
the **dense predictive loss**:

```
total loss = predict_loss  +  lambda * context_loss
             (hidden tokens)        (visible/context tokens)
```

- `predict_loss`: how wrong the guess is on the **hidden** tokens (the old loss).
- `context_loss`: how wrong the predictor is on the **visible** tokens (new!).
- `lambda`: how much we trust the context loss.

Because every token now gets checked, the model must keep real local detail
everywhere. The feature maps become clean and meaningful. This is what unlocks
good **dense** tasks: depth, segmentation, tracking.

### 6.1 Weighting near the holes

Not every visible token matters the same. A visible token that sits **next to**
a hole is very useful (it helps the guess). One far away matters less. So we
weight the context loss by distance:

```
weight_i = lambda / sqrt(distance to the nearest hidden token)
```

Closer visible tokens get a bigger weight. This is a small idea with a big
effect on quality.

### 6.2 A gentle warm-up for lambda

If we turn on the context loss too strongly at the start, the model can find a
lazy trick (just copy the visible features) and lose global understanding. So
we **warm up** lambda: it starts at 0, then grows slowly to its final value
over some steps. Slow and steady.

## 7. Deep self-supervision

A transformer has many layers. Early layers see fine detail; deep layers see
big meaning. V-JEPA 2.1 applies the loss not only at the last layer, but also
at a few **intermediate** layers (4 levels). This pushes good detail all the
way through the network. In code this is `n_output_distillation: 4`.

## 8. Multi-modal tokenizer

Images and videos are different. A video has time; an image does not. So we use
two "cutters":

- a **3D** convolution for video (it looks across space **and** time),
- a **2D** convolution for a single image.

We also add a small learned "modality" marker so the model knows if the input
is an image or a video. This lets one model handle both, cleanly.

## 9. How the training program works (step by step)

Now the practical side: the program in this repository. Here is the journey of
your data.

1. **Find the videos.** Your dataset is a folder or a zip file. It may have
   sub-folders. The finder walks everything and lists every video file.
2. **Clean and cache.** Some files are broken. We try to open each one; we keep
   only the good ones. We save the good list into a `*.cache.json` file next to
   your dataset. Next time, we read the list instead of scanning again (fast!).
3. **Split.** The test set is split into a **validation** part (a fraction,
   `val_prob`) and a **final test** part. Validation is used every epoch to
   watch progress; the final test is used once at the end.
4. **Transform and augment.** Each clip is resized, cropped and normalized.
   During training we also add random changes (flip, color jitter, blur) so the
   model does not memorize. All of this is controlled from the config file.
5. **(Optional) HDF5.** Decoding video is slow. You can pre-compute all clips
   once and store them in `train.h5` / `test.h5`. Then training reads ready
   clips and runs faster. Switch with `use_hdf5: true`.
6. **The resumable loader.** This is special. It is a data loader that
   remembers **where it is** inside an epoch. If your machine crashes after 3
   days, you do not restart the epoch from zero. It saves the shuffle order and
   the position, so it continues at the exact same batch.
7. **Gradient accumulation.** GPUs have limited memory. To act like a big batch
   without the memory cost, we add up gradients over several small batches, then
   do one optimizer step. We also make sure any leftover accumulation at the end
   of the epoch is applied.
8. **Checkpoints.** Every `ckpt_step` optimizer steps, we save everything:
   model, optimizer, scheduler, loader positions, meters, and the training
   state. We keep only the newest `max_checkpoint` files. If the run stops, it
   resumes from the newest checkpoint. **Priority: checkpoint first, then any
   init weights.**
9. **Best model.** After each validation, we look at one chosen metric. If it is
   the best so far, we save `best.pt`. You choose the metric and whether higher
   or lower is better.
10. **Plots.** After each epoch we draw train vs validation curves so you can
    see if the model is learning or overfitting.

## 10. The output folder

Every run writes into `runs/<run_name>/`:

```
runs/
  my_run/
    train/          # first training run (train2, train3, ... for the next ones)
      history.csv         # metrics per epoch
      config_used.yaml    # the exact config used
      weights/
        best.pt           # best model so far
        last.pt           # model at the last epoch
      checkpoints/
        epoch_000.pth ...
      plotes/
        training_history.jpg
      logs/
        train_YYYY-MM-DD_HH-MM-SS.log
    eval/           # first evaluation run (eval2, eval3, ...)
      results.csv
      renders/            # PCA feature-map images
      plotes/
      logs/
```

If `resume: true` and a checkpoint exists, the program reuses the **latest**
run folder and continues there. Otherwise it makes a new numbered folder.

## 11. Watching the training

The program prints two progress bars:

- a big **epoch** bar (how many epochs are done, time left, best score, lr),
- a small **step** bar (progress inside the current train or validation pass).

The bars use `█` for done and `░` for the background. Regular messages (the
`step 400/553824 | loss=...` lines) are printed with the logger, and they never
break the bars.

## 12. Reading the metrics

- `loss`: the total dense loss. Lower is better.
- `predict`: error on hidden tokens.
- `context`: error on visible tokens.
- `lambda`: the current weight of the context loss (grows during warm-up).
- `feat_std`: how spread out the features are. If it drops near zero, the model
  is collapsing (bad). Healthy training keeps it well above zero.
- `pred_cos`: cosine similarity between guess and answer. Closer to 1 is better.

## 13. A tiny recipe to try

```bash
# 1. (optional) pre-build HDF5 files for speed
buildh5ds --config cpu/configs/hdf5.yaml

# 2. train
trainvjepa --config cpu/configs/train.yaml

# 3. evaluate the best model
evalvjepa --config cpu/configs/eval.yaml

# 4. export the encoder to ONNX (see the export.yaml comments)
exportw runs/vjepa2_1_cpu/train/weights/best.pt -o encoder.onnx \
    --model-name vit_tiny --crop-size 128 --num-frames 16 --sdpa --single-file
```

That is the whole idea. Predict the meaning of what you hide, check every token,
push the signal through every layer, and keep your training safe so it can run
for weeks and always come back from a crash.
