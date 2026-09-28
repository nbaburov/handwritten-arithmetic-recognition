# YOLO Stage 1 — Symbol Detector

## What it is

YOLOv8n (You Only Look Once, version 8, nano variant) is a **pure CNN object detector**.

- **CNN** = Convolutional Neural Network. The entire model is made of convolution operations. No transformers, no attention mechanisms, no encoders in any NLP sense, no recurrence.
- **Object detector** = given an image, outputs a list of bounding boxes, where each box has a position, a size, a class label, and a confidence score.
- **"Only once"** = the entire image is processed in a single forward pass through the network. There is no separate "propose regions then classify" step like older detectors. One pass, all boxes out.
- **"nano"** = smallest model in the YOLOv8 family. Fast, low memory. 3,157,200 trainable parameters, 225 total modules.

---

## What a convolution is (baseline concept)

A convolution layer slides a small filter (e.g. 3×3 pixels) across the image. At each position it multiplies the filter weights by the pixel values underneath and sums them up. This produces one output value per position. The filter learns to activate strongly on specific patterns: edges, curves, corners. The more filters a layer has, the more patterns it can detect simultaneously. The result is called a **feature map** — a spatial grid of "how strongly did this pattern appear here".

Every Conv layer in YOLOv8 has three sub-operations applied in sequence:

| Sub-op | What it does |
|--------|-------------|
| **Conv2d** | The sliding filter multiplication described above |
| **BatchNorm** | Normalises each output channel to have mean≈0 and std≈1 across the batch. Stabilises training, reduces sensitivity to learning rate. |
| **SiLU** | Activation function. Squashes the output non-linearly: `x * sigmoid(x)`. Without this, stacking convolutions would just be one big linear operation and the network could not learn complex patterns. |

All three together are written as `Conv` in the model and appear at every stage.

---

## Building blocks

### C2f — Cross Stage Partial with 2 features

The main repeated unit in YOLOv8. It solves a problem: deep networks lose gradient signal during backprop. C2f mitigates this by splitting and merging feature paths:

1. **1×1 Conv** compresses the input channels.
2. The result is **split into two halves** along the channel dimension.
3. One half goes straight through (skip path).
4. The other half goes through one or more **Bottleneck** blocks (two 3×3 convolutions in sequence).
5. Both halves are **concatenated** back together.
6. A final **1×1 Conv** merges the channels.

A **Bottleneck** is just two 3×3 Conv layers in sequence with matching input/output channels. It extracts richer features while keeping parameter count low.

The skip path means the gradient can flow directly through during training without passing through all the heavy convolutions. This is why C2f trains more stably than a simple stack of convolutions.

### SPPF — Spatial Pyramid Pooling Fast

Used once, at the end of the backbone. It captures context at multiple scales without resizing the image:

1. A **1×1 Conv** halves the channels.
2. MaxPool2d (5×5, stride 1) is applied **three times in sequence** (reusing the same pooling layer), each time giving a larger effective receptive field.
3. All four outputs (original + 3 pooled) are **concatenated** along channels.
4. A **1×1 Conv** merges them back.

**MaxPool2d** = takes the maximum value in a 5×5 window at each position. Captures the strongest activation in a neighbourhood. Applying it three times in sequence simulates receptive fields of 5, 9, and 13 pixels without three separate pooling layers.

SPPF gives the model context about large-scale structure (e.g. the whole result bar) at the same time as fine detail (individual digit strokes).

---

## Full architecture (layer by layer)

Input: 512×512 grayscale image stacked to 3 channels → shape `[1, 3, 512, 512]`

### Backbone (layers 0–9) — extract features, progressively smaller

The backbone reads the image and compresses it into rich feature maps at 4 different spatial scales. Each strided Conv halves the spatial size and doubles the depth (number of channels).

| Layer | Operation | Input channels | Output channels | Spatial size (from 512×512) |
|-------|-----------|---------------|-----------------|------------------------------|
| 0 | Conv 3×3, stride 2 + BN + SiLU | 3 | 16 | 256×256 |
| 1 | Conv 3×3, stride 2 + BN + SiLU | 16 | 32 | 128×128 |
| 2 | C2f (1 bottleneck) | 32 | 32 | 128×128 |
| 3 | Conv 3×3, stride 2 + BN + SiLU | 32 | 64 | 64×64 |
| 4 | C2f (2 bottlenecks) | 64 | 64 | 64×64 — **P3 scale** |
| 5 | Conv 3×3, stride 2 + BN + SiLU | 64 | 128 | 32×32 |
| 6 | C2f (2 bottlenecks) | 128 | 128 | 32×32 — **P4 scale** |
| 7 | Conv 3×3, stride 2 + BN + SiLU | 128 | 256 | 16×16 |
| 8 | C2f (1 bottleneck) | 256 | 256 | 16×16 |
| 9 | SPPF | 256 | 256 | 16×16 — **P5 scale** |

P3, P4, P5 are the three scales saved and reused in the neck. Each pixel in the feature map at P5 (16×16) represents a 32×32 region of the original image. Each pixel at P3 (64×64) represents an 8×8 region. Smaller scale = larger context, coarser position. Larger scale = finer position, less context.

### Neck (layers 10–21) — merge scales top-down then bottom-up

The backbone only produces features that go downward (smaller and smaller). The neck merges them so that every output scale has both fine positional detail (from small strides) and broad context (from large strides). This is critical for detecting both small objects (single digit) and large objects (result bar spanning the full width).

**Top-down path (FPN — Feature Pyramid Network):** Takes P5 (16×16 coarse), upsamples it, and merges with P4, then P3.

| Layer | Operation | Input | Output |
|-------|-----------|-------|--------|
| 10 | Upsample ×2 (nearest neighbour) | P5: 256ch, 16×16 | 256ch, 32×32 |
| 11 | Concat with P4 (layer 6) | 256 + 128 = 384ch, 32×32 | 384ch, 32×32 |
| 12 | C2f (1 bottleneck) | 384ch | 128ch, 32×32 |
| 13 | Upsample ×2 | 128ch, 32×32 | 128ch, 64×64 |
| 14 | Concat with P3 (layer 4) | 128 + 64 = 192ch, 64×64 | 192ch, 64×64 |
| 15 | C2f (1 bottleneck) | 192ch | 64ch, 64×64 — **N3 output** |

**Bottom-up path (PAN — Path Aggregation Network):** Takes N3 and pushes information back down, merging with the top-down features.

| Layer | Operation | Input | Output |
|-------|-----------|-------|--------|
| 16 | Conv 3×3, stride 2 | 64ch, 64×64 | 64ch, 32×32 |
| 17 | Concat with layer 12 | 64 + 128 = 192ch, 32×32 | 192ch, 32×32 |
| 18 | C2f (1 bottleneck) | 192ch | 128ch, 32×32 — **N4 output** |
| 19 | Conv 3×3, stride 2 | 128ch, 32×32 | 128ch, 16×16 |
| 20 | Concat with SPPF (layer 9) | 128 + 256 = 384ch, 16×16 | 384ch, 16×16 |
| 21 | C2f (1 bottleneck) | 384ch | 256ch, 16×16 — **N5 output** |

**Upsample** = just stretches the feature map spatially (nearest neighbour = copy each pixel into a 2×2 block). No learned weights. Used purely to match spatial dimensions before concatenating.

**Concat** = stacks feature maps along the channel axis. Channels from two different layers are merged side-by-side so the next layer can see both.

Three output feature maps go to the head: N3 (64ch, 64×64), N4 (128ch, 32×32), N5 (256ch, 16×16).

### Head (layer 22) — predict boxes and classes

The detect head takes the three neck outputs and produces the final predictions. It is **decoupled**: box coordinates and class probabilities are predicted by separate branches, which trains more stably than a single shared branch.

For **each of the 3 scales** (N3, N4, N5), two parallel branches run:

**Box branch (cv2):** Conv3×3 → Conv3×3 → Conv1×1 → 64 output channels.
The 64 channels encode the bounding box as a distribution (DFL format, see below).

**Class branch (cv3):** Conv3×3 → Conv3×3 → Conv1×1 → 6 output channels (one per class, after fine-tuning).
Each channel gives the raw logit score for one class at that position.

#### What "grid cell" means

At scale N3 (64×64), the image is divided into a 64×64 grid. Each cell is responsible for predicting objects whose centre falls in that cell. That is 64×64 = 4096 candidate positions at this scale. At N4: 32×32 = 1024. At N5: 16×16 = 256. Total candidate positions: 5376.

Most predict nothing. After NMS (see inference section), only a handful of real boxes remain.

#### DFL — Distribution Focal Loss (box encoding)

Instead of predicting one number for each edge of the box (left, top, right, bottom), DFL predicts a **probability distribution** over 16 possible values for each edge. A 1×1 Conv then takes the weighted average of those 16 values to get the final distance. This makes training more stable because predicting a distribution is smoother than predicting a single number.

The `dfl` sub-module is a single `Conv2d(16, 1, 1×1)` — it computes the weighted sum.

---

## No transformers. No encoders. No attention.

To be explicit:

- No self-attention layers.
- No cross-attention layers.
- No positional embeddings.
- No transformer encoder or decoder blocks.
- No LSTM or recurrent units.
- SPPF uses MaxPool (pure sliding maximum), not any form of attention.

YOLOv8 is 100% convolutions, batch normalisation, and fixed upsample operations.

---

## Fine-tuning, not training from scratch

We start from `yolov8n.pt` pretrained on **COCO** (80-class general object dataset, ~118k real images). The backbone has already learned useful filters for edges, curves, and shapes from those 118k diverse images.

What Ultralytics does when `model.train(nc=6, ...)` is called:

1. The **backbone and neck weights are kept** (layers 0–21). These are the expensive learned representations.
2. The **detect head (layer 22) is reinitialised** with 6 output channels instead of 80. Fresh random weights.
3. The entire model is then trained end-to-end on our synthetic dataset for up to 50 epochs (configurable via `config.toml [yolo] epochs`), with early stopping if val loss does not improve for 10 consecutive epochs (configurable via `config.toml [yolo] patience`).

The backbone quickly adapts its filters for handwritten symbols. The head learns 6-class detection from scratch. Because the starting point is already strong, convergence is fast and the result is near-perfect on synthetic data.

---

## Hyperparameters

| Param | Value | Meaning |
|-------|-------|---------|
| `epochs` | 50 | Max passes through the full training set (configurable via `config.toml [yolo] epochs`) |
| `imgsz` | 512 | All images resized to 512×512 before entering the network |
| `batch` | 16 | Images per gradient update step |
| `patience` | 10 | Early stopping: halt if val fitness does not improve for 10 consecutive epochs (configurable via `config.toml [yolo] patience`) |
| `workers` | 8 | CPU threads loading and augmenting data in parallel |
| `device` | cpu | CPU-only training (MPS TAL crash recurs in ultralytics 8.4.54+). M4 Pro CPU achieves ~1–2 hours for full YOLO training. |
| `cache` | ram | Load all images into RAM once per run to speed up epoch iteration |

**Augmentation overrides (explicitly set, replacing Ultralytics defaults):**

| Param | Value | Rationale |
|-------|-------|-----------|
| `fliplr` | 0.0 | Horizontal flip is semantically wrong for arithmetic: it mirrors digits and operators, producing impossible equations |
| `flipud` | 0.0 | Vertical flip inverts the equation structure (result bar moves to top) |
| `mosaic` | 0.0 | Mosaic stitches four scenes into one; violates the one-equation-per-canvas assumption |
| `mixup` | 0.0 | Pixel blending of two scenes produces illegible overlaid equations |
| `hsv_h` | 0.0 | Hue shift is meaningless on grayscale input |
| `hsv_s` | 0.0 | Saturation shift is meaningless on grayscale input |
| `hsv_v` | 0.2 | Small brightness jitter only — keeps grayscale invariance while adding mild contrast variation |
| `degrees` | 3.0 | Modest rotation matches per-glyph writing angle variation in real tablet input |
| `translate` | 0.05 | Small translation for position invariance |
| `scale` | 0.4 | Scale jitter to handle size variation in children's handwriting |
| `perspective` | 0.0 | Tablet input has no perspective; simulating it would hurt rather than help |
| `copy_paste` | 0.0 | Segment paste is irrelevant for our bounding-box-only task |

These overrides are recorded in the `augmentation` field of every `run.json` manifest so training runs are reproducible and the rationale is visible in the report.

Ultralytics defaults (not overridden): `lr0=0.01`, `lrf=0.01` (cosine LR decay), `momentum=0.937`, `weight_decay=0.0005`, `warmup_epochs=3`.

---

## Loss function

Three losses summed during training:

| Loss | What it penalises |
|------|-------------------|
| **CIoU box loss** | Inaccurate box coordinates. CIoU = Complete IoU. Penalises: (1) low overlap area, (2) centre point distance, (3) aspect ratio mismatch. All three together guide the box to the right shape and position. |
| **DFL loss** | How spread out the distribution prediction is. Pushes the model to concentrate probability mass on the correct edge distance. |
| **BCE class loss** | Wrong class score. Binary cross-entropy applied independently per class. A class score should be 1.0 for the correct class and 0.0 for all others. |

---

## Inference: what happens at runtime

1. Grayscale input (H×W, 1 channel) is stacked to 3 identical channels to match the expected input shape.
2. Resized to 512×512.
3. Single forward pass through all 22 layers → 5376 raw predictions (one per grid cell across 3 scales).
4. Each prediction has: 4 box distribution values × 4 edges, plus 6 class logits.
5. DFL converts distributions to box coordinates. Sigmoid converts class logits to probabilities (0–1).
6. All predictions with max class confidence below `conf=0.25` are discarded. Most of the 5376 are eliminated here.
7. **NMS (Non-Maximum Suppression):** if two surviving boxes overlap by more than IoU 0.45, keep only the one with higher confidence. Prevents two boxes on the same symbol.
8. Output: list of `Detection(label, confidence, x0, y0, x1, y1)` objects.

If the output is empty (no confident detections), the pipeline falls back to connected-components flood-fill on the binary image.

---

## Results

The active YOLO detector is run `20260622T134925Z_fc69f9ef` (iter11 cloud, warm-start fine-tune from iter10-R6 `aa990906`, early-stopped at epoch 11). Its test-split metrics on synthetic data are the headline figures below. The live pointer is `artifacts/yolo/active.json`; see `docs/runs.md` for the full run registry.

| Metric | Value (active run) |
|--------|--------------------|
| mAP50 | 0.9648 |
| mAP50-95 | 0.8161 |

**Per-class AP50 (active run):**

| Class | AP50 |
|-------|------|
| digit_main | 0.981 |
| digit_carry | 0.955 |
| digit_borrow | 0.876 |
| operator | 0.986 |
| result_bar | 0.995 |
| divide_bracket | 0.995 |

**mAP50:** For each class, compute precision-recall curve at IoU threshold 0.5 (predicted box must overlap ground truth by ≥50%). Average Precision (AP) = area under that curve. mAP = mean AP across all 6 classes. 0.9648 = near-perfect detection at the loose overlap threshold.

**mAP50-95:** Same but averaged across IoU thresholds 0.50, 0.55, 0.60, ..., 0.95. Measures not just whether the box is roughly right but whether it is precisely placed. 0.8161 = boxes are tightly localised across strict thresholds.

**Iter11 promotion.** The warm-start fine-tune from iter10-R6 (`aa990906`) was promoted on real-data evaluation (190-scene set): it improves operator recognition on subtraction and division while maintaining spatial robustness. Synthetic metrics dipped slightly (mAP50 from 0.9705 to 0.9648) due to early-stopping at epoch 11, but real-data operator accuracy (addition +8.6pp, subtraction still weak at 0.11) improved overall system robustness. The prior iter10-R6 is retained as rollback target. Prior synthetic-eval peaks (mAP50 0.9949, mAP50-95 0.9789) came from superseded runs and should not be read as current performance.

---

## Role in the pipeline

YOLO outputs **where** and **what coarse type**. It does not read digits (0–9) or distinguish operator type (+/-/×/÷). That is Stage 2 (GNN).

```
Image
  → [YOLO]      → bounding boxes + coarse class (digit_main / operator / result_bar / ...)
  → [GNN]       → fine label (main_7 / op_plus / carry_3 / ...)
  → [Assembler] → structured JSON
```
