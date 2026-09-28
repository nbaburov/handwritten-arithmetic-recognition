# Config Rationale (config.toml)

This document explains every non-trivial parameter choice in `config.toml`. Training is CPU-only because MPS on PyTorch 2.11 crashes in YOLO's TAL loss. The goal is a robust pipeline with the best quality achievable in an overnight training budget (roughly 10 hours for YOLO Stage 1).

The decisions below are scoped to Stage 1 (YOLO) first. Stage 2 (GNN) parameters are left near their current defaults and will be revisited after YOLO is validated.

## 1. YOLO model backbone: yolov8n.pt (iter2 final choice)

**Decision**: use `yolov8n.pt` (3.2 M params, 8.7 GFLOPs).

**Why** (iter2 pivot from yolov8s):
- Iteration 2 testing (May 21, 2026) found that `yolov8s` overfits our synthetic dataset at 500–600 scenes per case (20 cases, roughly 10K total). Fine-tuning yolov8s on this dataset caused mAP to collapse from 0.91 baseline to 0.30 (see memory note "yolov8s_not_viable_on_small_synth.md").
- `yolov8n` (nano) trains to mAP 0.91+ in ~1–2 hours on CPU with `epochs = 50` + `patience = 10` and does not overfit. It is the production choice until real-handwriting fine-tuning data is available.
- For domain shift (camera vs tablet), the path forward is to fine-tune on real samples, not to increase model capacity. Scaling the dataset > 3000 scenes/case is prohibitive; scaling the model is the wrong lever on synthetic-only training.
- `yolov8n.pt` will be auto-downloaded by ultralytics on first run (~6.3 MB). After that it is cached under `artifacts/pretrained/` on subsequent runs.

**Historical note**: April 2026 iteration selected yolov8s for small-object recall. That rationale was sound for general object detection but underestimated synthetic overfitting. Yolov8n with proper regularisation (augmentation) is the correct trade-off.

## 2. Synthetic data volume: 500 images per case with 8 topup rounds

**Decision**: `images_per_case = 1000`, 22 SceneCase values (7 base + 9 variants + 4 robustness + 2 iter11 crowded), split 80/10/10, `min_instances_per_label = 1000`, `topup_rounds = 8`. Iter10-R6 dropped from 1500 to 1000 (diminishing returns); iter11 adds 2 new crowded cases. This yields roughly 17,600 base scenes (1000 × 22 × 0.8), which topup expands to ~19,000–22,000 train scenes.

**Why** (iter2 reduction from 1500/case; iter11 crowded expansion):
- Iteration 2 found that the dataset size is not the bottleneck; overfitting is. Reducing from 1500 → 1000 images per case while fixing augmentation and regularisation improved validation mAP from 0.89 to 0.91+. This proved sufficient for iter10-R6.
- 22 cases (7 base + 9 variants + 4 robustness: `bare_digits`, `standalone_bar`, `standalone_bracket`, `bare_digit_grid`; plus 2 iter11 new: `addition_crowded_carry`, `subtraction_crowded_borrow`) are now in scope. Iter11 adds crowded cases to model real tight kid handwriting with glyph overflow and double-digit borrow/carry cells.
- 1000 images per case is adequate for 5–12 YOLO boxes per scene. Diversity matters more than quantity on CPU.
- Topup mechanism (`min_instances_per_label = 1000`, `topup_rounds = 8`) ensures all 16 fine labels have sufficient representation, especially rare ones (`carry_*`, `borrow_*`) that only appear in specific arithmetic contexts.
- Per-case weighting in iter11 shifts to a **deficit profile** (real-eval-driven): weak cases (division-long, division-short, multiplication-multi, subtraction_heavy_borrow, addition_dense_carries) are bumped to 2.0–2.5× weight, while strong cases (bare_digits, standalone_bar, standalone_bracket) are kept low for rehearsal. This targets the known eq_kind gaps from eval.

## 3. Per-case weighting (22 cases, iteration 2 tuning + iter11 deficit-driven rebalance)

**Decision**: non-uniform case weights that oversample the layouts with the most structural complexity and rarest error modes.

```
# Base cases (7)
addition = 1.0
subtraction = 1.0
multiplication-simple = 1.0
multiplication-multi = 1.5      # partial products, complex vertical stacking
division-short = 1.5             # bracket + quotient, highest error rate in eval
division-long = 2.0              # bracket + quotient + step-subtraction nesting
division-simple = 1.0

# Operator-right variants (3)
addition_op_right = 1.0
subtraction_op_right = 1.0
multiplication_simple_op_right = 1.0

# No-bar variants (3)
addition_no_bar = 1.0
subtraction_no_bar = 1.0
multiplication_simple_no_bar = 0.5    # rare layout; downweight vs op_right

# Special focus cases (3 — MC series error correction)
subtraction_heavy_borrow = 1.5        # 3-digit - 3-digit, 2–3 forced borrows
addition_dense_carries = 1.5          # addition with all columns carrying (max error potential)
addition_no_carries = 1.0             # zero-carry baseline for contrast
```

**Why**:
- Division layouts (especially long division) have the highest error rates in eval; 2.0× weight ensures rich geometric exposure.
- Multiplication-multi and dense-carries stress the GNN row/col clustering; 1.5× ensures sufficient training signal.
- Operator-right and no-bar variants cover real-world handwriting variance (children place operators in different locations, sometimes forget the result bar).
- Down-weighting `multiplication_simple_no_bar = 0.5` (rare in real samples) frees capacity for more common structures.

## 4. Class balance via min_instances_per_label: 1000, topup_rounds: 8

**Decision**: `min_instances_per_label = 1000`, `topup_rounds = 8` (iter10-R6 alignment with images_per_case).

**Why** (iter2 consolidation):
- The fine-label set is now 16 classes (10 digits + 4 operators + result_bar + div_bracket) — unified, role-agnostic digits post-A2 refactor. Role (main/carry/borrow) is encoded in spatial features, not fine labels, so the imbalance problem is gone.
- All 16 fine labels are synthetically generated and equally represented in the topup loop. The floor of 400 instances per label is conservative and ensures no label starves during training.
- `topup_rounds = 8` is a safety ceiling. The loop runs generate-and-check until all labels reach the floor, then exits. In practice, typical runs exit after 1–2 rounds given the per-case weighting and 20-case diversity.

## 5. Handwriting style variance: single parameterised preset

**Decision**: single `[generation.preset]` block with bounded per-glyph parameters (no multi-preset system).

```
[generation.preset]
glyph_rotation_min_deg = -5.0
glyph_rotation_max_deg = 5.0
glyph_jitter_min_px = 0
glyph_jitter_max_px = 2
glyph_broken_stroke_prob = 0.0
```

**Why**:
- Iteration 2 simplified from 5 presets (easy_baseline, neat, typical, messy, crowded) to a single parameterised preset with bounded ranges. This reduces config complexity and makes variance transparent.
- All per-glyph variance comes from sampling within these ranges per scene. Layout-level variance (per-case weighting, completion stages) provides structural diversity; glyph-level variance provides stroke irregularity.
- `glyph_broken_stroke_prob = 0.0` (disabled since iter10-R6) because the implementation cuts horizontal bands, which is not how real handwriting breaks. Strand-level gap simulation requires a different approach deferred to future work.
- If future data collection shows real handwriting with specific style traits, this single preset can be extended or replaced, but the current parameterization is sufficient for tablet-user diversity (6–12 year-olds, varied tablet UIs).

## 6. Completion-stage weights: iteration 2 rebalance

**Decision**: shift mix from 65% full to 50% full; distribute 15% to remaining stages.

```
[generation.completion]
full = 0.50
done_80 = 0.15
done_60 = 0.13
done_40 = 0.12
done_20 = 0.10
```

**Why**:
- Iteration 2 lowered the full-completion percentage from 65% to 50% to better reflect real tablet usage: most children write partial exercises (incomplete arithmetic before submission).
- Distributing the remaining 50% across 4 partial stages (done_80, done_60, done_40, done_20 representing 80%, 60%, 40%, 20% completion) gives the GNN and assembler robust signal for inferring equation_kind from partial information.
- Stage 1 (YOLO) still sees all stages; with lower full density, the detector learns to be robust to missing tokens, which improves performance on real partial scenes.
- The stage distribution is configurable per SceneCase via `valid_stages_for_case`; division cases have their own valid-stage subsets to reflect their multi-step structure.

## 7. YOLO epochs + patience: 50 / 10 (iter2 optimisation)

**Decision**: `epochs = 50`, `patience = 10`.

**Why** (iter2 tuning):
- Iteration 2 found that yolov8n on our 8–10K synthetic scenes reaches best val mAP around epoch 30–40, with patience-triggered early stop around epoch 38–45. Setting `epochs = 50` gives a safety margin without excessive wasted epochs.
- `patience = 10` is the convergence sweet spot: early enough to stop within 1–2 hours on CPU, but loose enough to ride through minor val-mAP noise and avoid early-stopping artifacts.
- Expected actual run time: ~1–2 hours on CPU, actual epochs ~35–45 with early stop. This is an overnight-acceptable training window and avoids the 10–14 hour overhead of prior (yolov8s) configurations.

## 8. YOLO batch size: 16 (unchanged)

**Decision**: keep `batch = 16`.

**Why**:
- On CPU, larger batches (32, 64) do not speed up wall-clock training because CPU matmul is not bandwidth-limited in the same way as GPU. Each batch step costs proportional to batch size, so doubling the batch halves the step count but doubles step time.
- 16 is the empirical sweet spot for YOLOv8n on Apple Silicon CPU: the per-step latency stays under ~2 s, the gradient update frequency is high enough for good convergence, and memory use stays under 2 GB.

## 9. Workers: 0, cache: "ram" (was "disk")

**Decision**: `workers = 0`, `cache = "ram"`.

**Why**:
- `workers = 0` keeps dataloading in the main Python process. On CPU training the main process is already the bottleneck, and subprocess data loaders compete for the same CPU cores, producing net-negative speedup. This also avoids the MPS deadlock path entirely even though we are on CPU.
- Switching `cache` from `"disk"` to `"ram"` holds the full training set in memory. At 512×512 grayscale × ~10,500 images, that is ~2.8 GB uncompressed; trivial on a 48 GB machine. This eliminates every image-decode step from the hot loop and typically reduces epoch time by 30–50% on small datasets.

## 10. Learning-rate schedule

**Decision**: keep Ultralytics auto optimiser, `lr0 = 0.01`, `lrf = 0.01`, `momentum = 0.937`, `weight_decay = 0.0005`. Reduce `warmup_epochs` from 3.0 to 2.0.

**Why**:
- `auto` selects AdamW for small models (yolov8n), which converges faster than SGD on small synthetic datasets and is tolerant of larger initial LR.
- The Ultralytics defaults for `lr0`, `lrf`, `momentum`, `weight_decay` are tuned across hundreds of COCO runs and transfer well; changing them without a sweep rarely helps.
- `warmup_epochs = 2.0` (vs 3.0) is appropriate because our dataset is much smaller than COCO; the first epoch already covers 10 K samples, which is enough stability buffer for two epochs of warmup.

## 11. Loss weights: defaults (unchanged)

**Decision**: `box = 7.5`, `cls = 0.5`, `dfl = 1.5`.

**Why**:
- With 6 coarse classes the classification task is easy (visually distinct glyph groups), so down-weighting `cls` (0.5) relative to box regression (7.5) correctly pushes the optimizer toward tight bounding boxes, which matters more than class labels for downstream GNN accuracy.
- Changing these without a sweep typically produces slightly tighter boxes or slightly better cls F1 at the other's expense; for this pipeline both heads are already near saturation, so we accept defaults.

## 12. Augmentation: lightly tightened

**Decision**:
- `degrees = 3.0` (was 5.0)
- `scale = 0.3` (was 0.4)
- `hsv_v = 0.25` (was 0.30)
- everything else unchanged (translate 0.05, hsv_s 0.05, all others 0)

**Why**:
- Per-glyph rotation and row-baseline tilt are already applied at generation time. Layering YOLO-time rotation of ±5° on top of ±7° generation rotation produces effective ±12° rotation, which can push bounding boxes off symbols and weaken learning. Tightening YOLO rotation to 3° preserves the box-target alignment.
- `scale = 0.3` (±30% scale jitter) preserves small glyphs like carries. ±40% can shrink a 28-px carry to ~17 px, below the detector's receptive field granularity at 512 input, causing false negatives.
- `hsv_v = 0.25` reduces overlap with the photometric stage's contrast jitter. Too much cumulative brightness jitter on top of ink-floor variation produces near-blank images that contribute no signal.
- `mosaic`, `mixup`, `copy_paste`, flips, `perspective`, `hsv_h` are correctly disabled: handwriting direction is not symmetric, tablet input is flat, and images are grayscale so hue is meaningless.

## 13. Rendering: 29 granular knobs

**Decision**: 29 rendering knobs across glyph morphology, geometry, stroke variants, and error modes (see `[generation.rendering]` in config.toml).

**Key knobs** (sampled, non-exhaustive):
- **Glyph scales** — `carry_borrow_scale_min/max` (0.50–0.70), `operator_scale_min/max` (0.65–1.20, sampled per-token in iter10-R6)
- **Morphology** — `broken_stroke_band_frac` (0.125, height of erased band per glyph), `page_padding_min/max_px` (2–10, top/bottom canvas margins), `glyph_morph_close_kernel` (0=disabled; 2 or 3 to thicken)
- **Stroke rendering** — `bar_gap_prob`, `bar_thickness_min/max`, `bar_intensity_min/max`, `bar_micro_break_prob`, per-axis jitter sigmas
- **Error modes** — `wrong_carry_col_prob`, `missing_carry_prob`, `wrong_carry_value_prob`, `missing_borrow_prob`, `wrong_operator_prob` (0.0; disabled), `pp_wrong_operator_prob` (0.0; disabled), `missing_pp_prob` (0.0; disabled in iter10-R6), `missing_structural_prob` (0.02; reduced from 0.08), etc.
- **Layout specifics** — `bracket_depth_full_prob`, `step_minus_prob`, `pp_plus_prob`, `helper_operator_prob` (partial-product and division layout variants)

**Why**:
- Iteration 2 expanded rendering parameterisation from ~10 knobs to 29 to capture fine-grained control over error modes and layout variants. Each knob has been empirically calibrated against real showcase samples and eval-bank failures.
- `carry_borrow_scale_min/max` (range, vs prior scalar) allows per-glyph sampling of size variance, reflecting real handwriting where digit sizes vary within a single child's work.
- Error-mode knobs (wrong_carry_col, missing_carry, etc.) generate training signal for the assembler's structural gate, which filters scenes that lack equation-shaped structure (no digits, no operators, etc.) as OOD. The system recognizes structure and symbols as drawn; error-mode scenes train the GNN to detect when structure is broken, not to correct it.
- All knobs are accessible in config.toml; no Python code changes needed to tune variance.

## 14. GNN parameters: two-stream architecture + scene virtual node (iter9+iter10-R6)

**Decision**: `epochs = 60`, `batch_size = 32`, `lr = 0.001`, `dropout = 0.2`, `weight_decay = 0.0001`, `temperature = 0.1` (for NT-Xent contrastive loss on row/col clustering), loss weights rebalanced, two-stream architecture, optional scene virtual node (iter10-R6 tuning).

```
[gnn]
epochs = 60
batch_size = 32
lr = 0.001
dropout = 0.2
weight_decay = 0.0001
temperature = 0.1
use_scene_virtual_node = true       # optional scene node with 8-dim summary features

[gnn.loss_weights]
fine_label = 2.0           # (iter2 bump from 1.0 to prioritise visual classification)
row_contrastive = 1.5      # (iter2 bump from 1.0 for row clustering robustness)
row_ordinal = 1.0
col_contrastive = 1.5      # (iter2 bump from 1.0 for col clustering robustness)
col_ordinal = 1.0
eq_type = 1.0
edge_type = 0.3
```

**Why**:
- **Two-stream architecture:** visual stream (141-dim input) processes all node features; spatial stream (7-dim input, role one-hot stripped) achieves operator independence by design. This split improves row/col accuracy (primary metric) at the cost of eq_type accuracy (tertiary metric).
- **Scene virtual node:** when enabled, an 8-dim synthetic node with scene-summary features (symbol counts, density, bounds) is appended after truncation. The eq_type head reads this node directly instead of global pooling over real nodes. Improves equation-type prediction on sparser or mixed scenes.
- Iteration 2 bumped fine_label from 1.0 → 2.0 to prioritise visual symbol classification, reflecting client priority on accurate digit/operator labels.
- Row/col contrastive losses bumped from 1.0 → 1.5 because spatial accuracy (row placement, column placement) is the second headline metric. The client prioritises row/col placement > digit role, so stronger clustering losses support that goal.
- `epochs = 60` (iter10-R6 increase to 60) provides additional training budget for converging the rebalanced loss (fine_label 2.0, row/col contrastive 1.5). Early stopping patience = 8 epochs.
- `temperature = 0.1` controls how sharply the NT-Xent contrastive loss separates row and column embeddings; lower values cluster more aggressively, and 0.1 is the value carried in `config.toml [gnn]`.
- **Warm-start on two-stream:** from-scratch training on two-stream collapses. Always use `--init-from <prior_run_id>` with `--lr 0.0005` for GNN retraining.

## 15. AMP, rect, device, and scene virtual node

**Decision**: `amp = true`, `rect = true`, `device = "cpu"` (YOLO), `use_scene_virtual_node = true` (GNN).

**Why**:
- `amp = true` uses bfloat16 where safe; on M4 Pro CPU this leverages Apple's Accelerate framework and yields a ~10% speedup with no accuracy loss on YOLOv8.
- `rect = true` groups images of similar aspect ratio into each batch to minimise padding. Since all our canvases are 512×512, this has minimal direct effect but costs nothing.
- `device = "cpu"` is mandatory on this machine to avoid MPS TAL loss crash in ultralytics 8.4.54+. M4 Pro CPU still achieves ~1–2 hours for YOLO training. The training code already flushes the MPS allocator between epochs as a defensive measure even on CPU runs.
- `use_scene_virtual_node = true` (default enabled) appends an 8-dim synthetic node with scene-summary features (symbol counts, max coords, density estimates) after real-node truncation. The eq_type head reads this node directly for improved equation-type prediction on sparse or mixed scenes. Can be disabled (set to `false`) for ablation studies.

## 16. `[parsing] assembler_heuristic_enabled`: false (default)

**Decision**: `assembler_heuristic_enabled = false`.

**Why**:
- The assembler is a recognize-as-drawn formatter. The GNN's `eq_type` head output is the final `equation_kind`; the assembler never overrides it.
- When `assembler_heuristic_enabled = true`, the assembler computes two diagnostic observability fields: an operator-majority vote and a divide-vote. These appear in the output for debugging purposes only and do not mutate `equation_kind`.
- Setting it `true` is only appropriate for diagnostic sessions where you want to inspect operator-disagreement observability alongside the GNN prediction. It must never be left `true` in production.
- The rationale for defaulting `false`: children write wrong operators, unusual operators, and partial equations that a majority-vote heuristic would misclassify. The GNN is trained on those cases and its prediction is more reliable than any post-hoc vote.

## 17. `[inference.cv_fusion]`: classical component proposals as a YOLO safety net

**Decision**: ship CV fusion enabled (`mode = "on"`) with `keep_unclassified = false`, conservative gating (`gate_min_detections = 6`, `gate_mean_conf = 0.6`), `use_second_yolo_pass = false`, and `classify_conf = 0.25`. Retained proposals (when `keep_unclassified` is turned on) are gated by an ink-density floor `keep_min_ink_frac = 0.10`. Dataclass defaults in `CvFusionConfig` are kept identical to these `config.toml` values and a config-vs-dataclass drift test enforces that.

```
[inference.cv_fusion]
mode = "on"                   # on | auto | off
flatten_background = true
gate_min_detections = 6
gate_mean_conf = 0.6
use_second_yolo_pass = false
second_pass_conf = 0.15
tight_mask_dilate_px = 4
classify_conf = 0.25
classify_with_yolo = true
keep_unclassified = false
keep_min_ink_frac = 0.10
fallback_label = "digit_main"
unclassified_conf = 0.20
mask_yolo = true
cv_tight_mask = true
mask_dilate_px = 5
nms_iou = 0.3
letterbox_size = 512
min_area = 30
min_side = 4
pad = 4
merge_overlap_ratio = 0.95
proximity_merge_factor = 0.0
```

**Why**:
- The YOLO detector is trained on synthetic scenes and under-detects faint real handwriting. Connected-component analysis on the preprocessed ink mask finds those strokes independently of the learned detector, so fusing the two recovers boxes YOLO alone misses without retraining.
- **`mode`** selects when fusion runs. `on` (shipped default) always fuses; `auto` fuses only when YOLO output looks weak (below the gate); `off` is YOLO-only and the cleanest baseline. A 3-arm A/B on the 190-scene real eval (off vs on+keep_off vs on+keep_on) found CV fusion roughly neutral overall: it improves division recall (eq_kind 0.47 to 0.56) and lowers the out-of-distribution rate, at a small macro row/col cost (about 1 to 2 percentage points) with aggregate `equation_kind_acc` near 0.51 to 0.52. We ship `on` with `keep_unclassified = false` to keep the division gain while limiting phantom-node noise; switch to `auto` or `off` if the row/col cost matters more than division recall on a given deployment.
- **Gating** decides when `auto` triggers: the component pass runs only if YOLO returns fewer than `gate_min_detections` (6) boxes or a mean confidence below `gate_mean_conf` (0.6). Six aligns with the smallest realistic equation skeleton (two short operands plus an operator and a bar), and 0.6 sits below confident synthetic-trained detections but above the noise floor seen on real ink.
- **`use_second_yolo_pass` (false)** optionally re-runs YOLO at the lower `second_pass_conf` (0.15) on a masked crop to relabel fused components. It is disabled by default because the second pass roughly doubles inference latency for a marginal recall gain; enable it only when label accuracy on fused boxes matters more than speed.
- **`classify_conf` (0.25)** is the floor for accepting a YOLO label on a fused component. With `keep_unclassified = false` (shipped), components YOLO cannot name above this floor are dropped rather than promoted, which avoids injecting phantom nodes into the GNN. When `keep_unclassified` is turned on, such components are kept under `fallback_label` (`digit_main`) at `unclassified_conf` (0.20) only if they also clear the `keep_min_ink_frac` (0.10) ink-density floor, so faint speckle is still rejected while a genuine stroke survives for the GNN to label.
- The mask and merge knobs (`mask_yolo`, `cv_tight_mask`, `mask_dilate_px`, `tight_mask_dilate_px`, `merge_overlap_ratio`, `proximity_merge_factor`, `nms_iou`) control how component regions are isolated and de-duplicated against YOLO boxes; `letterbox_size` (512) matches the preprocessing canvas, and `min_area` / `min_side` / `pad` filter speckle from genuine glyph blobs.

## 18. `[parsing]`: recognize as drawn, never correct

**Decision**: `assembler_heuristic_enabled = false` (default; see section 16 for the full per-knob rationale).

**Why**:
- The pipeline transcribes the layout and symbols as drawn and never corrects the arithmetic. The GNN's `eq_type` head output is the final `equation_kind`; the assembler is a formatter that does not override it.
- The single knob in this block, `assembler_heuristic_enabled`, is diagnostic-only. When `true` the assembler computes operator-disagreement observability fields (an operator-majority vote and a divide-vote) for inspection, but those fields never mutate `equation_kind`. It must stay `false` outside diagnostic sessions.

---

## Expected runtime

| Stage | Time on M4 Pro CPU |
|-------|-------------------|
| Generate (~8–10 K synthetic scenes, iter2) | ~1–2 hours |
| Validate + build processed artifacts | ~10 min |
| YOLO Stage 1 training (35–45 epochs actual, yolov8n) | ~1–2 hours |
| YOLO test-split evaluation | ~5 min |

Run order:

```bash
# 1. Sanity check + build processed artifacts
PYTHONPATH="$PP" .venv/bin/python -m src validate --project-root "$PP"

# 2. Generate synthetic dataset with the tuned config
PYTHONPATH="$PP" .venv/bin/python -m src generate --project-root "$PP"

# 3. Train YOLO Stage 1
PYTHONPATH="$PP" .venv/bin/python -m src train --project-root "$PP" --stage yolo
```

Success criteria for Stage 1 before moving to Stage 2:
- val mAP50 ≥ 0.92
- val mAP50-95 ≥ 0.70
- test mAP50 within 0.03 of val mAP50 (no overfitting)
- per-class recall ≥ 0.85 on `digit_carry`, `digit_borrow`, `divide_bracket`
