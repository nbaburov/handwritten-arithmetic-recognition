# Handwriting recognition — case taxonomy

This document is the **contract** between the synthetic-data generator, the ontology, and the assembler. Every case listed here must be either reproducible synthetically, explicitly handled at inference, or explicitly rejected as `unknown` with a documented `ood_reason`.

This file is the active contract for the set of arithmetic layouts recognised by the system.

---

## Guiding principles

1. **Faithful transcription, not solving.** The pipeline never corrects, completes, or arithmetically validates the child's writing. A wrong answer is parsed as written.
2. **Partial is normal.** Most live samples are mid-write. The default assumption is "this is a partial equation", not "this is malformed". `equation_kind` must classify correctly even when the result is missing, the bar is missing, or only operands are present.
3. **No hallucinated decorations.** A carry must come from a stroke that is actually small and high; a borrow must come from a visible cross-out; a result bar must come from a long horizontal stroke. The model should not infer carries from row context alone. Showcase samples 1, 19, 26, 31, 35 all show spurious `carry_1` predictions on scenes with no carry stroke.
4. **Operator can be on either side.** Children write `+` to the left or to the right after the bottom operand. Both placements are legal. The canonical generator places `+` at the left; op-right variants are in the case set.
5. **Graceful OOD.** When the scene contains no operator, no bar, and no division bracket — and is not a recognisable arithmetic skeleton — emit `equation_kind = "unknown"` with one of the three documented `ood_reason` values. Never guess a kind.
6. **Visual ambiguity is real and must be modelled.** A horizontal stroke can be `op_minus`, `result_bar`, or part of a `+`. A vertical stroke can be `1`, part of `4`, or part of `div_bracket`. An `x` can be `op_times` or two cross-out strokes from a borrow. The architecture must use spatial context to disambiguate.
7. **Stray strokes are part of the signal.** Doodles, tail flicks, restarts, and re-traced glyphs occur in roughly half of real samples. The detector must either suppress them as non-tokens or label them and let the assembler filter by structural role.
8. **Row order can be inverted.** Some children write the bottom operand first, some write the top first. Row index assignment must use spatial y, not stroke order.
9. **Equality sign is real.** Children write `=` before the result roughly as often as they draw a result bar. The current ontology has no `op_equals` token; this is a gap (see Ontology gaps section).

---

## Hard rules

- Column arithmetic only. No inline single-row layouts.
- No empty / scribble / single-decoration scenes in `examples/`.
- No ontology-gap cases (`=`, `÷`, `/`, `*`/`·`, scratchout, decimal point) — they require ontology extension first.
- No scene-level rotation or mirroring.
- Every kept case has exactly one example PNG in `examples/<case_name>.png`.
- Adversarial / detector-stress / NMS edge cases are handled at training-noise / NMS / architecture layers, not as scene cases. See "Training noise dimensions" section.
- OOD rejection at inference is the assembler's structural gate, not a generator scene type.

---

## Canonical cases (78 total)

### Addition (10 baseline + 9 combo variants = 19)

| case_name | scene_case | stage | wrong | evidence_count |
|-----------|------------|-------|-------|----------------|
| `add_full_no_carry` | addition | full | False | 2 (s14, s33) |
| `add_full_with_carry` | addition | full | False | 0 |
| `add_full_wrong_answer` | addition | full | True | 0 |
| `add_operator_left` | addition | full | False | 0 |
| `add_operator_right` | addition_op_right | full | False | 2 (s30, s31) |
| `add_no_bar` | addition_no_bar | done_40 | False | 6 (s9,s10,s11,s13,s21,s27) |
| `add_partial_operands_only` | addition | done_20 | False | 0 |
| `add_carries_above_operands` | addition | done_80 | False | 0 |
| `add_dense_carries_full` | addition_dense_carries | full | False | 0 |
| `add_no_carries_full` | addition_no_carries | full | False | 0 |

#### Addition combo variants

| case_name | scene_case | stage | wrong | rendering_overrides |
|-----------|------------|-------|-------|---------------------|
| `add_full_op_right_with_carry` | addition_op_right | full | False | — |
| `add_full_op_right_no_carry` | addition_op_right | full | False | — |
| `add_no_bar_with_carry` | addition_no_bar | full | False | — |
| `add_partial_with_carry` | addition | done_80 | False | — |
| `add_wrong_carry_col` | addition | full | False | wrong_carry_col_prob=1.0 |
| `add_missing_carry` | addition | full | False | missing_carry_prob=1.0 |
| `add_wrong_carry_value` | addition | full | False | wrong_carry_value_prob=1.0 |
| `add_op_right_no_bar` | addition_op_right | full | False | — |
| `add_short_result_bar` | addition | full | False | short_bar_prob=1.0 |

### Subtraction (9 baseline + 5 combo variants = 14)

| case_name | scene_case | stage | wrong | evidence_count |
|-----------|------------|-------|-------|----------------|
| `sub_full_no_borrow` | subtraction | full | False | 0 |
| `sub_full_with_borrow` | subtraction | full | False | 0 |
| `sub_full_wrong_answer` | subtraction | full | True | 0 |
| `sub_operator_left` | subtraction | full | False | 0 |
| `sub_operator_right` | subtraction_op_right | full | False | 0 |
| `sub_no_bar` | subtraction_no_bar | done_40 | False | 0 |
| `sub_partial_borrows_only` | subtraction | done_60 | False | 0 |
| `sub_heavy_borrow_full` | subtraction_heavy_borrow | full | False | 0 |
| `sub_heavy_borrow_partial` | subtraction_heavy_borrow | done_60 | False | 0 |

#### Subtraction combo variants

| case_name | scene_case | stage | wrong | rendering_overrides |
|-----------|------------|-------|-------|---------------------|
| `sub_full_with_borrow_no_crossout` | subtraction | full | False | borrow_cross_prob=0.0 |
| `sub_full_with_borrow_full_crossout` | subtraction | full | False | borrow_cross_prob=1.0 |
| `sub_heavy_borrow_no_crossout` | subtraction_heavy_borrow | full | False | borrow_cross_prob=0.0 |
| `sub_full_op_right_with_borrow` | subtraction_op_right | full | False | — |
| `sub_missing_borrow` | subtraction | full | False | missing_borrow_prob=1.0 |
| `sub_no_bar_with_borrow` | subtraction_no_bar | full | False | — |

### Multiplication — simple (7 baseline + 3 combo variants = 10)

| case_name | scene_case | stage | wrong | evidence_count |
|-----------|------------|-------|-------|----------------|
| `mul_simple_full_no_carry` | multiplication-simple | full | False | 0 |
| `mul_simple_full_with_carry` | multiplication-simple | full | False | 0 |
| `mul_simple_full_wrong` | multiplication-simple | full | True | 0 |
| `mul_simple_operator_left` | multiplication-simple | full | False | 0 |
| `mul_simple_operator_right` | multiplication_simple_op_right | full | False | 0 |
| `mul_simple_no_bar` | multiplication_simple_no_bar | done_40 | False | 0 |
| `mul_simple_partial` | multiplication-simple | done_20 | False | 0 |

#### Multiplication-simple combo variants

| case_name | scene_case | stage | wrong | rendering_overrides |
|-----------|------------|-------|-------|---------------------|
| `mul_simple_op_right_with_carry` | multiplication_simple_op_right | full | False | — |
| `mul_simple_no_bar_with_carry` | multiplication_simple_no_bar | full | False | — |
| `mul_simple_full_no_carry_simple_ops` | multiplication-simple | full | False | — |

### Multiplication — multi-digit (5 baseline + 6 combo variants = 11)

| case_name | scene_case | stage | wrong | evidence_count |
|-----------|------------|-------|-------|----------------|
| `mul_multi_full` | multiplication-multi | full | False | 0 |
| `mul_multi_full_with_carries` | multiplication-multi | full | False | 0 |
| `mul_multi_wrong_final` | multiplication-multi | full | True | 0 |
| `mul_multi_first_pp_only` | multiplication-multi | done_20 | False | 0 |
| `mul_multi_pp_no_final_sum` | multiplication-multi | done_60 | False | 0 |

#### Multiplication-multi combo variants

| case_name | scene_case | stage | wrong | rendering_overrides |
|-----------|------------|-------|-------|---------------------|
| `mul_multi_with_pp_plus` | multiplication-multi | full | False | pp_plus_prob=1.0 |
| `mul_multi_with_pp_wrong_minus` | multiplication-multi | full | False | pp_plus_prob=1.0, pp_wrong_operator_prob=1.0 |
| `mul_multi_with_pp_mixed` | multiplication-multi | full | False | pp_plus_prob=1.0, pp_wrong_operator_prob=0.5 |
| `mul_multi_no_pp_operators` | multiplication-multi | full | False | pp_plus_prob=0.0 |
| `mul_multi_missing_pp_row` | multiplication-multi | full | False | missing_pp_prob=1.0 |
| `mul_multi_partial_pp_plus` | multiplication-multi | done_60 | False | pp_plus_prob=1.0 |

### Division — short (4 baseline + 3 combo variants = 7)

| case_name | scene_case | stage | wrong | evidence_count |
|-----------|------------|-------|-------|----------------|
| `div_short_full_with_remainder` | division-short | full | False | 1 (04-division-short-bracket) |
| `div_short_full_multi_step` | division-short | full | False | 0 |
| `div_short_wrong_quotient` | division-short | full | True | 0 |
| `div_short_first_step_only` | division-short | done_40 | False | 0 |

#### Division-short combo variants

| case_name | scene_case | stage | wrong | rendering_overrides |
|-----------|------------|-------|-------|---------------------|
| `div_short_with_step_minus` | division-short | full | False | step_minus_prob=1.0 |
| `div_short_with_step_borrow` | division-short | full | False | long_div_step_borrow_prob=1.0 |
| `div_short_with_step_minus_and_borrow` | division-short | full | False | step_minus_prob=1.0, long_div_step_borrow_prob=1.0 |

### Division — long (5 baseline + 7 combo variants = 12)

| case_name | scene_case | stage | wrong | evidence_count |
|-----------|------------|-------|-------|----------------|
| `div_long_full` | division-long | full | False | 0 |
| `div_long_step_k_of_K` | division-long | done_60 | False | 0 |
| `div_long_bracket_dividend_only` | division-long | done_20 | False | 0 |
| `div_long_wrong_quotient` | division-long | full | True | 0 |
| `div_long_no_final_remainder` | division-long | done_80 | False | 0 |

#### Division-long combo variants

| case_name | scene_case | stage | wrong | rendering_overrides |
|-----------|------------|-------|-------|---------------------|
| `div_long_with_step_minus` | division-long | full | False | step_minus_prob=1.0 |
| `div_long_with_step_borrow` | division-long | full | False | long_div_step_borrow_prob=1.0 |
| `div_long_with_step_minus_and_borrow` | division-long | full | False | step_minus_prob=1.0, long_div_step_borrow_prob=1.0 |
| `div_long_with_helper_operator` | division-long | full | False | helper_operator_prob=0.5 |
| `div_long_with_shallow_bracket` | division-long | full | False | bracket_depth_full_prob=0.0, bracket_depth_min_rows=2 |
| `div_long_missing_step_bar` | division-long | full | False | short_bar_prob=1.0, short_bar_shrink_frac=0.5 |
| `div_long_partial_with_step_minus` | division-long | done_60 | False | step_minus_prob=1.0 |

### Division — simple (3 baseline + 1 combo variant = 4)

| case_name | scene_case | stage | wrong | evidence_count |
|-----------|------------|-------|-------|----------------|
| `div_simple_full` | division-simple | full | False | 1 (06-division-simple-divsym) |
| `div_simple_wrong` | division-simple | full | True | 0 |
| `div_simple_partial` | division-simple | done_40 | False | 0 |

#### Division-simple combo variant

| case_name | scene_case | stage | wrong | rendering_overrides |
|-----------|------------|-------|-------|---------------------|
| `div_simple_wrong_partial` | division-simple | done_40 | True | — |

`evidence_count` is the number of eval-bank samples (`data/eval/bank/`) whose `equation_kind` and partial stage match this case. Cases with 0 evidence are theoretically sound and in scope but have not yet appeared in the bank. The bank has a heavy addition skew (25/37 samples) and is retained for regression testing. The primary measurement set is the 190-scene real eval set (`data/eval/real/`, not included in this repository) (run via `python -m src eval --samples-dir data/eval/real --config-id real`); non-addition evidence counts will grow as that set is curated.

---

## Generator implementation summary

The current `SceneCase` enum has **20 members** (verified in `src/generation/layouts_types.py`):

- 7 base cases: `addition`, `subtraction`, `multiplication-simple`, `multiplication-multi`, `division-short`, `division-long`, `division-simple`
- 9 variant cases: `addition_op_right`, `addition_no_bar`, `subtraction_op_right`, `subtraction_no_bar`, `multiplication_simple_op_right`, `multiplication_simple_no_bar`, `subtraction_heavy_borrow`, `addition_dense_carries`, `addition_no_carries`
- 4 robustness cases (iter10 W10-ROBUSTNESS): `bare_digits`, `standalone_bar`, `standalone_bracket`, `bare_digit_grid` (multi-row, multi-column block of bare main digits, added to exercise row/col-clustering on number grids)

`EquationKind.ood_unknown` is a runtime inference kind, not a `SceneCase`. The 4 OOD scene cases (`ood_two_numbers_stacked`, `ood_operator_only`, `ood_spurious_carry`, `ood_scribble`) were deleted from the generator in a prior iteration.

History of additions relative to the pre-canonical generator:
- **4 SceneCase values added** (iter7 Workstream B): `subtraction_op_right`, `subtraction_no_bar`, `multiplication_simple_op_right`, `multiplication_simple_no_bar`.
- **3 further SceneCase values added** (MC-3/MC-4 coverage axes): `subtraction_heavy_borrow`, `addition_dense_carries`, `addition_no_carries`.
- **4 SceneCase values deleted**: `ood_two_numbers_stacked`, `ood_operator_only`, `ood_spurious_carry`, `ood_scribble`.
- **4 robustness SceneCase values added** (iter10 W10-ROBUSTNESS): `bare_digits`, `standalone_bar`, `standalone_bracket`, `bare_digit_grid`.
- **1 flag added** to layout dispatch: `result_override: Optional[Sequence[int]]` for `*_wrong_*` cases (deterministic `+1 mod 10` on the units digit).
- **Examples driver:** `src/generation/render_examples.py` (CLI: `python -m src render-examples`) emits 78 PNGs into `examples/<case_name>.png` with deterministic seeds, plus `examples/manifest.json` mapping `case_name` to `equation_kind` and `completion_stage`.

Regenerate examples after any layout change:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src render-examples --project-root "$PP"
```

---

## Ontology gaps

The current 16-class fine_label set (10 digits + 4 operators + result_bar + div_bracket) cannot represent the following tokens. These are not in the current ontology, and each requires ontology extension before any case depending on it can be added to this contract.

| Gap token | Description | Status |
|-----------|-------------|--------|
| `op_equals` | Children write `=` where the result bar goes roughly as often as they draw the bar | Not in current ontology |
| `op_divide_obelus` | In project scope (simple division) via the `÷` obelus glyph | Not in current ontology |
| `op_slash` | `/` for inline division | Not in current ontology |
| `op_dot` | `*` or `·` multiplication alternative | Not in current ontology |
| `scratchout` | Generic crossed-out region | Not in current ontology |
| `decimal_point` | Decimal arithmetic | Out of scope for ages 6–12 |

Adding any of these requires: updating `src/core/ontology.py`, adding crop folders under `data/raw/pool_emnist_28/` or programmatic rendering, regenerating synthetic data, and retraining both YOLO and the GNN.

---

## Training noise dimensions (not case-level)

The following phenomena are **augmentation knobs and NMS configuration**, not generator scene cases. They appeared as `stress_*` entries in the original audit taxonomy but do not belong in the case dispatch. They are handled at the training-noise or post-processing layer.

| Phenomenon | Handling layer | Config knob or mechanism |
|------------|---------------|--------------------------|
| Stray dot / short stroke above column (spurious carry) | Synthetic noise injection | Add random small blobs above operand columns in `photometric.py` |
| Re-traced glyph (doubled stroke) | NMS at inference | `config.toml [yolo.inference] iou = 0.3` |
| Split glyph (disconnected strokes) | Preprocessing | Connected-component merge in `preprocessing.py` |
| Thin minus confused with noise | Pool diversity | CROHME pool already includes thin-stroke minus glyphs |
| Tall `1` confused with `7` or `4` | Pool diversity + GNN visual capacity | CROHME pool glyph variety; `CropBackbone` trained on diverse crops |

None of these require a new SceneCase or a new dispatch path in `layouts.py`.

---

## Case priority for next training cycle

Priority is driven by measured failure frequency and severity (kind misclassification > token error > cosmetic drop), read from the live eval metrics rather than a frozen list. The earlier priority tables in this document were keyed to specific 37-scene eval-bank sample indices that no longer map cleanly after the move to the 190-scene real eval set, so they have been removed.

For the current per-equation-kind and per-scene-case breakdowns (with Wilson confidence intervals), consult the latest evaluation summary at `reports/eval/latest.md`, regenerated by:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src eval --project-root "$PP" \
  --samples-dir data/eval/real --config-id real
```

Allocate the next training cycle against the equation kinds and scene cases with the lowest accuracy and tightest confidence intervals in that report.

### Out of scope (do not add)

Scene-level rotation, horizontal mirroring, paper textures, JPEG noise, shadow. Listed to prevent accidental inclusion by future contributors.
