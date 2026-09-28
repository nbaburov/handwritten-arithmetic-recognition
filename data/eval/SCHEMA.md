# Assembler Output Schema — schema_version=1

All fields listed below are present in every dict returned by `assemble_json`
in `src/parsing/assemble.py`. Schema version stays at 1; all additions are
backward-compatible (additive only).

## Recognize-as-drawn contract

The assembler structures symbols and equations **as drawn**, without correcting math or relabeling
operator type based on heuristics. `equation_kind` is the GNN's recognized kind; carries and
borrows are flagged if spurious (via `carry_conflict`) but never deleted. Structure-only scenes
(bare digits or grids without operators/equations) emit populated `rows` and `slots` with
`equation_kind="bare_digits"`. Only non-equation shapes (zero detections or operator-only scribbles)
are rejected to OOD with empty `rows` and `slots`.

## Top-level keys

| Key | Type | Always present | Description |
|-----|------|---------------|-------------|
| `schema_version` | int | yes | Always 1. |
| `equation_kind` | str | yes | One of `"add"`, `"subtract"`, `"multiply"`, `"divide"`, `"bare_digits"`, or `"unknown"` (OOD). The recognized kind from the GNN's eq_type head; never overridden by operator vote or heuristics. |
| `ood_reason` | str | OOD only | Reason the structural gate rejected the scene. One of the `OOD_REASON_*` constants in `assemble.py` (e.g., `"true_empty"` for zero detections, `"operator_only"` for operator-only scribbles). |
| `detections` | list | yes | Raw detected tokens with predicted row/col cluster IDs. Present in BOTH successful and OOD outputs. See below. |
| `rows` | list | yes | Structured row list. Empty only on hard-reject OOD (zero detections, operator-only). Populated even for `bare_digits`. |
| `slots` | dict | yes | Categorised token lists: `main_digits`, `carries`, `borrows`, `operators`, `structures`. Empty dict only on hard-reject OOD. Populated even for `bare_digits`. |
| `spatial_meta` | dict | yes | Parser diagnostics. On hard-reject OOD: `{"empty": True}`. On all other routes (success, bare_digits, entropy demotion): `{"parser_mode": "gnn", "spatial_conflict_count": int}`. |
| `row_count` | int | success only | Number of rows in the structured output. |

## detections[] entry schema

Each entry in `detections` corresponds to one `NodePrediction` after pre-gate
filtering (carry flagging, operator dedup, result-bar dedup). On hard-reject OOD this is
the set of predictions that triggered the rejection; on all other routes it mirrors the
full set passed to the row-builder.

| Key | Type | Description |
|-----|------|-------------|
| `label` | str | Fine-grained label (16-class unified ontology, e.g. `"main_2"`, `"op_plus"`, `"result_bar"`, `"div_bracket"`). Post-A2 refactor: role-agnostic digits (`main_0..9`) + 4 operators + 2 structural tokens. |
| `bbox` | list[float] | Bounding box `[x0, y0, x1, y1]` in 512x512 canvas pixels. |
| `row` | int | Predicted row cluster ID (0-based, assigned by GNN gap-clustering on y-centroids). |
| `col` | int | Predicted col cluster ID (0-based, assigned by GNN gap-clustering on x-centroids). |
| `conf` | float | GNN confidence for the predicted fine label. |

## Eval label sidecar schema (data/eval/bank/*.label.json)

```json
{
  "schema_version": 1,
  "sample": "<stem>",
  "equation_kind": "<addition|subtraction|multiplication|division>",
  "scene_case": "<addition|subtraction|multiplication_simple|...>",
  "symbols": [
    {"flattened": "<16-class label>", "bbox_px": [x1, y1, x2, y2]}
  ]
}
```

`scene_case` (optional, added phase 1) and `symbols` (optional) may be absent. When `symbols` is absent,
per-symbol spatial metrics (row_acc, col_acc, recall) are None for that sample. When present but
empty `[]`, recall is 0.0. Showcase-imported sidecars (sc-0001..sc-0030) also carry a `provenance`
block that `load_label_sidecars` silently ignores.

Note: label sidecars do not carry explicit `row`/`col` fields. The eval harness derives GT row/col
cluster IDs by gap-based clustering of `bbox_px` y-centroids (for row) and x-centroids (for col),
matching the GNN's internal assignment logic.
