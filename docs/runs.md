# Run registry

Human-readable map of training runs to what they are. Run IDs are timestamps, so this table
records the label, lineage, and status that the IDs alone do not convey. Metrics are test-split
unless noted. The live pointers are `artifacts/gnn/active.json` and `artifacts/yolo/active.json`;
machine-readable metric history is in each `artifacts/*/runs/index.json`.

Weights (`best.pt`) are gitignored. See `docs/sharing.md` for how to obtain them.

The short labels in the tables below are the single source of truth in
`artifacts/run_labels.json`. The GUI model picker reads that same file, so the dropdown tags
("iter7 · ACTIVE", "iter10-R6 GNN-B · ACTIVE", "iter10-R6 cloud", ...) always match this
registry. When promoting or adding a run, edit `artifacts/run_labels.json` and this table together.

Prior iteration runs (roughly seventeen YOLO and GNN training runs from iter7 through iter10-R5)
were removed in a disk cleanup, so their directories no longer exist and they are not listed here.
The four surviving runs below are the only ones present on disk. The machine source of truth for
which runs exist and which are active is `artifacts/yolo/active.json`, `artifacts/gnn/active.json`,
and `artifacts/run_labels.json`; this table should always agree with those files.

## GNN

| Run ID | Label | fine_acc / eq_type / row / col (test) | Status |
|--------|-------|---------------------------------------|--------|
| `20260622T123233Z_7dfc6c79` | iter11 cloud (warm-start fine-tune, ep25 early-stop) | 0.9875 / 0.9053 / 0.9348 / 0.7280 | active |
| `20260527T080712Z_14df2140` | iter10-R6 cloud (from-scratch, vast.ai GPU) | 0.9893 / 0.9006 / 0.9209 / 0.7630 | rollback |
| `20260525T224459Z_a7fd6e8f` | iter10-R6 GNN-B (from-scratch, local) | 0.9893 / 0.9006 / 0.9209 / 0.7630 | archive |

Iter11 warm-start from iter10-R6 `14df2140` was promoted on real-data evaluation (190-scene set).
The cloud iter11 improves row clustering and column precision on real handwriting (row_acc 0.854, col_acc 0.792)
and maintains equation-type accuracy. Full active-run metrics (fine_acc 0.9875, fine_macro_f1 0.9901, row 0.9348,
col 0.7280, within_row_ord 0.7445, within_col_ord 0.7546, eq_type 0.9053, exact_scene 0.5071) are in `docs/models/gnn.md`.

## YOLO

| Run ID | Label | mAP50 / mAP50-95 (test) | Status |
|--------|-------|-------------------------|--------|
| `20260622T134925Z_fc69f9ef` | iter11 cloud (warm-start fine-tune, ep11 early-stop) | 0.9648 / 0.8161 | active |
| `20260527T103315Z_aa990906` | iter10-R6 cloud (GPU, full 50ep) | 0.9705 / 0.8398 | rollback |
| `20260507T171441Z_e1060a61` | iter7 baseline (local) | 0.659 / 0.399 | archive |

Iter11 warm-start from iter10-R6 `aa990906` was promoted on real-data evaluation (190-scene set).
Synthetic metrics dipped slightly (mAP50 0.9648 vs 0.9705 due to early-stopping) but real-data operator
recognition improved substantially on addition (+8.6pp to 0.36) and division/multiplication gained small margins.
The prior iter10-R6 is retained as rollback target. The iter7 baseline is kept for reference; it generalised
differently than the later cloud models on the legacy addition-heavy eval bank (0.757 eq_kind historically).

## Current best combination

Active pipeline is the iter11 cloud pair: YOLO `20260622T134925Z_fc69f9ef` + GNN
`20260622T123233Z_7dfc6c79`. These were promoted on real-data evaluation (190-scene set, twice vision-reviewed).
Iter11 improves operator recognition (addition +8.6pp to 0.36) and strengthens row/column placement
(row_acc 0.854, col_acc 0.792) relative to iter10-R6 real-data baseline (row 0.849, col 0.730).
Equation-type accuracy on real is still 0.547, and subtraction operator remains weak (0.11);
the next iteration will target operator glyph disambiguation via real-data fine-tune volumes.

The current primary measurement set is the **190-scene real eval set** in `data/eval/real/` (not included in this repository)
(twice vision-reviewed). Run it with:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src eval --project-root "$PP" --samples-dir data/eval/real --config-id real
```

Real-data performance (iter11): row_acc 0.854, col_acc 0.792, eq_kind_acc 0.547.
The older 37-sample `data/eval/bank/` is retained for regression testing.

The prior iter10-R6 pair (YOLO `aa990906` + GNN `14df2140`) is retained for rollback.
The prior local pair (iter7 YOLO `20260507T171441Z_e1060a61` + GNN-B `20260525T224459Z_a7fd6e8f`) is kept for reference.
