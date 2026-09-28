# Sharing

This project has two sharing paths, both needing only the gitignored `*.pt` weights: a quick
client kit (run the models) and the full teammate setup (develop and retrain).

## Sharing with clients

A client who only needs to run the models, not develop or retrain, gets the git repo plus a
single weights zip and is running in four commands. The zip is built by a regenerable generator
and carries friendly-named weights, an installer, and its own README, so the client never has to
decode a timestamped run id.

### What to send

Two things: read access to the git repo (so they `git clone` the code, config, and `active.json`
pointers), and the one file `deploy/share/har_client_models.zip` (about 25 MB). Build or rebuild
it after promoting new models with:

```bash
python3 deploy/share/make_client_kit.py
```

The generator reads `active.json` and `run_labels.json`, so the packaged names always track the
current promotion. It dedups the weights on disk (nine files, six unique blobs) and labels each
by role: the two active iter10-R6 cloud models the pipeline loads, two retained rollbacks, the
pretrained YOLOv8n base, and one training last-epoch checkpoint. The zip also contains
`install.sh`, a human-readable `README.md`, and a `MANIFEST.json` of SHA-256 sums and
destinations for verification.

### Client setup (the tutorial to send them)

1. Clone and enter the repo:
   ```bash
   git clone <repo-url> handwritten-arithmetic-recognition
   cd handwritten-arithmetic-recognition
   ```
2. Create the environment (dedicated venv, not the repo-root one):
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. Plug in the weights. Unzip the kit, then run its installer from inside the repo. It copies
   every weight to the exact path `active.json` expects and refuses to run outside the repo:
   ```bash
   unzip har_client_models.zip
   bash har_client_models/install.sh
   ```
4. Run it:
   ```bash
   PYTHONPATH="$(pwd)" .venv/bin/python -m src gui --project-root "$(pwd)"    # draw or upload a sum
   PYTHONPATH="$(pwd)" .venv/bin/python -m src demo --project-root "$(pwd)"   # feedback demo
   ```

The client kit is the parallel path to the teammate bundle below: the same gitignored weights,
but named by role with an installer, so a non-developer can stand the models up without learning
the repo layout.

## Sharing with teammates

Everything in this repo is tracked in git except large binaries: the model weights (`*.pt`), the
synthetic and raw datasets (`data/raw/`, `data/generated/`), logs, and per-run YOLO training
visualisations. A teammate who clones the repo therefore has all the code, configuration,
`active.json` pointers, the evaluation bank, and this documentation.

To **draw real scenes for data collection** (the main task): a teammate needs only the gitignored
**weights bundle** (about 8 MB). The symbol pool (`data/raw/pool_emnist_28/`, 337 MB) is not needed
— the set-maker generates targets from layout geometry alone, and falling back to connected-component
detection if YOLO weights are absent still allows full drawing and labeling. The pool and manifest
are only needed for `validate` and `generate` (synthetic retraining), not for the set-maker.

## What to send

One file: `deploy/share/har_weights.tar.gz` (about 8 MB). It contains the two active weights:

- `artifacts/yolo/runs/20260622T134925Z_fc69f9ef/best.pt` (iter11 cloud YOLO)
- `artifacts/gnn/runs/20260622T123233Z_7dfc6c79/best.pt` (iter11 cloud GNN)

Regenerate it any time after promoting new models with:

```bash
bash deploy/share/make_weights_bundle.sh
```

The script reads `active.json`, so it always packs whatever is currently active, at the correct
repo-relative paths.

The training datasets are not shared. They are not needed for the GUI, evaluation, or real-data
fine-tuning. They are only needed to retrain on synthetic data, which can be regenerated locally
(see "Regenerating data" below).

## Teammate setup (the tutorial to send them)

### 1. Clone and enter the repo

```bash
git clone <repo-url> handwritten-arithmetic-recognition
cd handwritten-arithmetic-recognition
```

### 2. Create the environment

This project uses its own dedicated virtual environment (do not use any repo-root venv).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PP="$(pwd)"
```

### 3. Drop in the weights

Put the `har_weights.tar.gz` you were sent into the repo root, then extract it there. It unpacks
directly into the run directories that `active.json` already points to.

```bash
tar xzf har_weights.tar.gz
# verify
ls artifacts/yolo/runs/*/best.pt artifacts/gnn/runs/*/best.pt
```

### 4. Run it

Launch the GUI (draw or upload a handwritten sum):

```bash
PYTHONPATH="$PP" .venv/bin/python -m src gui --project-root "$PP"
# open http://127.0.0.1:7860
```

Run the evaluation harness over the 37-sample regression bank:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src eval --project-root "$PP"
# results: reports/eval/latest.md
```

Run against the 190-scene real eval set (primary measurement set):

```bash
PYTHONPATH="$PP" .venv/bin/python -m src eval --project-root "$PP" --samples-dir data/eval/real --config-id real
# results: reports/eval/latest.md  (iter11: row_acc 0.854, col_acc 0.792, eq_kind_acc 0.547)
```

Infer a single image:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src infer --project-root "$PP" --image path/to/img.png --output-json out.json
```

### 5. Make the training set (the main team task)

The model already places rows and columns well on real handwriting; the measured weakness is
operator recognition (it misreads `+` as `-`). To fix it we need real, hand-drawn training
scenes, and the set-maker decides exactly which ones.

Launch the set-maker and switch to Train:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src setmaker --project-root "$PP"
# open http://127.0.0.1:8000  -> click Train -> Start
```

The tool shows a round budget (default 300 scenes). This is the total for the **entire team** for
one round; split it across teammates by agreement (for example, if three of you are drawing, aim
for ~100 each). Served seeds prevent overlap, so do not draw 300 each. Once the round is complete,
run `prepare-realtrain` and fine-tune (see below) before starting another round.

Then just keep going until it says complete:

1. The tool shows the next scene to draw, picked to target the model's weakest cases. Some
   scenes are normal equations, some are multi-row number grids, and some ask you to draw a
   deliberate mistake (wrong operator, or a wrong/extra/missing carry or borrow).
2. Draw the scene on the canvas, then press Detect. Boxes and labels are pre-filled; fix only
   the ones flagged for review. The in-tool Help panel explains row index, column index,
   equation index, when a scene counts as an error, and what to do when unsure. The rule is
   always: label exactly what you drew, never correct the maths.
3. For an error scene, choose the error kind before saving (the tool will not let you save
   without it).
4. Save. The tool advances automatically and stops when the round is complete. Your scenes are
   written to `data/setmaker/train/<stem>.png`, `.gt.json`, and `.txt` (tracked and shared like
   the eval set).

### 6. Fine-tune on the collected scenes

Once a round of training scenes is drawn, build the merged dataset and warm-start fine-tune,
then re-measure against the real evaluation set:

```bash
# Merge the real scenes with the synthetic set (validation/test stay synthetic)
PYTHONPATH="$PP" .venv/bin/python -m src prepare-realtrain --project-root "$PP"

# Warm-start fine-tune both stages from the current active weights (low learning rate)
PYTHONPATH="$PP" .venv/bin/python -m src train --project-root "$PP" --stage gnn  --finetune-from-real --init-from <active_gnn_run> --lr 0.0005
PYTHONPATH="$PP" .venv/bin/python -m src train --project-root "$PP" --stage yolo --finetune-from-real

# Re-measure on the real eval set and compare to the prior baseline
PYTHONPATH="$PP" .venv/bin/python -m src eval --project-root "$PP" --samples-dir data/eval/real --config-id real
```

The active run to warm-start from is recorded in `artifacts/gnn/active.json` (see the model
registry). The default training path, without `--finetune-from-real`, still trains on synthetic
data only and is unchanged.

## Regenerating synthetic data (only if retraining from scratch)

The synthetic dataset and symbol pools are not shared. They are only needed if you are retraining
the models on synthetic data. To rebuild the synthetic dataset locally, the symbol pool must be
present under `data/raw/` (EMNIST or CROHME, per `config.toml` `source_pool`), then:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src validate --project-root "$PP"
PYTHONPATH="$PP" .venv/bin/python -m src generate --project-root "$PP"
```

For the real-data fine-tune workflow (collecting scenes with the set-maker and fine-tuning both
stages), you do NOT need to regenerate synthetic data — only `prepare-realtrain` to merge your
collected scenes with the existing synthetic validation/test splits.

## Which model is which

Two run directories are kept on disk per stage: the active model that the bundle ships, and one
rollback model for reverting cleanly. The active pair is the iter11 cloud YOLO
(`20260622T134925Z_fc69f9ef`) and GNN (`20260622T123233Z_7dfc6c79`). See `docs/runs.md` for the
full registry mapping run IDs to labels, lineage, and status.

## License

The code is released under the Apache License 2.0; the full text is in the `LICENSE` file at the
repo root. Contributors retain the right to use, modify, and redistribute the system under those
terms.
