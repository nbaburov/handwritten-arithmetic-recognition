# Docs index

## Start here
- **[getting-started.md](getting-started.md)** — quick start commands and full setup guide
- **[architecture.md](architecture.md)** — product context (problem, audience, scope), system overview, stage diagram, and end-to-end flow

## Working with the system
- **[runbook.md](runbook.md)** — prep → synth → train → eval workflow, CLI reference, GUI guide, and hyperparameter tables

## Deep dives
- **[models/gnn.md](models/gnn.md)** — GNN architecture: CropBackbone v2 (6-layer, 128-dim), GATv2, 16-class fine_label, multi-task loss, iter2 upgrade rationale
- **[models/yolo.md](models/yolo.md)** — YOLO Stage 1: detection, synthetic data strategy, layer-by-layer breakdown, results
- **[cases.md](cases.md)** — all recognised arithmetic layouts and edge cases
- **[config-rationale.md](config-rationale.md)** — justification for every non-trivial parameter in config.toml

## Models and sharing
- **[runs.md](runs.md)** — short run registry: the active GNN and YOLO run IDs with their labels and headline metrics
- **[sharing.md](sharing.md)** — how to share weights with teammates and the teammate setup tutorial (clone, venv, drop weights, run GUI/eval/fine-tune); includes the set-maker collection workflow and `prepare-realtrain` fine-tune loop

## Real-data loop
- Set-maker (`python -m src setmaker`) — local FastAPI + Konva web tool for building real eval and train sets; quota source is `data/setmaker/quota.json`
- `python -m src prepare-realtrain` — merges set-maker train output with the synthetic dataset for warm-start fine-tune
- Real eval set: 190 scenes in `data/eval/real/` (not included in this repository); run with `python -m src eval --samples-dir data/eval/real --config-id real`
- See [runbook.md](runbook.md) for the full fine-tune workflow and CLI flags

## Feedback demo
- feedback demo (`python -m src demo`) — FastAPI + Konva web app: a child draws a presented exercise, the recognizer reads it, and a swappable grading engine client (`config.toml [grading] mode = mock|http`) grades it with one on-canvas verdict. Runs with no API key via a high-fidelity mock. See [architecture.md](architecture.md)

## Development
- **[development.md](development.md)** — source layout (every package and file), test commands, contribution workflow

## History
