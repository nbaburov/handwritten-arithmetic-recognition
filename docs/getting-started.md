# Getting started

## Quick start

Run this to set up the venv and verify the CLI works.

```bash
./scripts/setup_venv.sh
export PP="$(pwd)"
PYTHONPATH="$PP" .venv/bin/python -m src --help
```

This guide assumes you have **unzipped** the folder `handwritten-arithmetic-recognition/` somewhere locally, and you are working **inside that folder**.

## What you need installed

- **Python 3.12** (see root `README.md` and `.python-version`).
- A working `python3.12` on your PATH (or set `PYTHON=...` for the setup script).

## Install Python packages (once)

From the project folder:

```bash
./scripts/setup_venv.sh
```

Or, if `.venv` already exists and you want to reinstall:

```bash
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

This installs PyTorch, PyTorch Geometric, and Ultralytics/Torch (Stage 1 YOLO + Stage 2 GNN) and optionally Gradio (GUI), as listed in this project's `requirements.txt`. There is no TensorFlow dependency.

## Tell Python where this project lives (`PYTHONPATH`)

All commands use the package name `src`. Python expects `src` to live under a folder you put on `PYTHONPATH`. That folder is the **project root** (the one that contains `data/`, `src/`, `reports/` side by side).

From the project root, a convenient shortcut is:

```bash
export PP="$(pwd)"
```

Then run anything as:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src <command> ...
```

See [runbook.md](runbook.md) for the full command list.

## Quick sanity check

List available CLI commands:

```bash
PYTHONPATH="$PP" .venv/bin/python -m src --help
```

You should see: `validate`, `generate`, `render-examples`, `train`, `infer`, `gui`, `setmaker`, `demo`, `eval`, `prepare-realtrain`, `prepare-pool-clean`, `prepare-pool-hires`.

The `demo` command launches the feedback demo (a FastAPI + Konva web app where a drawn exercise is recognized and graded); see [architecture.md](architecture.md).

The `infer` command accepts `--cv-fusion on|auto|off` to control the CV-fusion augmentation stage (see [architecture.md](architecture.md)); it defaults to the `config.toml [inference.cv_fusion]` mode.

## What to read next

| If you want to...                                                         | Read                                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Understand the product goals and system stages                            | [architecture.md](architecture.md)                           |
| Follow the correct order: prep → train                                    | [runbook.md](runbook.md)                                     |
| CLI reference and workflow order                                           | [runbook.md](runbook.md)                                     |
| Run the real eval set (190 scenes, `--samples-dir data/eval/real`)        | [runbook.md](runbook.md)                                     |
| Collect real handwriting scenes (setmaker) and fine-tune with `prepare-realtrain` | [sharing.md](sharing.md) + [runbook.md](runbook.md) |
| Share weights with a teammate / teammate setup                            | [sharing.md](sharing.md)                                     |
| GNN architecture detail (CropBackbone, GATv2, heads)                     | [models/gnn.md](models/gnn.md)                               |
| YOLO layer-by-layer breakdown                                             | [models/yolo.md](models/yolo.md)                             |
| Every folder and Python file                                              | [development.md](development.md)                     |
