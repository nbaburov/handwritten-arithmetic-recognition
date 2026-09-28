# Train on vast.ai (GPU) — simple path

Train the YOLO detector and/or the GNN on a rented cloud GPU instead of the M4 CPU.
A full 50-epoch YOLO run drops from **~50-70h on CPU** to **~2-3h on an RTX 4090**, for **~$2**.

This guide uses the **web GUI + Jupyter browser terminal** only. No CLI install, no SSH keys.
First-time setup is ~30 minutes; every run after that is one paste.

---

## What you train and on what data

- The repo is **private**, so the box does **not** clone it (cloning a private repo there needs auth = hassle). Instead you bundle code + data locally and upload one tarball.
- The bundle carries: `src/` code, `config.toml`, `requirements.txt`, both eval sets (`data/eval`: the 190-scene real set, primary, plus the 37-sample regression bank), the **already-generated iter10-R6 synthetic dataset** (`data/generated/synthetic/<run>`), and the **active iter10-R6 checkpoints** for warm-start (`artifacts/yolo/active.json` + `best.pt`, `artifacts/gnn/active.json` + `best.pt`). The box trains directly on this shipped synth (skipping validate+generate), so the data is byte-identical to the local iter10-R6 run and eval numbers stay comparable. Shipping pre-generated synth avoids needing the symbol pools on the box and removes any regeneration variability. Iter11 default is **fine-tune (warm-start)** from iter10-R6 checkpoints; from-scratch is a fallback if warm underperforms.

---

## Prereq (one-time, on your Mac after iter11 changes)

**Pack the bundle** (run from repo root):
```bash
bash deploy/vastai/prep_upload.sh
```
Produces `deploy/vastai/har_bundle.tar.gz` (~460 MB; mostly synth PNGs + active checkpoints for warm-start). That single file is everything the box needs. No push, no git auth on the box. Re-run this after any iter11+ config changes or weight promotions to update the shipped active.json pointers.

---

## Step 1 — Rent a GPU (console.vast.ai)

1. Sign up at [console.vast.ai](https://cloud.vast.ai/) and add ~$10 credit.
2. Click **"Edit Image & Config"**. Set the Docker image to:
   ```
   pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
   ```
   (Always use a pinned tag, never `latest`.)
3. **Launch Mode:** select **"Jupyter direct HTTPS"** (the leftmost option in the launch picker). Leave "Jupyter Lab" unchecked.
4. **Disk:** drag the storage slider to **30 GB**. (Default 10 GB is too small; cannot be changed after launch.)
5. **GPU filter:** `RTX 4090` or `RTX 3090` (both 24 GB VRAM, plenty). Pick a host with reliability ≥ 0.99.
6. **On-demand**, not interruptible (a 2-3h run should not risk reclamation).
7. Click **RENT**.

### macOS TLS certificate (one-time, required)

macOS blocks the Jupyter direct-HTTPS page without the cert.

1. Download the vast.ai Jupyter cert (link appears when you pick direct HTTPS mode, or under account/security).
2. Double-click the `.crt` → adds to Keychain.
3. Keychain Access → find "Vast.ai Jupyter" → Trust → **Always Trust**. No reboot.

---

## Step 2 — Open Jupyter + upload the bundle

1. On the instance card, click **"Open"** → Jupyter loads in the browser.
2. In Jupyter: drag `deploy/vastai/har_bundle.tar.gz` from your Mac into the file panel, into **`/workspace`**. Wait for upload (~5-15 min for 460 MB).
3. Jupyter → **New → Terminal**.

---

## Step 3 — Run training (in tmux, so a closed tab doesn't kill it)

In the Jupyter terminal:

```bash
# start a tmux session (survives browser disconnects)
tmux new -s train

# extract the bundle and run
cd /workspace
mkdir -p har && tar xzf har_bundle.tar.gz -C har

# Fine-tune from iter10-R6 (iter11 DEFAULT)
bash har/deploy/vastai/run_on_vast.sh all      # GNN + YOLO warm-start
bash har/deploy/vastai/run_on_vast.sh all warm # explicit fine-tune (same as default)
bash har/deploy/vastai/run_on_vast.sh yolo     # YOLO only, warm-start
bash har/deploy/vastai/run_on_vast.sh gnn      # GNN only, warm-start

# From-scratch fallback (if warm underperforms)
bash har/deploy/vastai/run_on_vast.sh all scratch

# Tune GNN learning rate for warm-start (env var override, default 0.0005)
GNN_LR=0.001 bash har/deploy/vastai/run_on_vast.sh gnn warm
```

Detach from tmux anytime with **Ctrl-b then d**; reattach with `tmux attach -t train`.

The script: installs deps → re-points the shipped synth manifest paths to the box → forces `device=0` →
skips validate+generate (synth is pre-built) → reads warm-start sources from shipped active.json → trains GNN (with `--init-from`) + YOLO (with `--model`) → evals → packs results.
No git, no auth — the bundle already carries the code and active checkpoints.

Watch for the final banner:
```
DONE. Download this file via Jupyter (right-click -> Download):
   /workspace/har_results.tar.gz
```

---

## Step 4 — Download results, then DESTROY

1. In the Jupyter file panel, navigate to `/workspace`, right-click **`har_results.tar.gz`** → **Download**.
   Contents: `gnn_best.pt`, `yolo_best.pt`, their `active.json` + `run.json` + `eval.md`, and `eval_latest.md` / `eval_history.jsonl`.
2. **Destroy the instance** in the console (billing stops only on destroy; "stop" still bills for disk).

---

## Step 5 — Use the weights locally

Drop the downloaded weights into the local repo:

```bash
# from repo root, after unpacking har_results.tar.gz
cp <unpacked>/yolo_best.pt artifacts/yolo/best.pt
cp <unpacked>/gnn_best.pt  artifacts/gnn/best.pt
# update active.json pointers from the downloaded *_active.json, then eval on the
# primary real set (the 37-sample bank is the secondary regression gate):
PYTHONPATH="$(pwd)" .venv/bin/python -m src eval --project-root "$(pwd)" \
  --samples-dir data/eval/real --config-id real
```

Compare `reports/eval/latest.md` against the current real-eval baseline (row/col macro
0.86 / 0.75, eq_kind 0.51). If a new run regresses, the prior active models are retained in
their run dirs under `artifacts/{yolo,gnn}/runs/`; roll back by repointing the matching
`active.json` at the kept run.

---

## Cost (May 2026)

| GPU | $/hr (on-demand) | 50-epoch YOLO + GNN | run cost |
|-----|------------------|---------------------|----------|
| RTX 3090 | $0.20-0.35 | ~3-5h | ~$1-2 |
| RTX 4090 | $0.30-0.55 | ~2-3h | ~$1-2.50 |
| A100 80GB | $0.80-1.50 | ~1.5-2h | ~$2-3 |

---

## Notes / gotchas

- **GNN install is simple here.** The code imports only pure `torch-geometric` (no `torch-scatter`/`torch-sparse`), so `pip install -r requirements.txt` is enough — the usual PyG CUDA-wheel dance does **not** apply.
- **CUDA avoids the MPS bug.** The TAL crash that forces `device=cpu` on the M4 is MPS-only; the CUDA path is clean.
- **`device=0` is set by the script** via `sed` on `config.toml`. It only edits the box's copy; your local config is untouched.
- **Determinism:** the exact local synth is shipped (not regenerated), so cloud-trained models are directly comparable to local iter10-R6.
- If `pip install` fails on `torch==2.11.0` CUDA wheels (very new), pin a slightly older torch in a throwaway edit on the box and reinstall PyG to match.
