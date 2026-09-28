#!/usr/bin/env bash
# iter10-R6 GPU training — run INSIDE the vast.ai Jupyter terminal.
#
# Precondition: you uploaded har_bundle.tar.gz and extracted it, e.g.
#   cd /workspace && mkdir -p har && tar xzf har_bundle.tar.gz -C har
# Then:
#   bash har/deploy/vastai/run_on_vast.sh all          # GNN + YOLO, WARM-start from iter10-R6 (default)
#   bash har/deploy/vastai/run_on_vast.sh all warm     # explicit fine-tune (same as default)
#   bash har/deploy/vastai/run_on_vast.sh all scratch  # from-zero fallback (later, if warm underperforms)
#   bash har/deploy/vastai/run_on_vast.sh yolo         # YOLO only (warm)
#   bash har/deploy/vastai/run_on_vast.sh gnn          # GNN only (warm)
# iter11 default is WARM (fine-tune). The bundle ships the active iter10-R6 checkpoints; warm mode reads
# their run ids from the shipped active.json and passes --init-from (GNN) / --model (YOLO).
#
# The bundle ships the already-generated iter10-R6 synth, so we SKIP validate+generate
# (no pools needed). We only re-point absolute paths in the synth manifests to this box.
set -euo pipefail

STAGE="${1:-all}"
MODE="${2:-warm}"   # iter11: warm = fine-tune from active iter10-R6 (default); scratch = from-zero (fallback)
GNN_LR="${GNN_LR:-0.0005}"   # warm-start GNN learning rate (override with env GNN_LR=...)

# Repo root = two levels up from this script (har/deploy/vastai/ -> har/)
cd "$(dirname "$0")/../.."
export PP="$(pwd)"
export PYTHONPATH="$PP"
echo "==== repo root: $PP ===="

echo "==== [0] python shim (some images ship python3 only) ===="
if ! command -v python >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python
fi
PY="$(command -v python3 || command -v python)"
echo "using: $PY ($($PY --version 2>&1))"

echo "==== [1] install deps (CUDA torch from PyPI; PyG is pure — no scatter/sparse) ===="
"$PY" -m pip install -q --upgrade pip || true
"$PY" -m pip install -q -r requirements.txt
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "No CUDA GPU visible — you rented a CPU box. Stop and re-rent a GPU."
PYEOF

echo "==== [2] locate shipped synth + re-point manifest paths to this box ===="
SYNTH_DIR="$(ls -d data/generated/synthetic/2026*/ 2>/dev/null | head -1)"
SYNTH_DIR="${SYNTH_DIR%/}"
if [ -z "$SYNTH_DIR" ] || [ ! -d "$SYNTH_DIR" ]; then
  echo "ERROR: no synth run dir under data/generated/synthetic/. Bundle extract failed?"; exit 1
fi
echo "synth: $SYNTH_DIR  ($(find "$SYNTH_DIR/train/images" -name '*.png' 2>/dev/null | wc -l) train imgs)"
# Manifests were written with the local Mac absolute prefix. Rewrite the prefix that
# precedes /data/generated/synthetic/ to this box's repo root. Per-CSV, all 3 columns.
for csv in "$SYNTH_DIR"/*.csv; do
  sed -i "s#/[^,]*/data/generated/synthetic/#$PP/data/generated/synthetic/#g" "$csv"
done
echo "manifest paths repointed to $PP"
# Recreate the 'latest' symlink (relative) so resolve_latest_run() finds the run.
ln -sfn "$(basename "$SYNTH_DIR")" data/generated/synthetic/latest
echo "latest -> $(readlink data/generated/synthetic/latest)"

echo "==== [3] force CUDA device in config (YOLO reads this; GNN auto-detects) ===="
sed -i 's/^device = "cpu"/device = "0"/' config.toml
grep -E '^device' config.toml

echo "==== [4] (validate + generate SKIPPED — synth shipped pre-built) ===="

# Capture warm-start sources from the SHIPPED active.json BEFORE training overwrites them.
GNN_INIT="$("$PY" -c "import json;print(json.load(open('artifacts/gnn/active.json'))['run_id'])" 2>/dev/null || echo "")"
YOLO_INIT_RID="$("$PY" -c "import json;print(json.load(open('artifacts/yolo/active.json'))['run_id'])" 2>/dev/null || echo "")"
YOLO_INIT_PT="$PP/artifacts/yolo/runs/$YOLO_INIT_RID/best.pt"
echo "mode=$MODE  gnn_init=$GNN_INIT  yolo_init=$YOLO_INIT_PT"
if [ "$MODE" = "warm" ]; then
  if [ -z "$GNN_INIT" ] || [ ! -f "$YOLO_INIT_PT" ]; then
    echo "ERROR: warm mode needs shipped active checkpoints (run prep_upload.sh after the iter11 changes)."; exit 1
  fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "gnn" ]; then
  if [ "$MODE" = "warm" ]; then
    echo "==== [5a] fine-tune GNN (warm-start from $GNN_INIT, lr=$GNN_LR) ===="
    "$PY" -m src train --project-root "$PP" --stage gnn --init-from "$GNN_INIT" --lr "$GNN_LR"
  else
    echo "==== [5a] train GNN from scratch ===="
    "$PY" -m src train --project-root "$PP" --stage gnn
  fi
fi
if [ "$STAGE" = "all" ] || [ "$STAGE" = "yolo" ]; then
  if [ "$MODE" = "warm" ]; then
    echo "==== [5b] fine-tune YOLO (warm-start from $YOLO_INIT_PT) ===="
    "$PY" -m src train --project-root "$PP" --stage yolo --model "$YOLO_INIT_PT"
  else
    echo "==== [5b] train YOLO from scratch (CUDA, full epochs) ===="
    "$PY" -m src train --project-root "$PP" --stage yolo
  fi
fi

echo "==== [6] eval on 37-sample bank ===="
"$PY" -m src eval --project-root "$PP" || echo "(eval non-fatal; check reports/eval/latest.md)"

echo "==== [7] package artifacts for download ===="
OUT="$PP/../har_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
OUT="$OUT" "$PY" - <<'PYEOF'
import json, glob, shutil, os
from pathlib import Path
out = os.environ["OUT"]
for stage in ("gnn", "yolo"):
    aj = Path(f"artifacts/{stage}/active.json")
    if not aj.exists():
        print(f"[warn] no {stage} active.json"); continue
    d = json.loads(aj.read_text())
    rid = d.get("run_id") or (d.get("outputs", {}) or {}).get("run_id")
    cands = glob.glob(f"artifacts/{stage}/runs/{rid}/**/best.pt", recursive=True) if rid else []
    if not cands:
        cands = glob.glob(f"artifacts/{stage}/best.pt")
    if cands:
        shutil.copy(cands[0], f"{out}/{stage}_best.pt"); print(f"[ok] {stage}_best.pt <- {cands[0]}")
    shutil.copy(aj, f"{out}/{stage}_active.json")
    if rid:
        for f in ("run.json", "eval.md"):
            p = glob.glob(f"artifacts/{stage}/runs/{rid}/{f}")
            if p: shutil.copy(p[0], f"{out}/{stage}_{f}")
for f in ("reports/eval/latest.md", "reports/eval/history.jsonl"):
    if Path(f).exists():
        shutil.copy(f, f"{out}/{Path(f).name}")
print("packaged into", out)
PYEOF
TARBALL="$PP/../har_results.tar.gz"
tar czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo ""
echo "================================================================"
echo " DONE. Download via Jupyter (right-click -> Download):"
echo "   $(cd "$(dirname "$TARBALL")" && pwd)/har_results.tar.gz"
tar tzf "$TARBALL" | sed 's/^/   /'
echo " Then DESTROY the instance to stop billing."
echo "================================================================"
