#!/usr/bin/env bash
# LOCAL helper (run on the M4, from the repo root).
# Bundles everything the GPU box needs into ONE tarball — no git clone on the box
# (repo is private). Ships the ALREADY-GENERATED iter10-R6 synthetic dataset, so the
# box skips validate+generate entirely and trains on data byte-identical to local.
# This avoids shipping the EMNIST/CROHME pools and any regeneration variability.
#
#   bash deploy/vastai/prep_upload.sh
#
# Produces: deploy/vastai/har_bundle.tar.gz  (~900 MB; mostly synth PNGs)
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root
OUT="deploy/vastai/har_bundle.tar.gz"

# Locate the synth run dir that `latest` points at.
SYNTH_LINK="data/generated/synthetic/latest"
if [ ! -e "$SYNTH_LINK" ]; then
  echo "ERROR: $SYNTH_LINK missing. Generate synth locally first."; exit 1
fi
SYNTH_RUN="$(basename "$(readlink "$SYNTH_LINK")")"
SYNTH_DIR="data/generated/synthetic/$SYNTH_RUN"
echo "Synth run: $SYNTH_RUN"

for p in src config.toml requirements.txt pyproject.toml data/eval "$SYNTH_DIR"; do
  if [ ! -e "$p" ]; then echo "ERROR: missing $p (run from repo root)"; exit 1; fi
done

# iter11 fine-tune: ship the ACTIVE (iter10-R6) checkpoints so the box can WARM-START from them.
# run_on_vast.sh reads the run ids from these active.json files and passes --init-from / --model.
WARM_FILES=()
for stage in yolo gnn; do
  aj="artifacts/$stage/active.json"
  if [ ! -f "$aj" ]; then echo "ERROR: missing $aj (need active checkpoint to warm-start)"; exit 1; fi
  rid="$(python3 -c "import json;print(json.load(open('$aj'))['run_id'])")"
  best="artifacts/$stage/runs/$rid/best.pt"
  if [ ! -f "$best" ]; then echo "ERROR: missing $best for warm-start"; exit 1; fi
  WARM_FILES+=("$aj" "$best")
  [ -f "artifacts/$stage/runs/$rid/run.json" ] && WARM_FILES+=("artifacts/$stage/runs/$rid/run.json")
  echo "warm-start $stage <- $rid"
done
# YOLO from-scratch fallback weights (if a from-scratch run is ever wanted on the box).
[ -f "artifacts/pretrained/yolov8n.pt" ] && WARM_FILES+=("artifacts/pretrained/yolov8n.pt")

echo "Bundling code + eval bank + synth ($SYNTH_RUN) + warm-start checkpoints ..."
# Exclude the examples gallery (not needed for training) and any stale YOLO caches.
tar czf "$OUT" \
  --exclude='**/__pycache__' \
  --exclude='**/.DS_Store' \
  --exclude="$SYNTH_DIR/examples" \
  --exclude='**/labels.cache' \
  src \
  config.toml \
  requirements.txt \
  pyproject.toml \
  data/eval \
  "$SYNTH_DIR" \
  "${WARM_FILES[@]}" \
  deploy/vastai/run_on_vast.sh

# Record which run id the box should activate (run_on_vast reads this).
echo "$SYNTH_RUN" > /tmp/har_synth_run.txt

echo ""
echo "Created: $OUT ($(du -sh "$OUT" | cut -f1))"
echo "Synth run id baked in: $SYNTH_RUN"
echo ""
echo "Next:"
echo "  1. Rent a GPU on vast.ai (Jupyter mode) — see deploy/vastai/README.md."
echo "  2. Drag $OUT into the Jupyter file panel, into /workspace."
echo "  3. In the Jupyter terminal:"
echo "       cd /workspace && mkdir -p har && tar xzf har_bundle.tar.gz -C har"
echo "       bash har/deploy/vastai/run_on_vast.sh all"

