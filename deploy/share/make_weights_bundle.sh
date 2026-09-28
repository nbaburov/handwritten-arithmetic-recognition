#!/usr/bin/env bash
# Build the weights bundle to share with teammates.
#
# Teammates already have everything tracked in git (code, config, active.json, eval bank, docs).
# The only thing they lack is the gitignored *.pt weights. This packs the active weights at their
# repo-relative paths so a teammate extracts the tar at the repo root and the weights land exactly
# where active.json already points.
#
#   bash deploy/share/make_weights_bundle.sh
#
# Produces: deploy/share/har_weights.tar.gz  (~12 MB)
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OUT="deploy/share/har_weights.tar.gz"

# Resolve the active run dirs from active.json (single source of truth).
GNN_RUN="$(python3 -c "import json;print(json.load(open('artifacts/gnn/active.json'))['run_id'])")"
YOLO_RUN="$(python3 -c "import json;print(json.load(open('artifacts/yolo/active.json'))['run_id'])")"

GNN_PT="artifacts/gnn/runs/${GNN_RUN}/best.pt"
YOLO_PT="artifacts/yolo/runs/${YOLO_RUN}/best.pt"

for p in "$GNN_PT" "$YOLO_PT"; do
  if [ ! -f "$p" ]; then echo "ERROR: missing $p"; exit 1; fi
done

echo "Active GNN  : $GNN_RUN"
echo "Active YOLO : $YOLO_RUN"
tar czf "$OUT" "$GNN_PT" "$YOLO_PT"

echo ""
echo "Created: $OUT ($(du -sh "$OUT" | cut -f1))"
echo "Contents:"; tar tzf "$OUT" | sed 's/^/  /'
echo ""
echo "Share this file. Teammate extracts it at the repo root:"
echo "  tar xzf har_weights.tar.gz"
echo "See docs/sharing.md for the full teammate setup."
