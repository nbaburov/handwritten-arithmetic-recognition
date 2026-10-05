#!/usr/bin/env bash
# Showcase CI: what "builds and runs from a clean checkout" means for this repository.
# Called by the shared workflow in nbaburov/.github; run it locally with `bash ci.sh`.
# Every check runs in a pinned container, so the result does not depend on the machine.
set -euo pipefail
cd "$(dirname "$0")"

docker run --rm -v "$PWD":/w -w /w -e PYTHONPATH=/w python:3.12-slim sh -c '
  apt-get update -qq && apt-get install -y -qq --no-install-recommends libxcb1 libgl1 libglib2.0-0 >/dev/null &&
  pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu &&
  python -m pytest tests/ -q -p no:cacheprovider'
