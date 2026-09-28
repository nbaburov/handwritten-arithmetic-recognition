#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3.12}"

if [[ ! -x "$PY" ]]; then
  if command -v "$PY" >/dev/null 2>&1; then
    :
  else
    echo "Could not find Python at '$PY'. Set PYTHON to a python3.12 executable." >&2
    exit 1
  fi
fi

if [[ ! -d ".venv" ]]; then
  "$PY" -m venv .venv
fi

. ".venv/bin/activate"
python -m pip install -U pip
python -m pip install -r requirements.txt

echo "Venv ready: $ROOT/.venv"

