"""F6 pytest regression gate — eval marker tests.

These tests are excluded from the default ``pytest tests/`` run via
``addopts = "-m 'not eval'"`` in pyproject.toml.  Run explicitly with::

    pytest tests/ -m eval

"""
from __future__ import annotations

import pytest
from pathlib import Path

from src.eval.harness import load_baseline_record

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Gate scope: this floor applies to the SYNTHETIC eval-bank run (the default `eval` command,
# config-id "bank"), which is the official gated "latest" record. The real-handwriting run
# (`--samples-dir data/eval/real --config-id real`) is a separate, deliberately un-gated
# measurement and is expected to sit below this floor (synth-to-real domain gap).
EQUATION_KIND_ACC_FLOOR: float = 0.70  # ratchet manually when sample set grows >=10 and floor sustained 2+ runs

# ---------------------------------------------------------------------------
# Project root derived from this file's location — no hardcoded paths.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_PATH = _PROJECT_ROOT / "reports" / "eval" / "history.jsonl"


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


@pytest.mark.eval
def test_equation_kind_acc_above_floor() -> None:
    """Latest baseline record must have equation_kind_acc >= EQUATION_KIND_ACC_FLOOR."""
    if not _HISTORY_PATH.exists():
        pytest.skip("No reports/eval/history.jsonl yet — run `python -m src eval --project-root $PP` first")

    record = load_baseline_record(_HISTORY_PATH, config_id="baseline")
    if record is None:
        pytest.skip(f"No reports/eval/history.jsonl yet — run `python -m src eval --project-root $PP` first")

    assert record.aggregate.equation_kind_acc >= EQUATION_KIND_ACC_FLOOR, (
        f"equation_kind_acc {record.aggregate.equation_kind_acc:.4f} is below floor {EQUATION_KIND_ACC_FLOOR} "
        f"(run_id={record.run_id})"
    )


@pytest.mark.eval
def test_no_unexpected_ood_on_valid_samples() -> None:
    """Phase 1.5 placeholder — OOD rate gate not yet implemented."""
    pytest.skip("phase 1.5 placeholder")
