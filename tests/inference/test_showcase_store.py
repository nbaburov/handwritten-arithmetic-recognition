"""Tests for ShowcaseStore — pure I/O, no Gradio imports."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.inference.showcase_store import (
    ShowcaseStore,
    STATUS_OK,
    STATUS_BLANK,
    STATUS_ERROR,
    STATUS_NO_DETECTIONS,
    STATUS_OOD,
    FEEDBACK_ENABLED_STATUSES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> ShowcaseStore:
    return ShowcaseStore(showcase_root=tmp_path / "showcase")


# ---------------------------------------------------------------------------
# Test 1: files written on record_prediction
# ---------------------------------------------------------------------------

def test_record_prediction_writes_files(store: ShowcaseStore) -> None:
    import io
    from PIL import Image
    import numpy as np

    # Build a tiny PNG in memory.
    arr = np.full((512, 512), 200, dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    outcome = {
        "status": STATUS_OK,
        "equation_kind": "addition",
        "num_tokens": 3,
        "timings": {"yolo_ms": 10.0, "stage2_ms": 5.0, "assemble_ms": 1.0},
    }

    sample_dir = store.record_prediction(gray_png_bytes=png_bytes, outcome=outcome)

    assert sample_dir.exists()
    assert (sample_dir / "input.png").exists()
    assert (sample_dir / "prediction.json").stat().st_size > 0

    pred = json.loads((sample_dir / "prediction.json").read_text())
    assert pred["schema_version"] == 1
    assert pred["status"] == STATUS_OK
    assert pred["equation_kind"] == "addition"
    assert "model_provenance" in pred
    assert "timestamp" in pred

    index_path = store.showcase_root / "index.jsonl"
    assert index_path.exists()
    lines = index_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "predict"
    assert rec["outcome_status"] == STATUS_OK


# ---------------------------------------------------------------------------
# Test 2: feedback appended to index.jsonl
# ---------------------------------------------------------------------------

def test_record_feedback_appends_to_index(store: ShowcaseStore, tmp_path: Path) -> None:
    outcome = {"status": STATUS_OK, "equation_kind": "subtraction", "num_tokens": 2}
    sample_dir = store.record_prediction(gray_png_bytes=b"\x89PNG", outcome=outcome)

    store.record_feedback(sample_dir=sample_dir, verdict="correct", intended_text="Nice!")

    assert (sample_dir / "feedback.json").exists()
    fb = json.loads((sample_dir / "feedback.json").read_text())
    assert fb["verdict"] == "correct"
    assert fb["intended_text"] == "Nice!"

    index_lines = (store.showcase_root / "index.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 2  # predict + feedback
    events = [json.loads(l)["event"] for l in index_lines]
    assert events == ["predict", "feedback"]


# ---------------------------------------------------------------------------
# Test 3: sequence numbers increment
# ---------------------------------------------------------------------------

def test_sequence_increments(store: ShowcaseStore) -> None:
    outcome_a = {"status": STATUS_OK, "equation_kind": "addition", "num_tokens": 1}
    outcome_b = {"status": STATUS_NO_DETECTIONS, "equation_kind": None, "num_tokens": 0}

    dir_a = store.record_prediction(gray_png_bytes=b"a", outcome=outcome_a)
    dir_b = store.record_prediction(gray_png_bytes=b"b", outcome=outcome_b)

    assert dir_a.name == "sample_0001"
    assert dir_b.name == "sample_0002"


# ---------------------------------------------------------------------------
# Test 4: blank_canvas still writes input.png and index entry
# ---------------------------------------------------------------------------

def test_record_prediction_blank_canvas(store: ShowcaseStore) -> None:
    outcome = {"status": STATUS_BLANK, "equation_kind": None, "num_tokens": 0}
    sample_dir = store.record_prediction(gray_png_bytes=None, outcome=outcome)

    # input.png must exist (zero-byte placeholder).
    assert (sample_dir / "input.png").exists()

    pred = json.loads((sample_dir / "prediction.json").read_text())
    assert pred["status"] == STATUS_BLANK

    lines = (store.showcase_root / "index.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["outcome_status"] == STATUS_BLANK


# ---------------------------------------------------------------------------
# Test 5: error outcome is persisted correctly
# ---------------------------------------------------------------------------

def test_record_prediction_error(store: ShowcaseStore) -> None:
    outcome = {
        "status": STATUS_ERROR,
        "equation_kind": None,
        "num_tokens": 0,
        "error_detail": "RuntimeError: CUDA OOM",
    }
    sample_dir = store.record_prediction(gray_png_bytes=b"\xff", outcome=outcome)

    pred = json.loads((sample_dir / "prediction.json").read_text())
    assert pred["status"] == STATUS_ERROR
    assert pred["error_detail"] == "RuntimeError: CUDA OOM"

    lines = (store.showcase_root / "index.jsonl").read_text().strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["outcome_status"] == STATUS_ERROR


# ---------------------------------------------------------------------------
# Test 6: feedback disabled for non-eligible statuses (logic check)
# ---------------------------------------------------------------------------

def test_feedback_enabled_statuses() -> None:
    assert STATUS_OK in FEEDBACK_ENABLED_STATUSES
    assert STATUS_NO_DETECTIONS in FEEDBACK_ENABLED_STATUSES
    assert STATUS_OOD in FEEDBACK_ENABLED_STATUSES
    assert STATUS_BLANK not in FEEDBACK_ENABLED_STATUSES
    assert STATUS_ERROR not in FEEDBACK_ENABLED_STATUSES


# ---------------------------------------------------------------------------
# Test 7: per_token persisted in feedback.json when supplied
# ---------------------------------------------------------------------------

def test_record_feedback_per_token(store: ShowcaseStore) -> None:
    outcome = {"status": STATUS_OK, "equation_kind": "addition", "num_tokens": 2}
    sample_dir = store.record_prediction(gray_png_bytes=b"\x89PNG", outcome=outcome)

    per_token_data = [
        {"index": 0, "predicted_label": "main_3", "gnn_conf": 0.95, "user_correct": "Yes", "user_should_be": ""},
        {"index": 1, "predicted_label": "op_plus", "gnn_conf": 0.88, "user_correct": "No", "user_should_be": "op_minus"},
    ]

    store.record_feedback(
        sample_dir=sample_dir,
        verdict="wrong",
        intended_text="operator mislabelled",
        per_token=per_token_data,
    )

    fb = json.loads((sample_dir / "feedback.json").read_text())
    assert fb["verdict"] == "wrong"
    assert "per_token" in fb
    assert len(fb["per_token"]) == 2
    assert fb["per_token"][1]["user_should_be"] == "op_minus"

    # Backward compat: omitting per_token leaves key absent
    outcome2 = {"status": STATUS_OK, "equation_kind": "subtraction", "num_tokens": 1}
    sample_dir2 = store.record_prediction(gray_png_bytes=b"\x89PNG", outcome=outcome2)
    store.record_feedback(sample_dir=sample_dir2, verdict="correct", intended_text="")
    fb2 = json.loads((sample_dir2 / "feedback.json").read_text())
    assert "per_token" not in fb2


# ---------------------------------------------------------------------------
# Test 8: second call to record_feedback overwrites (not appends)
# ---------------------------------------------------------------------------

def test_feedback_overwrite_on_second_call(store: ShowcaseStore) -> None:
    outcome = {"status": STATUS_OK, "equation_kind": "addition", "num_tokens": 2}
    sample_dir = store.record_prediction(gray_png_bytes=b"\x89PNG", outcome=outcome)

    # First call — quick wrong signal, no detail.
    store.record_feedback(
        sample_dir=sample_dir,
        verdict="wrong",
        intended_text="",
        per_token=None,
    )
    fb1 = json.loads((sample_dir / "feedback.json").read_text())
    assert fb1["verdict"] == "wrong"
    assert fb1["intended_text"] == ""
    assert "per_token" not in fb1

    # Second call — adds detail; must overwrite, not append.
    per_token_detail = [{"index": 0, "predicted_label": "main_5", "user_says_wrong": True}]
    store.record_feedback(
        sample_dir=sample_dir,
        verdict="wrong",
        intended_text="24 + 13 = 37",
        per_token=per_token_detail,
    )
    fb2 = json.loads((sample_dir / "feedback.json").read_text())
    assert fb2["verdict"] == "wrong"
    assert fb2["intended_text"] == "24 + 13 = 37"
    assert "per_token" in fb2
    assert fb2["per_token"][0]["user_says_wrong"] is True

    # File must be valid JSON (not two concatenated objects).
    raw = (sample_dir / "feedback.json").read_text()
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Test 9: get_latest_sample and latest_seq
# ---------------------------------------------------------------------------

def test_get_latest_sample_returns_last(store: ShowcaseStore) -> None:
    # Initially None and seq is 0.
    assert store.get_latest_sample() is None
    assert store.latest_seq == 0

    outcome_a = {"status": STATUS_OK, "equation_kind": "addition", "num_tokens": 2}
    dir_a = store.record_prediction(gray_png_bytes=b"\x89PNG", outcome=outcome_a)
    assert store.get_latest_sample() == dir_a
    assert store.latest_seq == 1

    outcome_b = {"status": STATUS_OK, "equation_kind": "subtraction", "num_tokens": 3}
    dir_b = store.record_prediction(gray_png_bytes=b"\x89PNG", outcome=outcome_b)
    assert store.get_latest_sample() == dir_b
    assert store.latest_seq == 2
    # dir_a must be a different path.
    assert dir_a != dir_b
