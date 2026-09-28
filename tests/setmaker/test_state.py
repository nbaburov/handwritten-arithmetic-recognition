"""Tests for the set-maker session-state module (WS-F).

Covers the plan's ``test_state.py`` line (save->load round-trip; cursor advance;
progress counts; resume trims met) plus the validation and atomicity guarantees
the module promises (fail-loud-with-path on malformed files, missing-file fresh
state, atomic temp-then-rename, schema tag).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.setmaker import state as st
from src.setmaker.types import SessionState, WorklistItem


def _item(case: str, seed: int, stage: str = "full", status: str = "pending") -> WorklistItem:
    return WorklistItem(case=case, seed=seed, completion_stage=stage, status=status)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def test_state_dir_is_under_project_data(tmp_path: Path) -> None:
    assert st.state_dir(tmp_path) == tmp_path / "data" / "setmaker" / "state"


def test_state_path_per_mode(tmp_path: Path) -> None:
    assert st.state_path(tmp_path, "eval") == tmp_path / "data" / "setmaker" / "state" / "eval.json"
    assert st.state_path(tmp_path, "train") == tmp_path / "data" / "setmaker" / "state" / "train.json"


def test_state_path_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not valid"):
        st.state_path(tmp_path, "bogus")  # type: ignore[arg-type]


def test_state_dir_not_created_by_path_helpers(tmp_path: Path) -> None:
    # Resolving a path must not touch the filesystem; only save() creates dirs.
    st.state_dir(tmp_path)
    st.state_path(tmp_path, "eval")
    assert not (tmp_path / "data").exists()


# --------------------------------------------------------------------------- #
# save -> load round-trip
# --------------------------------------------------------------------------- #


def test_save_returns_target_path_and_writes_file(tmp_path: Path) -> None:
    state = SessionState(mode="eval", worklist=[_item("addition", 1)])
    written = st.save(state, tmp_path)
    assert written == st.state_path(tmp_path, "eval")
    assert written.exists()


def test_save_load_round_trip_preserves_fields(tmp_path: Path) -> None:
    state = SessionState(
        mode="train",
        worklist=[
            _item("addition", 1, "full"),
            _item("division-short", 2, "done_60"),
        ],
        cursor=1,
        round_budget=40,
    )
    st.save(state, tmp_path)
    loaded = st.load("train", tmp_path)

    assert loaded.mode == "train"
    assert loaded.round_budget == 40
    assert loaded.cursor == 1
    assert [(w.case, w.seed, w.completion_stage, w.status) for w in loaded.worklist] == [
        ("addition", 1, "full", "pending"),
        ("division-short", 2, "done_60", "pending"),
    ]


def test_save_creates_state_dir_lazily(tmp_path: Path) -> None:
    assert not st.state_dir(tmp_path).exists()
    st.save(SessionState(mode="eval"), tmp_path)
    assert st.state_dir(tmp_path).is_dir()


def test_save_writes_schema_version(tmp_path: Path) -> None:
    st.save(SessionState(mode="eval", worklist=[_item("addition", 1)]), tmp_path)
    doc = json.loads(st.state_path(tmp_path, "eval").read_text(encoding="utf-8"))
    assert doc["schema_version"] == st.STATE_SCHEMA_VERSION


def test_round_budget_none_for_eval_survives_round_trip(tmp_path: Path) -> None:
    st.save(SessionState(mode="eval", worklist=[_item("addition", 1)]), tmp_path)
    loaded = st.load("eval", tmp_path)
    assert loaded.round_budget is None


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #


def test_save_stamps_created_and_last_saved(tmp_path: Path) -> None:
    written = st.save(SessionState(mode="eval"), tmp_path)
    doc = json.loads(written.read_text(encoding="utf-8"))
    assert doc["created_utc"]
    assert doc["last_saved_utc"]
    # ISO-8601 UTC strings carry a timezone designator.
    assert doc["created_utc"].endswith("+00:00") or doc["created_utc"].endswith("Z")


def test_created_utc_preserved_across_resave(tmp_path: Path) -> None:
    st.save(SessionState(mode="eval"), tmp_path)
    created_after_first = json.loads(st.state_path(tmp_path, "eval").read_text())["created_utc"]

    reloaded = st.load("eval", tmp_path)
    st.save(reloaded, tmp_path)
    doc = json.loads(st.state_path(tmp_path, "eval").read_text())

    assert doc["created_utc"] == created_after_first  # not reset on the second save
    assert doc["last_saved_utc"] >= created_after_first  # advanced (or equal) on resave


# --------------------------------------------------------------------------- #
# Missing file => fresh state
# --------------------------------------------------------------------------- #


def test_load_missing_file_returns_fresh_state(tmp_path: Path) -> None:
    loaded = st.load("eval", tmp_path)
    assert loaded.mode == "eval"
    assert loaded.worklist == []
    assert loaded.cursor == 0
    assert loaded.round_budget is None


def test_load_missing_file_does_not_create_anything(tmp_path: Path) -> None:
    st.load("train", tmp_path)
    assert not (tmp_path / "data").exists()


# --------------------------------------------------------------------------- #
# advance: cursor + status
# --------------------------------------------------------------------------- #


def test_advance_marks_cursor_item_done_and_increments(tmp_path: Path) -> None:
    state = SessionState(mode="eval", worklist=[_item("a", 1), _item("b", 2)], cursor=0)
    advanced = st.advance(state)
    assert advanced.worklist[0].status == "done"
    assert advanced.worklist[1].status == "pending"
    assert advanced.cursor == 1


def test_advance_is_pure_does_not_mutate_input(tmp_path: Path) -> None:
    state = SessionState(mode="eval", worklist=[_item("a", 1)], cursor=0)
    st.advance(state)
    # original frozen state untouched
    assert state.worklist[0].status == "pending"
    assert state.cursor == 0


def test_advance_past_end_is_noop(tmp_path: Path) -> None:
    state = SessionState(mode="eval", worklist=[_item("a", 1, status="done")], cursor=1)
    advanced = st.advance(state)
    assert advanced is state


def test_advance_empty_worklist_is_noop(tmp_path: Path) -> None:
    state = SessionState(mode="train", worklist=[], cursor=0)
    assert st.advance(state) is state


def test_advance_then_save_then_load_persists_done(tmp_path: Path) -> None:
    state = SessionState(mode="train", worklist=[_item("a", 1), _item("b", 2)], cursor=0, round_budget=20)
    advanced = st.advance(state)
    st.save(advanced, tmp_path)
    loaded = st.load("train", tmp_path)
    # item 0 done is the leading run => trimmed on resume; cursor re-based to 0.
    assert [w.case for w in loaded.worklist] == ["b"]
    assert loaded.cursor == 0


# --------------------------------------------------------------------------- #
# progress counts
# --------------------------------------------------------------------------- #


def test_progress_overall_counts(tmp_path: Path) -> None:
    state = SessionState(
        mode="eval",
        worklist=[
            _item("addition", 1, status="done"),
            _item("addition", 2),
            _item("subtraction", 3),
        ],
    )
    prog = st.progress(state)
    assert prog["mode"] == "eval"
    assert prog["overall"] == {"total": 3, "done": 1, "pending": 2, "percent": pytest.approx(100 / 3)}


def test_progress_per_case_counts(tmp_path: Path) -> None:
    state = SessionState(
        mode="train",
        worklist=[
            _item("addition", 1, status="done"),
            _item("addition", 2, status="done"),
            _item("addition", 3),
            _item("division-short", 4),
        ],
    )
    per_case = st.progress(state)["per_case"]
    assert per_case["addition"] == {"total": 3, "done": 2, "pending": 1}
    assert per_case["division-short"] == {"total": 1, "done": 0, "pending": 1}


def test_progress_empty_worklist(tmp_path: Path) -> None:
    prog = st.progress(SessionState(mode="eval"))
    assert prog["overall"] == {"total": 0, "done": 0, "pending": 0, "percent": 0.0}
    assert prog["per_case"] == {}


def test_progress_all_done_is_full_percent(tmp_path: Path) -> None:
    state = SessionState(
        mode="eval",
        worklist=[_item("a", 1, status="done"), _item("b", 2, status="done")],
    )
    assert st.progress(state)["overall"]["percent"] == pytest.approx(100.0)


def test_progress_per_case_order_is_first_seen(tmp_path: Path) -> None:
    state = SessionState(
        mode="train",
        worklist=[_item("subtraction", 1), _item("addition", 2), _item("subtraction", 3)],
    )
    assert list(st.progress(state)["per_case"].keys()) == ["subtraction", "addition"]


# --------------------------------------------------------------------------- #
# Resume trims met (leading-done prefix)
# --------------------------------------------------------------------------- #


def test_load_trims_leading_done_and_rebases_cursor(tmp_path: Path) -> None:
    state = SessionState(
        mode="eval",
        worklist=[
            _item("a", 1, status="done"),
            _item("b", 2, status="done"),
            _item("c", 3),
            _item("d", 4),
        ],
        cursor=2,
    )
    st.save(state, tmp_path)
    loaded = st.load("eval", tmp_path)
    assert [w.case for w in loaded.worklist] == ["c", "d"]
    assert loaded.cursor == 0  # 2 - 2 leading-done


def test_load_keeps_out_of_order_done(tmp_path: Path) -> None:
    # A done item that is NOT in the leading run must be preserved, not trimmed.
    state = SessionState(
        mode="train",
        worklist=[_item("a", 1), _item("b", 2, status="done"), _item("c", 3)],
        cursor=0,
        round_budget=20,
    )
    st.save(state, tmp_path)
    loaded = st.load("train", tmp_path)
    assert [(w.case, w.status) for w in loaded.worklist] == [
        ("a", "pending"),
        ("b", "done"),
        ("c", "pending"),
    ]
    assert loaded.cursor == 0


def test_load_all_done_trims_to_empty(tmp_path: Path) -> None:
    state = SessionState(
        mode="eval",
        worklist=[_item("a", 1, status="done"), _item("b", 2, status="done")],
        cursor=2,
    )
    st.save(state, tmp_path)
    loaded = st.load("eval", tmp_path)
    assert loaded.worklist == []
    assert loaded.cursor == 0
    assert st.progress(loaded)["overall"]["pending"] == 0


def test_load_no_done_prefix_is_unchanged(tmp_path: Path) -> None:
    state = SessionState(mode="eval", worklist=[_item("a", 1), _item("b", 2)], cursor=1)
    st.save(state, tmp_path)
    loaded = st.load("eval", tmp_path)
    assert [w.case for w in loaded.worklist] == ["a", "b"]
    assert loaded.cursor == 1


# --------------------------------------------------------------------------- #
# Atomicity
# --------------------------------------------------------------------------- #


def test_save_is_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    st.save(SessionState(mode="eval", worklist=[_item("a", 1)]), tmp_path)
    leftovers = list(st.state_dir(tmp_path).glob("*.tmp"))
    assert leftovers == []


def test_save_overwrites_existing_state(tmp_path: Path) -> None:
    st.save(SessionState(mode="eval", worklist=[_item("a", 1)]), tmp_path)
    st.save(SessionState(mode="eval", worklist=[_item("b", 2), _item("c", 3)]), tmp_path)
    loaded = st.load("eval", tmp_path)
    assert [w.case for w in loaded.worklist] == ["b", "c"]


def test_modes_are_isolated_files(tmp_path: Path) -> None:
    st.save(SessionState(mode="eval", worklist=[_item("a", 1)]), tmp_path)
    st.save(SessionState(mode="train", worklist=[_item("b", 2)], round_budget=20), tmp_path)
    assert [w.case for w in st.load("eval", tmp_path).worklist] == ["a"]
    assert [w.case for w in st.load("train", tmp_path).worklist] == ["b"]
    assert st.load("train", tmp_path).round_budget == 20


# --------------------------------------------------------------------------- #
# Malformed-file validation (fail loud, name the path)
# --------------------------------------------------------------------------- #


def _write_raw(tmp_path: Path, mode: str, text: str) -> Path:
    path = st.state_dir(tmp_path) / f"{mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_invalid_json_raises_with_path(tmp_path: Path) -> None:
    path = _write_raw(tmp_path, "eval", "{not json]")
    with pytest.raises(ValueError, match=str(path)):
        st.load("eval", tmp_path)


def test_load_non_object_top_level_raises(tmp_path: Path) -> None:
    _write_raw(tmp_path, "eval", "[1, 2, 3]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        st.load("eval", tmp_path)


def test_load_mode_mismatch_raises(tmp_path: Path) -> None:
    _write_raw(tmp_path, "eval", json.dumps({"mode": "train", "worklist": [], "cursor": 0}))
    with pytest.raises(ValueError, match="does not match requested mode"):
        st.load("eval", tmp_path)


def test_load_worklist_not_array_raises(tmp_path: Path) -> None:
    _write_raw(tmp_path, "eval", json.dumps({"mode": "eval", "worklist": {}, "cursor": 0}))
    with pytest.raises(ValueError, match="'worklist' must be a JSON array"):
        st.load("eval", tmp_path)


def test_load_worklist_entry_missing_key_raises(tmp_path: Path) -> None:
    _write_raw(
        tmp_path,
        "eval",
        json.dumps({"mode": "eval", "worklist": [{"case": "addition", "seed": 1}], "cursor": 0}),
    )
    with pytest.raises(ValueError, match="missing required key 'completion_stage'"):
        st.load("eval", tmp_path)


def test_load_worklist_entry_bad_status_raises(tmp_path: Path) -> None:
    _write_raw(
        tmp_path,
        "eval",
        json.dumps(
            {
                "mode": "eval",
                "worklist": [{"case": "addition", "seed": 1, "completion_stage": "full", "status": "wip"}],
                "cursor": 0,
            }
        ),
    )
    with pytest.raises(ValueError, match="status"):
        st.load("eval", tmp_path)


def test_load_worklist_entry_bad_seed_type_raises(tmp_path: Path) -> None:
    _write_raw(
        tmp_path,
        "eval",
        json.dumps(
            {
                "mode": "eval",
                "worklist": [{"case": "addition", "seed": "1", "completion_stage": "full"}],
                "cursor": 0,
            }
        ),
    )
    with pytest.raises(ValueError, match="'seed' must be an int"):
        st.load("eval", tmp_path)


def test_load_seed_bool_rejected(tmp_path: Path) -> None:
    # bool is a subclass of int; a stray ``true`` must not be silently read as 1.
    _write_raw(
        tmp_path,
        "eval",
        json.dumps(
            {
                "mode": "eval",
                "worklist": [{"case": "addition", "seed": True, "completion_stage": "full"}],
                "cursor": 0,
            }
        ),
    )
    with pytest.raises(ValueError, match="'seed' must be an int"):
        st.load("eval", tmp_path)


def test_load_negative_cursor_raises(tmp_path: Path) -> None:
    _write_raw(tmp_path, "eval", json.dumps({"mode": "eval", "worklist": [], "cursor": -1}))
    with pytest.raises(ValueError, match="non-negative"):
        st.load("eval", tmp_path)


def test_load_bad_round_budget_raises(tmp_path: Path) -> None:
    _write_raw(
        tmp_path,
        "train",
        json.dumps({"mode": "train", "worklist": [], "cursor": 0, "round_budget": "40"}),
    )
    with pytest.raises(ValueError, match="'round_budget' must be an int or null"):
        st.load("train", tmp_path)


# --------------------------------------------------------------------------- #
# FIX 2: served-seeds persistence helpers
# --------------------------------------------------------------------------- #


def test_served_seeds_path_per_mode(tmp_path: Path) -> None:
    assert st.served_seeds_path(tmp_path, "eval") == (
        tmp_path / "data" / "setmaker" / "state" / "eval_served_seeds.json"
    )
    assert st.served_seeds_path(tmp_path, "train") == (
        tmp_path / "data" / "setmaker" / "state" / "train_served_seeds.json"
    )


def test_served_seeds_path_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not valid"):
        st.served_seeds_path(tmp_path, "bogus")  # type: ignore[arg-type]


def test_load_served_seeds_missing_file_returns_empty_set(tmp_path: Path) -> None:
    result = st.load_served_seeds("eval", tmp_path)
    assert result == set()


def test_save_and_load_served_seeds_round_trip(tmp_path: Path) -> None:
    seeds = {1_000_000, 1_000_001, 1_000_010}
    st.save_served_seeds("eval", seeds, tmp_path)
    loaded = st.load_served_seeds("eval", tmp_path)
    assert loaded == seeds


def test_mark_seed_served_adds_to_existing(tmp_path: Path) -> None:
    st.save_served_seeds("eval", {1_000_000}, tmp_path)
    st.mark_seed_served("eval", 1_000_001, tmp_path)
    loaded = st.load_served_seeds("eval", tmp_path)
    assert loaded == {1_000_000, 1_000_001}


def test_mark_seed_served_is_idempotent(tmp_path: Path) -> None:
    st.mark_seed_served("eval", 1_000_000, tmp_path)
    st.mark_seed_served("eval", 1_000_000, tmp_path)
    loaded = st.load_served_seeds("eval", tmp_path)
    assert loaded == {1_000_000}


def test_served_seeds_modes_are_isolated(tmp_path: Path) -> None:
    st.mark_seed_served("eval", 1_000_000, tmp_path)
    st.mark_seed_served("train", 2_000_000, tmp_path)
    assert st.load_served_seeds("eval", tmp_path) == {1_000_000}
    assert st.load_served_seeds("train", tmp_path) == {2_000_000}


def test_save_served_seeds_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    st.save_served_seeds("eval", {1, 2, 3}, tmp_path)
    leftovers = list(st.state_dir(tmp_path).glob("*.tmp"))
    assert leftovers == []


def test_load_served_seeds_malformed_file_returns_empty(tmp_path: Path) -> None:
    path = st.served_seeds_path(tmp_path, "eval")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json]", encoding="utf-8")
    result = st.load_served_seeds("eval", tmp_path)
    assert result == set()  # corrupt file is silently ignored
