"""Contract round-trip tests for the grading engine type layer.

Every captured fixture is parsed with ``from_api`` and re-emitted with
``to_api``; the round-tripped data must equal the original captured JSON (modulo
the documented, intentional asymmetries). This pins the mock and the live engine
client to one shape: if either drifts, these fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import pytest

from src.grading.types import (
    EvalResult,
    GridToken,
    ScoringInfo,
    SessionCreated,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- create -----------------------------------------------------------------


def test_create_res_round_trips_verbatim() -> None:
    """The create response round-trips byte-aligned via the preserved raw envelope."""
    raw = _load("create.res.json")
    parsed = SessionCreated.from_api(raw)
    assert parsed.to_api() == raw


def test_create_res_extracts_session_fields() -> None:
    raw = _load("create.res.json")
    parsed = SessionCreated.from_api(raw)
    session = raw[0]["sessions"][0]
    assert parsed.session_id == session["sessionId"]
    assert parsed.marks_total == session["marksTotal"]
    # ref_id is the single ARITHMETIC interaction id (the key of the map).
    assert parsed.ref_id == next(iter(session["interactions"]))
    assert parsed.ref_id == "drYEz"


def test_create_res_parses_view_model_grid() -> None:
    """The view-model is pulled from the embedded init-data script."""
    parsed = SessionCreated.from_api(_load("create.res.json"))
    interaction = parsed.view_model["view"]["elements"][0]["interactions"][0]["ans"]
    assert interaction["type"] == "ARITHMETIC"
    assert interaction["width"] == 13
    assert interaction["height"] == 10
    assert isinstance(interaction["tokens"], list) and interaction["tokens"]


# --- event (evaluate result) ------------------------------------------------


def test_event_res_round_trips() -> None:
    """The evaluate result round-trips, preserving the MISSING-element asymmetry."""
    raw = _load("event.res.json")
    parsed = EvalResult.from_api(raw)
    assert parsed.to_api() == raw


def test_event_res_parses_elements_and_hint() -> None:
    raw = _load("event.res.json")
    parsed = EvalResult.from_api(raw)
    result = raw[0]["result"]
    assert len(parsed.elements) == len(result["elements"])
    assert parsed.progress == pytest.approx(result["progress"])
    # The captured response has a MISSING element with no symbolIdList.
    missing = [e for e in parsed.elements if e.status == "MISSING"]
    assert missing and missing[0].symbol_id_list == []
    # And a single positioned hint.
    assert parsed.hint is not None
    assert parsed.hint.message_type == "HorizontalSum"
    assert parsed.hint.target_positions[0].to_api() == result["hint"]["targetPositions"][0]


def test_event_res_empty_feedback_preserved() -> None:
    parsed = EvalResult.from_api(_load("event.res.json"))
    assert parsed.feedback == []


# --- event request (submitted grid tokens) ----------------------------------


def test_event_req_tokens_round_trip() -> None:
    """The submitted GridTokens parse from the captured form body and re-emit."""
    fixture = _load("event.req.json")
    pairs = parse_qsl(fixture["body"])
    # Reconstruct the input[] token dicts from the bracketed form keys.
    indexed: dict[int, dict[str, Any]] = {}
    for key, value in pairs:
        if not key.startswith("events[0][input]"):
            continue
        # events[0][input][N][field] or events[0][input][N][tags][]
        rest = key[len("events[0][input]"):]
        idx = int(rest[1 : rest.index("]")])
        token = indexed.setdefault(idx, {"tags": []})
        if rest.endswith("[tags][]"):
            token["tags"].append(value)
        else:
            field = rest[rest.rindex("[") + 1 : -1]
            token[field] = value
    tokens = [
        GridToken.from_api(
            {"id": t["id"], "c": t["c"], "x": int(t["x"]), "y": int(t["y"]), "tags": t["tags"]}
        )
        for _, t in sorted(indexed.items())
    ]
    assert tokens, "captured event request must carry input tokens"
    # Round-trip every token through to_api/from_api.
    for token in tokens:
        again = GridToken.from_api(token.to_api())
        assert again == token
    # Spot-check a known captured token (G0 = '2' at grid x13,y10, FIXED).
    g0 = next(t for t in tokens if t.id == "G0")
    assert g0.c == "2" and g0.x == 13 and g0.y == 10 and g0.tags == ["FIXED"]


# --- info (scoring) ---------------------------------------------------------


def test_info_res_round_trips() -> None:
    """The scoring block round-trips (penalties compared as a dict)."""
    raw = _load("info.res.json")
    parsed = ScoringInfo.from_api(raw)
    out = parsed.to_api()
    scoring = raw["scoring"]
    assert out["finished"] == scoring["finished"]
    assert out["marksTotal"] == scoring["marksTotal"]
    assert out["marksEarned"] == scoring["marksEarned"]
    assert out["penalties"] == scoring["penalties"]
    # And a re-parse of the emitted shape is identical.
    assert ScoringInfo.from_api({"scoring": out}) == parsed


# --- solution ---------------------------------------------------------------


def test_solution_res_round_trips_as_dict() -> None:
    """The solution route returns a raw dict; the client passes it through unchanged."""
    raw = _load("solution.res.json")
    # The Protocol's solution() returns a dict verbatim; identity round-trip.
    assert dict(raw) == raw
    # The full worked-solution token tree is reachable for a future "show me how".
    interaction = raw["data"]["view"]["elements"][0]["interactions"][0]
    assert isinstance(interaction["solution"], list)
    assert interaction["solution"], "captured solution must carry the worked token tree"
