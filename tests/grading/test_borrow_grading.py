"""Borrow / replacement cell support for short subtraction.

Verified against the live engine: grading engine grades
short-subtraction borrows when each is sent as a ``type=REPLACE`` token at the
borrow cell. The four cells for 256-89 (evaluate frame) are (13,11)=1,
(14,11)=4, (15,11)=16, (14,12)=14. Borrows are optional (answer-only scores
1.0); a wrong borrow drops progress and fires ``BorrowIncorrectFeedback``.

These tests exercise the contract through the mock client + grid mapper so they
run offline; the live behaviour they mirror is captured in the module docstring
above and the bank comment in ``exercises.py``.
"""

from pathlib import Path

from src.grading import exercises as ex
from src.grading.exercises import BankExercise
from src.grading.grid_mapping import GridCalibration, tokens_to_grid
from src.grading.mock_client import MockGradingClient
from src.grading.types import GridToken
from src.core.run_config import GradingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUBTRACT_ID = "subtract-256-89"


def _client() -> MockGradingClient:
    return MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)


def _derive_subtract_bank() -> tuple[MockGradingClient, BankExercise, str, str]:
    """Return (client, bank, session_id, ref_id) for subtract-256-89."""
    client = _client()
    spec = ex.get_exercise(_SUBTRACT_ID)
    created = client.create_session(spec)
    bank = client._sessions[created.session_id].exercise
    return client, bank, created.session_id, created.ref_id


def _borrows_from_bank() -> dict[tuple[int, int], str]:
    """Derive the native-frame borrow cells from the captured fixture.

    Filters ``bank.expected`` to ``role == "borrow"`` and returns a
    ``{(x, y): c}`` dict. This keeps ``_BORROWS`` in sync with the fixture
    automatically so a fixture change is caught by the tests rather than
    silently diverging.
    """
    _, bank, _, _ = _derive_subtract_bank()
    return {(t.x, t.y): t.c for t in bank.expected if t.role == "borrow"}


def _answer_tokens(bank: BankExercise) -> list[GridToken]:
    """The given scaffold (FIXED) plus the correct answer 167 (no borrows)."""
    tokens: list[GridToken] = []
    for i, t in enumerate(bank.expected):
        if t.role == "borrow":
            continue
        tags = ["FIXED"] if t.given else []
        tokens.append(GridToken(id=f"E{i}", c=t.c, x=t.x, y=t.y, tags=tags))
    return tokens


# --- GridToken.type round-trip + wire encoding ------------------------------


def test_gridtoken_type_defaults_to_symbol_and_is_omitted_on_wire() -> None:
    plain = GridToken(id="A", c="7", x=15, y=7)
    assert plain.type == "SYMBOL"
    assert "type" not in plain.to_api()  # SYMBOL never serialised


def test_gridtoken_replace_round_trips() -> None:
    borrow = GridToken(id="B", c="16", x=15, y=11, type="REPLACE")
    again = GridToken.from_api(borrow.to_api())
    assert again == borrow
    assert borrow.to_api()["type"] == "REPLACE"


# --- grid mapping: REPLACE tagging + two-digit grouping ---------------------


def _calibration() -> GridCalibration:
    # 13x10 native frame (subtract-256-89), 1:1 on a virtual canvas (40 px cells).
    # engine_x_left=0, engine_y_top=9 so row = engine_y_top - y = 9 - y, col = x - engine_x_left = x.
    return GridCalibration(
        grid_width=13,
        grid_height=10,
        cell_w_px=40.0,
        cell_h_px=40.0,
        origin_x_px=0.0,
        origin_y_px=0.0,
        engine_x_left=0,
        engine_y_top=9,
        invert_y=True,
    )


def _ink(label: str, cx: float, cy: float) -> dict:
    return {"label": label, "bbox": [cx - 8, cy - 8, cx + 8, cy + 8]}


def test_tokens_to_grid_tags_borrow_cell_as_replace() -> None:
    cal = _calibration()
    # Native frame cell (11,7): col=11, row = 9-7 = 2 -> center px (11*40+20, 2*40+20).
    cx, cy = 11 * 40 + 20, 2 * 40 + 20
    tokens = tokens_to_grid([_ink("main_4", cx, cy)], cal, replace_cells={(11, 7)})
    assert len(tokens) == 1
    assert tokens[0].type == "REPLACE"
    assert (tokens[0].x, tokens[0].y, tokens[0].c) == (11, 7, "4")


def test_tokens_to_grid_concatenates_two_digits_in_one_replace_cell() -> None:
    cal = _calibration()
    # Two glyphs '1' then '6' inside the units borrow cell (12,7) in native frame.
    # col=12, row = 9-7 = 2, base pixel x = 12*40 = 480, base y = 2*40+20 = 100.
    base_x, base_y = 12 * 40, 2 * 40 + 20
    left = _ink("main_1", base_x + 10, base_y)   # left half of the cell
    right = _ink("main_6", base_x + 32, base_y)  # right half of the cell
    tokens = tokens_to_grid([right, left], cal, replace_cells={(12, 7)})
    replace = [t for t in tokens if t.type == "REPLACE"]
    assert len(replace) == 1
    # Concatenated left-to-right regardless of input order.
    assert replace[0].c == "16"
    assert (replace[0].x, replace[0].y) == (12, 7)


def test_tokens_to_grid_leaves_non_replace_cells_untyped() -> None:
    cal = _calibration()
    # answer cell (12,3) in native frame: col=12, row = 9-3 = 6, center px (12*40+20, 6*40+20).
    cx, cy = 12 * 40 + 20, 6 * 40 + 20
    tokens = tokens_to_grid([_ink("main_7", cx, cy)], cal, replace_cells={(10, 7)})
    assert tokens[0].type == "SYMBOL"
    assert (tokens[0].x, tokens[0].y, tokens[0].c) == (12, 3, "7")


# --- mock grading: optional, value-checked borrows --------------------------


def _borrow_tokens(values: dict) -> list[GridToken]:
    return [
        GridToken(id=f"B{i}", c=c, x=x, y=y, type="REPLACE")
        for i, ((x, y), c) in enumerate(values.items())
    ]


def test_answer_only_scores_full_without_borrows() -> None:
    client, bank, session_id, ref_id = _derive_subtract_bank()
    tokens = _answer_tokens(bank)
    result = client.evaluate(session_id, ref_id, tokens)
    assert result.progress == 1.0
    assert not any(f.message_type == "BorrowIncorrectFeedback" for f in result.feedback)


def test_correct_borrows_keep_full_score() -> None:
    client, bank, session_id, ref_id = _derive_subtract_bank()
    tokens = _answer_tokens(bank) + _borrow_tokens(_borrows_from_bank())
    result = client.evaluate(session_id, ref_id, tokens)
    assert result.progress == 1.0
    assert all(e.status == "OK" for e in result.elements)


def test_wrong_borrow_drops_score_and_flags_feedback() -> None:
    client, bank, session_id, ref_id = _derive_subtract_bank()
    wrong = _borrows_from_bank()
    wrong[(12, 7)] = "15"  # should be "16" (units column borrow in native frame)
    tokens = _answer_tokens(bank) + _borrow_tokens(wrong)
    result = client.evaluate(session_id, ref_id, tokens)
    assert result.progress < 1.0
    fb_types = {f.message_type for f in result.feedback}
    assert "BorrowIncorrectFeedback" in fb_types
    bad = [e for e in result.elements if e.status == "ERROR"]
    assert any(e.position == (12, 7) for e in bad)


def test_partial_borrows_are_not_penalised() -> None:
    client, bank, session_id, ref_id = _derive_subtract_bank()
    # Only the single-digit reduced borrows drawn; the rest omitted -> still 1.0.
    tokens = _answer_tokens(bank) + _borrow_tokens({(10, 7): "1", (11, 7): "4"})
    result = client.evaluate(session_id, ref_id, tokens)
    assert result.progress == 1.0


def test_drifted_borrow_y_off_by_one_is_silently_dropped_mock_divergence() -> None:
    """A correct borrow value submitted at native_y+1 instead of native_y is
    silently dropped by the mock (treated as if the borrow was never drawn).

    Mock-only intentional divergence from live engine: live grading engine is
    translation-invariant and will accept a uniformly y-drifted borrow at the
    correct relative position. The mock looks up borrows by EXACT cell via
    ``borrow_subs.get(expected_cell)``; a token at y+1 misses this lookup and
    is discarded as an unmatched REPLACE token rather than accepted or flagged.

    This test makes the divergence visible in CI instead of only in the comment
    at mock_client.py ~line 614-629. When the TODO in that comment is resolved
    (borrow-specific _best_global_dy), update this test to assert OK instead.
    """
    client, bank, session_id, ref_id = _derive_subtract_bank()

    # Pick the first borrow cell from the fixture (deterministic: smallest x).
    borrows = _borrows_from_bank()
    first_cell = min(borrows.keys())  # e.g. (10, 7) in native frame
    first_value = borrows[first_cell]

    # Submit the correct borrow value but one row off (y+1 instead of y).
    drifted_x, drifted_y = first_cell[0], first_cell[1] + 1
    drifted_token = GridToken(
        id="B_drifted", c=first_value, x=drifted_x, y=drifted_y, type="REPLACE"
    )
    tokens = _answer_tokens(bank) + [drifted_token]
    result = client.evaluate(session_id, ref_id, tokens)

    # Mock divergence: the misplaced borrow is silently discarded.
    # No element is emitted at the expected borrow cell (neither OK nor ERROR).
    # Live engine would accept it (translation-invariant); mock does not.
    borrow_elements = [
        e for e in result.elements
        if e.position == first_cell and e.status in ("OK", "ERROR", "MISSING")
    ]
    assert not any(e.status == "OK" for e in borrow_elements), (
        "Mock must NOT accept a borrow at wrong y; live engine divergence guard"
    )
    # The drifted REPLACE token must not produce an UnexpectedSymbol either
    # (absorbed as a step-row token or ignored -- exact behaviour may vary).
    # The key assertion: progress == 1.0 because borrows are optional and the
    # drifted token is silently absorbed, not penalised.
    assert result.progress == 1.0, (
        "Drifted borrow must not penalise score (borrows optional in mock)"
    )
