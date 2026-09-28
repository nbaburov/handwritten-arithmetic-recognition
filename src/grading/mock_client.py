"""In-process grading engine mock (``mode = "mock"``).

A high-fidelity fake that needs no API key, so the demo runs end-to-end before a
real key arrives. It satisfies the :class:`~src.grading.client.GradingClient`
Protocol structurally and emits objects in the exact captured contract shape, so
swapping in a live engine adapter later changes nothing downstream.

The grader is a real grader, not a stub: each bank exercise
(:mod:`src.grading.exercises`) carries its complete expected worked solution
as grid tokens ``{x, y, c}`` (operands, every carry/borrow, partial rows, the
answer). On ``evaluate`` the mock diffs the submitted ``GridToken[]`` against
that expected set using two grading tiers:

* **Answer / operand cells** are graded strictly: the submitted character must
  match the expected character at the same right-aligned place-value slot.
  Wrong chars produce ``ERROR`` elements and ``IncorrectValue`` feedback.
* **Step cells** (carry, borrow, partial-product) are graded on *presence*
  only: any submitted token at the slot is accepted regardless of its value;
  an absent slot is ``MISSING`` (gentle nudge); ``ERROR`` is never emitted for
  a step cell.  Extra step-row tokens that land outside the expected column are
  silently absorbed (no ``UnexpectedSymbol`` for step rows).

The verdict in :func:`compute_feedback_summary` is answer-first: a correct
answer with shown working is ``"correct"``; a correct answer with a missing
step mark is ``"missing_step"`` (gentle); any wrong answer digit is
``"mistake"``; no answer yet is ``"incomplete"``.

Emits one positioned ``hint`` for the next missing cell nearest the answer
line, computes monotone ``progress``, and on ``info`` reports ``ScoringInfo``
with marks earned only once progress reaches 1.0. One generic rule covers
every template.

Grading is **structure-relative**: submitted tokens are matched to expected
slots by right-aligned place-value position within each structural row (carry
row, operand rows, answer row) rather than by absolute grading engine cell
coordinates. This makes grading robust to per-row horizontal drift: a correct
answer written one cell to the right of the expected column still grades OK
because both the submitted and expected tokens share the same right-aligned
slot within the answer band. See :func:`_structure_relative_match` for the
full algorithm.

``create_session`` replays the captured create view-model verbatim for the
256-89 fixture exercise (the golden capture under
``docs/research/engine-api-capture/``) and fabricates a view-model in the same shape
with scaffold tokens for the other three.

This module depends on :mod:`.types` and :mod:`.exercises`; it never touches
HTTP or the recognizer.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .client import GradingError
from .derive import derive_exercise
from .exercises import BankExercise, ExpectedToken
from .types import (
    EvalElement,
    EvalFeedback,
    EvalHint,
    EvalResult,
    ExerciseSpec,
    GridRect,
    GridToken,
    ScoringInfo,
    SessionCreated,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.run_config import GradingConfig

# Derive fixtures: captured create + solution responses per exercise, used to
# derive the BankExercise layout offline (no API key needed in mock mode).
# Each entry maps exercise_id -> (create fixture path, solution fixture path).
# These live under tests/ so they are always available in the dev tree;
# the mock never writes to them.
_DERIVE_FIXTURES: dict[str, tuple[Path, Path]] = {
    "subtract-256-89": (
        Path("tests") / "grading" / "fixtures" / "derive" / "subtract-256-89.create.json",
        Path("tests") / "grading" / "fixtures" / "derive" / "subtract-256-89.solution.json",
    ),
    "product-38-29": (
        Path("tests") / "grading" / "fixtures" / "derive" / "product-38-29.create.json",
        Path("tests") / "grading" / "fixtures" / "derive" / "product-38-29.solution.json",
    ),
}

# Legacy alias used by tests that pin the captured session_id / ref_id for
# the subtraction exercise.
_FIXTURE_EXERCISE_ID = "subtract-256-89"

# Element template types per expected-cell role, mirroring the captured event
# response (multi-digit numbers -> NumberElement, single operator ->
# SymbolElement, the result bar -> HorizontalLineElement).
_OPERATOR_ROLES = frozenset({"operator"})
# Structural roles whose chars are never digits or submitted by the child:
# result_bar (horizontal line) and div_bracket (L/bus-stop bracket).  Both are
# always given/pre-printed; neither is ever submitted in a GridToken submission,
# and neither should influence the global y-drift vote or count as a content row.
_BAR_ROLES = frozenset({"result_bar", "div_bracket"})


def _template_type(role: str) -> str:
    if role in _OPERATOR_ROLES:
        return "SymbolElement"
    if role in _BAR_ROLES:
        return "HorizontalLineElement"
    return "NumberElement"


def _is_bar_char(c: str) -> bool:
    """True when the char represents a structural bar/bracket (not a digit or operator).

    Covers ``"_"`` (result_bar) and ``""`` (div_bracket); both are pre-printed
    structural tokens that are never submitted by the child.
    """
    return c in ("", "_")


def _best_global_dy(
    submitted_tokens: list["GridToken"],
    expected_y_levels: list[int],
    max_dy: int = 6,
) -> int:
    """Find the integer dy in [-max_dy..max_dy] that maximises row-level alignment.

    Counts how many distinct submitted content-token y-values shift onto an
    expected content-row y-level after adding ``dy``.  A "content" token is
    any non-bar submitted token (``c not in ("", "_")``); bar rows are excluded
    so a short or absent bar does not bias the vote.

    Ties go to ``dy = 0`` (no shift preferred).  This corrects uniform y-drift
    (all rows shifted by the same amount) before the per-row x-alignment step.
    """
    content_sub_ys = {
        tok.y for tok in submitted_tokens if not _is_bar_char(tok.c)
    }
    if not content_sub_ys:
        return 0
    expected_y_set = set(expected_y_levels)
    best_score = -1
    best_dy = 0
    for dy in range(-max_dy, max_dy + 1):
        score = sum(1 for sy in content_sub_ys if (sy + dy) in expected_y_set)
        if score > best_score or (score == best_score and abs(dy) < abs(best_dy)):
            best_score = score
            best_dy = dy
    return best_dy


def _structure_relative_match(
    submitted_tokens: list["GridToken"],
    expected: tuple["ExpectedToken", ...],
) -> tuple[
    dict[tuple[int, int], "GridToken"],  # expected_cell -> matched submitted token
    list["GridToken"],                    # submitted tokens not matched to any expected cell
    int,                                  # global_dy applied to content tokens
]:
    """Match submitted tokens to expected cells by structure-relative position.

    This is the core grading primitive. It is tolerant to both:

    * **Uniform y-drift**: all rows written one or more cells above/below the
      expected y-levels (handled by :func:`_best_global_dy` which finds the
      single integer dy that best aligns the submission's row y-values to the
      expected y-levels, then applies that shift before row assignment).
    * **Per-row x-drift**: each row written at a different x-offset than the
      expected column (handled by right-aligned slot matching within each row,
      independent per row).

    Algorithm
    ---------
    1. Detect the global y-shift ``dy`` via :func:`_best_global_dy` and apply
       it to submitted y-coordinates (content tokens only; bar tokens are
       assigned separately to their nearest bar-compatible row).
    2. Group expected tokens by ``y``-level.
    3. Assign each submitted token (after y-shift) to the nearest expected
       y-row. Bar submitted tokens are only compatible with expected bar rows;
       content submitted tokens are only compatible with expected content rows.
       This prevents a digit from being eaten by the result-bar row even when
       they share the same y after shifting.
    4. Within each expected y-row, right-align expected tokens by x:
       rightmost = place-value slot 0 (units), next = slot 1 (tens), etc.
       Right-align the submitted tokens assigned to that row the same way.
       Match submitted slot k to expected slot k.
    5. Return:
       - ``matched``: ``expected_cell -> submitted_token`` for every expected
         cell that has a submitted token at the same right-aligned slot.
       - ``extras``: submitted tokens beyond the expected row width, or at
         a y-level too far from any expected row.

    The ``symbol_id_list`` in every ``EvalElement`` references the REAL
    submitted token id so popup anchoring always targets the actual glyph.
    """
    if not submitted_tokens:
        return {}, [], 0

    from collections import defaultdict

    # --- Step 1: group expected by y and detect global y-shift. -------------
    expected_by_y: dict[int, list["ExpectedToken"]] = defaultdict(list)
    for tok in expected:
        expected_by_y[tok.y].append(tok)

    expected_y_levels = sorted(expected_by_y.keys())
    if not expected_y_levels:
        # No expected y-levels (e.g. an all-borrow expected set): no content to
        # shift, so global_dy is 0. Must return the 3-tuple the callers unpack.
        return {}, list(submitted_tokens), 0

    # Classify expected y-levels as bar or content.
    _bar_y_levels: frozenset[int] = frozenset(
        ey
        for ey, toks in expected_by_y.items()
        if all(_is_bar_char(t.c) for t in toks)
    )
    _content_y_levels = [ey for ey in expected_y_levels if ey not in _bar_y_levels]

    # Find the best global dy using content tokens vs content y-levels only.
    global_dy = _best_global_dy(submitted_tokens, _content_y_levels)

    # --- Step 2: assign each submitted token to the nearest expected y-level,
    # after applying global_dy to content tokens. Bar tokens are not shifted
    # (their position relative to the bar row is usually exact or close). ----
    def _assign_cost(sub_c: str, sy_shifted: int, ey: int) -> tuple[int, int, int]:
        """Sort key (compat_penalty, distance, prefer_smaller_y)."""
        sub_is_bar = _is_bar_char(sub_c)
        row_is_bar = ey in _bar_y_levels
        compat_penalty = 0 if (sub_is_bar == row_is_bar) else 1
        dist = abs(sy_shifted - ey)
        return (compat_penalty, dist, ey)

    submitted_by_assigned_y: dict[int, list["GridToken"]] = defaultdict(list)
    for sub in submitted_tokens:
        # Apply global dy only to content tokens; bar tokens use their raw y.
        sy_shifted = sub.y + global_dy if not _is_bar_char(sub.c) else sub.y
        best_y = min(
            expected_y_levels,
            key=lambda ey: _assign_cost(sub.c, sy_shifted, ey),
        )
        submitted_by_assigned_y[best_y].append(sub)

    # --- Step 3: right-align within each row and match by slot. -------------
    matched: dict[tuple[int, int], "GridToken"] = {}
    extras: list["GridToken"] = []

    for ey, exp_row_tokens in expected_by_y.items():
        # Right-align expected: sort by x ascending; slot = (max_x - x).
        exp_sorted = sorted(exp_row_tokens, key=lambda t: t.x)
        max_exp_x = exp_sorted[-1].x
        exp_slot_to_token: dict[int, "ExpectedToken"] = {
            max_exp_x - t.x: t for t in exp_sorted
        }

        sub_row = submitted_by_assigned_y.get(ey, [])
        if not sub_row:
            continue  # all cells in this row will be MISSING

        # Right-align submitted tokens in this row.
        sub_sorted = sorted(sub_row, key=lambda t: t.x)
        max_sub_x = sub_sorted[-1].x
        sub_slot_to_token: dict[int, "GridToken"] = {
            max_sub_x - t.x: t for t in sub_sorted
        }

        for slot, sub_tok in sub_slot_to_token.items():
            exp_tok = exp_slot_to_token.get(slot)
            if exp_tok is not None:
                matched[exp_tok.cell()] = sub_tok
            else:
                extras.append(sub_tok)

    # Collect any submitted tokens not matched (defensive: catch anything missed).
    matched_sub_ids = {v.id for v in matched.values()}
    for sub in submitted_tokens:
        if sub.id not in matched_sub_ids and sub not in extras:
            extras.append(sub)

    return matched, extras, global_dy


def _group_expected_tokens(
    tokens: tuple[ExpectedToken, ...],
) -> list[list[ExpectedToken]]:
    """Group horizontally-adjacent, same-row, same-role expected cells.

    Cells that share the same ``y`` and ``role`` and whose ``x`` values are
    strictly consecutive (differ by 1 per step, sorted ascending) are merged
    into one group so the mock emits one multi-digit :class:`EvalElement` per
    operand/answer/carry row, matching the real grading engine ``NumberElement``
    grouping observed in the captured fixture.

    Each returned sub-list is sorted ascending by ``x``; its concatenated
    ``c`` values form the multi-character ``num``. Single-cell groups
    (operators, individual bar cells) also become a one-element list so the
    caller handles every case uniformly.
    """
    # Sort by (y desc, role, x asc) to keep natural left-to-right order within
    # each row; we group adjacently regardless of y ordering.
    sorted_tokens = sorted(tokens, key=lambda t: (t.y, t.role, t.x))
    groups: list[list[ExpectedToken]] = []
    current: list[ExpectedToken] = []
    for token in sorted_tokens:
        if (
            current
            and token.y == current[-1].y
            and token.role == current[-1].role
            and token.x == current[-1].x + 1
        ):
            current.append(token)
        else:
            if current:
                groups.append(current)
            current = [token]
    if current:
        groups.append(current)
    return groups


def _group_rect(group: list[ExpectedToken]) -> GridRect:
    """The :class:`GridRect` spanning the full cell group (engine exclusive bounds)."""
    left_x = group[0].x
    right_x = group[-1].x + 1  # exclusive
    y = group[0].y
    return GridRect(left=left_x, top=y + 1, bottom=y, right=right_x)


@dataclass
class _MockSession:
    """Mutable per-session state the mock keeps between create/evaluate/info.

    ``exercise`` is the graded bank entry; ``ref_id`` is the interaction id from
    the create view-model; ``marks_total`` is taken from the created session so
    ``info`` reports the same total the real engine would; ``last_result`` is the
    most recent ``evaluate`` outcome, used by ``info`` to decide marks earned.
    """

    exercise: BankExercise
    ref_id: str
    marks_total: int
    last_result: EvalResult | None


def _hint_message_for(
    exercise: BankExercise, missing_group: list[ExpectedToken]
) -> tuple[str, dict[str, Any]]:
    """Pick the captured-vocabulary ``messageType`` + ``messageArgs`` for a hint.

    Uses the same vocabulary observed in the captured real responses so the
    frontend catalog renders mock and real hints identically. The argument set
    names the operands and the missing piece. ``missing_group`` is a sorted
    list of horizontally-adjacent cells forming one logical token; ``missing``
    in the args carries the full concatenated string (e.g. ``["89"]``), matching
    the captured ``event.res`` fixture where the whole missing operand is named.
    """
    template = exercise.spec.template_name
    numbers = list(exercise.spec.template_args)
    missing_str = "".join(t.c for t in missing_group)
    role = missing_group[0].role
    args: dict[str, Any] = {"numbers": numbers, "missing": [missing_str], "role": role}
    if template.startswith("SUM"):
        return "HorizontalSum", args
    if template.startswith("SUBTRACT"):
        # The captured fixture used HorizontalSum for the subtraction column
        # interaction; keep that observed messageType so the catalog matches.
        return "HorizontalSum", args
    if template.startswith("PRODUCT"):
        return "HorizontalProduct", args
    if template.startswith("DIVISION"):
        return "HorizontalDivision", args
    return "HorizontalSum", args


class MockGradingClient:
    """In-process grader; satisfies :class:`GradingClient` structurally.

    Constructed by :func:`src.grading.client.make_client` with keyword args
    ``config`` and ``project_root``; both are accepted so the mock and the HTTP
    client share one construction signature. The mock keeps one in-memory map
    of ``session_id -> (BankExercise, last EvalResult)`` so ``info`` can report
    the score for the most recent evaluation without re-grading.
    """

    def __init__(self, *, config: "GradingConfig", project_root: Path) -> None:
        self._config = config
        self._project_root = Path(project_root)
        self._sessions: dict[str, _MockSession] = {}
        # Stash solution fixture data per session_id so solution() can return it.
        self._solution_data: dict[str, dict[str, Any]] = {}

    # --- create -------------------------------------------------------------

    def create_session(self, spec: ExerciseSpec) -> SessionCreated:
        """Start a session for one bank exercise; return the parsed create result.

        For every exercise that has derive fixtures, replays the captured create
        response verbatim (byte-aligned view-model) and derives the BankExercise
        from the captured create + solution fixtures via :func:`derive_exercise`.
        For exercises without derive fixtures, raises :class:`GradingError`
        (fail-fast: mock mode requires fixtures for every configured exercise).
        """
        exercise_id = spec.exercise_id
        if exercise_id not in _DERIVE_FIXTURES:
            raise GradingError(
                f"mock mode: no derive fixture for exercise {exercise_id!r}. "
                f"Known exercises: {list(_DERIVE_FIXTURES)}. "
                "Capture create+solution fixtures under "
                "tests/grading/fixtures/derive/ to use this exercise offline."
            )
        create_envelope, solution_data = self._load_derive_fixtures(exercise_id)
        created = SessionCreated.from_api(create_envelope)

        # Derive the BankExercise from the captured fixtures.
        exercise = derive_exercise(spec, created.view_model, solution_data)

        self._sessions[created.session_id] = _MockSession(
            exercise=exercise,
            ref_id=created.ref_id,
            marks_total=created.marks_total,
            last_result=None,
        )
        # Stash the solution data for the solution() method (keyed by session_id).
        self._solution_data[created.session_id] = solution_data
        return created

    def _load_derive_fixtures(
        self, exercise_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Load the derive create + solution fixtures for an exercise.

        Both fixtures are loaded relative to ``_project_root`` so tests and the
        running server both resolve to the same files regardless of cwd.
        """
        create_path_rel, solution_path_rel = _DERIVE_FIXTURES[exercise_id]
        create_path = self._project_root / create_path_rel
        solution_path = self._project_root / solution_path_rel
        if not create_path.exists():
            raise GradingError(
                f"mock fixture missing: {create_path}. "
                "Run the live capture script to regenerate."
            )
        if not solution_path.exists():
            raise GradingError(
                f"mock fixture missing: {solution_path}. "
                "Run the live capture script to regenerate."
            )
        create_envelope = json.loads(create_path.read_text(encoding="utf-8"))
        solution_data = json.loads(solution_path.read_text(encoding="utf-8"))
        return create_envelope, solution_data

    def _fabricate_create(self, exercise: BankExercise) -> list[dict[str, Any]]:
        """Build a create envelope in the captured shape for a non-fixture exercise.

        Mirrors the captured envelope: one ``SINGLE`` session with one
        ``ARITHMETIC`` interaction whose ``html`` carries an
        ``grading/init-data`` script with the grid ``width``/``height`` and
        the scaffold ``tokens[]`` (the given problem digits, operator, and result
        bar marked FIXED). The non-given expected cells are omitted from the
        scaffold (the child supplies them), exactly as the real scaffold does.
        """
        session_id = str(uuid.uuid4())
        ref_id = uuid.uuid4().hex[:5]
        scaffold_tokens = [
            {
                "id": f"G{i}",
                "type": "SYMBOL",
                "c": token.c,
                "x": token.x,
                "y": token.y,
                "clist": None,
                "tags": ["GENERATED", "FIXED"],
            }
            for i, token in enumerate(exercise.given_tokens())
        ]
        view_model = {
            "events": {"_main": [], ref_id: []},
            "view": {
                "i18n": ["en"],
                "showHints": True,
                "numeralSystem": "ARABIC_DECIMALPOINT",
                "elements": [
                    {
                        "id": "Q1",
                        "type": "QUESTION",
                        "interactions": [
                            {
                                "id": ref_id,
                                "ans": {
                                    "id": "ANS",
                                    "type": "ARITHMETIC",
                                    "i18n": ["en"],
                                    "width": exercise.grid_width,
                                    "height": exercise.grid_height,
                                    "attributeMap": {"areaMap": {}},
                                    "tokens": scaffold_tokens,
                                    "input": None,
                                    "thousandsSeparator": "NONE",
                                },
                                "marksTotal": 1,
                            }
                        ],
                    }
                ],
            },
            "attributes": {},
        }
        html = (
            f'<engine-exercise session-id="{session_id}" engine-init="inline">'
            f'<script type="grading/init-data">{json.dumps(view_model)}</script>'
            f"</engine-exercise>"
        )
        session = {
            "success": True,
            "sessionId": session_id,
            "type": "SINGLE",
            "solution": False,
            "marksTotal": 1,
            "interactions": {ref_id: {"type": "ARITHMETIC", "marks": 1, "scorable": True}},
            "html": html,
        }
        return [{"success": True, "sessions": [session]}]

    # --- evaluate -----------------------------------------------------------

    def evaluate(
        self, session_id: str, ref_id: str, tokens: list[GridToken]
    ) -> EvalResult:
        """Grade the submitted grid tokens against the full expected solution.

        Uses structure-relative grading (see :func:`_structure_relative_match`)
        so that per-row horizontal drift in handwriting does not cause correct
        digits to be scored MISSING. Each submitted token is assigned to the
        nearest expected y-level, then matched by right-aligned place-value
        slot within that level: rightmost token = units slot 0, next = tens
        slot 1, etc. A correct digit in the correct relative slot grades OK
        regardless of absolute column drift.

        Groups horizontally-adjacent, same-row, same-role expected cells into
        one :class:`EvalElement` each (matching the grading engine's ``NumberElement``
        multi-digit grouping from the captured fixture).

        **Answer / operand cells** use strict value grading. Within each group:

        * All cells submitted correctly (char matches at the right-aligned slot)
          -> one ``OK`` element, ``width`` = group size, ``num`` = concatenated
          chars, ``symbolIdList`` = real submitted token ids.
        * Any cell in the group is missing (no submitted token at that slot)
          -> one ``MISSING`` element; missing-to-fill count incremented per cell.
        * Any cell in the group has the wrong char at the matching slot
          -> one ``ERROR`` element + one ``EvalFeedback`` entry per wrong cell.

        **Step cells** (carry, borrow, partial-product) use presence-only
        grading.  Notation varies and small marks are easily misread by the
        recognizer, so the exact value is never checked:

        * Any token present at the slot -> one ``OK`` element (presence = OK).
        * No token present -> one ``MISSING`` element (gentle nudge to show
          the working).
        * ERROR and IncorrectValue feedback are never emitted for step cells.

        Extra submitted tokens that land on a step-row y-level are silently
        absorbed rather than surfaced as ``UnexpectedSymbol``, because a step
        mark placed slightly outside the expected column must never penalise
        otherwise correct work.  Extra tokens on non-step rows still produce
        ``UnexpectedSymbol`` feedback.

        Single-cell groups (operators, isolated carries) follow the same path
        with ``width=1``.

        ``symbol_id_list`` always references the REAL submitted token ids so
        popup anchoring targets the actual glyph on the canvas.

        Emits one positioned ``hint`` for the next missing group (preferring
        GIVEN scaffold cells) nearest the answer line, ``progress = matched /
        total`` over the child-must-supply cells, and the grading engine's likelihood.
        """
        exercise = self._require_session(session_id)

        # Borrow / replacement cells are OPTIONAL and graded separately (mirroring
        # Grading engine: answer-only scores 1.0; a drawn borrow is graded on value; an
        # absent borrow is never penalised). They are held out of the core
        # structure-relative match so adding them never disturbs the grading of
        # the answer / operand / bar cells.
        core_expected = [
            t for t in exercise.expected if not (t.role == "borrow" and not t.given)
        ]
        borrow_expected = [
            t for t in exercise.expected if t.role == "borrow" and not t.given
        ]

        # Borrow submissions (type=REPLACE, or any token landing on a borrow cell)
        # are pulled out BEFORE the core match so the structure-relative matcher
        # never mistakes a borrow row token for an operand/answer slot.
        borrow_cells = {t.cell(): t for t in borrow_expected}

        # Identify borrows by TYPE only: a global handwriting drift can push an
        # operand onto a borrow cell, so classifying by cell would misread it as a
        # borrow. type=REPLACE is how tokens_to_grid tags real borrow ink.
        def _is_borrow_sub(sub: GridToken) -> bool:
            return sub.type == "REPLACE"

        borrow_subs = {(s.x, s.y): s for s in tokens if _is_borrow_sub(s)}
        core_tokens = [s for s in tokens if not _is_borrow_sub(s)]

        # Returns expected_cell -> submitted_token plus extra tokens not mapped to any slot.
        matched_sub, extra_tokens, _core_global_dy = _structure_relative_match(
            core_tokens, core_expected
        )
        # NOTE (mock/live engine divergence): live engine is translation-invariant and will
        # accept a uniformly y-drifted borrow at the correct relative position. The mock
        # looks up borrows by EXACT cell: `borrow_subs.get(expected_cell)`. When the
        # child writes borrow ink at a different y from the expected native cell (but in
        # the right relative row), the mock marks it MISSING while live engine accepts it.
        #
        # Applying _core_global_dy (the content-row shift) to the borrow lookup is NOT
        # safe in general: tests show that children may drift content rows while keeping
        # borrow ink at the guide-box exact position. Applying the content shift would
        # then look for the borrow at the wrong cell and mark an actually-correct borrow
        # MISSING. Until the bank contains exercises with borrow cells that can be tested
        # end-to-end under drift, leave this path as exact-cell lookup.
        #
        # TODO: when the bank adds borrow-carrying exercises, add a borrow-specific
        # _best_global_dy computed only from REPLACE-token y-levels, and apply that
        # shift independently of the content shift.

        elements: list[EvalElement] = []
        feedback: list[EvalFeedback] = []
        # All missing groups for hint selection (GIVEN + to-fill).
        missing_all_groups: list[list[ExpectedToken]] = []
        missing_to_fill: list[ExpectedToken] = []
        matched_to_fill = 0
        total_to_fill = 0

        groups = _group_expected_tokens(core_expected)

        # Build the set of y-levels that belong exclusively to step (carry /
        # borrow / partial) rows.  Extra submitted tokens that land on these
        # rows are silently absorbed rather than surfaced as UnexpectedSymbol,
        # because the recognizer may place a step mark slightly outside the
        # expected column while the child's intent is clearly a step mark.
        _step_y_levels: frozenset[int] = frozenset(
            t.y
            for t in core_expected
            if t.role in _STEP_ROLES and not t.given
        )

        for group in groups:
            # Representative values for the group.
            role = group[0].role
            y = group[0].y
            left_x = group[0].x
            template_type = _template_type(role)
            template_id = f"{role}-{left_x}-{y}"
            width = len(group)
            num_str = "".join(t.c for t in group)
            is_given = all(t.given for t in group)

            # Collect submitted tokens for this group via structure-relative
            # match (None = absent at that slot).
            subs = [matched_sub.get(t.cell()) for t in group]

            # --- STEP CELLS: presence-only grading --------------------------
            # Carry, borrow, and partial-product marks are graded on PRESENCE
            # rather than exact value.  Notation varies and small marks are
            # easily misread by the recognizer, so a misread digit must never
            # flip a correct answer to a "mistake".  Any submitted token at the
            # step slot is accepted; an absent slot is MISSING (gentle nudge to
            # show the working); no ERROR is ever produced for a step cell.
            if role in _STEP_ROLES and not is_given:
                total_to_fill += len(group)
                any_present = any(s is not None for s in subs)
                if any_present:
                    # Presence = OK: count the full group as satisfied.
                    matched_to_fill += len(group)
                    symbol_ids = [s.id for s in subs if s is not None]
                    elements.append(
                        EvalElement(
                            template_type=template_type,
                            template_id=template_id,
                            attribute_map={"num": num_str, "SHOW": "USER"},
                            position=(left_x, y),
                            width=width,
                            height=1,
                            status="OK",
                            symbol_id_list=symbol_ids,
                            l=0.0,
                        )
                    )
                else:
                    # Nothing submitted: MISSING (gentle nudge).
                    elements.append(
                        EvalElement(
                            template_type=template_type,
                            template_id=template_id,
                            attribute_map={"num": num_str, "matched_num": "__"},
                            position=(left_x, y),
                            width=width,
                            height=1,
                            status="MISSING",
                            symbol_id_list=[],
                            l=-0.5,
                        )
                    )
                    missing_all_groups.append(group)
                    for t in group:
                        missing_to_fill.append(t)
                continue  # step group fully handled; skip strict grading below

            # --- ANSWER / OPERAND CELLS: strict value grading ---------------
            # Determine per-group status.
            all_correct = all(
                s is not None and s.c == t.c for s, t in zip(subs, group)
            )
            any_wrong = any(
                s is not None and s.c != t.c for s, t in zip(subs, group)
            )

            if all_correct:
                # Every cell matched correctly.
                for t in group:
                    if not t.given:
                        matched_to_fill += 1
                total_to_fill += sum(1 for t in group if not t.given)
                symbol_ids = [s.id for s in subs if s is not None]
                elements.append(
                    EvalElement(
                        template_type=template_type,
                        template_id=template_id,
                        attribute_map={
                            "num": num_str,
                            "SHOW": "FIXED" if is_given else "USER",
                        },
                        position=(left_x, y),
                        width=width,
                        height=1,
                        status="OK",
                        symbol_id_list=symbol_ids,
                        l=0.0,
                    )
                )
            elif any_wrong:
                # At least one cell has the wrong char (wrong before absent).
                total_to_fill += sum(1 for t in group if not t.given)
                for t, s in zip(group, subs):
                    if s is not None and s.c != t.c:
                        cell = t.cell()
                        elements.append(
                            EvalElement(
                                template_type=template_type,
                                template_id=f"{role}-{t.x}-{y}",
                                attribute_map={"num": t.c, "matched_num": s.c},
                                position=cell,
                                width=1,
                                height=1,
                                status="ERROR",
                                symbol_id_list=[s.id],
                                l=-0.5,
                            )
                        )
                        feedback.append(
                            EvalFeedback(
                                message_type="IncorrectValue",
                                message_args={
                                    "expected": t.c,
                                    "got": s.c,
                                    "role": role,
                                },
                                target_positions=[_cell_rect(cell)],
                                symbol_id_list=[s.id],
                            )
                        )
                    elif s is None:
                        # Absent within a partially-wrong group -> MISSING.
                        cell = t.cell()
                        elements.append(
                            EvalElement(
                                template_type=template_type,
                                template_id=f"{role}-{t.x}-{y}",
                                attribute_map={"num": t.c, "matched_num": "__"}
                                if not t.given
                                else {"num": t.c},
                                position=cell,
                                width=1,
                                height=1,
                                status="MISSING",
                                symbol_id_list=[],
                                l=-0.5,
                            )
                        )
                        if not t.given:
                            missing_to_fill.append(t)
            else:
                # Nothing submitted for this group (or only some absent, with no wrong chars):
                # treat as full-group MISSING, matching the captured shape.
                total_to_fill += sum(1 for t in group if not t.given)
                if is_given:
                    # Pre-printed scaffold/bar: always satisfied regardless of
                    # whether a submitted token matched it.  Emit OK (not MISSING)
                    # so a correct submission never shows a stray missing marker
                    # for given cells (e.g. the result bar, whose char '_' is
                    # excluded from grid tokens and therefore never submitted).
                    elements.append(
                        EvalElement(
                            template_type=template_type,
                            template_id=template_id,
                            attribute_map={"num": num_str, "SHOW": "FIXED"},
                            position=(left_x, y),
                            width=width,
                            height=1,
                            status="OK",
                            symbol_id_list=[],
                            l=0.0,
                        )
                    )
                else:
                    # Child-fill group with no submission: MISSING.
                    elements.append(
                        EvalElement(
                            template_type=template_type,
                            template_id=template_id,
                            attribute_map={"num": num_str, "matched_num": "__"},
                            position=(left_x, y),
                            width=width,
                            height=1,
                            status="MISSING",
                            symbol_id_list=[],
                            l=-0.5,
                        )
                    )
                    # Only child-fill (non-given) groups enter the hint pool
                    # and count toward missing_to_fill.
                    missing_all_groups.append(group)
                    for t in group:
                        if not t.given:
                            missing_to_fill.append(t)

        # Extra tokens: submitted tokens not mapped to any expected slot.
        # Step-row extras are silently absorbed: a child who draws a carry or
        # borrow mark slightly outside the expected column must never receive
        # an UnexpectedSymbol warning on otherwise correct work.
        for sub in extra_tokens:
            if sub.c in ("", "_"):
                continue  # skip structural tokens (bars) that were not matched
            if sub.y in _step_y_levels:
                continue  # absorbed: step-row token at unexpected column
            feedback.append(
                EvalFeedback(
                    message_type="UnexpectedSymbol",
                    message_args={"got": sub.c, "x": sub.x, "y": sub.y},
                    target_positions=[_cell_rect((sub.x, sub.y))],
                    symbol_id_list=[sub.id],
                )
            )

        # --- BORROW (optional REPLACE) cells: graded only when drawn ----------
        # Grading engine grades a borrow on VALUE (wrong borrow -> BorrowIncorrectFeedback
        # and a progress drop), but never penalises an absent borrow. Submissions
        # for these cells arrive as type=REPLACE tokens at the borrow cell; they
        # were left in extra_tokens by the core match, so claim them here.
        for cell, exp in borrow_cells.items():
            sub = borrow_subs.get(cell)
            if sub is None:
                continue  # optional: an undrawn borrow is never penalised
            total_to_fill += 1
            if sub.c == exp.c:
                matched_to_fill += 1
                elements.append(
                    EvalElement(
                        template_type="SpecialElement",
                        template_id=f"borrow-{cell[0]}-{cell[1]}",
                        attribute_map={"num": exp.c, "SHOW": "USER"},
                        position=cell,
                        width=1,
                        height=1,
                        status="OK",
                        symbol_id_list=[sub.id],
                        l=0.0,
                    )
                )
            else:
                elements.append(
                    EvalElement(
                        template_type="SpecialElement",
                        template_id=f"borrow-{cell[0]}-{cell[1]}",
                        attribute_map={"num": exp.c, "matched_num": sub.c},
                        position=cell,
                        width=1,
                        height=1,
                        status="ERROR",
                        symbol_id_list=[sub.id],
                        l=-0.5,
                    )
                )
                feedback.append(
                    EvalFeedback(
                        message_type="BorrowIncorrectFeedback",
                        message_args={"expected": exp.c, "got": sub.c, "role": "borrow"},
                        target_positions=[_cell_rect(cell)],
                        symbol_id_list=[sub.id],
                    )
                )

        progress = (matched_to_fill / total_to_fill) if total_to_fill else 1.0
        # Suppress hint when no child-fill cells remain (every required cell
        # is satisfied). missing_all_groups already excludes pure given/scaffold
        # groups, so this is normally redundant, but kept as an explicit guard.
        hint = self._next_hint(exercise, missing_all_groups) if missing_to_fill else None
        result = EvalResult(
            elements=elements,
            unmatched=[],
            feedback=feedback,
            hint=hint,
            progress=progress,
            l=-0.5 if progress < 1.0 else 0.0,
        )
        session = self._require_session_full(session_id)
        self._sessions[session_id] = _MockSession(
            exercise=session.exercise,
            ref_id=ref_id,
            marks_total=session.marks_total,
            last_result=result,
        )
        return result

    def _next_hint(
        self, exercise: BankExercise, missing_groups: list[list[ExpectedToken]]
    ) -> EvalHint | None:
        """Build one positioned hint for the next missing child-fill group.

        ``missing_groups`` contains only child-must-fill MISSING groups; pure
        given/scaffold groups (operands, result bar) are excluded by the caller
        because they are pre-printed and never fillable by the child. The hint
        targets the group closest to the answer row (breaking ties by rightmost
        ``x`` first, so the units column is prioritised). The returned
        :class:`EvalHint` spans the full group width in ``target_positions``
        and names the full multi-character value in ``messageArgs.missing``.
        Returns ``None`` when ``missing_groups`` is empty (nothing left to fill).
        """
        if not missing_groups:
            return None

        def _group_sort_key(group: list[ExpectedToken]) -> tuple[int, int]:
            # Sort by distance from answer row, then by rightmost x (units first).
            return (abs(group[0].y - exercise.answer_y), -group[-1].x)

        target_group = min(missing_groups, key=_group_sort_key)
        message_type, message_args = _hint_message_for(exercise, target_group)
        return EvalHint(
            source_token_ids=[],
            target_positions=[_group_rect(target_group)],
            message_type=message_type,
            message_args=message_args,
        )

    # --- solution -----------------------------------------------------------

    def solution(self, session_id: str) -> dict:
        """Return the captured solution fixture data for the session.

        Returns the raw ``session/solution`` response from the fixture captured
        at ``tests/grading/fixtures/derive/<exercise_id>.solution.json``.
        The demo's ``select_route`` calls ``client.solution(session_id)`` to
        derive the BankExercise; the mock returns the captured fixture so
        ``derive_exercise`` produces the same layout in both mock and http modes.
        Falls back to a minimal synthetic response when the stash is absent
        (e.g. the session was created outside the normal flow in a test).
        """
        sol_data = self._solution_data.get(session_id)
        if sol_data is not None:
            return sol_data
        # Fallback: minimal response compatible with the derive contract
        # (will fail if the caller tries to parse solution tokens from it).
        exercise = self._require_session(session_id)
        return {
            "success": True,
            "exerciseId": exercise.exercise_id,
        }

    # --- info ---------------------------------------------------------------

    def info(self, session_id: str) -> ScoringInfo:
        """Return the session's score from the most recent evaluation.

        Marks are earned only once the last evaluation reached full progress
        (every child-supplied cell correct). Before any evaluation the score is
        zero and the session is unfinished.
        """
        session = self._require_session_full(session_id)
        last = session.last_result
        finished = last is not None and last.progress >= 1.0
        marks_total = session.marks_total
        return ScoringInfo(
            finished=finished,
            marks_total=marks_total,
            marks_earned=marks_total if finished else 0,
            penalties={"marksPenalty": 0, "hintsRequested": 0, "mathErrors": 0},
        )

    # --- session lookup -----------------------------------------------------

    def _require_session(self, session_id: str) -> BankExercise:
        return self._require_session_full(session_id).exercise

    def _require_session_full(self, session_id: str) -> "_MockSession":
        session = self._sessions.get(session_id)
        if session is None:
            raise GradingError(
                f"unknown mock session {session_id!r}; call create_session first"
            )
        return session


# ---------------------------------------------------------------------------
# Feedback summary (verdict + focus), computed from the graded EvalResult
# ---------------------------------------------------------------------------

# Roles the child fills that constitute the FINAL ANSWER (quotient row for
# division; result-row digits below the bar for all other templates).
_ANSWER_ROLES: frozenset[str] = frozenset({"answer"})

# Roles the child fills that constitute INTERMEDIATE STEPS (carries, borrows,
# partial products). A submission that has the right answer but is missing one
# of these is "missing_step", not wrong.
_STEP_ROLES: frozenset[str] = frozenset({"carry", "borrow", "partial"})


@dataclass(frozen=True)
class _FocusItem:
    """One cell that the feedback layer wants to highlight.

    ``grid_cell`` is ``(x, y)`` in grading engine evaluate-frame coords when the
    focus falls on an empty expected cell (MISSING); ``None`` when the glyph
    bbox is used instead (ERROR on a drawn token).
    ``symbol_id`` is the submitted token id that was wrong (ERROR); ``None``
    for missing cells.
    ``kind`` is one of ``"error"`` / ``"missing_step"`` / ``"missing_answer"``.
    """

    grid_cell: tuple[int, int] | None
    symbol_id: str | None
    kind: str  # "error" | "missing_step" | "missing_answer"


@dataclass(frozen=True)
class FeedbackSummary:
    """Single-verdict feedback summary derived from a graded :class:`EvalResult`.

    ``outcome`` is one of ``"correct"`` / ``"mistake"`` / ``"missing_step"`` /
    ``"incomplete"``. ``headline`` is a short child-friendly message (no
    em-dashes, no loud punctuation). ``focus_items`` carries at most a handful
    of cells the frontend should highlight; empty when ``outcome == "correct"``.
    """

    outcome: str  # "correct" | "mistake" | "missing_step" | "incomplete"
    headline: str
    focus_items: tuple[_FocusItem, ...]


def compute_feedback_summary(
    exercise: "BankExercise",
    result: "EvalResult",
) -> "FeedbackSummary":
    """Derive the single-verdict :class:`FeedbackSummary` from a graded result.

    Classification logic
    --------------------
    The answer is the primary signal.  Carry, borrow, and partial-product
    marks are "shown working" graded on presence; their value is never
    checked, so they can never produce an ERROR and never drive a "mistake"
    verdict.

    1. Partition the exercise's non-given expected cells into ANSWER cells
       (role in ``_ANSWER_ROLES``) and STEP cells (role in ``_STEP_ROLES``).
    2. Walk the graded ``elements`` list and collect:
       - ERROR elements whose position maps to an ANSWER cell -> "mistake".
         (ERROR on a STEP cell is absorbed as defence-in-depth; the evaluator
         must not produce step ERRORs, but this guard ensures the verdict is
         never wrong even if it does.)
       - MISSING elements whose expected role is ANSWER -> "incomplete".
       - MISSING elements whose expected role is STEP -> "missing_step".
    3. Priority: mistake > incomplete > missing_step > correct. "correct"
       additionally requires the grading engine's ``result.progress`` to reach 1.0;
       otherwise the verdict is "incomplete" (keep going).
       "missing_step" only fires when there are NO answer errors AND NO
       missing answer cells (i.e. the answer is fully correct but a step mark
       was omitted).
    4. For "correct" the focus list is empty. For the other outcomes the focus
       carries only the first relevant cell(s) so the frontend highlights one
       clear target rather than a sea of markers.

    The function never mutates the result or the exercise; it is a pure
    projection from the graded state.
    """
    # Build a lookup: expected_cell (x,y) -> ExpectedToken for role queries.
    cell_role: dict[tuple[int, int], str] = {
        t.cell(): t.role for t in exercise.expected if not t.given
    }

    error_focus: list[_FocusItem] = []
    missing_answer_focus: list[_FocusItem] = []
    missing_step_focus: list[_FocusItem] = []

    for element in result.elements:
        status = element.status.upper()
        pos = element.position
        role = cell_role.get(pos, "")

        if status == "ERROR":
            # Only answer-cell errors count as a "mistake".  Step-cell errors
            # are absorbed here (defence in depth against an evaluator bug).
            if role in _ANSWER_ROLES:
                sid = element.symbol_id_list[0] if element.symbol_id_list else None
                error_focus.append(_FocusItem(grid_cell=None, symbol_id=sid, kind="error"))

        elif status == "MISSING" and role in _ANSWER_ROLES:
            missing_answer_focus.append(
                _FocusItem(grid_cell=pos, symbol_id=None, kind="missing_answer")
            )

        elif status == "MISSING" and role in _STEP_ROLES:
            missing_step_focus.append(
                _FocusItem(grid_cell=pos, symbol_id=None, kind="missing_step")
            )

    # Derive verdict: answer correctness is the sole gate for "mistake".
    if error_focus:
        return FeedbackSummary(
            outcome="mistake",
            headline="Almost. Take another look at the highlighted part.",
            focus_items=tuple(error_focus),
        )

    if missing_answer_focus:
        return FeedbackSummary(
            outcome="incomplete",
            headline="Keep going. Finish the answer here.",
            focus_items=tuple(missing_answer_focus),
        )

    if missing_step_focus:
        return FeedbackSummary(
            outcome="missing_step",
            headline="Your answer is right. Show your working here too.",
            focus_items=tuple(missing_step_focus),
        )

    # Correct only when grading engine reports full progress. ``result.progress`` is
    # the engine's authoritative completion signal (it also drives marks/finished); without
    # this gate an unfinished answer with no detected error or missing cell would
    # wrongly read as solved.
    if result.progress >= 1.0:
        return FeedbackSummary(
            outcome="correct",
            headline="You solved it. Nice working.",
            focus_items=(),
        )

    return FeedbackSummary(
        outcome="incomplete",
        headline="Keep going. Fill in the answer.",
        focus_items=(),
    )


def _cell_rect(cell: tuple[int, int]) -> GridRect:
    """A one-cell :class:`GridRect` in grading engine ``targetPositions`` shape.

    ``right``/``top`` are exclusive upper bounds (one past the cell), matching
    the captured hint rect ``{left:14, top:10, bottom:9, right:16}`` for the two
    cells x=14,15 at y=9: left=x, right=x+width, bottom=y, top=y+1.
    """
    x, y = cell
    return GridRect(left=x, top=y + 1, bottom=y, right=x + 1)
