"""Auto-deriver: build a BankExercise from grading engine create + solution responses.

``derive_exercise`` is the single pure function that owns the transformation::

    (spec, create_view_model, solution_data) -> BankExercise

It reads the grading engine's native solution frame (from ``session/solution``) and the
create view-model (from ``session/create``) to classify every solution token into
a full ``ExpectedToken`` set without any hand-written coordinates.

Contract invariants:

* The evaluate grader is TRANSLATION-INVARIANT and id-agnostic.  Submit the
  solution token list VERBATIM in the native frame. No offset constants.
* The FIXED scaffold tokens MUST always be submitted at evaluate.  Dropping them
  collapses progress to 0.  The deriver marks operands/operator/bar ``given=True``;
  ``evaluate_route`` already injects given as FIXED.
* ``solution`` tokens live at
  ``view.elements[0].interactions[0].solution`` in the solution response's init-data
  HTML (use ``_extract_init_data`` exactly as ``SessionCreated`` does).
* Area-type -> role: LINE=answer, REPLACE=borrow, OVERFLOW=carry.
  PLACE_VALUE / RIGHT_TO_LEFT / AUTOSHOW are structural/decorative -- exclude.
* Bottom operands (e.g. ``89``, ``38``/``29``) come from SOLUTION tokens
  (SYMBOL/GENERATED on/above bar row), NOT create-FIXED; mark them ``given=True``.
* Answer row = bottom-most LINE-band SYMBOL/GENERATED row (NOT a fixed offset
  from the bar; product has partial-product rows between).
* Multi-digit cell: ``len(str(c)) > 1`` (observed for borrows: ``16``, ``14``).

This module depends only on ``.types`` and ``.exercises``; it never calls
the network, renders anything, or grades.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Callable

from .exercises import BankExercise, ExpectedToken
from .types import ExerciseSpec, _extract_init_data

logger = logging.getLogger(__name__)

# Area types that produce gradable fill cells (the only ones we emit tokens for).
AREA_ROLE: dict[str, str] = {
    "LINE": "answer",
    "REPLACE": "borrow",
    "OVERFLOW": "carry",
}



# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _extract_question_prompt(view_model: dict) -> str:
    """Extract the plain-text prompt from the QUESTION element of the create view-model.

    Navigates ``view.elements[?type==QUESTION].content[0].content``, strips HTML
    tags, collapses whitespace, and strips trailing punctuation noise.
    Returns an empty string when the QUESTION element is absent or has no content.
    """
    elements = view_model.get("view", {}).get("elements", [])
    for element in elements:
        if element.get("type") == "QUESTION":
            content_list = element.get("content", [])
            if content_list:
                raw_html = content_list[0].get("content", "")
                # Strip HTML tags, collapse whitespace.
                plain = _HTML_TAG_RE.sub(" ", raw_html)
                plain = " ".join(plain.split()).strip()
                return plain
    return ""


def _title_from_prompt(prompt: str) -> str:
    """Derive a short title from a prompt string.

    Takes up to the first sentence (up to the first period or 60 chars),
    stripping trailing punctuation. Falls back to the full prompt when short.
    Examples:
      "Calculate 256 - 89." -> "Calculate 256 - 89"
      "Calculate 38 x 29." -> "Calculate 38 x 29"
    """
    if not prompt:
        return ""
    # Take up to first sentence-ending period.
    dot = prompt.find(".")
    candidate = prompt[:dot] if dot != -1 else prompt
    candidate = candidate.strip().rstrip(".")
    # If still long, truncate at 60 chars.
    return candidate[:60].strip()


def _find_ans(view_model: dict) -> dict:
    """Return the ARITHMETIC ``ans`` block from a create view-model.

    Navigates ``view.elements[0].interactions[0].ans``.
    Raises ``KeyError`` when the path is absent (malformed view-model).
    """
    view = view_model.get("view", {})
    elements = view.get("elements", [])
    if not elements:
        raise KeyError("view_model has no elements")
    for element in elements:
        for interaction in element.get("interactions", []):
            ans = interaction.get("ans")
            if isinstance(ans, dict):
                return ans
    raise KeyError("could not locate 'ans' block in view_model")


def _solution_tokens(solution_data: dict) -> list[dict]:
    """Extract the worked-solution token list from a ``session/solution`` response.

    The response is an HTML blob; the token list is embedded in the
    ``grading/init-data`` script at
    ``view.elements[0].interactions[0].solution``.
    Raises ``KeyError`` when the path is absent or empty.
    """
    html = solution_data.get("html", "")
    init = _extract_init_data(html)
    try:
        elements = init["view"]["elements"]
        if not elements:
            raise KeyError("no elements in solution init-data")
        interactions = elements[0].get("interactions", [])
        if not interactions:
            raise KeyError("no interactions in solution init-data")
        tokens = interactions[0].get("solution")
        if tokens is None:
            raise KeyError("'solution' key absent in interaction")
        return list(tokens)
    except (KeyError, IndexError, TypeError) as exc:
        raise KeyError(
            f"could not extract solution tokens from init-data: {exc}"
        ) from exc


def _role_for(
    token: dict,
    line_band: range,
    bar_row: int,
    is_fixed: bool,
) -> str | None:
    """Classify a single solution token into a role string, or ``None`` to exclude.

    Decision tree (matches the research recipe):

    1. FIXED-tagged tokens -> role by char (``_`` = result_bar, operator chars =
       operator, digit = operand). Always given=True.
    2. type=REPLACE -> ``"borrow"``
    3. type=OVERFLOW -> ``"carry"``
    4. SYMBOL/GENERATED, y >= bar_row (on or above bar), not AUTOSHOW
       -> ``"operand"`` (missing bottom operand / operator rows) given=True
    5. SYMBOL/GENERATED, y < bar_row, x in line_band
       -> ``"answer"`` (on the LINE band below the bar)
    6. SYMBOL/GENERATED, y < bar_row, x not in line_band
       -> ``"partial"`` (partial products, separators below bar outside LINE band)
    7. AUTOSHOW-tagged tokens -> ``None`` (exclude)

    For items 5 and 6, the caller identifies the actual answer row as the
    bottommost SYMBOL/GENERATED row inside the LINE band.  A partial-product
    row or separator row that happens to be inside the LINE band column range
    would pass test 5, but is then excluded from the ``answer`` role by the
    caller which only takes tokens on the *bottommost* such row.
    """
    tags = token.get("tags", [])
    tok_type = token.get("type", "SYMBOL")
    x = int(token["x"])
    y = int(token["y"])
    c = str(token["c"])

    # Exclude AUTOSHOW tokens entirely.
    if "AUTOSHOW" in tags:
        return None

    if is_fixed:
        # FIXED scaffold: classify by character.
        if c == "_":
            return "result_bar"
        if c in ("+", "-", "*", "/", "x", "×", "÷"):
            return "operator"
        return "operand"

    # Step cells by token type.
    if tok_type == "REPLACE":
        return "borrow"
    if tok_type == "OVERFLOW":
        return "carry"

    # SYMBOL/GENERATED: classify by position.
    if y >= bar_row:
        # On or above the bar row: the missing problem tokens (operands, plus a
        # GENERATED operator such as multiplication's *). Classify by character
        # so the operator renders as an operator glyph, not a digit box.
        if c == "_":
            return None  # separator underscore above bar: exclude
        if c in ("+", "-", "*", "/", "x", "×", "÷"):
            return "operator"
        return "operand"

    # Below the bar.
    if x in line_band:
        # Inside the LINE band: answer candidate or partial product.
        # Separator rows (c in ('+', '_') that are below bar and in band) are partial.
        if c in ("+", "_"):
            return "partial"
        return "answer"  # caller filters to bottommost row only
    else:
        # Outside LINE band below bar: partial product or separator.
        return "partial"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_exercise(
    spec: ExerciseSpec,
    create_view_model: dict,
    solution_data: dict,
) -> BankExercise:
    """Build a :class:`BankExercise` from grading engine native create + solution.

    Parameters
    ----------
    spec:
        The :class:`ExerciseSpec` the create call used (exercise_id, template, etc.).
    create_view_model:
        The parsed init-data from ``session/create`` (the ``view_model`` field of
        :class:`SessionCreated`). Supplies grid dims, FIXED scaffold, and areaMap.
    solution_data:
        The raw dict returned by ``client.solution(session_id)`` (an HTML blob
        containing an ``grading/init-data`` script with the worked solution).

    Returns
    -------
    BankExercise
        The complete exercise layout with ``given`` tokens (bar + operator +
        all operands) and child-fill cells (answer, optional borrows/carries,
        partial products), all in the grading engine's native solution frame.

    Raises
    ------
    GradingError
        When the view-model or solution response cannot be parsed (missing
        init-data, absent ``solution`` key, absent ``ans`` block).
    """
    from .client import GradingError

    # Local list of tokens excluded during derivation; used only for the debug
    # log at the end of this function (thread-safe, no module-level mutation).
    _excluded_debug: list[dict] = []

    exercise_id = spec.exercise_id

    # --- Step 1: Parse the create view-model. ---------------------------------
    try:
        ans = _find_ans(create_view_model)
    except KeyError as exc:
        raise GradingError(
            f"derive_exercise({exercise_id!r}): failed to locate 'ans' in "
            f"create view-model: {exc}"
        ) from exc

    grid_width = int(ans.get("width", 0))
    grid_height = int(ans.get("height", 0))
    if grid_width <= 0 or grid_height <= 0:
        raise GradingError(
            f"derive_exercise({exercise_id!r}): invalid grid dims "
            f"{grid_width}x{grid_height} from view-model"
        )

    # NTH-3: fill title/prompt from the view-model QUESTION element when the
    # spec carries empty strings (unknown UUID not in _CMS_METADATA). Known
    # exercises already have their human titles from _CMS_METADATA; those are
    # kept as-is so the picker labels are stable.
    if not spec.title or not spec.prompt:
        derived_prompt = _extract_question_prompt(create_view_model)
        if derived_prompt:
            new_prompt = spec.prompt or derived_prompt
            new_title = spec.title or _title_from_prompt(derived_prompt)
            spec = dataclasses.replace(spec, title=new_title, prompt=new_prompt)
            exercise_id = spec.exercise_id  # unchanged; re-bind for clarity

    area_map: dict[str, dict] = ans.get("attributeMap", {}).get("areaMap", {})

    # Locate the LINE area to find the answer column band.
    line_area = next(
        (a for a in area_map.values() if a.get("type") == "LINE"), None
    )
    if line_area is None:
        raise GradingError(
            f"derive_exercise({exercise_id!r}): no LINE area in areaMap; "
            "cannot locate answer band"
        )
    line_left = int(line_area["left"])
    line_width = int(line_area["width"])
    line_band = range(line_left, line_left + line_width)

    # --- Step 2: Extract solution tokens. ------------------------------------
    try:
        sol_tokens = _solution_tokens(solution_data)
    except KeyError as exc:
        raise GradingError(
            f"derive_exercise({exercise_id!r}): failed to extract solution "
            f"tokens: {exc}"
        ) from exc

    if not sol_tokens:
        raise GradingError(
            f"derive_exercise({exercise_id!r}): solution token list is empty"
        )

    # --- Step 3: Locate the result bar row. -----------------------------------
    # The bar row is the ``y`` of the FIXED ``_`` tokens.
    bar_row: int | None = None
    for t in sol_tokens:
        if "FIXED" in t.get("tags", []) and str(t["c"]) == "_":
            bar_row = int(t["y"])
            break
    if bar_row is None:
        raise GradingError(
            f"derive_exercise({exercise_id!r}): could not locate result bar "
            "(FIXED '_' token) in solution tokens"
        )

    # --- Step 4: Classify every solution token. ------------------------------
    expected: list[ExpectedToken] = []

    # First pass: collect all tokens classified by role.
    # We need to find the bottommost LINE-band row for answer_y determination.
    # Tokens classified as "answer" (LINE-band below bar, digit) are candidates;
    # then we take only those on the bottommost such row.
    answer_candidates: list[dict] = []
    non_answer_tokens: list[tuple[dict, str]] = []  # (token, role)

    for t in sol_tokens:
        tags = t.get("tags", [])
        is_fixed = "FIXED" in tags

        role = _role_for(t, line_band, bar_row, is_fixed)

        if role is None:
            _excluded_debug.append(t)
            continue

        if role == "answer":
            # Defer: only add from the bottommost row.
            answer_candidates.append(t)
        elif role == "partial" and str(t["c"]) in ("+", "_"):
            # Separator row tokens ('+' / '_' at separator rows) are excluded.
            # The separator row is an auto-rendered engine display artifact (product
            # y=1 row with '+' and '_'); the child never draws it.
            _excluded_debug.append(t)
        else:
            non_answer_tokens.append((t, role))

    # Determine answer_y: bottommost y among answer candidates inside LINE band.
    if answer_candidates:
        # Group by y; the bottommost (smallest y since y=0 is bottom) is answer_y.
        answer_y = min(int(t["y"]) for t in answer_candidates)
        # Only take tokens on the answer row itself.
        answer_row_tokens = [t for t in answer_candidates if int(t["y"]) == answer_y]
        # Any answer candidate on a row above answer_y becomes a partial product.
        partial_extras = [t for t in answer_candidates if int(t["y"]) != answer_y]
    else:
        # No answer candidates: degenerate case.
        answer_y = bar_row - 1
        answer_row_tokens = []
        partial_extras = []

    # Build ExpectedTokens from non-answer tokens.
    for t, role in non_answer_tokens:
        is_fixed = "FIXED" in t.get("tags", [])
        # A GENERATED operator (e.g. multiplication's *) is part of the given
        # problem just like the operands, so it is shown + submitted FIXED.
        given = is_fixed or role in ("operand", "operator")
        expected.append(
            ExpectedToken(
                x=int(t["x"]),
                y=int(t["y"]),
                c=str(t["c"]),
                role=role,
                given=given,
            )
        )

    # Build ExpectedTokens from partial extras (answer candidates above answer row).
    for t in partial_extras:
        expected.append(
            ExpectedToken(
                x=int(t["x"]),
                y=int(t["y"]),
                c=str(t["c"]),
                role="partial",
                given=False,
            )
        )

    # Build answer tokens.
    for t in answer_row_tokens:
        expected.append(
            ExpectedToken(
                x=int(t["x"]),
                y=int(t["y"]),
                c=str(t["c"]),
                role="answer",
                given=False,
            )
        )

    if not expected:
        raise GradingError(
            f"derive_exercise({exercise_id!r}): derived expected set is empty; "
            "check the solution token classification"
        )

    # Warn about any excluded tokens (for debugging).
    if _excluded_debug:
        logger.debug(
            "derive_exercise(%r): excluded %d tokens: %s",
            exercise_id,
            len(_excluded_debug),
            _excluded_debug,
        )

    # --- Step 5: Compute evaluate_engine_origin. ----------------------------------
    # evaluate_engine_origin = (min_x, max_y) of the non-bar, non-AUTOSHOW expected set.
    # This is used for canvas rendering (pixel-to-grid mapping), NOT for grading.
    # We include all expected tokens (given + child-fill) for the origin.
    all_x = [t.x for t in expected if t.role != "result_bar"]
    all_y = [t.y for t in expected if t.role != "result_bar"]
    if all_x and all_y:
        evaluate_engine_origin = (min(all_x), max(all_y))
    else:
        evaluate_engine_origin = (0, grid_height - 1)

    return BankExercise(
        spec=spec,
        expected=tuple(expected),
        grid_width=grid_width,
        grid_height=grid_height,
        answer_y=answer_y,
        evaluate_engine_origin=evaluate_engine_origin,
    )
