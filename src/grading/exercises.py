"""Exercise registry and contract types for the grading engine.

The exercise *layout* (which cells exist, in what roles, with what values) is
no longer hand-written here. It is derived at select-time by
:func:`src.grading.derive.derive_exercise` from the grading engine's own
``session/create`` + ``session/solution`` responses. This module owns:

* :class:`ExpectedToken` / :class:`BankExercise` — the internal contract shared
  by the deriver, the mock grader, the grid mapper, and the demo app. Never change
  these without updating every downstream consumer.
* :func:`expected_tokens_as_grid` — converts a :class:`BankExercise` to a
  submittable :class:`GridToken` list (borrow -> REPLACE, carry -> OVERFLOW).
* :func:`list_cms_exercises` — returns the configured CMS UUID list as
  :class:`ExerciseSpec` stubs (id + cms_exercise_id). Title/prompt are filled
  at derive time from the view-model; the stubs only need the UUID for
  ``session/create`` to resolve to the right exercise.
* :func:`build_exercise_spec` — validated spec constructor (decimal restrictions).

The static ``_BANK`` and the hand-written ``_build_subtract_256_89`` /
``_build_product_38_29`` builders have been removed. Their cell coordinates were
hand-mapped against a specific evaluate frame; the deriver works in the native
solution frame, which is translation-invariant (see
the derivation notes (not included)).

This module depends only on :mod:`.types`; it never grades or talks to HTTP.

**Exercise-authoring constraint (grid mapping):** the whole-row typed-area
partition in :func:`src.grading.grid_mapping.tokens_to_grid` assumes that
borrow/carry (typed) cells occupy y-rows that are strictly disjoint from
operand/answer y-rows. All current column-arithmetic exercises satisfy this
(borrows sit above operands). Do not add an exercise where a typed cell shares
an engine y-value with an operand or answer cell without revisiting the partition
logic first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import ExerciseSpec, GridToken

# Grading engine restrictions the bank must honour (engine documentation):
# product allows at most one decimal operand; division allows no decimal
# operands. The builder validates them so an out-of-spec edit fails loudly.
_DECIMAL = "."


def _has_decimal(arg: str) -> bool:
    return _DECIMAL in arg


def _validate_template_restrictions(template_name: str, template_args: list[str]) -> None:
    """Reject operands that violate the grading engine's per-template decimal rules.

    Product permits at most one decimal operand; division permits none. A
    violation is a bank authoring bug, so it raises at construction rather than
    surfacing later as a confusing grading mismatch.
    """
    if template_name.startswith("PRODUCT"):
        decimals = [a for a in template_args if _has_decimal(a)]
        if len(decimals) > 1:
            raise ValueError(
                f"PRODUCT templates allow at most one decimal operand; "
                f"got {template_args!r}"
            )
    if template_name.startswith("DIVISION"):
        if any(_has_decimal(a) for a in template_args):
            raise ValueError(
                f"DIVISION templates allow no decimal operands; got {template_args!r}"
            )


@dataclass(frozen=True)
class ExpectedToken:
    """One cell of an exercise's complete worked solution.

    ``x`` / ``y`` are grading engine native solution-frame grid coordinates (``y``
    is bottom-origin, increasing upward); ``c`` is the single character (or
    multi-digit string for REPLACE borrow cells) the cell must carry; ``role``
    labels the cell (``operand`` / ``operator`` / ``result_bar`` / ``borrow`` /
    ``carry`` / ``partial`` / ``answer``) so the grader can pick the most useful
    next-step hint without re-deriving structure. ``given`` marks scaffold cells
    grading engine pre-prints (the FIXED problem digits, operator, and result bar):
    they are part of the expected solution but are never something the child has
    to supply.

    Coordinates are in the native solution frame — the evaluate grader is
    translation-invariant, so these coords are submitted verbatim with no offset.
    """

    x: int
    y: int
    c: str
    role: str
    given: bool = False

    def cell(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BankExercise:
    """A bank entry: the create-call spec plus its complete expected solution.

    ``spec`` is the :class:`ExerciseSpec` sent to ``session/create``;
    ``expected`` is the full worked-solution token set the mock grades against;
    ``grid_width`` / ``grid_height`` size the interaction grid (matching the
    engine create view-model so the scaffold renders at the same pitch the grader
    expects); ``answer_y`` is the row the answer line occupies, used to pick the
    hint nearest the answer; ``evaluate_engine_origin`` is the ``(engine_x_left,
    engine_y_top)`` pair for canvas pixel-to-grid rendering (NOT for grading; the
    grader is frame-agnostic and accepts native coords verbatim).
    """

    spec: ExerciseSpec
    expected: tuple[ExpectedToken, ...]
    grid_width: int
    grid_height: int
    answer_y: int
    evaluate_engine_origin: tuple[int, int] = (0, 0)

    @property
    def exercise_id(self) -> str:
        return self.spec.exercise_id

    def expected_to_fill(self) -> tuple[ExpectedToken, ...]:
        """The expected cells the child must supply (everything not pre-given)."""
        return tuple(t for t in self.expected if not t.given)

    def given_tokens(self) -> tuple[ExpectedToken, ...]:
        """The scaffold cells grading engine pre-prints (FIXED problem + bar)."""
        return tuple(t for t in self.expected if t.given)


_DEFAULT_AUDIENCE = "uk_KS2"

_SHORT_FORM_ATTRIBUTES: dict[str, str] = {
    "SHOW_PROBLEM_VALUES": "all",
    "SHOW_PROBLEM_LINE": "true",
    "SHOW_PROBLEM_OPERATOR": "true",
    "SHOW_PLACE_VALUE": "true",
    "REQUIRE_OVERFLOW_BORROW": "required",
    "REQUIRE_PARTIAL_CALCULATION": "false",
}

# Known CMS exercise metadata keyed by UUID. Only the minimum needed to build
# a stub :class:`ExerciseSpec` for ``session/create``: the exercise_id slug
# (stable human key used throughout the demo), the template name (for decimal
# validation), and the template_args (for the same).
# Title and prompt are left empty here and filled at derive time from the
# view-model's QUESTION element content.
_CMS_METADATA: dict[str, dict[str, Any]] = {
    "489d0c1a-147f-47bc-86cb-6d7c8a90fe15": {
        "exercise_id": "subtract-256-89",
        "template_name": "SUBTRACT_SHORT",
        "template_args": ["256", "89"],
        "title": "Subtraction (256 - 89)",
        "prompt": "Calculate 256 - 89.",
    },
    "29612ec7-5d46-4816-8443-0ef16ae61cef": {
        "exercise_id": "product-38-29",
        "template_name": "PRODUCT_SHORT",
        "template_args": ["38", "29"],
        "title": "Multiplication (38 x 29)",
        "prompt": "Calculate 38 x 29.",
    },
}


def build_exercise_spec(
    *,
    exercise_id: str,
    title: str,
    prompt: str,
    template_name: str,
    template_args: list[str],
    attributes: dict[str, str],
    audience: str = _DEFAULT_AUDIENCE,
    cms_exercise_id: str = "",
) -> ExerciseSpec:
    """Build a validated :class:`ExerciseSpec` for one bank entry.

    Validates the grading engine's per-template decimal restrictions up front so an
    out-of-spec operand fails at construction, not at grading. Returns the
    frozen spec the create call serialises with :meth:`ExerciseSpec.to_api`.
    A ``cms_exercise_id`` is required for the HTTP client; the HTTP client
    raises :class:`~src.grading.client.GradingError` when missing.
    """
    _validate_template_restrictions(template_name, list(template_args))
    return ExerciseSpec(
        exercise_id=exercise_id,
        title=title,
        prompt=prompt,
        template_name=template_name,
        template_args=list(template_args),
        attributes=dict(attributes),
        audience=audience,
        cms_exercise_id=cms_exercise_id,
    )


def _spec_for_cms_id(cms_exercise_id: str) -> ExerciseSpec:
    """Build a stub :class:`ExerciseSpec` for a known CMS UUID.

    The stub carries the UUID so ``session/create`` resolves to the right
    exercise, plus enough metadata (template, operands) for decimal validation
    and the mock grader's hint catalog. Title/prompt come from ``_CMS_METADATA``
    and will be overridden at derive time when the view-model QUESTION content
    is available.

    Unknown UUIDs (not in ``_CMS_METADATA``) get a minimal stub with
    ``exercise_id = cms_exercise_id`` and empty template fields; the deriver
    still works because it reads everything it needs from the live responses.
    """
    meta = _CMS_METADATA.get(cms_exercise_id)
    if meta is None:
        # Unknown UUID: minimal stub; derive fills in the rest.
        return ExerciseSpec(
            exercise_id=cms_exercise_id,
            title="",
            prompt="",
            template_name="",
            template_args=[],
            attributes=dict(_SHORT_FORM_ATTRIBUTES),
            audience=_DEFAULT_AUDIENCE,
            cms_exercise_id=cms_exercise_id,
        )
    return build_exercise_spec(
        exercise_id=meta["exercise_id"],
        title=meta["title"],
        prompt=meta["prompt"],
        template_name=meta["template_name"],
        template_args=list(meta["template_args"]),
        attributes=_SHORT_FORM_ATTRIBUTES,
        audience=_DEFAULT_AUDIENCE,
        cms_exercise_id=cms_exercise_id,
    )


def list_cms_exercises(
    exercise_ids: list[str] | None = None,
) -> list[ExerciseSpec]:
    """Return stub :class:`ExerciseSpec` objects for the configured CMS UUIDs.

    ``exercise_ids`` is the list of CMS UUIDs to offer (from
    ``GradingConfig.exercise_ids`` or ``config.toml [grading]
    exercise_ids``). When ``None``, falls back to the two validated UUIDs from
    ``_CMS_METADATA`` (backward-compatible default for tests that do not pass
    the config).

    Each returned spec carries the UUID as ``cms_exercise_id`` so the HTTP
    client can pass it to ``session/create``. The exercise layout is NOT
    included here; it is derived at select time by
    :func:`src.grading.derive.derive_exercise`.
    """
    if exercise_ids is None:
        # Backward-compat default: return both validated exercises in insertion order.
        exercise_ids = list(_CMS_METADATA.keys())
    return [_spec_for_cms_id(eid) for eid in exercise_ids]


def get_exercise(exercise_id: str) -> ExerciseSpec:
    """Return the spec for a known ``exercise_id`` (slug or CMS UUID).

    Searches ``_CMS_METADATA`` by exercise_id slug first, then by UUID.
    Raises ``KeyError`` if not found.
    """
    # Search by exercise_id slug.
    for cms_id, meta in _CMS_METADATA.items():
        if meta["exercise_id"] == exercise_id:
            return _spec_for_cms_id(cms_id)
    # Search by UUID (in case a UUID was passed as exercise_id).
    if exercise_id in _CMS_METADATA:
        return _spec_for_cms_id(exercise_id)
    known = ", ".join(
        m["exercise_id"] for m in _CMS_METADATA.values()
    )
    raise KeyError(
        f"unknown exercise id {exercise_id!r}; known: {known}"
    )


def list_exercises() -> list[ExerciseSpec]:
    """Return the configured exercise specs (backward-compat alias for list_cms_exercises)."""
    return list_cms_exercises()


def expected_tokens_as_grid(
    exercise_id: str,
    bank_exercise: BankExercise | None = None,
) -> list[GridToken]:
    """Return the full expected solution as :class:`GridToken` objects.

    A convenience for tests and the demo: every expected cell (given and
    child-filled) rendered as a submittable token with a stable id.

    ``bank_exercise`` must be supplied post-refactor (the static bank is gone).
    When provided, the token list is built directly from ``bank_exercise.expected``.
    When ``None``, raises ``RuntimeError`` to surface callers that haven't
    been updated to pass the derived exercise.

    Borrow cells -> ``type="REPLACE"``; carry cells -> ``type="OVERFLOW"``.
    """
    if bank_exercise is None:
        raise RuntimeError(
            f"expected_tokens_as_grid({exercise_id!r}): bank_exercise must be "
            "supplied post-refactor. Pass the BankExercise from derive_exercise."
        )
    tokens: list[GridToken] = []
    for i, token in enumerate(bank_exercise.expected):
        tags = ["FIXED"] if token.given else []
        if token.role == "borrow":
            cell_type = "REPLACE"
        elif token.role == "carry":
            cell_type = "OVERFLOW"
        else:
            cell_type = "SYMBOL"
        tokens.append(
            GridToken(id=f"E{i}", c=token.c, x=token.x, y=token.y, tags=tags, type=cell_type)
        )
    return tokens
