"""Build a list of :class:`~src.inference.given_prior.GivenNode` from a
:class:`~src.grading.exercises.BankExercise`.

This module is the grading-side half of the given-equation prior feature
(Workstream B of the 22-Jun-26 plan). It reads ONLY the scaffold cells that
grading engine pre-prints (``bank_entry.given_tokens()``) and never touches the
child-fill / answer / carry / borrow / partial tokens.

Public API::

    build_given_prior(bank_entry, calibration) -> list[GivenNode]
    _render_tile(char, role) -> np.ndarray   # (28, 28) uint8 grayscale

See ``src/inference/given_prior.py`` for the :class:`GivenNode` dataclass
and the downstream merge / compose / pin helpers.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from ..core.ontology import YOLO_CLASS_NAMES, full_label_from_gnn_and_yolo, gnn_fine_labels_ordered
from ..inference.given_prior import GivenNode
from .exercises import BankExercise
from .grid_mapping import GridCalibration

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------

_GNN_FINE_LABELS: frozenset[str] = frozenset(gnn_fine_labels_ordered())
_YOLO_COARSE_LABELS: frozenset[str] = frozenset(YOLO_CLASS_NAMES)

# Operator character -> GNN fine label (16-class vocabulary).
_OP_CHAR_TO_FINE: dict[str, str] = {
    "+": "op_plus",
    "-": "op_minus",
    "*": "op_times",
    "×": "op_times",
    "x": "op_times",   # engine sometimes emits lowercase 'x' for multiplication
    "/": "op_divide",
    "÷": "op_divide",
}


def _resolve_labels(char: str, role: str) -> tuple[str, str]:
    """Map an :class:`~src.grading.exercises.ExpectedToken` char/role pair
    to a ``(coarse_yolo, fine_gnn)`` label pair from the ontology.

    Raises
    ------
    ValueError
        If the char/role combination cannot be resolved to a known ontology
        label (programming error — ontology must be extended before adding
        new exercise types).
    """
    if role == "result_bar":
        return "result_bar", "result_bar"

    if role == "div_bracket":
        return "divide_bracket", "div_bracket"

    if role == "operator":
        fine = _OP_CHAR_TO_FINE.get(char)
        if fine is None:
            raise ValueError(
                f"_resolve_labels: unknown operator char {char!r} (role={role!r}). "
                "Extend _OP_CHAR_TO_FINE or the exercise ontology."
            )
        return "operator", fine

    if role == "operand":
        # Given scaffold operand: a main digit printed by the exercise.
        if len(char) == 1 and char.isdigit():
            return "digit_main", full_label_from_gnn_and_yolo(char, "digit_main")
        raise ValueError(
            f"_resolve_labels: cannot resolve operand char {char!r}. "
            "Expected a single decimal digit."
        )

    # Defensive guard: child-fill roles must never reach the builder.
    # The honesty guard upstream blocks them; this is defense-in-depth.
    if role in ("carry", "borrow", "partial", "answer"):
        raise ValueError(
            f"_resolve_labels: role {role!r} is a child-fill role and must never "
            "reach build_given_prior. Only scaffold (given=True) tokens are "
            "permitted. Check the caller — given_tokens() must be used, not "
            "expected_to_fill()."
        )

    raise ValueError(
        f"_resolve_labels: unrecognised role {role!r} (char={char!r}). "
        "Extend _resolve_labels or the exercise ontology."
    )


# ---------------------------------------------------------------------------
# Tile renderer
# ---------------------------------------------------------------------------

# Resolved once at first render; None means PIL default font is in use.
# _TTF_SENTINEL is a unique object used to distinguish "not yet resolved"
# from None (which means "resolved but unavailable").
_TTF_SENTINEL = object()
_TTF_PATH: object = _TTF_SENTINEL  # starts as sentinel; set to str or None after first call
_TTF_WARNED: bool = False


def _get_ttf() -> str | None:
    """Return the resolved DejaVuSans TTF path, or None for PIL default font.

    Resolution is attempted once via ``matplotlib.font_manager``; the result is
    cached in a module-level variable.  A single warning is logged when the
    TTF is unavailable so tests remain quiet.
    """
    global _TTF_PATH, _TTF_WARNED  # noqa: PLW0603
    if _TTF_PATH is not _TTF_SENTINEL:
        return _TTF_PATH  # type: ignore[return-value]
    # First call — resolve.
    try:
        from matplotlib import font_manager as fm  # noqa: PLC0415

        candidate = fm.findfont("DejaVu Sans")
        # findfont never raises; it returns a fallback path when the requested
        # font is absent. Accept only if the path contains "DejaVu" to avoid
        # silently using whatever random system font matplotlib fell back to.
        if "DejaVu" in candidate or "dejavu" in candidate.lower():
            _TTF_PATH = candidate
        else:
            _TTF_PATH = None
    except Exception:  # noqa: BLE001
        _TTF_PATH = None

    if _TTF_PATH is None and not _TTF_WARNED:
        log.warning(
            "given_prior_builder: DejaVuSans TTF not found via matplotlib; "
            "falling back to PIL default font for tile rendering."
        )
        _TTF_WARNED = True

    return _TTF_PATH  # type: ignore[return-value]


# Representative glyphs for structural roles drawn without a character.
_STRUCTURAL_CHARS: dict[str, str | None] = {
    "result_bar": None,   # drawn as a horizontal bar
    "div_bracket": None,  # drawn as an L-shape
}


def _render_tile(char: str, role: str) -> np.ndarray:
    """Render a 28x28 uint8 greyscale tile for one given-equation token.

    Dark glyph on white background.  For structural roles (``result_bar``,
    ``div_bracket``) a simple representative mark is drawn instead of a
    character — a horizontal bar or an L-shaped bracket — giving the visual
    stream non-blank content without requiring a font.

    Parameters
    ----------
    char:
        The character string from the :class:`~src.grading.exercises.ExpectedToken`.
        Used for digit / operator rendering; ignored for structural roles.
    role:
        The token role (``operand``, ``operator``, ``result_bar``,
        ``div_bracket``, etc.).
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    TILE_PX = 28

    if role == "result_bar":
        tile = Image.new("L", (TILE_PX, TILE_PX), 255)
        d = ImageDraw.Draw(tile)
        yc = TILE_PX // 2
        d.rectangle([2, yc - 1, TILE_PX - 3, yc + 1], fill=0)
        return np.array(tile, dtype=np.uint8)

    if role == "div_bracket":
        tile = Image.new("L", (TILE_PX, TILE_PX), 255)
        d = ImageDraw.Draw(tile)
        # Left vertical stroke.
        d.rectangle([3, 2, 5, TILE_PX - 3], fill=0)
        # Top horizontal stroke.
        d.rectangle([3, 2, TILE_PX - 3, 4], fill=0)
        return np.array(tile, dtype=np.uint8)

    # Character-bearing token (digit or operator).
    # `char` is the raw ExpectedToken.c value (e.g. "+", "3") — use it directly.
    display_char = char if char else "?"

    tile = Image.new("L", (TILE_PX, TILE_PX), 255)
    d = ImageDraw.Draw(tile)

    ttf_path = _get_ttf()
    if ttf_path is not None:
        try:
            font = ImageFont.truetype(ttf_path, size=max(8, int(TILE_PX * 0.85)))
            bb = d.textbbox((0, 0), display_char, font=font)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            ox = (TILE_PX - w) / 2 - bb[0]
            oy = (TILE_PX - h) / 2 - bb[1]
            d.text((ox, oy), display_char, font=font, fill=0)
            return np.array(tile, dtype=np.uint8)
        except Exception:  # noqa: BLE001
            pass  # fall through to default font

    # PIL default font fallback.
    try:
        font = ImageFont.load_default()
        bb = d.textbbox((0, 0), display_char, font=font)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        ox = (TILE_PX - w) / 2 - bb[0]
        oy = (TILE_PX - h) / 2 - bb[1]
        d.text((ox, oy), display_char, font=font, fill=0)
    except Exception:  # noqa: BLE001
        # Absolute last resort: return a white tile (non-blank content not
        # available; the visual stream degrades gracefully).
        pass

    return np.array(tile, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_given_prior(
    bank_entry: BankExercise,
    calibration: GridCalibration,
) -> list[GivenNode]:
    """Build the given-equation prior node list from a :class:`BankExercise`.

    Consumes ONLY ``bank_entry.given_tokens()`` — the scaffold cells grading engine
    pre-prints at exercise start (operands, operator, result bar).  Never reads
    expected-to-fill / answer / carry / borrow / partial values.

    Honesty guard: asserts every consumed token has ``given is True``.  Raises
    :class:`ValueError` if a non-given token leaks in (programming error in the
    exercise definition or in the caller).

    Parameters
    ----------
    bank_entry:
        The derived exercise description.  Must have at least one given token.
    calibration:
        The pixel-space calibration for this exercise's 512x512 canvas.

    Returns
    -------
    list[GivenNode]
        One :class:`~src.inference.given_prior.GivenNode` per given token, in
        the same order as ``given_tokens()`` returns them.
    """
    given = bank_entry.given_tokens()
    nodes: list[GivenNode] = []

    for token in given:
        # Honesty guard: every token from given_tokens() MUST be given=True.
        # This protects against exercises where a mis-tagged token (given=False)
        # slips into the scaffold set, which would inject answer values into the
        # prior and corrupt grading.
        if not token.given:
            raise ValueError(
                f"build_given_prior: token {token!r} has given=False but was "
                "returned by given_tokens(). This indicates a bug in the exercise "
                "definition. Only scaffold (given=True) tokens are permitted here. "
                "Never call expected_to_fill() to build the prior."
            )

        # Map engine cell coordinates to 512-space pixel rect.
        left, top, right, bottom = calibration.cell_to_pixels(token.x, token.y)
        x0 = int(left)
        y0 = int(top)
        x1 = int(right)
        y1 = int(bottom)

        # Resolve ontology labels from char + role.
        coarse_label, fine_label = _resolve_labels(token.c, token.role)

        # Render the font tile.
        tile = _render_tile(token.c, token.role)

        nodes.append(
            GivenNode(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                coarse_label=coarse_label,
                fine_label=fine_label,
                tile=tile,
            )
        )

    return nodes
