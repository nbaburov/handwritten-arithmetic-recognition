"""The single pixel<->grid transform and recognizer-to-guide reconciliation.

This module owns the only place the demo converts between 512-space pixel
coordinates (where the recognizer and the Konva canvas live) and the grading engine's
integer W x H grid (where the engine grades). The scaffold renderer, the faint
writing-guide grid, and this mapper all consume one :class:`GridCalibration`, so
a clean drawing makes the recognizer's grid assignment and the visible guide
cells agree by construction.

Two orientation facts drive every conversion:

* the grading engine's grid is bottom-origin: ``y`` increases upward, so the top row of
  the scaffold has the largest ``y``.
* Our recognizer rows are top-down: ``row`` 0 is the topmost row and increases
  downward.

The calibration inverts ``y`` once, in one place, so callers never repeat the
sign flip. The ink's real 512-space bbox position is the source of truth for
grid assignment: :func:`tokens_to_grid` projects each token's bbox center
through :meth:`GridCalibration.pixel_cell_at` to determine which guide cell the
ink physically occupies, then converts that to a grading engine ``(x, y)`` via
:meth:`GridCalibration.grid_coords_for`. This makes placement robust to
partial drawings where the GNN's scene-relative ``row`` / ``grid_col`` cluster
indices would be offset when the child draws only a subset of columns (e.g. only
the answer digits, not the operand columns the scaffold pre-prints). The
recognizer's ``row`` / ``col`` fields are kept on every token for display and
the soft ``reconcile_with_guide`` cross-check, but they are not used for grading
cell assignment.

This module depends only on :mod:`src.grading.types` and
:mod:`src.inference.palette` (for the glyph character). It never calls the
grading engine client and contains no HTTP, grading, or app logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.grading.types import GridRect, GridToken
from src.inference.palette import display_glyph

# 512-space is the fixed pipeline canvas size (see preprocessing.py). The demo
# renders the guide grid and the scaffold inside this square, so the mapper's
# default canvas geometry is also 512x512 unless a caller overrides it.
DEFAULT_CANVAS_PX: float = 512.0


@dataclass(frozen=True)
class GridCalibration:
    """The pixel<->grid affine transform shared by every demo renderer.

    The transform is axis-aligned: a single pixel pitch per axis plus the pixel
    origin of the *top-left* grid cell (recognizer ``row`` 0, recognizer
    ``col`` 0). ``grid_width`` / ``grid_height`` come from the create view-model;
    ``engine_x_left`` / ``engine_y_top`` are the grading engine grid coordinates of that same
    top-left cell, so recognizer ``(row, col)`` maps to grading engine ``(x, y)`` via
    ``x = engine_x_left + col`` and (because grading engine ``y`` is bottom-origin)
    ``y = engine_y_top - row``.

    Keeping ``engine_x_left`` / ``engine_y_top`` explicit lets WS5 absorb the captured
    create-vs-event offset (the view-model tokens sit at ``y`` 4-6 while the
    evaluate echo sits at ``y`` 8-10) by adjusting two constants, never the math.
    """

    grid_width: int
    grid_height: int
    cell_w_px: float
    cell_h_px: float
    origin_x_px: float
    origin_y_px: float
    engine_x_left: int = 0
    engine_y_top: int = 0
    invert_y: bool = True

    # -- recognizer (row, col) <-> grading engine (x, y) -----------------------------

    def grid_coords_for(self, row: int, col: int) -> tuple[int, int]:
        """Map a recognizer ``(row, col)`` to grading engine grid ``(x, y)``."""
        x = self.engine_x_left + int(col)
        if self.invert_y:
            y = self.engine_y_top - int(row)
        else:
            y = self.engine_y_top + int(row)
        return x, y

    def row_col_for(self, x: int, y: int) -> tuple[int, int]:
        """Inverse of :meth:`grid_coords_for`: grading engine ``(x, y)`` -> ``(row, col)``."""
        col = int(x) - self.engine_x_left
        if self.invert_y:
            row = self.engine_y_top - int(y)
        else:
            row = int(y) - self.engine_y_top
        return row, col

    # -- grading engine grid <-> 512-space pixels -------------------------------------

    def cell_to_pixels(self, x: int, y: int) -> tuple[float, float, float, float]:
        """Pixel rect ``(left, top, right, bottom)`` for grading engine cell ``(x, y)``.

        ``y`` is bottom-origin, so a larger ``y`` sits higher on the canvas
        (smaller pixel ``top``). The returned rect is the single guide cell.
        """
        row, col = self.row_col_for(x, y)
        left = self.origin_x_px + col * self.cell_w_px
        top = self.origin_y_px + row * self.cell_h_px
        return left, top, left + self.cell_w_px, top + self.cell_h_px

    def pixel_cell_at(self, px: float, py: float) -> tuple[int, int]:
        """The guide cell a pixel point falls in, as recognizer ``(row, col)``.

        Used by :func:`reconcile_with_guide` to ask which guide square a token's
        bbox center physically sits in. Result is clamped to the grid so ink that
        drifts past an edge maps to the nearest border cell rather than off-grid.
        """
        col = int((px - self.origin_x_px) // self.cell_w_px)
        row = int((py - self.origin_y_px) // self.cell_h_px)
        col = max(0, min(self.grid_width - 1, col))
        row = max(0, min(self.grid_height - 1, row))
        return row, col


@dataclass(frozen=True)
class ReconciledToken:
    """A recognizer token paired with the guide cell its ink physically occupies.

    ``recognizer_row`` / ``recognizer_col`` is the model's answer and the source
    of truth; ``guide_row`` / ``guide_col`` is where the bbox center sits in the
    faint writing grid. ``disagreement`` is ``True`` when those differ. The model
    answer is never changed; this only surfaces a soft warning + overlay.
    """

    token_id: str
    char: str
    recognizer_row: int
    recognizer_col: int
    guide_row: int
    guide_col: int
    grid_x: int
    grid_y: int
    disagreement: bool


@dataclass(frozen=True)
class CalibrationReport:
    """Result of cross-checking sent tokens against the grading engine's echoed positions.

    ``aligned`` is ``True`` when every matched token's sent grid coordinate equals
    the position grading engine echoed for it (zero offset). A consistent non-zero
    ``(dx, dy)`` across all matched tokens is a *systematic* offset and sets
    ``systematic_offset`` to that pair; inconsistent per-token offsets leave it
    ``None`` and set ``aligned`` ``False`` regardless. ``matched`` / ``unmatched``
    count tokens that could / could not be paired by id to an echoed element.
    """

    aligned: bool
    matched: int
    unmatched: int
    systematic_offset: tuple[int, int] | None = None
    per_token_offsets: dict[str, tuple[int, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# View-model parsing
# ---------------------------------------------------------------------------


def interaction_answer(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the single ARITHMETIC interaction's ``ans`` block from a create view-model.

    Stable public name for :func:`_interaction_answer`; will not be renamed.
    """
    return _interaction_answer(view_model)


def _interaction_answer(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the single ARITHMETIC interaction's ``ans`` block from a view-model.

    The create view-model nests the grid spec under
    ``view.elements[QUESTION].interactions[0].ans`` (carrying ``width`` /
    ``height`` / ``tokens``). A fabricated view-model (the mock builds three of
    the four bank exercises directly) may instead expose ``ans`` / ``width`` /
    ``height`` at the top level; both shapes are accepted so the mapper does not
    care which path produced the view-model.
    """
    if not view_model:
        return {}
    # Fabricated flat shape: width/height (and optionally tokens) at top level.
    if "width" in view_model and "height" in view_model:
        return dict(view_model)
    if "ans" in view_model and isinstance(view_model["ans"], Mapping):
        return dict(view_model["ans"])
    view = view_model.get("view")
    if not isinstance(view, Mapping):
        return {}
    for element in view.get("elements", []):
        if not isinstance(element, Mapping):
            continue
        for interaction in element.get("interactions", []):
            if isinstance(interaction, Mapping) and isinstance(interaction.get("ans"), Mapping):
                return dict(interaction["ans"])
    return {}


def build_calibration(
    view_model: Mapping[str, Any],
    canvas_geometry: Mapping[str, Any] | None = None,
) -> GridCalibration:
    """Build the shared :class:`GridCalibration` from a create view-model.

    ``view_model`` supplies the grid ``width`` / ``height`` (and, for the real
    capture, the grading engine grid origin of the scaffold tokens). ``canvas_geometry``
    supplies the pixel frame the demo draws into; it accepts:

    * ``canvas_px`` (square side, default 512), or ``canvas_w_px`` /
      ``canvas_h_px`` for a non-square frame;
    * optional ``origin_x_px`` / ``origin_y_px`` (top-left pixel of the grid,
      default 0) and ``cell_w_px`` / ``cell_h_px`` to override the derived pitch;
    * optional ``engine_x_left`` / ``engine_y_top`` to pin the grading engine grid coordinate
      of the top-left cell (defaults derived from the view-model token extents).

    The default pitch divides the available pixel span evenly across the grid, so
    one drawn cell is one guide square and one grading engine cell.
    """
    geom: dict[str, Any] = dict(canvas_geometry or {})
    answer = _interaction_answer(view_model)

    grid_width = int(geom.get("grid_width", answer.get("width", 1) or 1))
    grid_height = int(geom.get("grid_height", answer.get("height", 1) or 1))
    grid_width = max(1, grid_width)
    grid_height = max(1, grid_height)

    canvas_px = float(geom.get("canvas_px", DEFAULT_CANVAS_PX))
    canvas_w_px = float(geom.get("canvas_w_px", canvas_px))
    canvas_h_px = float(geom.get("canvas_h_px", canvas_px))

    origin_x_px = float(geom.get("origin_x_px", 0.0))
    origin_y_px = float(geom.get("origin_y_px", 0.0))

    cell_w_px = float(geom.get("cell_w_px", (canvas_w_px - origin_x_px) / grid_width))
    cell_h_px = float(geom.get("cell_h_px", (canvas_h_px - origin_y_px) / grid_height))

    engine_x_left, engine_y_top = _default_engine_origin(answer, grid_width, grid_height)
    engine_x_left = int(geom.get("engine_x_left", engine_x_left))
    engine_y_top = int(geom.get("engine_y_top", engine_y_top))
    invert_y = bool(geom.get("invert_y", True))

    return GridCalibration(
        grid_width=grid_width,
        grid_height=grid_height,
        cell_w_px=cell_w_px,
        cell_h_px=cell_h_px,
        origin_x_px=origin_x_px,
        origin_y_px=origin_y_px,
        engine_x_left=engine_x_left,
        engine_y_top=engine_y_top,
        invert_y=invert_y,
    )


def _default_engine_origin(
    answer: Mapping[str, Any], grid_width: int, grid_height: int
) -> tuple[int, int]:
    """Derive the grading engine grid coords of the top-left cell from the view-model.

    When the view-model carries scaffold ``tokens`` with grid coords, the left
    column is the minimum ``x`` and the top row is the maximum ``y`` (bottom-origin
    grid). AUTOSHOW tokens are decorative UI helpers added by the grading engine's renderer
    that do not correspond to scaffold input cells; they are excluded so they cannot
    skew the origin toward a lower-than-expected ``y``. Absent non-AUTOSHOW tokens,
    fall back to ``x_left = 0`` and ``y_top = height - 1``, which keeps a fabricated
    grid bottom-origin with row 0 at the top.
    """
    tokens = answer.get("tokens")
    if isinstance(tokens, Sequence) and tokens:
        non_autoshow = [
            t for t in tokens
            if isinstance(t, Mapping)
            and "x" in t
            and "y" in t
            and "AUTOSHOW" not in (t.get("tags") or [])
        ]
        if non_autoshow:
            xs = [int(t["x"]) for t in non_autoshow]
            ys = [int(t["y"]) for t in non_autoshow]
            return min(xs), max(ys)
        # All tokens were AUTOSHOW; fall back so the grid is at least anchored to the scaffold.
        plain = [t for t in tokens if isinstance(t, Mapping) and "x" in t and "y" in t]
        if plain:
            return min(int(t["x"]) for t in plain), max(int(t["y"]) for t in plain)
    return 0, max(0, grid_height - 1)


# ---------------------------------------------------------------------------
# Token -> grid
# ---------------------------------------------------------------------------


def token_id_for(token: Mapping[str, Any], index: int) -> str:
    """Stable id for a token: uses the token's own id or a synthetic ``T{index}``.

    The synthetic id lets grading engine echo it back so :func:`verify_calibration`
    can pair the echo to the sent token. Public API; callers outside this module
    should use this rather than the private alias below.
    """
    raw = token.get("id")
    return str(raw) if raw not in (None, "") else f"T{index}"


# Internal alias kept for any remaining internal call-sites.
_token_id = token_id_for


def _token_char(token: Mapping[str, Any]) -> str:
    """The engine grid character for a token, via :func:`palette.display_glyph`.

    ``result_bar`` returns ``""`` (same as ``div_bracket``): sending the bar's
    multi-dash glyph to a single grid cell produces spurious ERROR/MISSING
    against the bank's ``"_"`` char. Skipping it lets the GIVEN scaffold handle
    the bar, matching real grading engine behaviour.

    ``carry_N`` / ``borrow_N`` labels encode a digit with a role prefix;
    ``display_glyph`` has no entry for these, so we extract the bare digit
    directly (e.g. ``carry_1`` -> ``"1"``).
    """
    from src.core.ontology import parse_flattened_label  # local to avoid circular at module level
    label = str(token.get("label", ""))
    if label == "result_bar":
        return ""
    role, detail = parse_flattened_label(label)
    if role in ("carry", "borrow") and detail is not None:
        return detail
    return display_glyph(label)


def _token_grid_col(token: Mapping[str, Any]) -> int:
    """The recognizer column for a token, preferring ``grid_col`` over ``col``.

    The assembler emits both ``col`` (the within-row ordinal position) and
    ``grid_col`` (the geometric ``col_cluster_id``). Grid assignment uses the
    geometric column so a token lands in the correct place-value column even when
    earlier columns are blank; ``col`` is a fallback for tokens without it.
    """
    if "grid_col" in token and token["grid_col"] is not None:
        return int(token["grid_col"])
    return int(token.get("col", 0))


def tokens_to_grid(
    assembled_tokens: Sequence[Mapping[str, Any]],
    calibration: GridCalibration,
    replace_cells: "set[tuple[int, int]] | None" = None,
    overflow_cells: "set[tuple[int, int]] | None" = None,
) -> list[GridToken]:
    """Map recognizer tokens to grading engine :class:`GridToken` list.

    Each token's engine ``(x, y)`` is derived from its ink's 512-space bbox center
    via :meth:`GridCalibration.pixel_cell_at` -> :meth:`GridCalibration.grid_coords_for`.
    Using the physical pixel position rather than the GNN's scene-relative
    ``row`` / ``grid_col`` cluster indices makes grading robust to partial
    drawings: if the child only draws the answer digits (columns 2-3 of the
    guide) the GNN assigns cluster_id 0,1 to those two columns and the
    old index-based approach would map them to engine x=12,13 instead of the
    correct 14,15. The bbox-first path projects through the same transform
    the scaffold renderer uses, so correctly placed ink always lands on the
    correct engine cell regardless of how many columns the child drew.

    ``replace_cells`` are the engine ``(x, y)`` cells that hold borrow/replacement
    values in column subtraction. Ink landing on one of these is emitted with
    ``type="REPLACE"`` (grading engine grades a borrow sent as a plain SYMBOL as
    wrong), and multiple glyphs that land on the same replace cell are
    concatenated left-to-right into one value (so a hand-written ``1`` then ``6``
    in the units borrow box becomes the single cell value ``16`` engine expects).

    ``overflow_cells`` are the engine ``(x, y)`` cells that hold carry values in
    column addition and multiplication. Ink landing on one of these is emitted
    with ``type="OVERFLOW"`` (grading engine grades a carry sent as a plain SYMBOL
    as wrong). Multiple glyphs on the same overflow cell are concatenated
    left-to-right (same logic as replace cells).

    **Cluster-aware grouping for typed cells (replace + overflow only):**
    When a child writes a two-digit borrow/carry value (e.g. "16") wide enough
    that the two glyphs straddle a cell boundary, the pixel-floor projection
    would send them to different adjacent cells, splitting "16" into two wrong
    single-digit values. To rescue this, glyphs whose pixel-floor cell is in or
    adjacent (same row) to a typed area, and which share the same recognizer
    cluster key ``(int(row), grid_col)``, are grouped together and their centroid
    is snapped to the nearest typed cell within a tolerance of 1.5 cell widths.
    Glyphs whose centroid is too far from any typed cell fall back to their
    pixel-floor cell as a plain SYMBOL. This logic is scoped to the typed area
    only; answer/operand/bar binning is exactly as before.

    **Honesty constraints (never violated):**
    - Char values come from ``_token_char`` and are never changed. A wrong digit
      goes as the wrong char; the mapper only decides placement and grouping.
    - No expected/solution values are read. Only geometry (pixel centroid) and
      the model's own ``row``/``grid_col`` cluster keys are used.
    - When both ``replace_cells`` and ``overflow_cells`` are empty, behaviour is
      identical to the pre-cluster implementation.

    The recognizer's ``row`` / ``col`` fields remain on the token dict for
    display and the :func:`reconcile_with_guide` soft cross-check; they are
    not used here for cell assignment. The char comes from
    :func:`palette.display_glyph`. ``tags`` is left empty (child-drawn).
    Tokens with an empty rendered char (e.g. ``div_bracket``) are skipped.
    """
    replace_cells = replace_cells or set()
    overflow_cells = overflow_cells or set()
    typed_cells: set[tuple[int, int]] = replace_cells | overflow_cells

    # First pass: project every renderable token to its pixel-floor engine cell,
    # keeping center-x/y and the model's cluster key for the typed-area path.
    # placed entry: (pixel_floor_cell, cx, cy, char, token_id, cluster_key)
    #   cluster_key = (int(row), int(grid_col)) from the model's own fields.
    placed: list[tuple[tuple[int, int], float, float, str, str, tuple[int, int]]] = []
    for index, token in enumerate(assembled_tokens):
        char = _token_char(token)
        if char == "":
            continue
        cx, cy = _bbox_center(token)
        guide_row, guide_col = calibration.pixel_cell_at(cx, cy)
        cell = calibration.grid_coords_for(guide_row, guide_col)
        cluster_key = (int(token.get("row", 0)), _token_grid_col(token))
        placed.append((cell, cx, cy, char, _token_id(token, index), cluster_key))

    # Fast path: no typed cells means replace_cells and overflow_cells are both
    # empty, so every projected token is a plain SYMBOL at its pixel-floor cell.
    # The replace/overflow bucket logic below is unreachable in this branch and
    # has been removed; the result is trivially identical to the original behaviour.
    if not typed_cells:
        return [
            GridToken(id=tid, c=char, x=cell[0], y=cell[1], tags=[])
            for cell, _cx, _cy, char, tid, _ck in placed
        ]

    # Determine which engine-row values correspond to typed cells so we can
    # identify the "typed area" (same engine y as any replace/overflow cell).
    # Safe because engine column-arithmetic exercises place borrow/carry rows strictly
    # ABOVE operand/answer rows (no shared y). Revisit (make cell-exact or
    # x-bounded) before adding any exercise where a typed cell shares a y-row
    # with an operand/answer cell.
    typed_ys: set[int] = {cell[1] for cell in typed_cells}

    # Tolerance for centroid-to-typed-cell snapping: 1.5 cell widths.
    snap_tolerance_x = 1.5 * calibration.cell_w_px
    snap_tolerance_y = 1.5 * calibration.cell_h_px

    # Helper: pixel center of an engine cell.
    def _cell_pixel_center(engine_cell: tuple[int, int]) -> tuple[float, float]:
        row, col = calibration.row_col_for(engine_cell[0], engine_cell[1])
        px = calibration.origin_x_px + col * calibration.cell_w_px + calibration.cell_w_px / 2.0
        py = calibration.origin_y_px + row * calibration.cell_h_px + calibration.cell_h_px / 2.0
        return px, py

    # Precompute pixel centers for all typed cells.
    typed_cell_centers: dict[tuple[int, int], tuple[float, float]] = {
        c: _cell_pixel_center(c) for c in typed_cells
    }

    # Partition: glyph is "typed-area" if its pixel-floor cell is a typed cell OR
    # shares the same engine y-row as any typed cell (catches spilled glyphs that
    # landed one cell to the left/right of a typed cell).
    typed_placed: list[tuple[tuple[int, int], float, float, str, str, tuple[int, int]]] = []
    plain_placed: list[tuple[tuple[int, int], float, float, str, str, tuple[int, int]]] = []
    for entry in placed:
        cell, cx, cy, char, tid, cluster_key = entry
        if cell in typed_cells or cell[1] in typed_ys:
            typed_placed.append(entry)
        else:
            plain_placed.append(entry)

    # Group typed-area glyphs by model cluster key.
    # cluster_groups: cluster_key -> list of (cx, cy, char, tid)
    cluster_groups: dict[tuple[int, int], list[tuple[float, float, str, str]]] = {}
    for _cell, cx, cy, char, tid, cluster_key in typed_placed:
        cluster_groups.setdefault(cluster_key, []).append((cx, cy, char, tid))

    # For each cluster, compute the centroid and snap to the nearest typed cell
    # within tolerance. Clusters too far from any typed cell fall back individually
    # to their per-glyph pixel-floor cells as plain SYMBOLs.
    replace_buckets_2: dict[tuple[int, int], list[tuple[float, str, str]]] = {}
    overflow_buckets_2: dict[tuple[int, int], list[tuple[float, str, str]]] = {}
    extra_symbols: list[GridToken] = []

    for cluster_key, glyphs in cluster_groups.items():
        centroid_x = sum(g[0] for g in glyphs) / len(glyphs)
        centroid_y = sum(g[1] for g in glyphs) / len(glyphs)

        # Find nearest typed cell by centroid distance.
        best_cell: tuple[int, int] | None = None
        best_dist = float("inf")
        for tc, (tcx, tcy) in typed_cell_centers.items():
            dist = ((centroid_x - tcx) ** 2 + (centroid_y - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_cell = tc

        within_tolerance = (
            best_cell is not None
            and abs(centroid_x - typed_cell_centers[best_cell][0]) <= snap_tolerance_x
            and abs(centroid_y - typed_cell_centers[best_cell][1]) <= snap_tolerance_y
        )

        if within_tolerance and best_cell is not None:
            # Snap the whole cluster to the nearest typed cell.
            if best_cell in replace_cells:
                for g in glyphs:
                    replace_buckets_2.setdefault(best_cell, []).append((g[0], g[2], g[3]))
            else:
                for g in glyphs:
                    overflow_buckets_2.setdefault(best_cell, []).append((g[0], g[2], g[3]))
        else:
            # Centroid too far from any typed cell: emit each glyph at its own
            # pixel-floor cell as a plain SYMBOL (fallback, never degrades).
            for cx_g, cy_g, char_g, tid_g in glyphs:
                gr, gc = calibration.pixel_cell_at(cx_g, cy_g)
                gx, gy = calibration.grid_coords_for(gr, gc)
                extra_symbols.append(GridToken(id=tid_g, c=char_g, x=gx, y=gy, tags=[]))

    # Assemble output.
    grid_tokens_out: list[GridToken] = []

    # Plain (non-typed-area) tokens: unchanged from original behaviour.
    for cell, cx, _cy, char, tid, _ck in plain_placed:
        grid_tokens_out.append(GridToken(id=tid, c=char, x=cell[0], y=cell[1], tags=[]))

    # Fallback symbols from typed-area clusters too far from any typed cell.
    grid_tokens_out.extend(extra_symbols)

    # Concatenate replace clusters left-to-right.
    for cell, glyphs in replace_buckets_2.items():
        glyphs.sort(key=lambda g: g[0])
        value = "".join(g[1] for g in glyphs)
        token_id = glyphs[0][2]
        grid_tokens_out.append(
            GridToken(id=token_id, c=value, x=cell[0], y=cell[1], tags=[], type="REPLACE")
        )

    # Concatenate overflow clusters left-to-right.
    for cell, glyphs in overflow_buckets_2.items():
        glyphs.sort(key=lambda g: g[0])
        value = "".join(g[1] for g in glyphs)
        token_id = glyphs[0][2]
        grid_tokens_out.append(
            GridToken(id=token_id, c=value, x=cell[0], y=cell[1], tags=[], type="OVERFLOW")
        )

    return grid_tokens_out


def _bbox_center(token: Mapping[str, Any]) -> tuple[float, float]:
    """The 512-space center of an assembled token's ``bbox`` ``[x0, y0, x1, y1]``."""
    bbox = token.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def reconcile_with_guide(
    assembled_tokens: Sequence[Mapping[str, Any]],
    calibration: GridCalibration,
) -> list[ReconciledToken]:
    """Cross-check each token's recognizer cell against the guide cell its ink occupies.

    For every token, the recognizer's ``(row, grid_col)`` is the answer (and the
    grid coords sent to grading engine). Separately, the bbox center is projected to
    the guide cell it physically sits in via :meth:`GridCalibration.pixel_cell_at`.
    ``disagreement`` is set when the two cells differ. The model is never moved to
    the guide cell; this only feeds the soft warning + glyph overlay.

    Tokens without a renderable char are skipped (nothing to overlay or warn on),
    matching :func:`tokens_to_grid` so the two lists stay index-comparable by id.
    """
    reconciled: list[ReconciledToken] = []
    for index, token in enumerate(assembled_tokens):
        char = _token_char(token)
        if char == "":
            continue
        rec_row = int(token.get("row", 0))
        rec_col = _token_grid_col(token)
        cx, cy = _bbox_center(token)
        guide_row, guide_col = calibration.pixel_cell_at(cx, cy)
        grid_x, grid_y = calibration.grid_coords_for(rec_row, rec_col)
        reconciled.append(
            ReconciledToken(
                token_id=_token_id(token, index),
                char=char,
                recognizer_row=rec_row,
                recognizer_col=rec_col,
                guide_row=guide_row,
                guide_col=guide_col,
                grid_x=grid_x,
                grid_y=grid_y,
                disagreement=(rec_row != guide_row or rec_col != guide_col),
            )
        )
    return reconciled


# ---------------------------------------------------------------------------
# Grid rect -> pixels (popup placement)
# ---------------------------------------------------------------------------


def grid_rect_to_pixels(
    grid_rect: GridRect,
    calibration: GridCalibration,
) -> tuple[float, float, float, float]:
    """Convert a grading engine :class:`GridRect` to a 512-space pixel rect.

    grading engine ``targetPositions`` rects are bottom-origin with exclusive upper
    bounds: a single cell at grid ``(x, y)`` echoes ``top = y + 1`` (exclusive)
    and ``bottom = y`` (inclusive), and a 2-wide number at ``left = 14`` echoes
    ``right = 16`` (exclusive). The inclusive cell span is therefore columns
    ``left`` through ``right - 1`` and rows ``bottom`` through ``top - 1``. The
    returned rect is ``(left_px, top_px, right_px, bottom_px)`` with ``top_px`` <
    ``bottom_px`` (pixel y grows downward), ready for a Konva ``Rect``.
    """
    # Inclusive cell corners after dropping the grading engine's exclusive upper bounds.
    left_x = grid_rect.left
    right_x = max(grid_rect.left, grid_rect.right - 1)
    high_y = max(grid_rect.top, grid_rect.bottom) - 1  # topmost inclusive row
    low_y = min(grid_rect.top, grid_rect.bottom)  # bottommost inclusive row
    high_y = max(high_y, low_y)

    # Topmost cell sits highest on the canvas (largest grid y -> smallest px top).
    top_left = calibration.cell_to_pixels(left_x, high_y)
    bottom_right = calibration.cell_to_pixels(right_x, low_y)

    left_px = min(top_left[0], bottom_right[0])
    right_px = max(top_left[2], bottom_right[2])
    top_px = min(top_left[1], bottom_right[1])
    bottom_px = max(top_left[3], bottom_right[3])
    return left_px, top_px, right_px, bottom_px


# ---------------------------------------------------------------------------
# Calibration verification against the grading engine's echoed positions
# ---------------------------------------------------------------------------


def _element_position_by_token(
    eval_elements: Sequence[Any],
) -> dict[str, tuple[int, int]]:
    """Map each echoed token id to the grid ``(x, y)`` of the element it landed in.

    grading engine echoes the submitted token ids in each element's ``symbolIdList``
    and reports that element's grid ``position``. Accepts either parsed
    ``EvalElement`` objects (``symbol_id_list`` / ``position`` attributes) or raw
    API dicts (``symbolIdList`` / ``position{x,y}``), so callers can verify before
    or after parsing.
    """
    mapping: dict[str, tuple[int, int]] = {}
    for element in eval_elements:
        ids, position = _element_ids_and_position(element)
        if position is None:
            continue
        for token_id in ids:
            mapping[str(token_id)] = position
    return mapping


def _element_ids_and_position(element: Any) -> tuple[list[str], tuple[int, int] | None]:
    """Extract ``(symbol ids, position)`` from a parsed or raw element."""
    if isinstance(element, Mapping):
        ids = [str(s) for s in element.get("symbolIdList", [])]
        raw_pos = element.get("position")
        if isinstance(raw_pos, Mapping) and "x" in raw_pos and "y" in raw_pos:
            return ids, (int(raw_pos["x"]), int(raw_pos["y"]))
        return ids, None
    ids = [str(s) for s in getattr(element, "symbol_id_list", [])]
    pos = getattr(element, "position", None)
    if pos is None:
        return ids, None
    return ids, (int(pos[0]), int(pos[1]))


def verify_calibration(
    sent_tokens: Sequence[GridToken],
    eval_elements: Sequence[Any],
) -> CalibrationReport:
    """Detect a systematic px<->grid offset using the grading engine's echoed positions.

    Each sent token carries the grid ``(x, y)`` we computed; grading engine echoes the
    token id inside the element it matched, at that element's grid ``position``.
    For every token paired by id, the offset is ``sent - echoed``. When all matched
    offsets are equal and zero, the calibration is aligned. When all matched
    offsets are equal and non-zero, that pair is reported as a ``systematic_offset``
    (the constant WS5 would fold into ``engine_x_left`` / ``engine_y_top``). Mixed offsets
    are inconsistent: ``aligned`` is ``False`` and ``systematic_offset`` is
    ``None``.

    A token with no echoed match counts toward ``unmatched`` and does not affect
    alignment (grading engine drops tokens it cannot place, for example an extra digit
    in a full cell).
    """
    echoed = _element_position_by_token(eval_elements)
    per_token: dict[str, tuple[int, int]] = {}
    unmatched = 0
    for token in sent_tokens:
        position = echoed.get(token.id)
        if position is None:
            unmatched += 1
            continue
        per_token[token.id] = (token.x - position[0], token.y - position[1])

    matched = len(per_token)
    if matched == 0:
        return CalibrationReport(
            aligned=False,
            matched=0,
            unmatched=unmatched,
            systematic_offset=None,
            per_token_offsets={},
        )

    distinct = set(per_token.values())
    if len(distinct) == 1:
        offset = next(iter(distinct))
        return CalibrationReport(
            aligned=(offset == (0, 0)),
            matched=matched,
            unmatched=unmatched,
            systematic_offset=None if offset == (0, 0) else offset,
            per_token_offsets=per_token,
        )

    return CalibrationReport(
        aligned=False,
        matched=matched,
        unmatched=unmatched,
        systematic_offset=None,
        per_token_offsets=per_token,
    )
