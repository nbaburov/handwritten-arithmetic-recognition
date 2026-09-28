"""Label sidecar schema and loader for the iter6 evaluation harness.

Public API
----------
EQUATION_KINDS : frozenset[str]
    Canonical set of valid equation kind strings.

SCENE_CASES : frozenset[str]
    Canonical set of valid scene_case strings, sourced from the generation
    ``SceneCase`` enum (single source of truth).

ERROR_KINDS : frozenset[str]
    Canonical set of valid ``error_kind`` tags for as-drawn error scenes.

SymbolLabel : dataclass (frozen)
    A single symbol ground-truth entry: flattened label + bounding box.

SampleLabel : dataclass (frozen)
    Ground truth for one sample image.

load_label_sidecars(samples_dir) -> Dict[str, SampleLabel]
    Scan a directory for ``*.label.json`` files, parse and validate each,
    return a dict keyed by filename stem (without ``.label.json``).

Schema (``schema_version = 1``)
--------------------------------
{
    "schema_version": 1,
    "sample": "<stem>",
    "equation_kind": "<addition|subtraction|multiplication|division>",
    "scene_case": "<one of SCENE_CASES>",   # optional; absent or null => None
    "error_kind": "<one of ERROR_KINDS>",   # optional; absent or null => None
    "symbols": [                             # optional key; absent => None
        {"flattened": "<label>", "bbox_px": [x1, y1, x2, y2]}
    ]
}

Validation rules
----------------
- ``schema_version`` must equal 1.
- ``sample`` must match the filename stem.
- ``equation_kind`` must be in EQUATION_KINDS.
- ``scene_case``, when present and non-null, must be in SCENE_CASES; an absent
  key or an explicit ``null`` both map to None.
- ``error_kind``, when present and non-null, must be in ERROR_KINDS; an absent
  key or an explicit ``null`` both map to None.  It tags how a scene is wrong
  as-drawn (the system recognises errors, it never corrects them).
- Each ``symbols[].flattened`` must be in the canonical 36-class ontology.
- Each ``bbox_px`` must satisfy ``0 <= x1 < x2 <= 512`` and
  ``0 <= y1 < y2 <= 512``.
- Empty ``symbols: []`` is allowed and maps to an empty tuple (not None).
- Absent ``symbols`` key maps to ``symbols=None``.
- The ``provenance`` block (present on showcase-imported sidecars) is silently
  ignored.

All validation failures raise ``ValueError`` with the file path in the message.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.core.ontology import stage2_labels_ordered
from src.generation.layouts_types import SceneCase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EQUATION_KINDS: frozenset[str] = frozenset(
    {"addition", "subtraction", "multiplication", "division", "bare_digits"}
)

# Single source of truth: built from the generation SceneCase enum so that
# adding a new SceneCase automatically extends the valid set here.
SCENE_CASES: frozenset[str] = frozenset(sc.value for sc in SceneCase)

# Canonical tags for as-drawn error scenes. The system recognises how a scene
# is wrong as drawn (wrong result, misplaced carry, etc.); it never corrects
# the arithmetic. ``other`` is the catch-all for errors outside this list.
ERROR_KINDS: frozenset[str] = frozenset(
    {
        "wrong_result",
        "wrong_operator",
        "wrong_carry",
        "spurious_carry",
        "missing_carry",
        "wrong_borrow",
        "spurious_borrow",
        "missing_borrow",
        "wrong_digit",
        "other",
    }
)

_SCHEMA_VERSION: int = 1
_CANVAS_SIZE: float = 512.0

# Build once at import time; lookup is O(1).
_VALID_FLATTENED: frozenset[str] = frozenset(stage2_labels_ordered())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolLabel:
    """Ground-truth for a single handwritten symbol in one sample image.

    Parameters
    ----------
    flattened:
        36-class fine label string (e.g. ``"main_3"``, ``"op_plus"``).
    bbox_px:
        Bounding box in pixel coordinates on the 512x512 canvas as
        ``(x1, y1, x2, y2)`` with ``x1 < x2`` and ``y1 < y2``.
    """

    flattened: str
    bbox_px: Tuple[float, float, float, float]


@dataclass(frozen=True)
class SampleLabel:
    """Ground truth for one sample image.

    Parameters
    ----------
    schema_version:
        Must be 1 for this module; reserved for future migrations.
    sample:
        Stem of the label file (without ``.label.json``), must match the
        filename it was loaded from.
    equation_kind:
        Equation category; must be in ``EQUATION_KINDS``.
    symbols:
        Tuple of ``SymbolLabel`` entries, or ``None`` when the ``symbols``
        key is absent from the sidecar JSON.  An empty tuple (``()``) is
        distinct from ``None`` and indicates zero-symbol ground truth.
    scene_case:
        Optional sub-case identifier from the ``SceneCase`` enum (e.g.
        ``"addition_dense_carries"``), or ``None`` when the key is absent or
        explicitly ``null``.  Used for per-case diagnostic aggregation in the
        eval harness.  When present and non-null, must be a member of
        ``SCENE_CASES``.
    error_kind:
        Optional tag describing how the scene is wrong as-drawn (e.g.
        ``"missing_carry"``, ``"wrong_result"``), or ``None`` when the key is
        absent or explicitly ``null``.  Used to break out as-drawn error scenes
        in the eval harness.  The system recognises the error; it never corrects
        it.  When present and non-null, must be a member of ``ERROR_KINDS``.
    """

    schema_version: int
    sample: str
    equation_kind: str
    symbols: Optional[Tuple[SymbolLabel, ...]]
    scene_case: Optional[str] = field(default=None)
    error_kind: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_bbox(raw: list, file_path: Path) -> Tuple[float, float, float, float]:
    """Parse and validate a ``bbox_px`` list from JSON.

    Parameters
    ----------
    raw:
        List of four numeric values ``[x1, y1, x2, y2]``.
    file_path:
        Source file used in error messages.

    Returns
    -------
    Tuple[float, float, float, float]
        Validated bounding box.

    Raises
    ------
    ValueError
        If the list length is wrong, coordinates violate ordering, or any
        coordinate falls outside ``[0, 512]``.
    """
    if len(raw) != 4:
        raise ValueError(
            f"{file_path}: bbox_px must have exactly 4 elements, got {len(raw)}"
        )
    x1, y1, x2, y2 = (float(v) for v in raw)
    if not (0.0 <= x1 < x2 <= _CANVAS_SIZE):
        raise ValueError(
            f"{file_path}: bbox_px x-coords must satisfy 0 <= x1 < x2 <= 512, "
            f"got x1={x1}, x2={x2}"
        )
    if not (0.0 <= y1 < y2 <= _CANVAS_SIZE):
        raise ValueError(
            f"{file_path}: bbox_px y-coords must satisfy 0 <= y1 < y2 <= 512, "
            f"got y1={y1}, y2={y2}"
        )
    return (x1, y1, x2, y2)


def _parse_symbol(raw: dict, file_path: Path) -> SymbolLabel:
    """Parse and validate a single symbol entry from JSON.

    Parameters
    ----------
    raw:
        Dict with ``flattened`` and ``bbox_px`` keys.
    file_path:
        Source file used in error messages.

    Returns
    -------
    SymbolLabel
        Validated symbol label.

    Raises
    ------
    ValueError
        If ``flattened`` is not in the canonical 36-class ontology or
        ``bbox_px`` fails validation.
    """
    flattened = raw.get("flattened", "")
    if flattened not in _VALID_FLATTENED:
        raise ValueError(
            f"{file_path}: unknown flattened label '{flattened}'; "
            f"must be one of the 36-class ontology"
        )
    bbox = _parse_bbox(raw.get("bbox_px", []), file_path)
    return SymbolLabel(flattened=flattened, bbox_px=bbox)


def _parse_sidecar(doc: dict, stem: str, file_path: Path) -> SampleLabel:
    """Parse and validate a complete sidecar JSON document.

    Parameters
    ----------
    doc:
        Parsed JSON dict.
    stem:
        Expected filename stem (without ``.label.json``).
    file_path:
        Source file used in error messages.

    Returns
    -------
    SampleLabel
        Validated sample label.

    Raises
    ------
    ValueError
        If any validation rule is violated.
    """
    # schema_version
    schema_version = doc.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"{file_path}: schema_version must be {_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )

    # sample name must match file stem
    sample = doc.get("sample", "")
    if sample != stem:
        raise ValueError(
            f"{file_path}: 'sample' field '{sample}' does not match "
            f"filename stem '{stem}'"
        )

    # equation_kind
    equation_kind = doc.get("equation_kind", "")
    if equation_kind not in EQUATION_KINDS:
        raise ValueError(
            f"{file_path}: equation_kind '{equation_kind}' is not valid; "
            f"must be one of {sorted(EQUATION_KINDS)}"
        )

    # scene_case — absent key (or explicit null) => None; present non-null but
    # invalid => ValueError.
    scene_case: Optional[str] = None
    raw_scene_case = doc.get("scene_case")
    if raw_scene_case is not None:
        if raw_scene_case not in SCENE_CASES:
            raise ValueError(
                f"{file_path}: scene_case '{raw_scene_case}' is not valid; "
                f"must be one of {sorted(SCENE_CASES)}"
            )
        scene_case = raw_scene_case

    # error_kind — absent key (or explicit null) => None; present non-null but
    # invalid => ValueError. Tags how the scene is wrong as-drawn.
    error_kind: Optional[str] = None
    raw_error_kind = doc.get("error_kind")
    if raw_error_kind is not None:
        if raw_error_kind not in ERROR_KINDS:
            raise ValueError(
                f"{file_path}: error_kind '{raw_error_kind}' is not valid; "
                f"must be one of {sorted(ERROR_KINDS)}"
            )
        error_kind = raw_error_kind

    # symbols — absent key => None; present key (even empty list) => tuple
    if "symbols" not in doc:
        symbols: Optional[Tuple[SymbolLabel, ...]] = None
    else:
        raw_symbols = doc["symbols"]
        symbols = tuple(_parse_symbol(s, file_path) for s in raw_symbols)

    return SampleLabel(
        schema_version=schema_version,
        sample=sample,
        equation_kind=equation_kind,
        symbols=symbols,
        scene_case=scene_case,
        error_kind=error_kind,
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_label_sidecars(samples_dir: Path) -> Dict[str, SampleLabel]:
    """Scan a directory for ``*.label.json`` files and load all of them.

    Only files whose names end with ``.label.json`` are considered.  The
    stem (everything before ``.label.json``) becomes the dict key.

    Parameters
    ----------
    samples_dir:
        Directory to scan (non-recursive).

    Returns
    -------
    Dict[str, SampleLabel]
        Mapping from filename stem to validated ``SampleLabel``.

    Raises
    ------
    ValueError
        If any sidecar file fails schema or content validation.  The error
        message always includes the absolute file path.
    FileNotFoundError
        If ``samples_dir`` does not exist.
    """
    result: Dict[str, SampleLabel] = {}
    for file_path in sorted(samples_dir.glob("*.label.json")):
        # Strip the compound extension ".label.json"
        stem = file_path.name[: -len(".label.json")]
        with file_path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        label = _parse_sidecar(doc, stem, file_path)
        result[stem] = label
    return result
