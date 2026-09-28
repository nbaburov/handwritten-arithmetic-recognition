"""Edge-type vocabulary for the dense N×N graph used by the GNN.

This module is the single source of truth for the 8-token edge-type vocabulary.
It maps pairs of YOLO coarse class names to edge-type indices (0–7).

All classification is done via a pre-built lookup dict so that ``pair_edge_type``
runs in O(1) regardless of the number of classes.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, FrozenSet, Tuple

from ..core.ontology import YOLO_CLASS_NAMES

# ---------------------------------------------------------------------------
# Vocabulary size
# ---------------------------------------------------------------------------

NUM_EDGE_TYPES: int = 8

# ---------------------------------------------------------------------------
# Role sets used to classify pairs
# ---------------------------------------------------------------------------

_IS_DIGIT: FrozenSet[str] = frozenset({"digit_main", "digit_carry", "digit_borrow"})
_IS_ANNOTATION: FrozenSet[str] = frozenset({"digit_carry", "digit_borrow"})
_IS_STRUCTURAL: FrozenSet[str] = frozenset({"result_bar", "divide_bracket"})

# ---------------------------------------------------------------------------
# Internal helper — classify a *sorted* (canonical) pair
# ---------------------------------------------------------------------------


def _classify_sorted_pair(a: str, b: str) -> int:
    """Return the edge-type index for a pair that is already in canonical order.

    Canonical order means ``a <= b`` (string comparison), so the function only
    needs to handle each unordered pair once.

    Args:
        a: YOLO coarse class name (lexicographically ≤ b).
        b: YOLO coarse class name (lexicographically ≥ a).

    Returns:
        Edge-type index in [0, NUM_EDGE_TYPES).
    """
    # 0 — both are digit_main
    if a == "digit_main" and b == "digit_main":
        return 0

    # 1 — one is digit_main, other is a carry/borrow annotation
    if (a == "digit_main" and b in _IS_ANNOTATION) or (
        b == "digit_main" and a in _IS_ANNOTATION
    ):
        return 1

    # 2 — both are annotations (carry or borrow)
    if a in _IS_ANNOTATION and b in _IS_ANNOTATION:
        return 2

    # 3 — one is any digit, other is operator
    if (a in _IS_DIGIT and b == "operator") or (b in _IS_DIGIT and a == "operator"):
        return 3

    # 4 — one is any digit, other is result_bar
    if (a in _IS_DIGIT and b == "result_bar") or (b in _IS_DIGIT and a == "result_bar"):
        return 4

    # 5 — one is any digit, other is divide_bracket
    if (a in _IS_DIGIT and b == "divide_bracket") or (
        b in _IS_DIGIT and a == "divide_bracket"
    ):
        return 5

    # 6 — operator paired with anything structural or another operator
    _op_neighbours = _IS_STRUCTURAL | {"operator"}
    if (a == "operator" and b in _op_neighbours) or (
        b == "operator" and a in _op_neighbours
    ):
        return 6

    # 7 — everything else
    return 7


# ---------------------------------------------------------------------------
# Pre-build O(1) lookup table at import time
# ---------------------------------------------------------------------------

_EDGE_TYPE_LOOKUP: Dict[Tuple[str, str], int] = {}

for _a, _b in product(YOLO_CLASS_NAMES, repeat=2):
    _key: Tuple[str, str] = (_a, _b) if _a <= _b else (_b, _a)
    if _key not in _EDGE_TYPE_LOOKUP:
        _EDGE_TYPE_LOOKUP[_key] = _classify_sorted_pair(*_key)
    # Store both orderings so the public function needs no extra sort
    _EDGE_TYPE_LOOKUP[(_a, _b)] = _EDGE_TYPE_LOOKUP[_key]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pair_edge_type(src_role: str, dst_role: str) -> int:
    """Map a pair of YOLO coarse labels to an edge-type index in [0, NUM_EDGE_TYPES).

    The function is symmetric: ``pair_edge_type(a, b) == pair_edge_type(b, a)``.
    Unknown labels fall back to edge-type 7 (``other``).

    Args:
        src_role: YOLO coarse class name of the source node.
        dst_role: YOLO coarse class name of the destination node.

    Returns:
        Edge type index in [0, NUM_EDGE_TYPES).
    """
    result = _EDGE_TYPE_LOOKUP.get((src_role, dst_role))
    if result is not None:
        return result
    # Graceful fallback for labels not in the known vocabulary
    return NUM_EDGE_TYPES - 1
