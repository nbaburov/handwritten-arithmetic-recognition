from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.ontology import parse_flattened_label

# ---------------------------------------------------------------------------
# OOD gate constants — no magic strings anywhere else in this file.
# ---------------------------------------------------------------------------

OOD_REASON_TOO_FEW_SYMBOLS: str = "too_few_symbols"
OOD_REASON_NO_DIGITS: str = "no_digits"
OOD_REASON_NO_STRUCTURAL_TOKEN: str = "no_structural_token"
# W10 new constants
OOD_REASON_TRUE_EMPTY: str = "true_empty"
OOD_REASON_SINGLE_NON_DIGIT: str = "single_non_digit"
# W1 new constants
OOD_REASON_HIGH_ENTROPY: str = "high_entropy_eq_type"
OOD_REASON_KIND_OVERRIDE_NO_OPERATOR: str = "kind_override_no_structural_operator"
OOD_REASON_OPERATOR_ONLY: str = "operator_only"
OOD_REASON_SINGLE_ROW: str = "single_row"
OOD_REASON_DECORATION_WITHOUT_EQUATION: str = "decoration_without_equation"
OOD_REASON_LOW_CONFIDENCE: str = "low_confidence"

OOD_EQUATION_KIND: str = "unknown"

# Entropy threshold for eq_type head: 85% of max entropy for 4-class uniform.
# ln(4) ≈ 1.3863. Threshold = 0.85 * ln(4).
_ENTROPY_OOD_THRESHOLD: float = 0.85 * math.log(4)

# Fine-label roles that count as "digit" for structural validation.
_DIGIT_ROLES: frozenset = frozenset({"main", "carry", "borrow"})

# Fine-label roles that count as "structural token" for structural validation.
_STRUCTURAL_ROLES: frozenset = frozenset({"operator", "structure"})

# div_bracket parses as role="unknown" via parse_flattened_label (it has no
# underscore-separated digit suffix). Enumerate it explicitly so it still
# counts as a structural token.
#
# Single registry: _STRUCTURAL_FINE_LABELS is the one place to add or remove
# fine labels that are treated as structural tokens. All four code paths that
# classify a symbol as structural depend on it:
#   _validate_equation_structure  — counts structural tokens for the gate
#   _is_structural_pred           — broad check (includes operators)
#   _is_structural_mark           — strict check (marks only, not operators)
#   _build_bare_digits_json       — routes standalone-mark scenes to bare_digits
_STRUCTURAL_FINE_LABELS: frozenset = frozenset({"div_bracket"})

# Fine labels excluded from spatial monotonicity check (structurally placed below rows).
_SPATIAL_SANITY_EXCLUDE: frozenset = frozenset({"result_bar", "div_bracket"})


@dataclass(frozen=True)
class NodePrediction:
    fine_label: str
    row_cluster_id: int
    col_cluster_id: int
    within_row_ord: int
    within_col_ord: int
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    spatial_conflict: bool = False  # True if row_cluster_id contradicts y-position rank
    carry_conflict: bool = False  # True if a carry/borrow has no main in its column
                                  # or sits above the lowest main (recognize-as-drawn:
                                  # the token is KEPT and flagged, never deleted)
    given: bool = False  # True if this node originated from a given/pre-printed equation token


def _validate_equation_structure(
    predictions: List[NodePrediction],
) -> Tuple[bool, Optional[str]]:
    """Check that the scene is equation-SHAPED. Return (is_valid, reason_or_None).

    This is a token-presence / shape check only. It is the anti-hallucination
    gate: it guards against inventing structure from noise (a lone scribble or
    an operator-only blob), NOT against drawn math being wrong. It must never
    inspect operand values or verify arithmetic.

    Shape invariants (all must hold for an equation-shaped scene):
      - len(predictions) >= 2  — rejects single-symbol scribbles
      - n_digits >= 1          — at least one main/carry/borrow digit present
      - n_structural >= 1      — at least one operator, result_bar, or div_bracket present

    Returns (False, OOD_REASON_*) on the first failing invariant.
    Returns (True, None) when all invariants hold.
    """
    if len(predictions) < 2:
        return False, OOD_REASON_TOO_FEW_SYMBOLS

    n_digits = 0
    n_structural = 0
    for pred in predictions:
        role, _ = parse_flattened_label(pred.fine_label)
        if role in _DIGIT_ROLES:
            n_digits += 1
        elif role in _STRUCTURAL_ROLES or pred.fine_label in _STRUCTURAL_FINE_LABELS:
            n_structural += 1

    if n_digits < 1:
        return False, OOD_REASON_NO_DIGITS
    if n_structural < 1:
        return False, OOD_REASON_NO_STRUCTURAL_TOKEN
    return True, None


def _is_digit_pred(pred: NodePrediction) -> bool:
    """Return True if this prediction counts as a digit (main/carry/borrow)."""
    role, _ = parse_flattened_label(pred.fine_label)
    return role in _DIGIT_ROLES


def _is_structural_pred(pred: NodePrediction) -> bool:
    """Return True if this prediction counts as a structural token (operator,
    result_bar, or div_bracket). Used by the equation-shape gate, where an
    operator is a legitimate structural token (e.g. ``3 + 5``)."""
    role, _ = parse_flattened_label(pred.fine_label)
    return role in _STRUCTURAL_ROLES or pred.fine_label in _STRUCTURAL_FINE_LABELS


def _is_structural_mark(pred: NodePrediction) -> bool:
    """Return True only for a drawn structural mark (result_bar / div_bracket).

    Stricter than ``_is_structural_pred``: operators do NOT count. This is the
    predicate that decides whether a digitless scene is real structure
    (standalone_bar / standalone_bracket → bare_digits) or operator-only
    scribble that must stay OOD (anti-hallucination split)."""
    role, _ = parse_flattened_label(pred.fine_label)
    return role == "structure" or pred.fine_label in _STRUCTURAL_FINE_LABELS


# Gate routing outcome literals.
_GATE_OOD = "ood"
_GATE_BARE_DIGITS = "bare_digits"
_GATE_VALID = "valid"


def _classify_gate(
    predictions: List[NodePrediction],
) -> Tuple[str, Optional[str]]:
    """Classify a prediction list and return (route, ood_reason).

    Route values:
      _GATE_OOD         — emit OOD JSON; ood_reason carries the reason string.
      _GATE_BARE_DIGITS — emit bare_digits JSON; ood_reason is None.
      _GATE_VALID       — scene is equation-SHAPED (token presence/shape only);
                          proceed normally. This is NOT an arithmetic-validity
                          check; it never inspects operand values.

    Routing table (first matching row wins):

    | Scene shape                                       | Route        | ood_reason          |
    |---------------------------------------------------|--------------|---------------------|
    | Zero detections                                   | ood          | true_empty          |
    | 1 detection, digit                                | bare_digits  | —                   |
    | 1 detection, non-digit (lone bar/bracket/op)      | bare_digits  | —                   |
    | ≥ 2 detections, no digits, ≥ 1 structural token   | bare_digits  | —                   |
    | ≥ 2 detections, no digits, no structural token    | ood          | no_digits           |
    | ≥ 2 detections, ≥ 1 digit, no structural token    | bare_digits  | —                   |
    | ≥ 2 detections, ≥ 1 digit, ≥ 1 structural token   | valid        | —                   |

    Anti-hallucination split (load-bearing): a digitless scene flips to
    bare_digits ONLY when it carries a structural token (result_bar /
    div_bracket), which is the standalone_bar / standalone_bracket shape.
    A digitless, structure-less scene (operators only, e.g. op_plus +
    op_minus, or a zero-detection scene) stays OOD so spurious scribbles
    are never promoted to a populated equation.

    OOD_REASON_SINGLE_NON_DIGIT is reserved: production routing no longer
    emits it (a lone non-digit token now routes to bare_digits), but the
    constant is retained for the direct _validate_equation_structure tests
    and as documentation of the prior contract.
    """
    n = len(predictions)

    if n == 0:
        return _GATE_OOD, OOD_REASON_TRUE_EMPTY

    if n == 1:
        # A lone token — digit, operator, result_bar, or div_bracket — is real
        # structure the GNN already placed. Route it to bare_digits so its
        # row/col/fine_label surface as populated rows/slots rather than an
        # empty OOD payload (the standalone_bar / standalone_bracket shape).
        return _GATE_BARE_DIGITS, None

    # n >= 2
    n_digits = sum(1 for p in predictions if _is_digit_pred(p))
    n_structural = sum(1 for p in predictions if _is_structural_pred(p))

    if n_digits < 1:
        # No digits: a drawn structural mark (result_bar / div_bracket) means the
        # scene carries real structure (grouped standalone marks) → populate it.
        # Operators alone do NOT count here — an operator-only scene is the
        # "scribble that looks like operators" garbage path and stays OOD. This
        # split uses _is_structural_mark (marks only), NOT _is_structural_pred
        # (which also counts operators for the equation-shaped path below).
        n_marks = sum(1 for p in predictions if _is_structural_mark(p))
        if n_marks >= 1:
            return _GATE_BARE_DIGITS, None
        return _GATE_OOD, OOD_REASON_NO_DIGITS
    if n_structural < 1:
        return _GATE_BARE_DIGITS, None
    return _GATE_VALID, None


def _build_bare_digits_json(
    predictions: List[NodePrediction],
) -> Dict[str, object]:
    """Build a schema_version=1 bare_digits response for scenes with no structural token.

    Populates rows and slots using _cluster_by_coordinate-equivalent logic: the
    row_cluster_id and col_cluster_id on each NodePrediction are already assigned
    by the graph builder via _cluster_by_coordinate(bbox, axis), so we use them
    directly. No re-clustering needed here.
    """
    rows_map: Dict[int, List[NodePrediction]] = {}
    for pred in predictions:
        rows_map.setdefault(pred.row_cluster_id, []).append(pred)

    mains: List[Dict[str, object]] = []
    carries: List[Dict[str, object]] = []
    borrows: List[Dict[str, object]] = []
    operators: List[Dict[str, object]] = []
    structures: List[Dict[str, object]] = []
    rows_out: List[Dict[str, object]] = []

    for out_row_idx, row_key in enumerate(sorted(rows_map.keys())):
        row_preds = sorted(rows_map[row_key], key=lambda p: p.col_cluster_id)
        tokens: List[Dict[str, object]] = []
        for col_pos, pred in enumerate(row_preds):
            role, detail = parse_flattened_label(pred.fine_label)
            token: Dict[str, object] = {
                "label": pred.fine_label,
                "role": role,
                "detail": detail,
                "row": out_row_idx,
                "col": col_pos,
                "grid_col": pred.col_cluster_id,
                "within_row_ord": pred.within_row_ord,
                "within_col_ord": pred.within_col_ord,
                "bbox": [pred.x0, pred.y0, pred.x1, pred.y1],
                "confidence": pred.confidence,
                "spatial_conflict": pred.spatial_conflict,
                "carry_conflict": pred.carry_conflict,
                "given": pred.given,
            }
            tokens.append(token)
            if role == "carry":
                carries.append({**token, "digit": detail})
            elif role == "borrow":
                borrows.append({**token, "digit": detail})
            elif role == "operator":
                operators.append({**token, "operator": detail})
            elif role == "structure" or pred.fine_label in _STRUCTURAL_FINE_LABELS:
                # div_bracket parses as role="unknown" (no digit suffix), so the
                # _STRUCTURAL_FINE_LABELS fallback keeps it in the structures slot
                # rather than mis-bucketing it as a main digit on the
                # standalone_bracket route.
                structures.append({**token, "kind": detail or pred.fine_label})
            else:
                mains.append({**token, "digit": detail})
        rows_out.append({"row": out_row_idx, "tokens": tokens})

    detections = [
        {
            "label": p.fine_label,
            "bbox": [p.x0, p.y0, p.x1, p.y1],
            "row": int(p.row_cluster_id),
            "col": int(p.col_cluster_id),
            "conf": float(p.confidence),
        }
        for p in predictions
    ]

    return {
        "schema_version": 1,
        "spatial_meta": {
            "parser_mode": "gnn",
            "spatial_conflict_count": 0,
        },
        "equation_kind": "bare_digits",
        "row_count": len(rows_out),
        "detections": detections,
        "rows": rows_out,
        "slots": {
            "main_digits": mains,
            "carries": carries,
            "borrows": borrows,
            "operators": operators,
            "structures": structures,
        },
    }


def _deduplicate_result_bars(predictions: List[NodePrediction]) -> List[NodePrediction]:
    """Keep at most one result_bar per unique row cluster (highest confidence)."""
    bars_by_row: Dict[int, NodePrediction] = {}
    others: List[NodePrediction] = []
    for pred in predictions:
        if pred.fine_label == "result_bar":
            existing = bars_by_row.get(pred.row_cluster_id)
            if existing is None or pred.confidence > existing.confidence:
                bars_by_row[pred.row_cluster_id] = pred
        else:
            others.append(pred)
    return others + list(bars_by_row.values())


# ---------------------------------------------------------------------------
# Rule 1: Spatial carry diagnostic (recognize-as-drawn — flag, never delete)
# ---------------------------------------------------------------------------

def _flag_spurious_carries(predictions: List[NodePrediction]) -> List[NodePrediction]:
    """Flag (never delete) carry_* / borrow_* tokens that sit in a spurious
    spatial position, mirroring how _check_spatial_consistency sets
    spatial_conflict without removing detections.

    Recognize-as-drawn principle: if a child writes a carry where no main digit
    sits below it, that carry was genuinely drawn and must be surfaced in the
    output, not erased. The token is KEPT in rows/slots and gets
    carry_conflict=True so downstream observers can see the placement is odd.

    A carry/borrow is flagged when either:
    - No main_* token exists in the same col_cluster_id, OR
    - The carry/borrow bbox (y0) is entirely above (smaller y value) the
      minimum y0 of all main_* tokens in the same col_cluster_id.

    'Above' on a canvas means a smaller y coordinate.

    Returns the prediction list unchanged in length; flagged tokens are
    rebuilt (frozen dataclass) with carry_conflict=True.
    """
    # Build per-column map of min y0 among main_* tokens.
    main_min_y0_by_col: Dict[int, float] = {}
    for pred in predictions:
        role, _ = parse_flattened_label(pred.fine_label)
        if role == "main":
            col = pred.col_cluster_id
            if col not in main_min_y0_by_col or pred.y0 < main_min_y0_by_col[col]:
                main_min_y0_by_col[col] = pred.y0

    flagged: List[NodePrediction] = []
    for pred in predictions:
        role, _ = parse_flattened_label(pred.fine_label)
        is_conflict = False
        if role in ("carry", "borrow"):
            col = pred.col_cluster_id
            if col not in main_min_y0_by_col:
                # No main in this column → spurious placement, but keep + flag.
                is_conflict = True
            elif pred.y0 < main_min_y0_by_col[col]:
                # Carry sits entirely above the lowest main in its column.
                is_conflict = True

        if is_conflict:
            flagged.append(NodePrediction(
                fine_label=pred.fine_label,
                row_cluster_id=pred.row_cluster_id,
                col_cluster_id=pred.col_cluster_id,
                within_row_ord=pred.within_row_ord,
                within_col_ord=pred.within_col_ord,
                x0=pred.x0,
                y0=pred.y0,
                x1=pred.x1,
                y1=pred.y1,
                confidence=pred.confidence,
                spatial_conflict=pred.spatial_conflict,
                carry_conflict=True,
                given=pred.given,
            ))
        else:
            flagged.append(pred)
    return flagged


# ---------------------------------------------------------------------------
# Rule 2: Operator deduplication
# ---------------------------------------------------------------------------

def _x_spans_overlap(a: NodePrediction, b: NodePrediction) -> bool:
    """True if two bbox x-spans overlap (touching counts as overlap)."""
    return a.x0 <= b.x1 and b.x0 <= a.x1


def _deduplicate_operators(predictions: List[NodePrediction]) -> List[NodePrediction]:
    """For each operator fine_label, keep only the highest-confidence token when
    multiple instances of the same label share the same row_cluster_id AND have
    overlapping x-spans. Non-overlapping duplicates in the same row are kept
    (e.g. two genuine op_minus in different horizontal positions).
    """
    # Separate operators from everything else.
    op_labels = {"op_plus", "op_minus", "op_times", "op_divide"}
    operators: List[NodePrediction] = []
    others: List[NodePrediction] = []
    for pred in predictions:
        if pred.fine_label in op_labels:
            operators.append(pred)
        else:
            others.append(pred)

    if not operators:
        return predictions

    # Greedy dedup: mark operators that are dominated by a higher-conf overlapping
    # sibling with the same label in the same row.
    kept_ops: List[NodePrediction] = []
    dropped: set[int] = set()
    for i, a in enumerate(operators):
        if i in dropped:
            continue
        best = a
        for j, b in enumerate(operators):
            if j <= i or j in dropped:
                continue
            if (
                b.fine_label == a.fine_label
                and b.row_cluster_id == a.row_cluster_id
                and _x_spans_overlap(a, b)
            ):
                if b.confidence > best.confidence:
                    dropped.add(operators.index(best))
                    best = b
                else:
                    dropped.add(j)
        kept_ops.append(best)

    # Re-check: remove duplicates in kept_ops that were already added via best swap.
    seen_ids: set[int] = set()
    deduped_ops: List[NodePrediction] = []
    for op in kept_ops:
        oid = id(op)
        if oid not in seen_ids:
            seen_ids.add(oid)
            deduped_ops.append(op)

    return others + deduped_ops


# ---------------------------------------------------------------------------
# Rule 3: Kind diagnostic (recognize-as-drawn — passthrough, never relabel)
# ---------------------------------------------------------------------------

def _apply_kind_override(predictions: List[NodePrediction], equation_type: str) -> str:
    """Recognize-as-drawn passthrough: return the recognized equation_type
    unchanged.

    Previously this relabelled a 'divide' scene by operator vote (or forced
    'unknown' when no bracket was found). Both decisions imposed what the
    equation "should" be on the structure the GNN already recognized, which is a
    correction, not a recognition. The kind is now whatever the eq_type head
    predicted; a missing bracket is a detection gap to fix by training, not a
    relabel to apply here.

    The diagnostic signal (divide kind with no bracket) is computed separately
    by _kind_low_confidence_reason and surfaced as an observability flag without
    mutating equation_kind. A populated scene is NEVER collapsed to OOD on a
    structural-shape mismatch.
    """
    return equation_type


def _kind_low_confidence_reason(
    predictions: List[NodePrediction], equation_type: str
) -> Optional[str]:
    """Return a non-destructive diagnostic reason when the recognized kind looks
    low-confidence against the detected tokens; else None.

    Currently flags only the divide-without-bracket case: the eq_type head said
    'divide' but no div_bracket token was detected. This is observability only;
    the caller keeps equation_kind == 'divide' and emits the populated scene.
    """
    if equation_type != "divide":
        return None
    has_bracket = any(p.fine_label == "div_bracket" for p in predictions)
    if has_bracket:
        return None
    return "divide_without_bracket"


# ---------------------------------------------------------------------------
# Rule 4: Entropy OOD gate
# ---------------------------------------------------------------------------

def _compute_softmax_entropy(logits: List[float]) -> float:
    """Compute Shannon entropy of softmax distribution from raw logits."""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def _check_entropy_ood(eq_type_logits: Optional[List[float]]) -> Optional[str]:
    """Return OOD_REASON_HIGH_ENTROPY if entropy exceeds threshold; else None."""
    if eq_type_logits is None:
        return None
    entropy = _compute_softmax_entropy(eq_type_logits)
    if entropy > _ENTROPY_OOD_THRESHOLD:
        return OOD_REASON_HIGH_ENTROPY
    return None


# ---------------------------------------------------------------------------
# Rule 5: Spatial sanity check
# ---------------------------------------------------------------------------

def _check_spatial_consistency(
    predictions: List[NodePrediction],
) -> Tuple[List[NodePrediction], int]:
    """Flag tokens whose row_cluster_id order is inconsistent with y-position rank.

    Groups tokens by row_cluster_id, computes median y-centroid per group, checks
    monotonicity (lower row_cluster_id should have smaller median y). Tokens in a
    violating group get spatial_conflict=True. result_bar and div_bracket are excluded.

    Returns (updated_predictions, spatial_conflict_count).
    """
    # Split into candidates (included in check) and excluded.
    included = [p for p in predictions if p.fine_label not in _SPATIAL_SANITY_EXCLUDE]
    excluded = [p for p in predictions if p.fine_label in _SPATIAL_SANITY_EXCLUDE]

    if len(included) == 0:
        return predictions, 0

    # Compute median y-centroid per row_cluster_id.
    rows_map: Dict[int, List[float]] = {}
    for p in included:
        y_center = (p.y0 + p.y1) / 2.0
        rows_map.setdefault(p.row_cluster_id, []).append(y_center)

    median_y: Dict[int, float] = {}
    for row_id, ys in rows_map.items():
        sorted_ys = sorted(ys)
        n = len(sorted_ys)
        mid = n // 2
        median_y[row_id] = (sorted_ys[mid - 1] + sorted_ys[mid]) / 2.0 if n % 2 == 0 else sorted_ys[mid]

    # Check monotonicity: sort row_cluster_ids and verify median y is non-decreasing.
    sorted_row_ids = sorted(median_y.keys())
    conflicting_rows: set[int] = set()
    for i in range(len(sorted_row_ids) - 1):
        r_a = sorted_row_ids[i]
        r_b = sorted_row_ids[i + 1]
        if median_y[r_a] > median_y[r_b]:
            # Both groups violate: the lower-indexed row has higher y than expected.
            conflicting_rows.add(r_a)
            conflicting_rows.add(r_b)

    if not conflicting_rows:
        return predictions, 0

    # Rebuild predictions: flag tokens in conflicting rows.
    spatial_conflict_count = 0
    updated: List[NodePrediction] = []
    for p in included:
        if p.row_cluster_id in conflicting_rows:
            # frozen dataclass: rebuild with spatial_conflict=True
            updated.append(NodePrediction(
                fine_label=p.fine_label,
                row_cluster_id=p.row_cluster_id,
                col_cluster_id=p.col_cluster_id,
                within_row_ord=p.within_row_ord,
                within_col_ord=p.within_col_ord,
                x0=p.x0,
                y0=p.y0,
                x1=p.x1,
                y1=p.y1,
                confidence=p.confidence,
                spatial_conflict=True,
                carry_conflict=p.carry_conflict,
                given=p.given,
            ))
            spatial_conflict_count += 1
        else:
            updated.append(p)

    return updated + excluded, spatial_conflict_count


# ---------------------------------------------------------------------------
# W10: Heuristic override constants and helper
# ---------------------------------------------------------------------------

# Minimum div_bracket confidence to trigger the division override.
# Kept low (0.3) because YOLO under-confidences real handwriting.
LOW_DIV_THRESHOLD: float = 0.3


def _compute_op_counts(predictions: List[NodePrediction]) -> Dict[str, int]:
    """Count each operator type in the prediction list."""
    counts = {"op_plus": 0, "op_minus": 0, "op_times": 0, "op_divide": 0}
    for p in predictions:
        if p.fine_label in counts:
            counts[p.fine_label] += 1
    return counts


def _apply_heuristic_overrides(
    predictions: List[NodePrediction],
    equation_type: str,
) -> Tuple[str, Optional[str], Optional[str], float, Dict[str, int]]:
    """Compute the operator/structural-token-derived kind as a DIAGNOSTIC only;
    NEVER mutate the recognized equation_kind.

    Recognize-as-drawn principle: equation_kind is the scene kind the GNN eq_type
    head recognized (passed in as ``equation_type``). The per-glyph operator
    tokens carry their own signal, but using a token majority to flip the kind
    "corrects" the scene to what it should be (the documented +/- harm: a misread
    '+' detected as op_minus would silently convert a drawn addition into a
    subtraction). A misread operator is a model error to fix by fine-tuning, not
    a patch to apply in the assembler.

    This function therefore always returns the head's kind unchanged. It still
    derives a token-implied kind so the caller can record observability fields
    (eq_kind_overridden_from / eq_kind_override_reason) when the tokens disagree
    with the head, but those fields are diagnostic and do not affect the emitted
    equation_kind.

    Token-implied-kind derivation (first match wins; diagnostic only):
      1. Division   — a div_bracket above LOW_DIV_THRESHOLD, or any op_divide present.
      2. Multiplication — any op_times present.
      3. Subtraction — op_minus tokens outnumber op_plus.
      4. Addition    — op_plus tokens outnumber op_minus.
      5. No signal   — no operator/structural token detected; no diagnostic.

    Returns:
        (equation_type, overridden_from, override_reason, div_bracket_max_conf, op_counts)
        equation_type is ALWAYS the input head kind (unchanged). overridden_from /
        override_reason are populated only when the token-derived kind differs from
        the head's kind, recording the disagreement WITHOUT acting on it.
    """
    op_counts = _compute_op_counts(predictions)

    div_bracket_max_conf: float = max(
        (p.confidence for p in predictions if p.fine_label == "div_bracket"),
        default=0.0,
    )

    head_kind = equation_type

    # Token-implied kind (diagnostic only; first match wins).
    derived: Optional[str] = None
    reason: Optional[str] = None
    if div_bracket_max_conf > LOW_DIV_THRESHOLD or op_counts["op_divide"] > 0:
        derived, reason = "divide", "div_signal"
    elif op_counts["op_times"] > 0:
        derived, reason = "multiply", "op_times_present"
    elif op_counts["op_minus"] > op_counts["op_plus"]:
        derived, reason = "subtract", "op_minus_majority"
    elif op_counts["op_plus"] > op_counts["op_minus"]:
        derived, reason = "add", "op_plus_majority"

    if derived is None or derived == head_kind:
        # No token signal, or tokens agree with the head — nothing to report.
        return head_kind, None, None, div_bracket_max_conf, op_counts

    # Tokens disagree with the head. Record the disagreement as a diagnostic but
    # KEEP the recognized (head) kind — never flip equation_kind.
    return head_kind, head_kind, reason, div_bracket_max_conf, op_counts


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------

def assemble_json(
    predictions: List[NodePrediction],
    equation_type: str,
    eq_type_logits: Optional[List[float]] = None,
    heuristic_enabled: bool = False,
) -> Dict[str, object]:
    """Convert GNN node predictions to schema_version=1 JSON.

    Recognize-as-drawn contract: equation_kind is the scene kind the GNN eq_type
    head recognized (the ``equation_type`` argument). The assembler structures
    the recognized symbols; it does NOT correct the math, relabel the kind by
    operator vote, or delete recognized tokens. The only hard rejections are the
    anti-hallucination shape gates (zero detections, operator-only scribble),
    which guard against inventing structure, not against drawn math being wrong.

    Rule application order (load-bearing):
      1. carry diagnostic    — flag spurious carries/borrows (never delete)
      2. op dedup            — same-label NMS-style dedupe (overlapping x, same row)
      3. result_bar dedup    — one result_bar per row cluster
      4. kind passthrough    — equation_kind = recognized kind (no relabel)
      4b. kind diagnostics   — compute observability-only disagreement fields
      5. shape gate          — rejects only non-equation-shaped scenes (OOD)
      6. entropy diagnostic  — uncertain eq_type → equation_kind='unknown' but the
                               recognized rows/slots are KEPT (not blanked)
      7. spatial sanity      — diagnostic only, never rejects

    row_cluster_id values are renumbered 0-based after sorting.
    col_cluster_id becomes grid_col in output tokens.
    """
    # Rule 1: flag spurious carries/borrows (recognize-as-drawn: keep + flag).
    predictions = _flag_spurious_carries(predictions)

    # Rule 2: deduplicate stacked operators
    predictions = _deduplicate_operators(predictions)

    # Existing dedup: result bars
    predictions = _deduplicate_result_bars(predictions)

    # Rule 3: kind passthrough — equation_kind stays the recognized kind.
    equation_type = _apply_kind_override(predictions, equation_type)

    # Rule 4b: diagnostics only. equation_type is NEVER reassigned from the
    # token-derived kind; we only record observability fields. The
    # heuristic_enabled knob now toggles whether the operator-disagreement
    # diagnostic is computed — it can never again mutate equation_kind.
    eq_kind_overridden_from: Optional[str] = None
    eq_kind_override_reason: Optional[str] = None
    div_bracket_max_conf: float = max(
        (p.confidence for p in predictions if p.fine_label == "div_bracket"),
        default=0.0,
    )
    op_counts: Dict[str, int] = _compute_op_counts(predictions)
    # Non-destructive kind diagnostic (e.g. divide recognized with no bracket).
    kind_low_conf_reason: Optional[str] = _kind_low_confidence_reason(
        predictions, equation_type
    )
    if heuristic_enabled:
        # _apply_heuristic_overrides returns the head kind UNCHANGED; the first
        # return value is ignored on purpose (equation_type is not reassigned).
        _, eq_kind_overridden_from, eq_kind_override_reason, div_bracket_max_conf, op_counts = (
            _apply_heuristic_overrides(predictions, equation_type)
        )

    # Build detections list from predictions (used in both OOD and success paths).
    # This is computed before the gates so OOD callers still see raw token placements.
    def _build_detections(preds: List[NodePrediction]) -> List[Dict[str, object]]:
        return [
            {
                "label": p.fine_label,
                "bbox": [p.x0, p.y0, p.x1, p.y1],
                "row": int(p.row_cluster_id),
                "col": int(p.col_cluster_id),
                "conf": float(p.confidence),
            }
            for p in preds
        ]

    def _observability_fields() -> Dict[str, object]:
        return {
            "eq_kind_overridden_from": eq_kind_overridden_from,
            "eq_kind_override_reason": eq_kind_override_reason,
            "div_bracket_max_conf": div_bracket_max_conf,
            "op_counts": op_counts,
            "kind_low_conf_reason": kind_low_conf_reason,
        }

    # Structural gate (W10: uses _classify_gate which implements the full routing table)
    gate_route, ood_reason = _classify_gate(predictions)
    if gate_route == _GATE_OOD:
        return {
            "schema_version": 1,
            "equation_kind": OOD_EQUATION_KIND,
            "ood_reason": ood_reason,
            "detections": _build_detections(predictions),
            "rows": [],
            "slots": [],
            "spatial_meta": {"empty": True},
            **_observability_fields(),
        }
    if gate_route == _GATE_BARE_DIGITS:
        result = _build_bare_digits_json(predictions)
        result.update(_observability_fields())
        return result

    # Rule 4: entropy diagnostic (only runs on equation-shaped scenes).
    # Recognize-as-drawn: an uncertain eq_type head no longer blanks the scene.
    # The independently recognized rows/slots are KEPT; only equation_kind is
    # demoted to 'unknown' with a diagnostic ood_reason so the GNN's row/col/
    # fine_label placements stay scorable.
    entropy_ood = _check_entropy_ood(eq_type_logits)

    # Rule 5: spatial sanity check (diagnostic, never rejects)
    predictions, spatial_conflict_count = _check_spatial_consistency(predictions)

    rows_map: Dict[int, List[NodePrediction]] = {}
    for pred in predictions:
        rows_map.setdefault(pred.row_cluster_id, []).append(pred)

    carries: List[Dict[str, object]] = []
    borrows: List[Dict[str, object]] = []
    mains: List[Dict[str, object]] = []
    operators: List[Dict[str, object]] = []
    structures: List[Dict[str, object]] = []
    rows_out: List[Dict[str, object]] = []

    for out_row_idx, row_key in enumerate(sorted(rows_map.keys())):
        row_preds = sorted(rows_map[row_key], key=lambda p: p.col_cluster_id)
        tokens: List[Dict[str, object]] = []
        for col_pos, pred in enumerate(row_preds):
            role, detail = parse_flattened_label(pred.fine_label)
            token: Dict[str, object] = {
                "label": pred.fine_label,
                "role": role,
                "detail": detail,
                "row": out_row_idx,
                "col": col_pos,
                "grid_col": pred.col_cluster_id,
                "within_row_ord": pred.within_row_ord,
                "within_col_ord": pred.within_col_ord,
                "bbox": [pred.x0, pred.y0, pred.x1, pred.y1],
                "confidence": pred.confidence,
                "spatial_conflict": pred.spatial_conflict,
                "carry_conflict": pred.carry_conflict,
                "given": pred.given,
            }
            tokens.append(token)
            if role == "carry":
                carries.append({**token, "digit": detail})
            elif role == "borrow":
                borrows.append({**token, "digit": detail})
            elif role == "operator":
                operators.append({**token, "operator": detail})
            elif role == "structure" or pred.fine_label in _STRUCTURAL_FINE_LABELS:
                # div_bracket parses as role="unknown" (no digit suffix), so the
                # _STRUCTURAL_FINE_LABELS fallback keeps it in the structures slot
                # rather than mis-bucketing it as a main digit on the valid
                # division success path (mirrors the bare_digits standalone route).
                structures.append({**token, "kind": detail or pred.fine_label})
            else:
                mains.append({**token, "digit": detail})
        rows_out.append({"row": out_row_idx, "tokens": tokens})

    # Entropy demotion: when the eq_type head is uncertain, emit equation_kind=
    # 'unknown' but KEEP the populated rows/slots (recognize-as-drawn). The
    # diagnostic reason is surfaced via ood_reason without blanking structure.
    emitted_kind: str = OOD_EQUATION_KIND if entropy_ood is not None else equation_type

    result: Dict[str, object] = {
        "schema_version": 1,
        "spatial_meta": {
            "parser_mode": "gnn",
            "spatial_conflict_count": spatial_conflict_count,
        },
        "equation_kind": emitted_kind,
        "row_count": len(rows_out),
        "detections": _build_detections(predictions),
        "rows": rows_out,
        "slots": {
            "main_digits": mains,
            "carries": carries,
            "borrows": borrows,
            "operators": operators,
            "structures": structures,
        },
        **_observability_fields(),
    }
    if entropy_ood is not None:
        result["ood_reason"] = entropy_ood
    return result
