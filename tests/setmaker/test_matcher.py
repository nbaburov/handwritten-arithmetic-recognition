"""Tests for the set-maker target<->detection matcher (WS-D, ``src/setmaker/matcher.py``).

The matcher is a pure function: it fuses a known ``TargetScene`` with a
geometry-only ``List[Detection]`` into a prefilled ``List[AnnotationDraft]`` via a
Hungarian (``scipy.optimize.linear_sum_assignment``) centroid-distance assignment.
These tests pin the contract the set-maker relies on:

* an exact-count, well-aligned scene assigns every target label onto a detection
  (``source="matched"``) and copies ``fine_label``/``row``/``col``/``equation_idx``
  from the *target* while keeping the *detection* box + confidence;
* an extra detection (more boxes than targets) yields a flagged ``"detected"``
  draft with no label;
* an unmatched target produces NO draft -- ghost boxes are suppressed entirely
  (FIX 1: previously a ``"manual"`` draft was emitted for every unmatched target,
  flooding the canvas with phantom boxes when the human drew few strokes);
* the global assignment beats greedy on a dense adjacent-pair scene (the swap
  that greedy makes is the failure mode this design exists to kill);
* a pair beyond the distance ceiling is rejected (the detection falls through to
  ``"detected"``; the target is silently dropped);
* the soft distance flag and the confidence floor each flag a kept match;
* output ordering is stable (detections in input order only) and deterministic
  across repeated calls;
* degenerate inputs (no detections, no symbols, both empty) are handled without a
  solve.

Tests build ``Detection`` and ``TargetSymbol`` directly (both are tiny frozen
dataclasses), so no weights, model, or disk are needed.
"""

from __future__ import annotations

import unittest

from src.parsing.detection import Detection
from src.setmaker.matcher import (
    CONFIDENCE_FLOOR,
    DISTANCE_CEILING_PX,
    DISTANCE_FLAG_PX,
    UNASSIGNED_INDEX,
    UNLABELLED,
    match,
)
from src.setmaker.types import AnnotationDraft, TargetScene, TargetSymbol


# ── Builders ──────────────────────────────────────────────────────────────────

def _symbol(
    fine_label: str,
    *,
    cx: float,
    cy: float,
    row: int = 0,
    col: int = 0,
    equation_idx: int = 0,
    half: float = 14.0,
) -> TargetSymbol:
    """A ``TargetSymbol`` centred at ``(cx, cy)`` with a square ``2*half`` bbox.

    ``glyph_key``/``yolo_class`` are filled with plausible values; the matcher
    only reads ``bbox`` plus the four label/grid fields it copies, but the full
    seven-field shape is built so the draft contract stays honest.
    """
    return TargetSymbol(
        fine_label=fine_label,
        glyph_key=f"{fine_label}/sample",
        yolo_class="digit_main",
        row_index=row,
        col_index=col,
        equation_idx=equation_idx,
        bbox=(cx - half, cy - half, cx + half, cy + half),
    )


def _scene(symbols: list[TargetSymbol]) -> TargetScene:
    """Wrap symbols in a minimal single-equation ``TargetScene`` for matching.

    Only ``symbols`` is read by the matcher; the rest are valid placeholders.
    """
    return TargetScene(
        case="addition",
        equation_type="addition",
        completion_stage="full",
        seed=0,
        reference="1 + 2 = 3",
        symbols=symbols,
    )


def _detection(
    *,
    cx: float,
    cy: float,
    confidence: float = 0.9,
    half: float = 14.0,
    label: str = "digit_main",
) -> Detection:
    """A ``Detection`` centred at ``(cx, cy)`` with a square ``2*half`` bbox."""
    return Detection(
        label=label,
        confidence=confidence,
        x0=cx - half,
        y0=cy - half,
        x1=cx + half,
        y1=cy + half,
    )


# ── Module surface / constants ────────────────────────────────────────────────

class TestModuleSurface(unittest.TestCase):
    """Documented thresholds and sentinels have the expected, ordered values."""

    def test_distance_flag_below_ceiling(self) -> None:
        """The soft review bound must sit strictly below the hard reject radius."""
        self.assertLess(DISTANCE_FLAG_PX, DISTANCE_CEILING_PX)

    def test_thresholds_positive(self) -> None:
        self.assertGreater(DISTANCE_CEILING_PX, 0.0)
        self.assertGreater(DISTANCE_FLAG_PX, 0.0)
        self.assertGreater(CONFIDENCE_FLOOR, 0.0)
        self.assertLess(CONFIDENCE_FLOOR, 1.0)

    def test_unlabelled_is_empty_string(self) -> None:
        self.assertEqual(UNLABELLED, "")

    def test_unassigned_index_is_negative(self) -> None:
        """A negative sentinel can never collide with a real 0-based index."""
        self.assertLess(UNASSIGNED_INDEX, 0)


# ── Exact-count match ─────────────────────────────────────────────────────────

class TestExactCountMatch(unittest.TestCase):
    """A well-aligned, equal-count scene matches every symbol."""

    def setUp(self) -> None:
        # Three symbols on a row, three detections almost exactly on them.
        self.symbols = [
            _symbol("main_1", cx=100.0, cy=100.0, col=0),
            _symbol("op_plus", cx=160.0, cy=100.0, col=1),
            _symbol("main_2", cx=220.0, cy=100.0, col=2),
        ]
        self.detections = [
            _detection(cx=101.0, cy=99.0),
            _detection(cx=159.0, cy=101.0),
            _detection(cx=221.0, cy=100.0),
        ]
        self.drafts = match(_scene(self.symbols), self.detections)

    def test_returns_list_of_drafts(self) -> None:
        self.assertIsInstance(self.drafts, list)
        self.assertTrue(all(isinstance(d, AnnotationDraft) for d in self.drafts))

    def test_one_draft_per_symbol(self) -> None:
        self.assertEqual(len(self.drafts), len(self.symbols))

    def test_all_matched(self) -> None:
        self.assertTrue(all(d.source == "matched" for d in self.drafts))

    def test_no_flags_when_close_and_confident(self) -> None:
        """Tight, high-confidence pairs need no human review."""
        self.assertTrue(all(not d.flagged for d in self.drafts))

    def test_labels_copied_from_target(self) -> None:
        """Each detection (by input order) wears its nearest target's label."""
        self.assertEqual([d.fine_label for d in self.drafts], ["main_1", "op_plus", "main_2"])

    def test_grid_fields_copied_from_target(self) -> None:
        self.assertEqual([d.col_index for d in self.drafts], [0, 1, 2])
        self.assertTrue(all(d.row_index == 0 for d in self.drafts))
        self.assertTrue(all(d.equation_idx == 0 for d in self.drafts))

    def test_bbox_is_detection_geometry_not_target(self) -> None:
        """``bbox_px`` is the detected box (real pixels), not the target box."""
        first = self.drafts[0]
        self.assertEqual(first.bbox_px, (101.0 - 14.0, 99.0 - 14.0, 101.0 + 14.0, 99.0 + 14.0))

    def test_confidence_is_detection_confidence(self) -> None:
        self.assertTrue(all(abs(d.confidence - 0.9) < 1e-9 for d in self.drafts))


# ── Extra detection (more boxes than targets) ─────────────────────────────────

class TestExtraDetection(unittest.TestCase):
    """A detection with no target becomes a flagged ``"detected"`` draft."""

    def setUp(self) -> None:
        self.symbols = [_symbol("main_7", cx=100.0, cy=100.0, col=0)]
        self.detections = [
            _detection(cx=100.0, cy=100.0),       # matches the only target
            _detection(cx=400.0, cy=400.0),       # extra, far away
        ]
        self.drafts = match(_scene(self.symbols), self.detections)

    def test_one_draft_per_detection(self) -> None:
        """Every detection is represented (matched or detected)."""
        self.assertEqual(len(self.drafts), 2)

    def test_sources_are_matched_then_detected(self) -> None:
        self.assertEqual([d.source for d in self.drafts], ["matched", "detected"])

    def test_extra_detection_is_flagged(self) -> None:
        extra = self.drafts[1]
        self.assertTrue(extra.flagged)

    def test_extra_detection_has_no_label(self) -> None:
        extra = self.drafts[1]
        self.assertEqual(extra.fine_label, UNLABELLED)

    def test_extra_detection_grid_is_unassigned(self) -> None:
        extra = self.drafts[1]
        self.assertEqual(extra.row_index, UNASSIGNED_INDEX)
        self.assertEqual(extra.col_index, UNASSIGNED_INDEX)
        self.assertEqual(extra.equation_idx, UNASSIGNED_INDEX)

    def test_extra_detection_keeps_its_geometry(self) -> None:
        extra = self.drafts[1]
        self.assertEqual(extra.bbox_px, (386.0, 386.0, 414.0, 414.0))

    def test_matched_one_has_the_label(self) -> None:
        self.assertEqual(self.drafts[0].fine_label, "main_7")


# ── Missing detection (fewer boxes than targets) ──────────────────────────────

class TestMissingDetection(unittest.TestCase):
    """Unmatched targets produce NO draft (FIX 1: ghost-box suppression).

    Previously a ``"manual"`` draft was emitted for every unmatched target,
    causing phantom boxes on the canvas when the human drew fewer symbols than
    the full target. The new contract: only detections produce drafts.
    """

    def setUp(self) -> None:
        self.symbols = [
            _symbol("main_1", cx=100.0, cy=100.0, col=0),
            _symbol("main_2", cx=300.0, cy=100.0, col=1),  # human never drew this
        ]
        self.detections = [_detection(cx=100.0, cy=100.0)]
        self.drafts = match(_scene(self.symbols), self.detections)

    def test_only_detection_backed_drafts_emitted(self) -> None:
        """Unmatched target is silently dropped; only the one detection makes a draft."""
        self.assertEqual(len(self.drafts), 1)

    def test_no_manual_drafts(self) -> None:
        """``"manual"`` source must never appear when there are detections."""
        self.assertFalse(any(d.source == "manual" for d in self.drafts))

    def test_matched_draft_has_correct_label(self) -> None:
        """The one detection is matched to the nearest target and gets its label."""
        self.assertEqual(self.drafts[0].source, "matched")
        self.assertEqual(self.drafts[0].fine_label, "main_1")

    def test_unmatched_target_does_not_appear(self) -> None:
        """``main_2`` was never detected so it produces no draft at all."""
        labels = [d.fine_label for d in self.drafts]
        self.assertNotIn("main_2", labels)


# ── Hungarian beats greedy on a dense adjacent pair ───────────────────────────

class TestGlobalAssignmentBeatsGreedy(unittest.TestCase):
    """The global optimum avoids the adjacent-swap a greedy rule would make.

    Two target slots at x=100 (``main_3``) and x=130 (``main_5``). Two detections
    drawn slightly right of centre at x=118 and x=131. A greedy first-detection
    rule assigns det@118 to its nearest target (x=130, ``main_5``) and then forces
    det@131 onto x=100 (``main_3``) -- a swap. The Hungarian solve minimises total
    distance and pairs det@118->x=100 and det@131->x=130, the correct labelling.
    """

    def setUp(self) -> None:
        self.symbols = [
            _symbol("main_3", cx=100.0, cy=100.0, col=0),
            _symbol("main_5", cx=130.0, cy=100.0, col=1),
        ]
        self.detections = [
            _detection(cx=118.0, cy=100.0),
            _detection(cx=131.0, cy=100.0),
        ]
        self.drafts = match(_scene(self.symbols), self.detections)

    def test_no_adjacent_swap(self) -> None:
        """det@118 -> main_3 (col 0), det@131 -> main_5 (col 1)."""
        self.assertEqual([d.fine_label for d in self.drafts], ["main_3", "main_5"])
        self.assertEqual([d.col_index for d in self.drafts], [0, 1])

    def test_both_matched(self) -> None:
        self.assertTrue(all(d.source == "matched" for d in self.drafts))


# ── Distance ceiling rejection ────────────────────────────────────────────────

class TestDistanceCeiling(unittest.TestCase):
    """A solved pair beyond the ceiling is rejected, not silently accepted."""

    def test_over_ceiling_detection_becomes_detected_target_dropped(self) -> None:
        """A single detection far from a single target: Hungarian pairs them, but
        the pair is over the ceiling, so the detection becomes ``"detected"`` and
        the target is silently dropped (no ghost box). Never a wrong ``"matched"``."""
        symbols = [_symbol("main_4", cx=100.0, cy=100.0)]
        far = DISTANCE_CEILING_PX + 50.0
        detections = [_detection(cx=100.0 + far, cy=100.0)]
        drafts = match(_scene(symbols), detections)

        # Only the detection produces a draft; the unmatched target is dropped.
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].source, "detected")
        self.assertTrue(drafts[0].flagged)
        self.assertEqual(drafts[0].fine_label, UNLABELLED)

    def test_just_inside_ceiling_is_matched(self) -> None:
        """A pair just under the ceiling is kept (flagged, but matched)."""
        symbols = [_symbol("main_4", cx=100.0, cy=100.0)]
        inside = DISTANCE_CEILING_PX - 1.0
        detections = [_detection(cx=100.0 + inside, cy=100.0)]
        drafts = match(_scene(symbols), detections)

        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].source, "matched")
        self.assertEqual(drafts[0].fine_label, "main_4")

    def test_exactly_at_ceiling_is_kept(self) -> None:
        """Distance == ceiling is inclusive (``<=``), so the pair survives."""
        symbols = [_symbol("main_4", cx=100.0, cy=100.0)]
        detections = [_detection(cx=100.0 + DISTANCE_CEILING_PX, cy=100.0)]
        drafts = match(_scene(symbols), detections)

        self.assertEqual([d.source for d in drafts], ["matched"])


# ── Flagging logic on kept matches ────────────────────────────────────────────

class TestMatchFlagging(unittest.TestCase):
    """A kept match is flagged on distance OR confidence; clean only when both ok."""

    def test_high_distance_flags_match(self) -> None:
        """Distance above the soft flag bound (but under the ceiling) flags it."""
        symbols = [_symbol("main_8", cx=100.0, cy=100.0)]
        between = (DISTANCE_FLAG_PX + DISTANCE_CEILING_PX) / 2.0
        detections = [_detection(cx=100.0 + between, cy=100.0, confidence=0.99)]
        drafts = match(_scene(symbols), detections)

        self.assertEqual(drafts[0].source, "matched")
        self.assertTrue(drafts[0].flagged)

    def test_low_confidence_flags_match(self) -> None:
        """A perfectly placed but low-confidence detection is still flagged."""
        symbols = [_symbol("main_8", cx=100.0, cy=100.0)]
        detections = [_detection(cx=100.0, cy=100.0, confidence=CONFIDENCE_FLOOR / 2.0)]
        drafts = match(_scene(symbols), detections)

        self.assertEqual(drafts[0].source, "matched")
        self.assertTrue(drafts[0].flagged)

    def test_close_and_confident_is_not_flagged(self) -> None:
        symbols = [_symbol("main_8", cx=100.0, cy=100.0)]
        detections = [_detection(cx=101.0, cy=100.0, confidence=0.95)]
        drafts = match(_scene(symbols), detections)

        self.assertEqual(drafts[0].source, "matched")
        self.assertFalse(drafts[0].flagged)

    def test_confidence_just_below_floor_flags(self) -> None:
        symbols = [_symbol("main_8", cx=100.0, cy=100.0)]
        detections = [_detection(cx=100.0, cy=100.0, confidence=CONFIDENCE_FLOOR - 0.01)]
        drafts = match(_scene(symbols), detections)
        self.assertTrue(drafts[0].flagged)

    def test_confidence_at_floor_is_not_flagged_by_confidence(self) -> None:
        """The floor is a strict ``<`` check, so confidence == floor is acceptable."""
        symbols = [_symbol("main_8", cx=100.0, cy=100.0)]
        detections = [_detection(cx=100.0, cy=100.0, confidence=CONFIDENCE_FLOOR)]
        drafts = match(_scene(symbols), detections)
        self.assertFalse(drafts[0].flagged)


# ── Ordering + determinism ────────────────────────────────────────────────────

class TestOrderingAndDeterminism(unittest.TestCase):
    """Output order is stable: detections in input order, then manual drafts."""

    def test_matched_and_detected_follow_detection_order(self) -> None:
        symbols = [
            _symbol("main_1", cx=100.0, cy=100.0, col=0),
            _symbol("main_2", cx=200.0, cy=100.0, col=1),
        ]
        # Detections deliberately given out of spatial order: the second symbol's
        # box first, then an extra, then the first symbol's box.
        detections = [
            _detection(cx=200.0, cy=100.0),   # -> main_2
            _detection(cx=450.0, cy=450.0),   # -> detected (extra)
            _detection(cx=100.0, cy=100.0),   # -> main_1
        ]
        drafts = match(_scene(symbols), detections)
        self.assertEqual(
            [(d.source, d.fine_label) for d in drafts],
            [("matched", "main_2"), ("detected", UNLABELLED), ("matched", "main_1")],
        )

    def test_unmatched_targets_produce_no_draft(self) -> None:
        """Unmatched targets are silently dropped; only detection-backed drafts appear."""
        symbols = [
            _symbol("main_1", cx=100.0, cy=100.0, col=0),   # matched
            _symbol("main_2", cx=300.0, cy=100.0, col=1),   # unmatched -> dropped
            _symbol("main_3", cx=400.0, cy=100.0, col=2),   # unmatched -> dropped
        ]
        detections = [_detection(cx=100.0, cy=100.0)]
        drafts = match(_scene(symbols), detections)
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].source, "matched")
        self.assertEqual(drafts[0].fine_label, "main_1")

    def test_repeated_calls_are_identical(self) -> None:
        """Deterministic: the same inputs always produce the same drafts."""
        symbols = [
            _symbol("main_1", cx=100.0, cy=100.0, col=0),
            _symbol("op_plus", cx=160.0, cy=100.0, col=1),
            _symbol("main_2", cx=220.0, cy=100.0, col=2),
        ]
        detections = [
            _detection(cx=159.0, cy=101.0),
            _detection(cx=221.0, cy=100.0),
            _detection(cx=101.0, cy=99.0),
        ]
        scene = _scene(symbols)
        first = match(scene, detections)
        second = match(scene, detections)
        self.assertEqual(first, second)


# ── Degenerate inputs ─────────────────────────────────────────────────────────

class TestDegenerateInputs(unittest.TestCase):
    """No detections, no symbols, or both empty are handled without a solve."""

    def test_no_detections_yields_empty_list(self) -> None:
        """No detections means no drafts at all -- no ghost boxes for the targets."""
        symbols = [
            _symbol("main_1", cx=100.0, cy=100.0, col=0),
            _symbol("main_2", cx=200.0, cy=100.0, col=1),
        ]
        drafts = match(_scene(symbols), [])
        self.assertEqual(drafts, [])

    def test_no_symbols_yields_detected_per_detection(self) -> None:
        detections = [
            _detection(cx=100.0, cy=100.0),
            _detection(cx=200.0, cy=100.0),
        ]
        drafts = match(_scene([]), detections)
        self.assertEqual(len(drafts), 2)
        self.assertTrue(all(d.source == "detected" for d in drafts))
        self.assertTrue(all(d.flagged for d in drafts))
        self.assertTrue(all(d.fine_label == UNLABELLED for d in drafts))

    def test_both_empty_yields_empty_list(self) -> None:
        drafts = match(_scene([]), [])
        self.assertEqual(drafts, [])

    def test_empty_inputs_return_a_list(self) -> None:
        """Even the empty case is a list, never ``None`` (route relies on it)."""
        self.assertIsInstance(match(_scene([]), []), list)


# ── A realistic dense mixed scene ─────────────────────────────────────────────

class TestDenseMixedScene(unittest.TestCase):
    """A two-row scene with one extra detection and one missing detection.

    Targets: row 0 = [main_1@(100,100), op_plus@(160,100), main_2@(220,100)];
    row 1 = [main_3@(160,200)]. Detections cover row 0 and add a spurious far box,
    but never cover row 1. Expect 3 matched + 1 detected (extra); the uncovered
    row-1 target is silently dropped (no ghost box, FIX 1).
    """

    def setUp(self) -> None:
        self.symbols = [
            _symbol("main_1", cx=100.0, cy=100.0, row=0, col=0),
            _symbol("op_plus", cx=160.0, cy=100.0, row=0, col=1),
            _symbol("main_2", cx=220.0, cy=100.0, row=0, col=2),
            _symbol("main_3", cx=160.0, cy=200.0, row=1, col=1),
        ]
        self.detections = [
            _detection(cx=100.0, cy=101.0),
            _detection(cx=161.0, cy=100.0),
            _detection(cx=219.0, cy=100.0),
            _detection(cx=480.0, cy=30.0),   # spurious extra
        ]
        self.drafts = match(_scene(self.symbols), self.detections)

    def test_total_draft_count(self) -> None:
        """4 detections only = 4 drafts; uncovered row-1 target dropped (no ghost)."""
        self.assertEqual(len(self.drafts), 4)

    def test_source_breakdown(self) -> None:
        counts = {"matched": 0, "detected": 0}
        for d in self.drafts:
            counts[d.source] = counts.get(d.source, 0) + 1
        self.assertEqual(counts, {"matched": 3, "detected": 1})

    def test_no_manual_drafts(self) -> None:
        """Unmatched row-1 target must not appear as a ``"manual"`` ghost."""
        self.assertFalse(any(d.source == "manual" for d in self.drafts))

    def test_row0_labels_are_correct(self) -> None:
        matched = [d for d in self.drafts if d.source == "matched"]
        self.assertEqual(
            sorted(d.fine_label for d in matched),
            ["main_1", "main_2", "op_plus"],
        )


# ── FIX 1 regression: no ghost boxes when few detections vs many targets ──────

class TestNoGhostBoxes(unittest.TestCase):
    """FIX 1 regression: only detection-backed drafts appear regardless of target count.

    The original bug: 3 detected vs 32 target symbols emitted 29 phantom
    ``"manual"`` drafts at expected target positions, flooding the canvas.
    The fixed contract: only the 3 detections produce drafts.
    """

    def _many_symbols(self, n: int) -> list:
        """Build ``n`` symbols spread along a row."""
        return [_symbol(f"main_{i % 10}", cx=50.0 + i * 15.0, cy=100.0, col=i) for i in range(n)]

    def test_few_detections_vs_many_targets_no_ghost(self) -> None:
        symbols = self._many_symbols(32)
        detections = [
            _detection(cx=50.0, cy=100.0),
            _detection(cx=65.0, cy=100.0),
            _detection(cx=80.0, cy=100.0),
        ]
        drafts = match(_scene(symbols), detections)
        # Exactly 3 drafts -- one per detection. No phantom boxes.
        self.assertEqual(len(drafts), 3)
        self.assertFalse(any(d.source == "manual" for d in drafts))

    def test_zero_detections_zero_drafts(self) -> None:
        symbols = self._many_symbols(10)
        drafts = match(_scene(symbols), [])
        self.assertEqual(drafts, [])

    def test_draft_count_equals_detection_count(self) -> None:
        """For any number of targets, len(drafts) == len(detections)."""
        for n_det in (0, 1, 5, 10):
            symbols = self._many_symbols(20)
            detections = [_detection(cx=50.0 + i * 15.0, cy=100.0) for i in range(n_det)]
            drafts = match(_scene(symbols), detections)
            self.assertEqual(len(drafts), n_det, f"n_det={n_det}")


if __name__ == "__main__":
    unittest.main()
