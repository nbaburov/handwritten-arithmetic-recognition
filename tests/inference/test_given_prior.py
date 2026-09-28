"""Tests for src/inference/given_prior.py — Workstream A helpers.

Covers:
  merge_given   — centre-in-cell dedupe, no-overlap, overflow cap
  compose_into_scene — child ink preserved, tiles painted, shape assert
  pin_given_labels   — only given preds overridden, missing key no-op
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.inference.given_prior import (
    GivenNode,
    MergeResult,
    _bbox_key,
    compose_into_scene,
    merge_given,
    pin_given_labels,
)
from src.parsing.assemble import NodePrediction
from src.parsing.detection import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_tile() -> np.ndarray:
    """28x28 all-white tile (blank given node)."""
    return np.full((28, 28), 255, dtype=np.uint8)


def _dark_tile() -> np.ndarray:
    """28x28 all-dark tile (fully inked given node)."""
    return np.zeros((28, 28), dtype=np.uint8)


def _make_given(x0: int, y0: int, x1: int, y1: int,
                coarse: str = "digit_main", fine: str = "main_3",
                tile: np.ndarray | None = None) -> GivenNode:
    if tile is None:
        tile = _white_tile()
    return GivenNode(x0=x0, y0=y0, x1=x1, y1=y1,
                     coarse_label=coarse, fine_label=fine, tile=tile)


def _make_det(x0: float, y0: float, x1: float, y1: float,
              label: str = "digit_main", conf: float = 0.9,
              given: bool = False) -> Detection:
    return Detection(label=label, confidence=conf,
                     x0=x0, y0=y0, x1=x1, y1=y1, given=given)


def _make_pred(x0: float, y0: float, x1: float, y1: float,
               fine: str = "main_5", given: bool = False) -> NodePrediction:
    return NodePrediction(
        fine_label=fine,
        row_cluster_id=0,
        col_cluster_id=0,
        within_row_ord=0,
        within_col_ord=0,
        x0=x0, y0=y0, x1=x1, y1=y1,
        confidence=0.9,
        given=given,
    )


# ---------------------------------------------------------------------------
# merge_given — centre-in-cell dedupe
# ---------------------------------------------------------------------------

class TestMergeCentreInCell:
    def test_child_centre_inside_given_bbox_merges_to_one_node(self):
        """Child detection centred inside a given bbox -> exactly one node at that
        cell, given=True, child geometry kept; the synthetic for that cell is dropped;
        known_fine has the child bbox key."""
        given_node = _make_given(100, 100, 150, 150, fine="main_7")
        # Child centre: (120, 120) -- well inside [100,150]x[100,150]
        child = _make_det(110.0, 110.0, 130.0, 130.0)
        result = merge_given([child], [given_node], max_nodes=80)

        # Exactly one detection total (no synthetic added for the matched cell).
        assert len(result.detections) == 1
        det = result.detections[0]
        # Child geometry is preserved.
        assert det.x0 == pytest.approx(110.0)
        assert det.y0 == pytest.approx(110.0)
        assert det.x1 == pytest.approx(130.0)
        assert det.y1 == pytest.approx(130.0)
        # Re-emitted with given=True.
        assert det.given is True
        # compose_nodes should be empty (the given node was matched).
        assert result.compose_nodes == []
        # known_fine has entry keyed by the CHILD's integer bbox.
        child_key = _bbox_key(110.0, 110.0, 130.0, 130.0)
        assert child_key in result.known_fine
        assert result.known_fine[child_key] == "main_7"

    def test_child_centre_exactly_on_edge_counts_as_inside(self):
        """A centre exactly on the given node boundary (inclusive) is a match."""
        given_node = _make_given(50, 50, 100, 100, fine="op_plus")
        # Centre at (50, 50) -- on the corner boundary
        child = _make_det(50.0, 50.0, 50.0, 50.0)  # degenerate but centre is on boundary
        result = merge_given([child], [given_node], max_nodes=80)
        assert len(result.detections) == 1
        assert result.detections[0].given is True

    def test_child_centre_outside_given_bbox_no_match(self):
        """Child centre outside all given bboxes -> child unmodified, synthetic added."""
        given_node = _make_given(100, 100, 150, 150, fine="main_2")
        # Child centre at (200, 200) -- outside [100,150]x[100,150]
        child = _make_det(180.0, 180.0, 220.0, 220.0)
        result = merge_given([child], [given_node], max_nodes=80)

        assert len(result.detections) == 2
        child_det = result.detections[0]
        assert child_det.given is False  # unchanged
        synthetic_det = result.detections[1]
        assert synthetic_det.given is True
        assert synthetic_det.x0 == pytest.approx(100.0)
        # compose_nodes has the unmatched given node
        assert len(result.compose_nodes) == 1
        assert result.compose_nodes[0] is given_node

    def test_each_given_cell_absorbs_at_most_one_child(self):
        """Two children both inside the same given bbox -- only the first is matched;
        the second remains unmatched (given node already claimed)."""
        given_node = _make_given(100, 100, 200, 200, fine="main_1")
        child1 = _make_det(130.0, 130.0, 160.0, 160.0)  # centre (145,145) inside
        child2 = _make_det(140.0, 140.0, 170.0, 170.0)  # centre (155,155) inside
        result = merge_given([child1, child2], [given_node], max_nodes=80)
        # One matched (given=True), one unmatched (given=False).
        given_flags = [d.given for d in result.detections]
        assert given_flags.count(True) == 1
        assert given_flags.count(False) == 1
        # No synthetic added for the already-matched cell.
        assert len(result.compose_nodes) == 0


# ---------------------------------------------------------------------------
# merge_given — no-overlap scenario
# ---------------------------------------------------------------------------

class TestMergeNoOverlap:
    def test_child_plus_three_disjoint_given_nodes(self):
        """Child + 3 disjoint given nodes -> 4 detections, 3 synthetics in
        compose_nodes, known_fine has 3 entries (one per given)."""
        gn1 = _make_given(0, 0, 50, 50, fine="main_1")
        gn2 = _make_given(100, 0, 150, 50, fine="main_2")
        gn3 = _make_given(200, 0, 250, 50, fine="op_plus", coarse="operator")
        # Child is far away from all given cells
        child = _make_det(400.0, 400.0, 450.0, 450.0)
        result = merge_given([child], [gn1, gn2, gn3], max_nodes=80)

        assert len(result.detections) == 4
        assert len(result.compose_nodes) == 3
        # known_fine has one entry per given node bbox
        assert len(result.known_fine) == 3
        assert _bbox_key(0, 0, 50, 50) in result.known_fine
        assert _bbox_key(100, 0, 150, 50) in result.known_fine
        assert _bbox_key(200, 0, 250, 50) in result.known_fine

    def test_synthetic_detection_fields(self):
        """Synthetic given detection has the given node's coarse label, given=True,
        confidence=1.0, and the given node's bbox as floats."""
        gn = _make_given(10, 20, 60, 70, coarse="operator", fine="op_minus")
        result = merge_given([], [gn], max_nodes=80)
        assert len(result.detections) == 1
        syn = result.detections[0]
        assert syn.given is True
        assert syn.confidence == pytest.approx(1.0)
        assert syn.label == "operator"
        assert syn.x0 == pytest.approx(10.0)
        assert syn.y0 == pytest.approx(20.0)
        assert syn.x1 == pytest.approx(60.0)
        assert syn.y1 == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# merge_given — overflow cap
# ---------------------------------------------------------------------------

class TestMergeOverflowCap:
    def test_child_fills_budget_all_given_dropped(self):
        """Child count == max_nodes -> all given nodes dropped, all children kept."""
        max_nodes = 5
        children = [_make_det(float(i * 60), 400.0, float(i * 60 + 50), 450.0)
                    for i in range(max_nodes)]
        given_nodes = [_make_given(i * 60, 0, i * 60 + 50, 50, fine=f"main_{i}")
                       for i in range(3)]
        result = merge_given(children, given_nodes, max_nodes=max_nodes)

        assert len(result.detections) == max_nodes
        assert all(not d.given for d in result.detections)
        assert result.compose_nodes == []

    def test_operator_survives_when_non_ops_evicted(self):
        """With max_nodes-1 children and 3 given nodes (1 operator, 2 non-ops),
        exactly 1 given survives and it is the operator."""
        max_nodes = 5
        # 4 children (none overlap with given cells)
        children = [_make_det(float(i * 10), 400.0, float(i * 10 + 8), 410.0)
                    for i in range(max_nodes - 1)]
        gn_digit1 = _make_given(0, 0, 40, 40, coarse="digit_main", fine="main_1")
        gn_digit2 = _make_given(50, 0, 90, 40, coarse="digit_main", fine="main_2")
        gn_op = _make_given(100, 0, 140, 40, coarse="operator", fine="op_times")
        result = merge_given(children, [gn_digit1, gn_digit2, gn_op], max_nodes=max_nodes)

        # Total must not exceed max_nodes.
        assert len(result.detections) <= max_nodes
        # Children all kept.
        assert sum(1 for d in result.detections if not d.given) == max_nodes - 1
        # Exactly one given survivor.
        given_dets = [d for d in result.detections if d.given]
        assert len(given_dets) == 1
        # The survivor is the operator.
        assert given_dets[0].label == "operator"

    def test_zero_budget_for_given_all_evicted(self):
        """When max_nodes exactly matches child count, no room for any given."""
        max_nodes = 3
        children = [_make_det(float(i * 100), 400.0, float(i * 100 + 50), 450.0)
                    for i in range(max_nodes)]
        gn = _make_given(0, 0, 40, 40, coarse="operator", fine="op_plus")
        result = merge_given(children, [gn], max_nodes=max_nodes)
        assert len(result.detections) == max_nodes
        assert result.compose_nodes == []


# ---------------------------------------------------------------------------
# compose_into_scene
# ---------------------------------------------------------------------------

class TestComposeIntoScene:
    def _white_scene(self) -> np.ndarray:
        return np.full((512, 512), 255, dtype=np.uint8)

    def test_returns_512x512_copy_of_input(self):
        gray = self._white_scene()
        gray_id = id(gray)
        result = compose_into_scene(gray, [])
        assert result.shape == (512, 512)
        # Input not mutated.
        assert id(result) != gray_id

    def test_input_not_mutated(self):
        gray = self._white_scene()
        gray_copy = gray.copy()
        gn = _make_given(10, 10, 40, 40, tile=_dark_tile())
        compose_into_scene(gray, [gn])
        np.testing.assert_array_equal(gray, gray_copy)

    def test_wrong_shape_raises(self):
        bad = np.full((256, 256), 255, dtype=np.uint8)
        with pytest.raises(AssertionError):
            compose_into_scene(bad, [])

    def test_given_cell_region_gets_dark_pixels(self):
        """A dark tile painted into an all-white scene -> the region has dark pixels."""
        gray = self._white_scene()
        gn = _make_given(50, 50, 100, 100, tile=_dark_tile())
        result = compose_into_scene(gray, [gn])
        region = result[50:100, 50:100]
        assert region.min() == 0  # dark pixels present

    def test_child_ink_outside_given_region_preserved(self):
        """Pixels outside any given node region are untouched."""
        gray = self._white_scene()
        gray[200, 200] = 30  # child ink at (200, 200)
        gn = _make_given(50, 50, 100, 100, tile=_dark_tile())
        result = compose_into_scene(gray, [gn])
        assert result[200, 200] == 30  # child ink intact

    def test_child_ink_inside_given_region_preserved_by_min(self):
        """Child ink (dark pixel) inside a given region stays dark; darken-only."""
        gray = self._white_scene()
        gray[70, 70] = 20  # very dark child ink
        gn = _make_given(50, 50, 100, 100, tile=_white_tile())  # white tile
        result = compose_into_scene(gray, [gn])
        # White tile (255) min dark child ink (20) = 20 -- child ink preserved.
        assert result[70, 70] == 20

    def test_degenerate_bbox_skipped(self):
        """A GivenNode with zero width/height is silently skipped."""
        gray = self._white_scene()
        gn_zero_w = _make_given(50, 50, 50, 100, tile=_dark_tile())  # w=0
        gn_zero_h = _make_given(50, 50, 100, 50, tile=_dark_tile())  # h=0
        result = compose_into_scene(gray, [gn_zero_w, gn_zero_h])
        # Scene is still all-white (nothing was painted).
        assert result.min() == 255

    def test_multiple_given_nodes_all_painted(self):
        """Multiple given nodes all get their dark tiles painted."""
        gray = self._white_scene()
        gn1 = _make_given(0, 0, 50, 50, tile=_dark_tile())
        gn2 = _make_given(200, 200, 250, 250, tile=_dark_tile())
        result = compose_into_scene(gray, [gn1, gn2])
        assert result[25, 25] == 0    # gn1 region dark
        assert result[225, 225] == 0  # gn2 region dark
        assert result[100, 100] == 255  # untouched region still white


# ---------------------------------------------------------------------------
# pin_given_labels
# ---------------------------------------------------------------------------

class TestPinGivenLabels:
    def test_only_given_preds_overridden(self):
        """Non-given preds are untouched; given preds get their fine_label overridden."""
        child_pred = _make_pred(10.0, 10.0, 50.0, 50.0, fine="main_9", given=False)
        given_pred = _make_pred(100.0, 100.0, 140.0, 140.0, fine="main_0", given=True)
        known_fine = {_bbox_key(100.0, 100.0, 140.0, 140.0): "op_plus"}
        pin_given_labels([child_pred, given_pred], known_fine)
        assert child_pred.fine_label == "main_9"  # unchanged
        assert given_pred.fine_label == "op_plus"  # pinned

    def test_missing_key_is_noop(self):
        """A given pred whose bbox is absent from known_fine is silently left alone."""
        given_pred = _make_pred(100.0, 100.0, 140.0, 140.0, fine="main_0", given=True)
        pin_given_labels([given_pred], known_fine={})
        assert given_pred.fine_label == "main_0"  # unchanged

    def test_multiple_given_preds_all_pinned(self):
        """All given preds in the list are pinned to their respective fine labels."""
        gp1 = _make_pred(0.0, 0.0, 40.0, 40.0, fine="main_0", given=True)
        gp2 = _make_pred(50.0, 0.0, 90.0, 40.0, fine="main_0", given=True)
        known_fine = {
            _bbox_key(0.0, 0.0, 40.0, 40.0): "main_3",
            _bbox_key(50.0, 0.0, 90.0, 40.0): "op_minus",
        }
        pin_given_labels([gp1, gp2], known_fine)
        assert gp1.fine_label == "main_3"
        assert gp2.fine_label == "op_minus"

    def test_empty_preds_list_is_noop(self):
        """Empty preds list does not raise."""
        pin_given_labels([], {})

    def test_no_given_preds_nothing_changed(self):
        """When all preds are non-given, nothing is mutated."""
        preds = [_make_pred(float(i * 50), 0.0, float(i * 50 + 40), 40.0, fine="main_1")
                 for i in range(5)]
        known_fine = {_bbox_key(float(i * 50), 0.0, float(i * 50 + 40), 40.0): "main_9"
                      for i in range(5)}
        pin_given_labels(preds, known_fine)
        for p in preds:
            assert p.fine_label == "main_1"  # all unchanged (not given)

    def test_bbox_key_used_for_lookup(self):
        """Float coords are rounded to ints for the lookup key (sub-pixel safety)."""
        given_pred = _make_pred(100.4, 100.6, 139.5, 139.7, fine="main_0", given=True)
        # The key as _bbox_key would compute it.
        key = _bbox_key(100.4, 100.6, 139.5, 139.7)
        known_fine = {key: "main_7"}
        pin_given_labels([given_pred], known_fine)
        assert given_pred.fine_label == "main_7"
