"""Tests for src/grading/given_prior_builder.py (Workstream B).

Three test groups:

1. honesty  — build_given_prior raises ValueError when a non-given token leaks
              in; expected_to_fill() is never called.
2. correctness — every returned GivenNode has the correct bbox (== cell_to_pixels
                 as ints), valid ontology coarse + fine labels, and a (28,28)
                 uint8 tile.
3. count    — len(result) == len(given_tokens()).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.grading import exercises as ex
from src.grading.exercises import BankExercise, ExpectedToken
from src.grading.given_prior_builder import _render_tile, _resolve_labels, build_given_prior
from src.grading.grid_mapping import GridCalibration, build_calibration
from src.grading.mock_client import MockGradingClient
from src.grading.types import ExerciseSpec
from src.core.ontology import (
    YOLO_CLASS_NAMES,
    full_label_from_gnn_and_yolo,
    gnn_fine_labels_ordered,
)
from src.core.run_config import GradingConfig
from src.inference.given_prior import GivenNode

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_GNN_FINE_LABELS: frozenset[str] = frozenset(gnn_fine_labels_ordered())
_YOLO_COARSE_LABELS: frozenset[str] = frozenset(YOLO_CLASS_NAMES)

# GivenNode.fine_label carries the EXPANDED assemble form (e.g. "main_3", "op_plus"),
# NOT the 16-class bare GNN vocab ("3").  Build the valid set by crossing all
# GNN vocab entries against all YOLO coarse classes via full_label_from_gnn_and_yolo.
_EXPANDED_FINE_LABELS: frozenset[str] = frozenset(
    full_label_from_gnn_and_yolo(gnn, yolo)
    for gnn in gnn_fine_labels_ordered()
    for yolo in YOLO_CLASS_NAMES
)


# ---------------------------------------------------------------------------
# Shared fixture helpers (pattern from test_calibration_integration.py)
# ---------------------------------------------------------------------------


def _derive_bank(exercise_id: str) -> BankExercise:
    """Derive BankExercise from captured fixtures via MockGradingClient."""
    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    return client._sessions[created.session_id].exercise


def _calibration_from_evaluate_origin(entry: BankExercise) -> GridCalibration:
    """Build the standard 512x512 calibration for an exercise."""
    engine_x_left, engine_y_top = entry.evaluate_engine_origin
    return build_calibration(
        {"width": entry.grid_width, "height": entry.grid_height},
        {
            "canvas_px": 512.0,
            "engine_x_left": engine_x_left,
            "engine_y_top": engine_y_top,
        },
    )


# ---------------------------------------------------------------------------
# 1. Honesty tests
# ---------------------------------------------------------------------------


class _FakeSpec:
    """Minimal stub for ExerciseSpec (only exercise_id is used by BankExercise)."""

    exercise_id = "fake-exercise"


def _make_minimal_bank(tokens: tuple[ExpectedToken, ...]) -> BankExercise:
    """Build a minimal BankExercise with a given token set for testing."""
    spec = MagicMock(spec=ExerciseSpec)
    spec.exercise_id = "fake-exercise"
    return BankExercise(
        spec=spec,
        expected=tokens,
        grid_width=5,
        grid_height=5,
        answer_y=0,
        evaluate_engine_origin=(0, 4),
    )


def _simple_calibration(grid_w: int = 5, grid_h: int = 5) -> GridCalibration:
    """Build a simple 512x512 calibration over a small grid."""
    return build_calibration(
        {"width": grid_w, "height": grid_h},
        {"canvas_px": 512.0, "engine_x_left": 0, "engine_y_top": grid_h - 1},
    )


class TestHonesty:
    """Honesty guard: build_given_prior must raise when a non-given token leaks."""

    def test_raises_on_non_given_token(self) -> None:
        """A bank entry whose given_tokens() yields a token with given=False
        must cause build_given_prior to raise ValueError."""
        # Construct a token that claims to be a scaffold token but has given=False.
        # This simulates a bug where expected_to_fill() content slips into the
        # given set (which should be impossible via the normal API, but we guard
        # against it defensively).
        bad_token = ExpectedToken(x=0, y=4, c="3", role="operand", given=False)
        good_token = ExpectedToken(x=1, y=4, c="8", role="operand", given=True)

        # Patch given_tokens() to return a mix that includes the bad token.
        bank = _make_minimal_bank((good_token, bad_token))
        calibration = _simple_calibration()

        # Monkey-patch given_tokens to return both tokens (bypassing the normal
        # filter that would exclude non-given tokens, simulating a definition bug).
        bank_patched = MagicMock(wraps=bank)
        bank_patched.given_tokens.return_value = (good_token, bad_token)

        with pytest.raises(ValueError, match="given=False"):
            build_given_prior(bank_patched, calibration)

    def test_expected_to_fill_never_called(self) -> None:
        """build_given_prior must never call expected_to_fill on the bank entry."""
        token = ExpectedToken(x=0, y=4, c="5", role="operand", given=True)
        bank = _make_minimal_bank((token,))
        calibration = _simple_calibration()

        bank_mock = MagicMock(wraps=bank)
        bank_mock.given_tokens.return_value = (token,)

        build_given_prior(bank_mock, calibration)

        bank_mock.expected_to_fill.assert_not_called()

    def test_raises_with_clear_message(self) -> None:
        """The ValueError message must clearly indicate the given=False token."""
        bad = ExpectedToken(x=2, y=3, c="7", role="answer", given=False)
        bank = _make_minimal_bank((bad,))
        calibration = _simple_calibration()

        bank_mock = MagicMock(wraps=bank)
        bank_mock.given_tokens.return_value = (bad,)

        with pytest.raises(ValueError) as exc_info:
            build_given_prior(bank_mock, calibration)

        # The message must mention given=False so it is diagnosable.
        assert "given=False" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Correctness tests — product-38-29
# ---------------------------------------------------------------------------


class TestCorrectnessProduct:
    """Correctness checks on the multiplication (38 x 29) exercise."""

    @pytest.fixture(scope="class")
    def bank(self) -> BankExercise:
        return _derive_bank("product-38-29")

    @pytest.fixture(scope="class")
    def calibration(self, bank: BankExercise) -> GridCalibration:
        return _calibration_from_evaluate_origin(bank)

    @pytest.fixture(scope="class")
    def nodes(self, bank: BankExercise, calibration: GridCalibration) -> list[GivenNode]:
        return build_given_prior(bank, calibration)

    def test_count_equals_given_tokens(
        self, bank: BankExercise, nodes: list[GivenNode]
    ) -> None:
        """Number of GivenNodes must equal number of given tokens."""
        assert len(nodes) == len(bank.given_tokens())

    def test_all_nodes_are_given_node_instances(self, nodes: list[GivenNode]) -> None:
        assert all(isinstance(n, GivenNode) for n in nodes)

    def test_bboxes_match_cell_to_pixels(
        self, bank: BankExercise, calibration: GridCalibration, nodes: list[GivenNode]
    ) -> None:
        """Every GivenNode bbox must equal cell_to_pixels(token.x, token.y) as ints."""
        for token, node in zip(bank.given_tokens(), nodes):
            left, top, right, bottom = calibration.cell_to_pixels(token.x, token.y)
            assert node.x0 == int(left), f"x0 mismatch for token {token}"
            assert node.y0 == int(top), f"y0 mismatch for token {token}"
            assert node.x1 == int(right), f"x1 mismatch for token {token}"
            assert node.y1 == int(bottom), f"y1 mismatch for token {token}"

    def test_coarse_labels_are_valid_yolo_classes(self, nodes: list[GivenNode]) -> None:
        """Every coarse_label must be a member of the YOLO ontology."""
        for node in nodes:
            assert node.coarse_label in _YOLO_COARSE_LABELS, (
                f"coarse_label {node.coarse_label!r} not in YOLO_CLASS_NAMES"
            )

    def test_fine_labels_are_valid_expanded_labels(self, nodes: list[GivenNode]) -> None:
        """Every fine_label must be a valid expanded assemble-form label (e.g. "main_3").

        GivenNode.fine_label is the full_label_from_gnn_and_yolo output that goes
        directly into NodePrediction.fine_label via pin_given_labels.  It must
        match the expanded form, not the bare 16-class GNN vocab.
        """
        for node in nodes:
            assert node.fine_label in _EXPANDED_FINE_LABELS, (
                f"fine_label {node.fine_label!r} not in expanded fine-label set"
            )

    def test_tile_shape_and_dtype(self, nodes: list[GivenNode]) -> None:
        """Every tile must be (28, 28) uint8."""
        for node in nodes:
            assert node.tile.shape == (28, 28), (
                f"tile shape {node.tile.shape!r} != (28, 28)"
            )
            assert node.tile.dtype == np.uint8, (
                f"tile dtype {node.tile.dtype} != uint8"
            )

    def test_operand_digits_map_to_digit_main(
        self, bank: BankExercise, nodes: list[GivenNode]
    ) -> None:
        """Operand digits must get coarse=digit_main and fine=main_<digit>.

        The expanded form (e.g. "main_3") matches what a child-detected digit
        carries on NodePrediction.fine_label after full_label_from_gnn_and_yolo,
        so pin_given_labels produces an identical label string for both paths.
        """
        for token, node in zip(bank.given_tokens(), nodes):
            if token.role == "operand" and len(token.c) == 1 and token.c.isdigit():
                assert node.coarse_label == "digit_main", token
                assert node.fine_label == f"main_{token.c}", token

    def test_result_bar_maps_correctly(
        self, bank: BankExercise, nodes: list[GivenNode]
    ) -> None:
        """result_bar tokens must map to coarse=result_bar, fine=result_bar."""
        bar_pairs = [
            (t, n) for t, n in zip(bank.given_tokens(), nodes)
            if t.role == "result_bar"
        ]
        assert bar_pairs, "expected at least one result_bar given token for product-38-29"
        for token, node in bar_pairs:
            assert node.coarse_label == "result_bar", token
            assert node.fine_label == "result_bar", token

    def test_operator_times_maps_correctly(
        self, bank: BankExercise, nodes: list[GivenNode]
    ) -> None:
        """The multiplication operator token must map to op_times."""
        op_pairs = [
            (t, n) for t, n in zip(bank.given_tokens(), nodes)
            if t.role == "operator"
        ]
        assert op_pairs, "expected at least one operator given token for product-38-29"
        for token, node in op_pairs:
            assert node.coarse_label == "operator", token
            assert node.fine_label == "op_times", token


# ---------------------------------------------------------------------------
# 3. Correctness tests — subtract-256-89
# ---------------------------------------------------------------------------


class TestCorrectnessSubtract:
    """Correctness checks on the subtraction (256 - 89) exercise."""

    @pytest.fixture(scope="class")
    def bank(self) -> BankExercise:
        return _derive_bank("subtract-256-89")

    @pytest.fixture(scope="class")
    def calibration(self, bank: BankExercise) -> GridCalibration:
        return _calibration_from_evaluate_origin(bank)

    @pytest.fixture(scope="class")
    def nodes(self, bank: BankExercise, calibration: GridCalibration) -> list[GivenNode]:
        return build_given_prior(bank, calibration)

    def test_count_equals_given_tokens(
        self, bank: BankExercise, nodes: list[GivenNode]
    ) -> None:
        assert len(nodes) == len(bank.given_tokens())

    def test_bboxes_match_cell_to_pixels(
        self, bank: BankExercise, calibration: GridCalibration, nodes: list[GivenNode]
    ) -> None:
        for token, node in zip(bank.given_tokens(), nodes):
            left, top, right, bottom = calibration.cell_to_pixels(token.x, token.y)
            assert node.x0 == int(left)
            assert node.y0 == int(top)
            assert node.x1 == int(right)
            assert node.y1 == int(bottom)

    def test_coarse_and_fine_labels_valid(self, nodes: list[GivenNode]) -> None:
        for node in nodes:
            assert node.coarse_label in _YOLO_COARSE_LABELS
            assert node.fine_label in _EXPANDED_FINE_LABELS, (
                f"fine_label {node.fine_label!r} not in expanded fine-label set"
            )

    def test_tile_shape_and_dtype(self, nodes: list[GivenNode]) -> None:
        for node in nodes:
            assert node.tile.shape == (28, 28)
            assert node.tile.dtype == np.uint8

    def test_operator_minus_maps_correctly(
        self, bank: BankExercise, nodes: list[GivenNode]
    ) -> None:
        op_pairs = [
            (t, n) for t, n in zip(bank.given_tokens(), nodes)
            if t.role == "operator"
        ]
        assert op_pairs, "expected at least one operator token for subtract-256-89"
        for token, node in op_pairs:
            assert node.coarse_label == "operator"
            assert node.fine_label == "op_minus"

    def test_result_bar_present(
        self, bank: BankExercise, nodes: list[GivenNode]
    ) -> None:
        bar_nodes = [n for t, n in zip(bank.given_tokens(), nodes) if t.role == "result_bar"]
        assert bar_nodes, "expected result_bar tokens for subtract-256-89"
        for node in bar_nodes:
            assert node.coarse_label == "result_bar"
            assert node.fine_label == "result_bar"


# ---------------------------------------------------------------------------
# 4. _render_tile unit tests
# ---------------------------------------------------------------------------


class TestRenderTile:
    """Unit tests for the _render_tile helper."""

    def test_digit_tile_shape_dtype(self) -> None:
        tile = _render_tile("5", "operand")
        assert tile.shape == (28, 28)
        assert tile.dtype == np.uint8

    def test_result_bar_tile(self) -> None:
        tile = _render_tile("_", "result_bar")
        assert tile.shape == (28, 28)
        assert tile.dtype == np.uint8
        # A horizontal bar means there should be some dark pixels (< 200) in
        # the vertical centre.
        yc = 14
        assert tile[yc, :].min() < 100, "result_bar tile should have dark pixels at centre"

    def test_div_bracket_tile(self) -> None:
        tile = _render_tile("", "div_bracket")
        assert tile.shape == (28, 28)
        assert tile.dtype == np.uint8
        # Left vertical stroke and top horizontal stroke should have dark pixels.
        assert tile[:, 4].min() < 100, "div_bracket tile should have dark pixels in left column"
        assert tile[3, :].min() < 100, "div_bracket tile should have dark pixels in top row"

    def test_operator_tile(self) -> None:
        tile = _render_tile("+", "operator")
        assert tile.shape == (28, 28)
        assert tile.dtype == np.uint8

    def test_tile_values_in_range(self) -> None:
        for char, role in [("3", "operand"), ("_", "result_bar"), ("", "div_bracket"), ("*", "operator")]:
            tile = _render_tile(char, role)
            assert tile.min() >= 0
            assert tile.max() <= 255


# ---------------------------------------------------------------------------
# 5. Schema-consistency: given digit label == child-detected label
# ---------------------------------------------------------------------------


class TestSchemaConsistency:
    """Given-path tokens must produce the same assembled token ``label`` as
    the child-detected path for the same digit.

    The child path in gnn.py calls ``full_label_from_gnn_and_yolo(gnn_char, yolo_coarse)``
    which produces e.g. ``"main_3"`` for a main-digit 3.  The given path sets
    ``GivenNode.fine_label`` to the same string; ``pin_given_labels`` then
    writes it to ``NodePrediction.fine_label``; and ``assemble_json`` emits it
    verbatim as the assembled token ``"label"``.  These tests verify that chain.
    """

    @pytest.mark.parametrize("digit", list("0123456789"))
    def test_given_operand_fine_label_matches_child_path(self, digit: str) -> None:
        """_resolve_labels for an operand digit must produce the same fine_label
        as full_label_from_gnn_and_yolo produces for a child-detected main digit."""
        # Child path: GNN predicts bare digit, YOLO says digit_main.
        child_fine = full_label_from_gnn_and_yolo(digit, "digit_main")

        # Given path: _resolve_labels maps the char + role.
        _coarse, given_fine = _resolve_labels(digit, "operand")

        assert given_fine == child_fine, (
            f"Given digit {digit!r}: _resolve_labels returned fine_label={given_fine!r} "
            f"but full_label_from_gnn_and_yolo returns {child_fine!r}. "
            "The two paths must produce identical labels so assembled tokens are "
            "indistinguishable for given vs child-detected digits."
        )
        assert given_fine == f"main_{digit}", (
            f"Expected 'main_{digit}', got {given_fine!r}"
        )

    def test_given_digit_assemble_label_equals_child_assemble_label(self) -> None:
        """A given digit GivenNode, after pin_given_labels, must emit the same
        assembled token ``label`` as a child-detected NodePrediction for the
        same digit position.

        This is the end-to-end schema-consistency check: build a GivenNode for
        digit '3' at an arbitrary bbox, create a matching NodePrediction (as the
        GNN would produce for the same symbol), call pin_given_labels, then verify
        the NodePrediction.fine_label equals what the child path would have set.
        """
        from src.inference.given_prior import GivenNode, pin_given_labels
        from src.parsing.assemble import NodePrediction

        digit = "3"
        x0, y0, x1, y1 = 100, 50, 140, 90  # arbitrary 512-space bbox

        # Resolve the given node's labels.
        coarse, fine = _resolve_labels(digit, "operand")

        # Build the GivenNode (tile is irrelevant for label check).
        tile = np.zeros((28, 28), dtype=np.uint8)
        gn = GivenNode(x0=x0, y0=y0, x1=x1, y1=y1,
                       coarse_label=coarse, fine_label=fine, tile=tile)

        # Build a NodePrediction as the GNN would produce (fine_label initially
        # holds the GNN-predicted value; pin_given_labels will overwrite it).
        pred = NodePrediction(
            fine_label="main_0",  # GNN made a wrong prediction — will be pinned
            row_cluster_id=0,
            col_cluster_id=0,
            within_row_ord=0,
            within_col_ord=0,
            x0=float(x0),
            y0=float(y0),
            x1=float(x1),
            y1=float(y1),
            confidence=0.9,
            given=True,
        )

        # Pin the given label (simulating run_inference_with_models).
        known_fine = {(x0, y0, x1, y1): gn.fine_label}
        pin_given_labels([pred], known_fine)

        # The child-detected equivalent: full_label_from_gnn_and_yolo would give:
        child_fine = full_label_from_gnn_and_yolo(digit, "digit_main")

        assert pred.fine_label == child_fine, (
            f"After pin_given_labels, pred.fine_label={pred.fine_label!r} but "
            f"child-path full_label_from_gnn_and_yolo returns {child_fine!r}. "
            "Schema consistency is broken."
        )
        assert pred.fine_label == f"main_{digit}"

    def test_child_fill_roles_raise_in_resolve_labels(self) -> None:
        """Defense-in-depth: _resolve_labels must raise ValueError for child-fill
        roles (carry, borrow, partial, answer) to prevent them from reaching the
        prior builder."""
        for role in ("carry", "borrow", "partial", "answer"):
            with pytest.raises(ValueError, match="child-fill role"):
                _resolve_labels("5", role)
