"""Tests for src.eval.labels — sidecar schema, loader, and validation.

TDD: tests written first; implementation in src/eval/labels.py.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.eval.labels import (
    EQUATION_KINDS,
    ERROR_KINDS,
    SCENE_CASES,
    SampleLabel,
    SymbolLabel,
    load_label_sidecars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_sidecar(
    sample: str = "01-addition",
    equation_kind: str = "addition",
    symbols: list | None = None,
    include_symbols_key: bool = True,
    schema_version: int = 1,
    scene_case: str | None = None,
    error_kind: str | None = None,
) -> dict:
    """Build a minimal valid sidecar dict."""
    doc: dict = {
        "schema_version": schema_version,
        "sample": sample,
        "equation_kind": equation_kind,
    }
    if scene_case is not None:
        doc["scene_case"] = scene_case
    if error_kind is not None:
        doc["error_kind"] = error_kind
    if include_symbols_key:
        doc["symbols"] = symbols if symbols is not None else []
    return doc


def _write_sidecar(directory: Path, stem: str, doc: dict) -> Path:
    """Write a sidecar JSON file and return its path."""
    p = directory / f"{stem}.label.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# SymbolLabel / SampleLabel frozen dataclass tests
# ---------------------------------------------------------------------------

class TestSymbolLabel(unittest.TestCase):
    """SymbolLabel frozen dataclass contracts."""

    def test_symbol_label_creation(self) -> None:
        sym = SymbolLabel(flattened="main_3", bbox_px=(10.0, 20.0, 50.0, 80.0))
        self.assertEqual(sym.flattened, "main_3")
        self.assertEqual(sym.bbox_px, (10.0, 20.0, 50.0, 80.0))

    def test_sample_label_is_frozen(self) -> None:
        """Mutation of a frozen SampleLabel raises FrozenInstanceError."""
        label = SampleLabel(
            schema_version=1,
            sample="01-addition",
            equation_kind="addition",
            symbols=(),
            scene_case=None,
        )
        with self.assertRaises(FrozenInstanceError):
            label.equation_kind = "subtraction"  # type: ignore[misc]

    def test_symbol_label_is_frozen(self) -> None:
        sym = SymbolLabel(flattened="op_plus", bbox_px=(0.0, 0.0, 30.0, 30.0))
        with self.assertRaises(FrozenInstanceError):
            sym.flattened = "op_minus"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SampleLabel round-trip
# ---------------------------------------------------------------------------

class TestSampleLabelRoundTrip(unittest.TestCase):
    """Valid dict → SampleLabel → re-serialised fields match input."""

    def test_sample_label_round_trip(self) -> None:
        """A valid sidecar dict survives a parse-then-field-check round trip."""
        doc = {
            "schema_version": 1,
            "sample": "01-addition",
            "equation_kind": "addition",
            "symbols": [
                {"flattened": "main_3", "bbox_px": [10.0, 20.0, 50.0, 80.0]},
                {"flattened": "op_plus", "bbox_px": [55.0, 20.0, 90.0, 80.0]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_sidecar(d, "01-addition", doc)
            sidecars = load_label_sidecars(d)
            label = sidecars["01-addition"]

            self.assertEqual(label.schema_version, doc["schema_version"])
            self.assertEqual(label.sample, doc["sample"])
            self.assertEqual(label.equation_kind, doc["equation_kind"])
            assert label.symbols is not None
            self.assertEqual(len(label.symbols), 2)
            self.assertEqual(label.symbols[0].flattened, "main_3")
            self.assertAlmostEqual(label.symbols[0].bbox_px[2], 50.0)
            self.assertEqual(label.symbols[1].flattened, "op_plus")


# ---------------------------------------------------------------------------
# load_label_sidecars — happy path
# ---------------------------------------------------------------------------

class TestLoadLabelSidecarsFindsAllFiles(unittest.TestCase):
    """Three valid sidecars → dict size 3 keyed by filename stem."""

    def test_load_label_sidecars_finds_all_label_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            stems = ["01-addition", "02-subtraction", "03-multiplication"]
            for stem in stems:
                kind = stem.split("-", 1)[1]
                _write_sidecar(d, stem, _make_valid_sidecar(sample=stem, equation_kind=kind))
            result = load_label_sidecars(d)
            self.assertEqual(set(result.keys()), set(stems))
            self.assertEqual(len(result), 3)


class TestLoadLabelSidecarsAcceptsNoSymbolsField(unittest.TestCase):
    """Labels without symbols[] key parse; SampleLabel.symbols is None (not empty tuple)."""

    def test_load_label_sidecars_accepts_no_symbols_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(
                sample="01-addition",
                include_symbols_key=False,
            )
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            label = result["01-addition"]
            self.assertIsNone(label.symbols)


class TestLoadLabelSidecarsAcceptsEmptySymbolsList(unittest.TestCase):
    """symbols: [] parses to an empty tuple — distinct from None."""

    def test_load_label_sidecars_accepts_empty_symbols_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition", symbols=[])
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            label = result["01-addition"]
            self.assertIsNotNone(label.symbols)
            self.assertEqual(label.symbols, ())


# ---------------------------------------------------------------------------
# load_label_sidecars — rejection / validation failures
# ---------------------------------------------------------------------------

class TestLoadLabelSidecarsRejectsUnknownEquationKind(unittest.TestCase):
    """equation_kind not in EQUATION_KINDS raises ValueError mentioning the file path."""

    def test_load_label_sidecars_rejects_unknown_equation_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-foo", equation_kind="exponentiation")
            p = _write_sidecar(d, "01-foo", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))


class TestLoadLabelSidecarsRejectsUnknownFlattened(unittest.TestCase):
    """A symbol with flattened 'main_99' raises ValueError."""

    def test_load_label_sidecars_rejects_unknown_flattened(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(
                sample="01-addition",
                symbols=[{"flattened": "main_99", "bbox_px": [0.0, 0.0, 50.0, 50.0]}],
            )
            p = _write_sidecar(d, "01-addition", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))


class TestLoadLabelSidecarsBboxWithin512(unittest.TestCase):
    """bbox_px exceeding 512 raises ValueError."""

    def test_bbox_px_validates_within_512(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(
                sample="01-addition",
                symbols=[{"flattened": "main_3", "bbox_px": [0.0, 0.0, 600.0, 100.0]}],
            )
            p = _write_sidecar(d, "01-addition", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))


class TestLoadLabelSidecarsBboxX1LtX2(unittest.TestCase):
    """bbox_px with x1 >= x2 raises ValueError."""

    def test_bbox_px_validates_x1_less_than_x2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(
                sample="01-addition",
                symbols=[{"flattened": "main_3", "bbox_px": [50.0, 0.0, 30.0, 100.0]}],
            )
            p = _write_sidecar(d, "01-addition", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))


class TestLoadLabelSidecarsRejectsSampleNameMismatch(unittest.TestCase):
    """File 01-foo.label.json with sample: '02-bar' raises ValueError."""

    def test_load_label_sidecars_rejects_sample_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="02-bar", equation_kind="addition")
            p = _write_sidecar(d, "01-foo", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))


class TestLoadLabelSidecarsRejectsSchemaVersionMismatch(unittest.TestCase):
    """schema_version != 1 raises ValueError."""

    def test_load_label_sidecars_rejects_schema_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition", schema_version=99)
            p = _write_sidecar(d, "01-addition", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))


# ---------------------------------------------------------------------------
# EQUATION_KINDS constant
# ---------------------------------------------------------------------------

class TestEquationKindsConstant(unittest.TestCase):
    """EQUATION_KINDS is a frozenset with the four expected values."""

    def test_equation_kinds_is_frozenset(self) -> None:
        self.assertIsInstance(EQUATION_KINDS, frozenset)

    def test_equation_kinds_contains_expected_values(self) -> None:
        # W10: bare_digits added as a new equation_kind value.
        self.assertEqual(
            EQUATION_KINDS,
            frozenset({"addition", "subtraction", "multiplication", "division", "bare_digits"}),
        )


# ---------------------------------------------------------------------------
# Sentinel bbox [0.0, 0.0, 0.1, 0.1] — used for unlocated missed tokens
# ---------------------------------------------------------------------------

class TestSentinelBboxParsesWithoutError(unittest.TestCase):
    """Near-zero sentinel bbox [0.0, 0.0, 0.1, 0.1] must parse correctly.

    Missed tokens from feedback.json have no real bounding box.  The
    derivation script writes them with a sentinel bbox [0.0, 0.0, 0.1, 0.1]
    (x1 < x2, y1 < y2) so the existing _parse_bbox validator accepts them.
    The IoU of this sentinel against any real detection is effectively zero
    (area = 0.01 px) and will never exceed the 0.35 match threshold.
    """

    def test_sentinel_bbox_parses_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="sc-0001", equation_kind="addition")
            doc["symbols"] = [
                {"flattened": "op_plus", "bbox_px": [0.0, 0.0, 0.1, 0.1]}
            ]
            _write_sidecar(d, "sc-0001", doc)
            labels = load_label_sidecars(d)
            self.assertIn("sc-0001", labels)
            syms = labels["sc-0001"].symbols
            self.assertIsNotNone(syms)
            self.assertEqual(len(syms), 1)
            self.assertEqual(syms[0].flattened, "op_plus")
            self.assertAlmostEqual(syms[0].bbox_px[0], 0.0)
            self.assertAlmostEqual(syms[0].bbox_px[2], 0.1)

    def test_sentinel_bbox_iou_against_real_bbox_is_below_threshold(self) -> None:
        """IoU between sentinel and a typical 40x40 symbol bbox is negligible."""
        from src.parsing.match_tokens import greedy_match_scene
        gt = [{"flattened": "op_plus", "bbox_px": [0.0, 0.0, 0.1, 0.1]}]
        pred = [{"flattened": "op_plus", "bbox_px": [50.0, 50.0, 90.0, 90.0]}]
        result = greedy_match_scene(gt, pred)
        # Sentinel has near-zero overlap with a symbol elsewhere on canvas
        self.assertEqual(result["matched"], 0)


# ---------------------------------------------------------------------------
# SCENE_CASES constant
# ---------------------------------------------------------------------------

class TestSceneCasesConstant(unittest.TestCase):
    """SCENE_CASES is a frozenset derived from the SceneCase enum."""

    def test_scene_cases_is_frozenset(self) -> None:
        self.assertIsInstance(SCENE_CASES, frozenset)

    def test_scene_cases_contains_known_values(self) -> None:
        """A sample of known SceneCase values must be in SCENE_CASES."""
        for expected in ("addition", "subtraction", "division-short", "division-long",
                         "multiplication-simple", "addition_dense_carries"):
            self.assertIn(expected, SCENE_CASES, f"{expected!r} missing from SCENE_CASES")

    def test_scene_cases_non_empty(self) -> None:
        self.assertGreater(len(SCENE_CASES), 0)


# ---------------------------------------------------------------------------
# Phase 0: scene_case sidecar field
# ---------------------------------------------------------------------------

class TestSceneCaseAbsentLoadsAsNone(unittest.TestCase):
    """Sidecar without scene_case key loads with SampleLabel.scene_case == None."""

    def test_absent_scene_case_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition")  # no scene_case
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            self.assertIsNone(result["01-addition"].scene_case)


class TestSceneCaseExplicitNullLoadsAsNone(unittest.TestCase):
    """An explicit ``"scene_case": null`` is treated like a missing key (None).

    Regression: previously the ``in doc`` membership check fired on an explicit
    null and, since ``None not in SCENE_CASES``, raised ValueError, which aborted
    the entire worklist build. A null must be indistinguishable from absent.
    """

    def test_explicit_null_scene_case_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition")
            doc["scene_case"] = None  # explicit JSON null
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            self.assertIsNone(result["01-addition"].scene_case)


class TestSceneCaseValidLoads(unittest.TestCase):
    """Sidecar with valid scene_case loads with the correct value."""

    def test_valid_scene_case_loads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            valid_case = next(iter(sorted(SCENE_CASES)))  # pick any valid value
            doc = _make_valid_sidecar(sample="01-addition", scene_case=valid_case)
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            self.assertEqual(result["01-addition"].scene_case, valid_case)

    def test_addition_dense_carries_loads(self) -> None:
        """Specific real-world scene_case value loads correctly."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition", scene_case="addition_dense_carries")
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            self.assertEqual(result["01-addition"].scene_case, "addition_dense_carries")


class TestSceneCaseUnknownRaisesValueError(unittest.TestCase):
    """Sidecar with unknown scene_case raises ValueError mentioning the file path."""

    def test_unknown_scene_case_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition", scene_case="not_a_real_case")
            p = _write_sidecar(d, "01-addition", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))
            self.assertIn("not_a_real_case", str(ctx.exception))


class TestSceneCaseInDataclass(unittest.TestCase):
    """SampleLabel.scene_case default is None and field is present."""

    def test_sample_label_scene_case_default_none(self) -> None:
        label = SampleLabel(
            schema_version=1,
            sample="x",
            equation_kind="addition",
            symbols=None,
        )
        self.assertIsNone(label.scene_case)

    def test_sample_label_scene_case_set(self) -> None:
        label = SampleLabel(
            schema_version=1,
            sample="x",
            equation_kind="addition",
            symbols=None,
            scene_case="division-short",
        )
        self.assertEqual(label.scene_case, "division-short")


# ---------------------------------------------------------------------------
# ERROR_KINDS constant
# ---------------------------------------------------------------------------

class TestErrorKindsConstant(unittest.TestCase):
    """ERROR_KINDS is a frozenset with the expected as-drawn error tags."""

    def test_error_kinds_is_frozenset(self) -> None:
        self.assertIsInstance(ERROR_KINDS, frozenset)

    def test_error_kinds_contains_expected_values(self) -> None:
        self.assertEqual(
            ERROR_KINDS,
            frozenset(
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
            ),
        )


# ---------------------------------------------------------------------------
# error_kind sidecar field — as-drawn error tagging
# ---------------------------------------------------------------------------

class TestErrorKindAbsentLoadsAsNone(unittest.TestCase):
    """Sidecar without error_kind key loads with SampleLabel.error_kind == None."""

    def test_absent_error_kind_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition")  # no error_kind
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            self.assertIsNone(result["01-addition"].error_kind)


class TestErrorKindExplicitNullLoadsAsNone(unittest.TestCase):
    """An explicit ``"error_kind": null`` is treated like a missing key (None)."""

    def test_explicit_null_error_kind_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition")
            doc["error_kind"] = None  # explicit JSON null
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            self.assertIsNone(result["01-addition"].error_kind)


class TestErrorKindValidLoads(unittest.TestCase):
    """Sidecar with valid error_kind loads with the correct value."""

    def test_valid_error_kind_loads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition", error_kind="missing_carry")
            _write_sidecar(d, "01-addition", doc)
            result = load_label_sidecars(d)
            self.assertEqual(result["01-addition"].error_kind, "missing_carry")

    def test_every_error_kind_value_loads(self) -> None:
        """Each canonical ERROR_KINDS member parses back to itself."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i, kind in enumerate(sorted(ERROR_KINDS)):
                stem = f"{i:02d}-addition"
                doc = _make_valid_sidecar(sample=stem, error_kind=kind)
                _write_sidecar(d, stem, doc)
            result = load_label_sidecars(d)
            loaded = {lbl.error_kind for lbl in result.values()}
            self.assertEqual(loaded, set(ERROR_KINDS))


class TestErrorKindUnknownRaisesValueError(unittest.TestCase):
    """Sidecar with unknown error_kind raises ValueError mentioning the file path."""

    def test_unknown_error_kind_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            doc = _make_valid_sidecar(sample="01-addition", error_kind="not_a_real_error")
            p = _write_sidecar(d, "01-addition", doc)
            with self.assertRaises(ValueError) as ctx:
                load_label_sidecars(d)
            self.assertIn(str(p), str(ctx.exception))
            self.assertIn("not_a_real_error", str(ctx.exception))


class TestErrorKindInDataclass(unittest.TestCase):
    """SampleLabel.error_kind default is None and field is present + settable."""

    def test_sample_label_error_kind_default_none(self) -> None:
        label = SampleLabel(
            schema_version=1,
            sample="x",
            equation_kind="addition",
            symbols=None,
        )
        self.assertIsNone(label.error_kind)

    def test_sample_label_error_kind_set(self) -> None:
        label = SampleLabel(
            schema_version=1,
            sample="x",
            equation_kind="addition",
            symbols=None,
            scene_case="division-short",
            error_kind="wrong_result",
        )
        self.assertEqual(label.error_kind, "wrong_result")

    def test_positional_construction_preserved(self) -> None:
        """error_kind appends after scene_case; positional construction still works."""
        label = SampleLabel(1, "x", "addition", None, "division-short", "wrong_borrow")
        self.assertEqual(label.scene_case, "division-short")
        self.assertEqual(label.error_kind, "wrong_borrow")


if __name__ == "__main__":
    unittest.main()
