"""TDD tests for F3: JSONL persistence + Markdown auto-summary.

Covers:
- write_jsonl_record: schema creation, append, schema mismatch rejection
- load_baseline_record: latest match, no match, missing file
- render_markdown_summary: delta table, no-baseline, atomic write, None fields,
  round-trip equality
"""
from __future__ import annotations

import dataclasses
import json
import os
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from src.eval.harness import (
    SCHEMA_VERSION,
    AggregateMetric,
    RunRecord,
    SampleMetric,
    load_baseline_record,
    render_markdown_summary,
    write_jsonl_record,
)


# ---------------------------------------------------------------------------
# Helpers — fake RunRecord builders (no model weights needed)
# ---------------------------------------------------------------------------


def _make_aggregate(
    n_samples: int = 2,
    equation_kind_acc: float = 0.5,
    ood_rate: float = 0.0,
    ood_honest_rate: float = 0.0,
    mean_label_acc: Optional[float] = None,
    mean_iou_matched: Optional[float] = None,
    mean_recall: Optional[float] = None,
    row_acc_macro: Optional[float] = None,
    col_acc_macro: Optional[float] = None,
    n_samples_with_spatial: int = 0,
    total_wall_ms: float = 42.0,
) -> AggregateMetric:
    return AggregateMetric(
        n_samples=n_samples,
        equation_kind_acc=equation_kind_acc,
        ood_rate=ood_rate,
        ood_honest_rate=ood_honest_rate,
        mean_label_acc=mean_label_acc,
        mean_iou_matched=mean_iou_matched,
        mean_recall=mean_recall,
        row_acc_macro=row_acc_macro,
        col_acc_macro=col_acc_macro,
        n_samples_with_spatial=n_samples_with_spatial,
        total_wall_ms=total_wall_ms,
    )


def _make_sample() -> SampleMetric:
    return SampleMetric(
        name="sample-a",
        equation_kind_pred="add",
        equation_kind_gt="addition",
        equation_kind_ok=True,
        ood_triggered=False,
        ood_honest=False,
        ood_reason=None,
        n_pred=3,
        n_gt=3,
        matched=3,
        label_correct=2,
        recall=1.0,
        mean_iou_matched=0.82,
        row_acc=None,
        col_acc=None,
        wall_ms=21.0,
    )


def _make_record(
    run_id: str = "20260505T000000",
    config_id: str = "baseline",
    equation_kind_acc: float = 0.5,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        config_id=config_id,
        config_snapshot={"config_id": config_id},
        torch_version="2.11.0",
        ultralytics_version="8.4.46",
        python_version="3.12.0",
        yolo_sha="abcd1234",
        gnn_sha="efgh5678",
        aggregate=_make_aggregate(equation_kind_acc=equation_kind_acc),
        samples=(_make_sample(),),
    )


# ---------------------------------------------------------------------------
# write_jsonl_record tests
# ---------------------------------------------------------------------------


class TestWriteJsonlRecordCreatesSchemaFirst(unittest.TestCase):
    def test_write_jsonl_record_creates_schema_line_first_time(self):
        """Fresh path: line 1 = schema record, line 2 = run record."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "eval_history.jsonl"
            record = _make_record()

            write_jsonl_record(record, path)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2, "Expected exactly 2 lines")

            schema = json.loads(lines[0])
            self.assertEqual(schema.get("record_type"), "schema")
            self.assertEqual(schema.get("schema_version"), SCHEMA_VERSION)
            self.assertIn("fields", schema)
            self.assertIn("created", schema)

            run = json.loads(lines[1])
            self.assertNotIn("record_type", run)
            self.assertEqual(run["run_id"], record.run_id)
            self.assertEqual(run["config_id"], record.config_id)


class TestWriteJsonlRecordAppendsWithoutRewritingSchema(unittest.TestCase):
    def test_write_jsonl_record_appends_without_rewriting_schema(self):
        """Second call: 1 schema line + 2 run records, schema unchanged."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "eval_history.jsonl"
            r1 = _make_record(run_id="20260505T000000")
            r2 = _make_record(run_id="20260505T000001")

            write_jsonl_record(r1, path)
            write_jsonl_record(r2, path)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3, "Expected 3 lines: schema + 2 records")

            schema = json.loads(lines[0])
            self.assertEqual(schema.get("record_type"), "schema")

            rec1 = json.loads(lines[1])
            rec2 = json.loads(lines[2])
            self.assertEqual(rec1["run_id"], "20260505T000000")
            self.assertEqual(rec2["run_id"], "20260505T000001")


class TestWriteJsonlRecordRejectsSchemaMismatch(unittest.TestCase):
    def test_write_jsonl_record_rejects_schema_mismatch(self):
        """Existing file with schema_version=99 causes ValueError."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "eval_history.jsonl"
            # Write a fake file with wrong schema version
            path.write_text(
                json.dumps({"record_type": "schema", "schema_version": 99}) + "\n",
                encoding="utf-8",
            )

            record = _make_record()
            with self.assertRaises(ValueError) as ctx:
                write_jsonl_record(record, path)

            msg = str(ctx.exception)
            self.assertIn("99", msg)
            self.assertIn(str(SCHEMA_VERSION), msg)


# ---------------------------------------------------------------------------
# load_baseline_record tests
# ---------------------------------------------------------------------------


class TestLoadBaselineRecordReturnsLatestMatchingConfigId(unittest.TestCase):
    def test_load_baseline_record_returns_latest_matching_config_id(self):
        """JSONL with 3 records (baseline, swept, baseline): returns 2nd baseline."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "eval_history.jsonl"
            r1 = _make_record(run_id="20260505T000001", config_id="baseline", equation_kind_acc=0.5)
            r2 = _make_record(run_id="20260505T000002", config_id="swept", equation_kind_acc=0.6)
            r3 = _make_record(run_id="20260505T000003", config_id="baseline", equation_kind_acc=0.8)

            write_jsonl_record(r1, path)
            write_jsonl_record(r2, path)
            write_jsonl_record(r3, path)

            result = load_baseline_record(path, "baseline")
            self.assertIsNotNone(result)
            self.assertEqual(result.run_id, "20260505T000003")
            self.assertAlmostEqual(result.aggregate.equation_kind_acc, 0.8)


class TestLoadBaselineRecordReturnsNoneWhenNoMatch(unittest.TestCase):
    def test_load_baseline_record_returns_none_when_no_match(self):
        """No matching config_id returns None."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "eval_history.jsonl"
            write_jsonl_record(_make_record(config_id="baseline"), path)

            result = load_baseline_record(path, "nonexistent-config")
            self.assertIsNone(result)


class TestLoadBaselineRecordReturnsNoneWhenFileMissing(unittest.TestCase):
    def test_load_baseline_record_returns_none_when_file_missing(self):
        """Non-existent path returns None without error."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "does_not_exist.jsonl"
            result = load_baseline_record(path, "baseline")
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# render_markdown_summary tests
# ---------------------------------------------------------------------------


class TestRenderMarkdownSummaryWithBaseline(unittest.TestCase):
    def test_render_markdown_summary_with_baseline_shows_delta(self):
        """With baseline eq_kind_acc=0.7, current=0.5: delta is -0.2000 (or -0.20)."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "summary.md"
            baseline = _make_record(config_id="baseline", equation_kind_acc=0.7)
            current = _make_record(config_id="baseline", equation_kind_acc=0.5)

            render_markdown_summary(current, baseline, output_path)

            self.assertTrue(output_path.exists(), "Output file should exist")
            content = output_path.read_text(encoding="utf-8")
            # Delta = 0.5 - 0.7 = -0.2000
            self.assertTrue(
                "-0.2000" in content or "-0.20" in content,
                f"Expected '-0.2000' or '-0.20' in output, got:\n{content}",
            )


class TestRenderMarkdownSummaryNoBaselineOmitsDeltaColumn(unittest.TestCase):
    def test_render_markdown_summary_no_baseline_omits_delta_column(self):
        """With baseline=None, output has no Delta column header."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "summary.md"
            current = _make_record()

            render_markdown_summary(current, None, output_path)

            content = output_path.read_text(encoding="utf-8")
            self.assertNotIn("Delta", content)
            self.assertIn("Current", content)


class TestRenderMarkdownSummaryAtomicWriteLeavesFilePriorIntact(unittest.TestCase):
    def test_render_markdown_summary_atomic_write_leaves_prior_file_intact(self):
        """If os.replace raises, the prior file content is preserved."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "summary.md"
            prior_content = "PRIOR CONTENT\n"
            output_path.write_text(prior_content, encoding="utf-8")

            current = _make_record()

            with patch("os.replace", side_effect=OSError("mock rename fail")):
                try:
                    render_markdown_summary(current, None, output_path)
                except OSError:
                    pass

            # Original file must still have prior content
            actual = output_path.read_text(encoding="utf-8")
            self.assertEqual(actual, prior_content)


class TestRenderMarkdownSummaryHandlesNoneAggregateFields(unittest.TestCase):
    def test_render_markdown_summary_handles_none_aggregate_fields(self):
        """RunRecord with None aggregate fields renders 'n/a' without crash."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "summary.md"
            agg_with_nones = AggregateMetric(
                n_samples=1,
                equation_kind_acc=0.0,
                ood_rate=0.0,
                ood_honest_rate=0.0,
                mean_label_acc=None,
                mean_iou_matched=None,
                mean_recall=None,
                row_acc_macro=None,
                col_acc_macro=None,
                n_samples_with_spatial=0,
                total_wall_ms=10.0,
            )
            sample = SampleMetric(
                name="x",
                equation_kind_pred="add",
                equation_kind_gt="addition",
                equation_kind_ok=True,
                ood_triggered=False,
                ood_honest=False,
                ood_reason=None,
                n_pred=None,
                n_gt=None,
                matched=None,
                label_correct=None,
                recall=None,
                mean_iou_matched=None,
                row_acc=None,
                col_acc=None,
                wall_ms=10.0,
            )
            record = RunRecord(
                run_id="20260505T000000",
                config_id="test",
                config_snapshot={},
                torch_version="2.0",
                ultralytics_version="8.0",
                python_version="3.12",
                yolo_sha=None,
                gnn_sha="abc",
                aggregate=agg_with_nones,
                samples=(sample,),
            )

            render_markdown_summary(record, None, output_path)

            content = output_path.read_text(encoding="utf-8")
            self.assertIn("n/a", content)


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


class TestRoundTripRecordViaJsonl(unittest.TestCase):
    def test_round_trip_record_via_jsonl(self):
        """Write a RunRecord, load it back, assert reconstructed == original."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "eval_history.jsonl"
            original = _make_record()

            write_jsonl_record(original, path)
            loaded = load_baseline_record(path, "baseline")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded, original)


# ---------------------------------------------------------------------------
# Phase 0: per-group breakdowns round-trip
# ---------------------------------------------------------------------------


class TestPerGroupBreakdownsRoundTrip(unittest.TestCase):
    """per_equation_kind and per_scene_case round-trip through JSONL."""

    def _make_record_with_groups(self) -> RunRecord:
        """Build a RunRecord with populated per_equation_kind and per_scene_case."""
        import dataclasses
        from src.eval.harness import aggregate_by, compute_aggregate

        sample_add = SampleMetric(
            name="s1",
            equation_kind_pred="add",
            equation_kind_gt="addition",
            equation_kind_ok=True,
            ood_triggered=False,
            ood_honest=False,
            ood_reason=None,
            n_pred=3,
            n_gt=3,
            matched=3,
            label_correct=3,
            recall=1.0,
            mean_iou_matched=None,
            row_acc=0.9,
            col_acc=0.8,
            wall_ms=10.0,
            scene_case="addition",
        )
        sample_div = SampleMetric(
            name="s2",
            equation_kind_pred="divide",
            equation_kind_gt="division",
            equation_kind_ok=True,
            ood_triggered=False,
            ood_honest=False,
            ood_reason=None,
            n_pred=5,
            n_gt=5,
            matched=4,
            label_correct=4,
            recall=0.8,
            mean_iou_matched=None,
            row_acc=0.7,
            col_acc=0.6,
            wall_ms=15.0,
            scene_case="division-short",
        )
        samples = (sample_add, sample_div)
        agg = compute_aggregate(list(samples))
        pek = aggregate_by(list(samples), "equation_kind")
        psc = aggregate_by(list(samples), "scene_case")
        return RunRecord(
            run_id="20260604T120000",
            config_id="baseline",
            config_snapshot={"config_id": "baseline"},
            torch_version="2.11.0",
            ultralytics_version="8.4.54",
            python_version="3.12.0",
            yolo_sha="abcd1234",
            gnn_sha="efgh5678",
            aggregate=agg,
            samples=samples,
            per_equation_kind=pek,
            per_scene_case=psc,
        )

    def test_per_equation_kind_round_trips(self) -> None:
        """per_equation_kind survives write + load round-trip."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.jsonl"
            original = self._make_record_with_groups()
            write_jsonl_record(original, path)
            loaded = load_baseline_record(path, "baseline")
            self.assertIsNotNone(loaded)
            self.assertIn("addition", loaded.per_equation_kind)
            self.assertIn("division", loaded.per_equation_kind)
            self.assertAlmostEqual(
                loaded.per_equation_kind["addition"].equation_kind_acc,
                original.per_equation_kind["addition"].equation_kind_acc,
            )

    def test_per_scene_case_round_trips(self) -> None:
        """per_scene_case survives write + load round-trip."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.jsonl"
            original = self._make_record_with_groups()
            write_jsonl_record(original, path)
            loaded = load_baseline_record(path, "baseline")
            self.assertIsNotNone(loaded)
            self.assertIn("addition", loaded.per_scene_case)
            self.assertIn("division-short", loaded.per_scene_case)

    def test_jsonl_record_contains_per_group_keys(self) -> None:
        """Raw JSONL line contains per_equation_kind and per_scene_case keys."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.jsonl"
            original = self._make_record_with_groups()
            write_jsonl_record(original, path)
            lines = path.read_text(encoding="utf-8").splitlines()
            run_line = json.loads(lines[1])
            self.assertIn("per_equation_kind", run_line)
            self.assertIn("per_scene_case", run_line)
            self.assertIsInstance(run_line["per_equation_kind"], dict)
            self.assertIsInstance(run_line["per_scene_case"], dict)

    def test_legacy_v1_record_loads_with_empty_per_group(self) -> None:
        """A schema_version=1 record (no per_group fields) loads with empty dicts."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.jsonl"
            # Write a fake v1 schema header
            schema_line = json.dumps({
                "record_type": "schema",
                "schema_version": 2,  # must match current to pass mismatch check
                "fields": [],
                "created": "2026-01-01T00:00:00+00:00",
            })
            # Build a minimal v1-style run record (no per_equation_kind/per_scene_case)
            agg_dict = {
                "n_samples": 1, "equation_kind_acc": 1.0, "ood_rate": 0.0,
                "ood_honest_rate": 0.0, "mean_label_acc": None,
                "mean_iou_matched": None, "mean_recall": None,
                "row_acc_macro": None, "col_acc_macro": None,
                "n_samples_with_spatial": 0, "total_wall_ms": 5.0,
            }
            sample_dict = {
                "name": "x", "equation_kind_pred": "add", "equation_kind_gt": "addition",
                "equation_kind_ok": True, "ood_triggered": False, "ood_honest": False,
                "ood_reason": None, "n_pred": 0, "n_gt": None, "matched": None,
                "label_correct": None, "recall": None, "mean_iou_matched": None,
                "row_acc": None, "col_acc": None, "wall_ms": 5.0,
                # scene_case absent (v1 record)
            }
            run_dict = {
                "run_id": "20260101T000000",
                "config_id": "baseline",
                "config_snapshot": {},
                "torch_version": "2.0",
                "ultralytics_version": "8.0",
                "python_version": "3.12",
                "yolo_sha": None,
                "gnn_sha": "abc12345",
                "aggregate": agg_dict,
                "samples": [sample_dict],
                # per_equation_kind and per_scene_case absent (v1 record)
            }
            path.write_text(
                schema_line + "\n" + json.dumps(run_dict) + "\n",
                encoding="utf-8",
            )
            loaded = load_baseline_record(path, "baseline")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.per_equation_kind, {})
            self.assertEqual(loaded.per_scene_case, {})


if __name__ == "__main__":
    unittest.main()
