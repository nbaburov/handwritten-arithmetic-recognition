"""Tests for src.setmaker.worklist — the deficit/gap worklist planner (WS-C).

TDD-style behavioural coverage for the keystone planner:
- Eval is deficit-driven: existing per-case counts are subtracted from the quota,
  met cases are skipped, and the order is biggest deficit first.
- Train is gap-driven: ``round_budget`` is allocated proportional to a weakness
  score in which row/col placement dominates, clamped by a per-case floor and cap.
- No usable eval history triggers a logged warning and a provisional-weights
  fallback rather than a crash.
- Eval and train seeds occupy disjoint namespaces (risk R7), and within a mode each
  ``(case, completion_stage)`` bucket carves out its own seed sub-range.

Tests build a self-contained temporary project tree (``data/setmaker/quota.json``,
``data/eval/real``, ``data/eval/bank``, ``reports/eval/history.jsonl``) and pass it
via the ``project_root`` keyword, so nothing reads the real repository state.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional

from src.setmaker.types import WorklistItem
from src.setmaker.worklist import (
    ERROR_SEED_BASE,
    EVAL_SEED_BASE,
    SEED_BLOCK,
    TRAIN_ERROR_SEED_BASE,
    TRAIN_PER_CASE_FLOOR,
    TRAIN_SEED_BASE,
    WEIGHT_COL,
    WEIGHT_EQ_KIND,
    WEIGHT_LABEL,
    WEIGHT_ROW,
    _all_buckets,
    _seed_for,
    _weakness_score,
    build_eval_worklist,
    build_train_worklist,
)
from src.generation.completion_stages import valid_stages_for_case
from src.generation.layouts_types import SceneCase


# ---------------------------------------------------------------------------
# Helpers — build a minimal but valid temporary project tree
# ---------------------------------------------------------------------------

def _write_quota(
    root: Path,
    eval_cases: Dict[str, int],
    train_cases: Dict[str, int],
    *,
    schema_version: int = 1,
) -> Path:
    """Write a ``data/setmaker/quota.json`` with the given per-case blocks."""
    quota_path = root / "data" / "setmaker" / "quota.json"
    quota_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": schema_version,
        "eval": {"scene_cases": dict(eval_cases)},
        "train": {"scene_cases": dict(train_cases)},
    }
    quota_path.write_text(json.dumps(doc), encoding="utf-8")
    return quota_path


def _write_sidecar(
    samples_dir: Path,
    stem: str,
    equation_kind: str,
    scene_case: Optional[str],
) -> None:
    """Write a schema-1 label sidecar (no symbols key) into ``samples_dir``."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "schema_version": 1,
        "sample": stem,
        "equation_kind": equation_kind,
    }
    if scene_case is not None:
        doc["scene_case"] = scene_case
    (samples_dir / f"{stem}.label.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_history(
    root: Path,
    records: List[dict],
    *,
    schema_version: int = 2,
) -> Path:
    """Write ``reports/eval/history.jsonl``: a schema line then run records."""
    history_path = root / "reports" / "eval" / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"record_type": "schema", "schema_version": schema_version})]
    lines.extend(json.dumps(r) for r in records)
    history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return history_path


def _aggregate_metric(
    *,
    row_acc_macro: Optional[float] = 1.0,
    col_acc_macro: Optional[float] = 1.0,
    equation_kind_acc: float = 1.0,
    mean_label_acc: Optional[float] = 1.0,
) -> dict:
    """A serialised AggregateMetric dict (only the fields the planner reads)."""
    return {
        "row_acc_macro": row_acc_macro,
        "col_acc_macro": col_acc_macro,
        "equation_kind_acc": equation_kind_acc,
        "mean_label_acc": mean_label_acc,
    }


def _by_case(items: List[WorklistItem]) -> Dict[str, int]:
    """Count worklist items per case."""
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.case] = counts.get(it.case, 0) + 1
    return counts


# ===========================================================================
# Eval: deficit-driven planner
# ===========================================================================

class TestEvalDeficitMath(unittest.TestCase):
    """build_eval_worklist subtracts existing counts and skips met cases."""

    def test_deficit_subtracts_existing_and_skips_met(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 5, "subtraction": 3, "division-short": 4},
                train_cases={"addition": 40},
            )
            real = root / "data" / "eval" / "real"
            # addition: 2 existing -> deficit 3. subtraction: 3 existing -> met (0).
            _write_sidecar(real, "re-addition-001", "addition", "addition")
            _write_sidecar(real, "re-addition-002", "addition", "addition")
            for i in range(3):
                _write_sidecar(real, f"re-sub-{i}", "subtraction", "subtraction")

            items = build_eval_worklist(project_root=root)
            counts = _by_case(items)

            # addition short by 3, division-short short by full 4, subtraction met.
            self.assertEqual(counts.get("addition"), 3)
            self.assertEqual(counts.get("division-short"), 4)
            self.assertNotIn("subtraction", counts)
            self.assertEqual(len(items), 7)

    def test_existing_at_or_over_quota_yields_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 2},
                train_cases={"addition": 40},
            )
            real = root / "data" / "eval" / "real"
            # Three existing for a quota of two -> over quota, no deficit.
            for i in range(3):
                _write_sidecar(real, f"re-add-{i}", "addition", "addition")

            self.assertEqual(build_eval_worklist(project_root=root), [])

    def test_counts_both_real_and_bank(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 5},
                train_cases={"addition": 40},
            )
            real = root / "data" / "eval" / "real"
            bank = root / "data" / "eval" / "bank"
            _write_sidecar(real, "re-add-1", "addition", "addition")
            _write_sidecar(bank, "ba-add-1", "addition", "addition")
            _write_sidecar(bank, "ba-add-2", "addition", "addition")

            # 1 (real) + 2 (bank) = 3 existing; quota 5 -> deficit 2.
            items = build_eval_worklist(project_root=root)
            self.assertEqual(_by_case(items).get("addition"), 2)

    def test_same_stem_in_both_dirs_counts_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 5},
                train_cases={"addition": 40},
            )
            real = root / "data" / "eval" / "real"
            bank = root / "data" / "eval" / "bank"
            # Identical stem present in BOTH dirs must be merged, not double-counted.
            _write_sidecar(real, "re-add-shared", "addition", "addition")
            _write_sidecar(bank, "re-add-shared", "addition", "addition")
            # Plus one unique scene so the total existing count is exactly 2.
            _write_sidecar(bank, "ba-add-1", "addition", "addition")

            # Shared stem counts once + 1 unique = 2 existing; quota 5 -> deficit 3.
            items = build_eval_worklist(project_root=root)
            self.assertEqual(_by_case(items).get("addition"), 3)

    def test_sidecar_without_scene_case_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 3},
                train_cases={"addition": 40},
            )
            bank = root / "data" / "eval" / "bank"
            # Bank-style sidecar with no scene_case must not subtract from any case.
            _write_sidecar(bank, "ba-add-1", "addition", None)
            _write_sidecar(bank, "ba-add-2", "addition", None)

            items = build_eval_worklist(project_root=root)
            self.assertEqual(_by_case(items).get("addition"), 3)

    def test_missing_eval_dirs_treated_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 4},
                train_cases={"addition": 40},
            )
            # Neither data/eval/real nor data/eval/bank exists -> full deficit.
            items = build_eval_worklist(project_root=root)
            self.assertEqual(_by_case(items).get("addition"), 4)


class TestEvalOrdering(unittest.TestCase):
    """Cases are emitted biggest deficit first, ties broken by name."""

    def test_biggest_deficit_first(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 2, "subtraction": 9, "division-short": 5},
                train_cases={"addition": 40},
            )
            items = build_eval_worklist(project_root=root)
            # First-seen case order should be deficit-descending: sub(9), div(5), add(2).
            seen: List[str] = []
            for it in items:
                if it.case not in seen:
                    seen.append(it.case)
            self.assertEqual(seen, ["subtraction", "division-short", "addition"])

    def test_equal_deficit_tie_broken_by_case_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"subtraction": 3, "addition": 3},
                train_cases={"addition": 40},
            )
            items = build_eval_worklist(project_root=root)
            seen: List[str] = []
            for it in items:
                if it.case not in seen:
                    seen.append(it.case)
            # Equal deficit (3 each) -> alphabetical: addition before subtraction.
            self.assertEqual(seen, ["addition", "subtraction"])


class TestEvalCompletionStagesPinned(unittest.TestCase):
    """Items pin a valid completion stage; multi-stage cases cycle them."""

    def test_stage_is_valid_for_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 5, "bare_digits": 3},
                train_cases={"addition": 40},
            )
            items = build_eval_worklist(project_root=root)
            for it in items:
                valid = valid_stages_for_case(SceneCase(it.case))
                self.assertIn(
                    it.completion_stage,
                    valid,
                    f"{it.completion_stage} not valid for {it.case}",
                )

    def test_multi_stage_case_cycles_stages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # addition has 5 valid stages; ask for 5 -> each stage used once.
            _write_quota(
                root,
                eval_cases={"addition": 5},
                train_cases={"addition": 40},
            )
            items = build_eval_worklist(project_root=root)
            stages = [it.completion_stage for it in items]
            self.assertEqual(set(stages), set(valid_stages_for_case(SceneCase.addition)))

    def test_full_only_case_uses_full(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # bare_digits is a full-only structure case.
            _write_quota(
                root,
                eval_cases={"bare_digits": 4},
                train_cases={"bare_digits": 20},
            )
            items = build_eval_worklist(project_root=root)
            self.assertTrue(all(it.completion_stage == "full" for it in items))


class TestBareDigitGridCase(unittest.TestCase):
    """The multi-row/col bare_digit_grid case (gap 1) is servable by the planner."""

    def test_grid_quota_is_served(self) -> None:
        # A quota with bare_digit_grid yields exactly that many worklist items,
        # all in the bare_digit_grid case (no existing scenes => full deficit).
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"bare_digit_grid": 5},
                train_cases={"bare_digit_grid": 20},
            )
            items = build_eval_worklist(project_root=root)
            self.assertEqual(_by_case(items), {"bare_digit_grid": 5})

    def test_grid_stage_pinned_to_full(self) -> None:
        # bare_digit_grid is a full-only case (no progression); every planned item
        # pins the "full" completion stage, matching valid_stages_for_case.
        self.assertEqual(
            valid_stages_for_case(SceneCase.bare_digit_grid), ["full"]
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"bare_digit_grid": 3},
                train_cases={"bare_digit_grid": 20},
            )
            items = build_eval_worklist(project_root=root)
            self.assertTrue(all(it.completion_stage == "full" for it in items))

    def test_grid_deficit_drops_existing(self) -> None:
        # Existing bare_digit_grid sidecars count against the quota deficit.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"bare_digit_grid": 4},
                train_cases={"bare_digit_grid": 20},
            )
            real_dir = root / "data" / "eval" / "real"
            _write_sidecar(real_dir, "re-grid-001", "bare_digits", "bare_digit_grid")
            items = build_eval_worklist(project_root=root)
            self.assertEqual(_by_case(items), {"bare_digit_grid": 3})


class TestEvalDeterminism(unittest.TestCase):
    """Two builds on the same tree produce identical worklists."""

    def test_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 4, "division-short": 6},
                train_cases={"addition": 40},
            )
            first = build_eval_worklist(project_root=root)
            second = build_eval_worklist(project_root=root)
            self.assertEqual(first, second)


# ===========================================================================
# Train: gap-driven planner
# ===========================================================================

class TestTrainGapAllocation(unittest.TestCase):
    """Allocation is proportional, floored, capped, and budget-exact."""

    def test_proportional_with_budget_exact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60, "subtraction": 60, "division-short": 60},
            )
            # addition weak on row/col; others strong -> addition gets the bulk.
            per_scene_case = {
                "addition": _aggregate_metric(row_acc_macro=0.2, col_acc_macro=0.2),
                "subtraction": _aggregate_metric(),  # all 1.0 -> weakness 0
                "division-short": _aggregate_metric(),
            }
            _write_history(root, [{"per_scene_case": per_scene_case}])

            items = build_train_worklist(30, project_root=root)
            counts = _by_case(items)
            self.assertEqual(sum(counts.values()), 30)
            # Weak case dominates the share.
            self.assertGreater(counts["addition"], counts["subtraction"])
            self.assertGreater(counts["addition"], counts["division-short"])

    def test_row_col_weighting_dominates_label_eqkind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60, "subtraction": 60},
            )
            # addition: bad row/col, perfect label+eqk.
            # subtraction: perfect row/col, bad label+eqk (same raw 0.0 accuracies).
            per_scene_case = {
                "addition": _aggregate_metric(
                    row_acc_macro=0.0, col_acc_macro=0.0,
                    equation_kind_acc=1.0, mean_label_acc=1.0,
                ),
                "subtraction": _aggregate_metric(
                    row_acc_macro=1.0, col_acc_macro=1.0,
                    equation_kind_acc=0.0, mean_label_acc=0.0,
                ),
            }
            _write_history(root, [{"per_scene_case": per_scene_case}])

            items = build_train_worklist(40, project_root=root)
            counts = _by_case(items)
            # Row/col are weighted heavier, so the row/col-weak case wins more budget.
            self.assertGreater(counts["addition"], counts["subtraction"])

    def test_floor_guarantees_coverage_for_weak_signal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60, "subtraction": 60, "division-short": 60},
            )
            # addition extremely weak; others perfect -> floor still covers them.
            per_scene_case = {
                "addition": _aggregate_metric(
                    row_acc_macro=0.0, col_acc_macro=0.0,
                    equation_kind_acc=0.0, mean_label_acc=0.0,
                ),
                "subtraction": _aggregate_metric(),
                "division-short": _aggregate_metric(),
            }
            _write_history(root, [{"per_scene_case": per_scene_case}])

            items = build_train_worklist(40, project_root=root)
            counts = _by_case(items)
            self.assertGreaterEqual(counts.get("subtraction", 0), TRAIN_PER_CASE_FLOOR)
            self.assertGreaterEqual(counts.get("division-short", 0), TRAIN_PER_CASE_FLOOR)

    def test_cap_limits_per_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 5, "subtraction": 5},
            )
            # Both weak, but caps are small -> total bounded by sum of caps.
            per_scene_case = {
                "addition": _aggregate_metric(row_acc_macro=0.0, col_acc_macro=0.0),
                "subtraction": _aggregate_metric(row_acc_macro=0.0, col_acc_macro=0.0),
            }
            _write_history(root, [{"per_scene_case": per_scene_case}])

            items = build_train_worklist(1000, project_root=root)
            counts = _by_case(items)
            self.assertLessEqual(counts.get("addition", 0), 5)
            self.assertLessEqual(counts.get("subtraction", 0), 5)
            self.assertEqual(sum(counts.values()), 10)  # both caps saturated

    def test_uses_latest_history_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60, "subtraction": 60},
            )
            # Older record says subtraction weak; newer says addition weak.
            older = {"per_scene_case": {
                "addition": _aggregate_metric(),
                "subtraction": _aggregate_metric(row_acc_macro=0.0, col_acc_macro=0.0),
            }}
            newer = {"per_scene_case": {
                "addition": _aggregate_metric(row_acc_macro=0.0, col_acc_macro=0.0),
                "subtraction": _aggregate_metric(),
            }}
            _write_history(root, [older, newer])

            counts = _by_case(build_train_worklist(40, project_root=root))
            # The newer record must drive the plan: addition gets more.
            self.assertGreater(counts["addition"], counts["subtraction"])

    def test_zero_budget_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60},
            )
            _write_history(root, [{"per_scene_case": {
                "addition": _aggregate_metric(row_acc_macro=0.0),
            }}])
            self.assertEqual(build_train_worklist(0, project_root=root), [])
            self.assertEqual(build_train_worklist(-5, project_root=root), [])


class TestTrainStagesAndDeterminism(unittest.TestCase):
    """Train items pin valid stages and the plan is reproducible."""

    def test_stage_is_valid_for_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"division-short": 60, "addition": 60},
            )
            _write_history(root, [{"per_scene_case": {
                "division-short": _aggregate_metric(row_acc_macro=0.1, col_acc_macro=0.1),
                "addition": _aggregate_metric(row_acc_macro=0.3, col_acc_macro=0.3),
            }}])
            items = build_train_worklist(30, project_root=root)
            for it in items:
                self.assertIn(it.completion_stage, valid_stages_for_case(SceneCase(it.case)))

    def test_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60, "subtraction": 60},
            )
            _write_history(root, [{"per_scene_case": {
                "addition": _aggregate_metric(row_acc_macro=0.2, col_acc_macro=0.4),
                "subtraction": _aggregate_metric(row_acc_macro=0.6, col_acc_macro=0.7),
            }}])
            first = build_train_worklist(30, project_root=root)
            second = build_train_worklist(30, project_root=root)
            self.assertEqual(first, second)


# ===========================================================================
# Train: no-history fallback
# ===========================================================================

class TestTrainFallback(unittest.TestCase):
    """No usable per_scene_case -> warn + provisional-weights fallback."""

    def test_missing_history_file_warns_and_allocates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 40, "subtraction": 60},
            )
            # No reports/eval/history.jsonl at all.
            with self.assertLogs("src.setmaker.worklist", level="WARNING") as cm:
                items = build_train_worklist(30, project_root=root)
            self.assertTrue(any("provisional" in m for m in cm.output))
            self.assertEqual(sum(_by_case(items).values()), 30)

    def test_v1_history_without_per_scene_case_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 40, "subtraction": 60},
            )
            # A run record with no per_scene_case key (schema-1 shape).
            _write_history(
                root,
                [{"run_id": "x", "aggregate": {"equation_kind_acc": 0.7}}],
                schema_version=1,
            )
            with self.assertLogs("src.setmaker.worklist", level="WARNING") as cm:
                items = build_train_worklist(30, project_root=root)
            self.assertTrue(any("provisional" in m for m in cm.output))
            self.assertEqual(sum(_by_case(items).values()), 30)

    def test_empty_per_scene_case_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 40, "subtraction": 60},
            )
            _write_history(root, [{"per_scene_case": {}}])
            with self.assertLogs("src.setmaker.worklist", level="WARNING") as cm:
                build_train_worklist(30, project_root=root)
            self.assertTrue(any("provisional" in m for m in cm.output))

    def test_fallback_weights_by_provisional_quota(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # subtraction has the larger provisional target -> larger share.
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 40, "subtraction": 60},
            )
            with self.assertLogs("src.setmaker.worklist", level="WARNING"):
                items = build_train_worklist(30, project_root=root)
            counts = _by_case(items)
            self.assertGreater(counts["subtraction"], counts["addition"])

    def test_malformed_history_lines_skipped_then_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 40, "subtraction": 60},
            )
            history_path = root / "reports" / "eval" / "history.jsonl"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            # Schema line, a junk line, then a record with no per_scene_case.
            history_path.write_text(
                json.dumps({"record_type": "schema", "schema_version": 2}) + "\n"
                + "{not valid json}\n"
                + json.dumps({"run_id": "x"}) + "\n",
                encoding="utf-8",
            )
            with self.assertLogs("src.setmaker.worklist", level="WARNING"):
                items = build_train_worklist(20, project_root=root)
            self.assertEqual(sum(_by_case(items).values()), 20)


class TestTrainUnmeasuredCase(unittest.TestCase):
    """A case present in quota but absent from per_scene_case is treated weak."""

    def test_unmeasured_case_gets_at_least_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60, "subtraction": 60},
            )
            # Only addition measured; subtraction missing from the record.
            _write_history(root, [{"per_scene_case": {
                "addition": _aggregate_metric(row_acc_macro=0.5, col_acc_macro=0.5),
            }}])
            items = build_train_worklist(40, project_root=root)
            counts = _by_case(items)
            # subtraction is unmeasured -> treated maximally weak, gets >= floor.
            self.assertGreaterEqual(counts.get("subtraction", 0), TRAIN_PER_CASE_FLOOR)


# ===========================================================================
# Weakness score unit behaviour
# ===========================================================================

class TestWeaknessScore(unittest.TestCase):
    """The composite weakness score weights row/col heavy and is null-safe."""

    def test_perfect_metrics_score_zero(self) -> None:
        self.assertEqual(_weakness_score(_aggregate_metric()), 0.0)

    def test_row_col_weighted_heavier(self) -> None:
        # Equal accuracy deficit on row vs eq_kind -> row contributes more.
        row_only = _weakness_score(_aggregate_metric(
            row_acc_macro=0.0, col_acc_macro=1.0,
            equation_kind_acc=1.0, mean_label_acc=1.0,
        ))
        eqk_only = _weakness_score(_aggregate_metric(
            row_acc_macro=1.0, col_acc_macro=1.0,
            equation_kind_acc=0.0, mean_label_acc=1.0,
        ))
        self.assertAlmostEqual(row_only, WEIGHT_ROW)
        self.assertAlmostEqual(eqk_only, WEIGHT_EQ_KIND)
        self.assertGreater(row_only, eqk_only)

    def test_none_metric_counts_as_fully_weak(self) -> None:
        # All-None metric -> sum of all weights (each axis fully weak).
        score = _weakness_score({})
        self.assertAlmostEqual(
            score, WEIGHT_ROW + WEIGHT_COL + WEIGHT_EQ_KIND + WEIGHT_LABEL
        )


# ===========================================================================
# Seed namespaces (risk R7) — disjoint across modes and buckets
# ===========================================================================

class TestSeedNamespaces(unittest.TestCase):
    """Eval and train seeds never collide; per-bucket sub-ranges are disjoint."""

    def test_buckets_enumeration_is_collision_free(self) -> None:
        buckets = _all_buckets()
        self.assertEqual(len(buckets), len(set(buckets)))

    def test_eval_and_train_seed_blocks_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 6, "subtraction": 6, "division-short": 6},
                train_cases={"addition": 60, "subtraction": 60, "division-short": 60},
            )
            _write_history(root, [{"per_scene_case": {
                "addition": _aggregate_metric(row_acc_macro=0.2, col_acc_macro=0.2),
                "subtraction": _aggregate_metric(row_acc_macro=0.3, col_acc_macro=0.3),
                "division-short": _aggregate_metric(row_acc_macro=0.4, col_acc_macro=0.4),
            }}])
            eval_seeds = {it.seed for it in build_eval_worklist(project_root=root)}
            train_seeds = {it.seed for it in build_train_worklist(40, project_root=root)}
            self.assertTrue(eval_seeds)
            self.assertTrue(train_seeds)
            self.assertEqual(eval_seeds & train_seeds, set())

    def test_eval_seeds_within_eval_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 5},
                train_cases={"addition": 40},
            )
            for it in build_eval_worklist(project_root=root):
                self.assertGreaterEqual(it.seed, EVAL_SEED_BASE)
                self.assertLess(it.seed, TRAIN_SEED_BASE)

    def test_train_seeds_within_train_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 60},
            )
            _write_history(root, [{"per_scene_case": {
                "addition": _aggregate_metric(row_acc_macro=0.2, col_acc_macro=0.2),
            }}])
            for it in build_train_worklist(20, project_root=root):
                self.assertGreaterEqual(it.seed, TRAIN_SEED_BASE)

    def test_no_repeated_seed_within_a_worklist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 5, "subtraction": 7, "division-long": 9},
                train_cases={"addition": 40},
            )
            items = build_eval_worklist(project_root=root)
            seeds = [it.seed for it in items]
            self.assertEqual(len(seeds), len(set(seeds)))

    def test_distinct_stages_of_same_case_use_distinct_seeds(self) -> None:
        # Two different stages of addition must land in different seed sub-ranges.
        seed_full = _seed_for(EVAL_SEED_BASE, "addition", "full", 0)
        seed_d80 = _seed_for(EVAL_SEED_BASE, "addition", "done_80", 0)
        self.assertNotEqual(seed_full, seed_d80)

    def test_seed_counter_overflow_raises(self) -> None:
        with self.assertRaises(ValueError):
            _seed_for(EVAL_SEED_BASE, "addition", "full", SEED_BLOCK)


# ===========================================================================
# Quota file validation
# ===========================================================================

class TestQuotaValidation(unittest.TestCase):
    """Missing or malformed quota files fail loudly with the path."""

    def test_missing_quota_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(FileNotFoundError) as ctx:
                build_eval_worklist(project_root=root)
            self.assertIn("quota.json", str(ctx.exception))

    def test_wrong_schema_version_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 1},
                schema_version=99,
            )
            with self.assertRaises(ValueError) as ctx:
                build_eval_worklist(project_root=root)
            self.assertIn("schema_version", str(ctx.exception))

    def test_missing_block_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            quota_path = root / "data" / "setmaker" / "quota.json"
            quota_path.parent.mkdir(parents=True, exist_ok=True)
            # Missing the 'train' block entirely.
            quota_path.write_text(
                json.dumps({"schema_version": 1, "eval": {"scene_cases": {"addition": 1}}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                build_train_worklist(10, project_root=root)
            self.assertIn("train", str(ctx.exception))


# ===========================================================================
# Return-type contract
# ===========================================================================

class TestReturnTypes(unittest.TestCase):
    """Both builders return WorklistItem instances in pending status."""

    def test_eval_returns_worklist_items_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 2},
                train_cases={"addition": 40},
            )
            items = build_eval_worklist(project_root=root)
            self.assertTrue(items)
            for it in items:
                self.assertIsInstance(it, WorklistItem)
                self.assertEqual(it.status, "pending")

    def test_train_returns_worklist_items_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_quota(
                root,
                eval_cases={"addition": 1},
                train_cases={"addition": 40},
            )
            with self.assertLogs("src.setmaker.worklist", level="WARNING"):
                items = build_train_worklist(10, project_root=root)
            self.assertTrue(items)
            for it in items:
                self.assertIsInstance(it, WorklistItem)
                self.assertEqual(it.status, "pending")


# ===========================================================================
# FIX 2: served-seeds persistence — three fresh sessions yield distinct seeds
# ===========================================================================

class TestServedSeedPersistence(unittest.TestCase):
    """FIX 2: served_seeds argument ensures each session gets a distinct equation.

    Root cause: seeds are only persisted as "used" on SAVE, so three consecutive
    fresh /api/mode + /api/next sessions (no saves) all serve the same first seed.
    The fix: pass the durable served_seeds set when building the worklist so the
    planner skips seeds already delivered to the annotator.
    """

    def _quota(self, root: Path) -> None:
        _write_quota(root, eval_cases={"addition": 100}, train_cases={"addition": 100})

    def test_three_fresh_sessions_produce_distinct_first_seeds(self) -> None:
        """Simulating 3 consecutive fresh worklist builds without saving.

        Each build receives the seeds served in prior sessions (the durable record)
        and must serve a different first seed.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            served: set = set()

            first_seeds = []
            for _ in range(3):
                items = build_eval_worklist(project_root=root, served_seeds=served)
                self.assertTrue(items, "worklist must not be empty")
                first_seed = items[0].seed
                first_seeds.append(first_seed)
                # Simulate /api/next serving the first item: record as served.
                served.add(first_seed)

            self.assertEqual(len(set(first_seeds)), 3,
                             f"expected 3 distinct seeds, got {first_seeds}")

    def test_served_seeds_skipped_in_subsequent_build(self) -> None:
        """A seed in served_seeds must not appear in the new worklist."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            # Build once to find the seeds that would be served.
            items_first = build_eval_worklist(project_root=root, served_seeds=set())
            self.assertTrue(items_first)
            served = {it.seed for it in items_first}
            # Second build skips all of the first build's seeds.
            items_second = build_eval_worklist(project_root=root, served_seeds=served)
            new_seeds = {it.seed for it in items_second}
            overlap = served & new_seeds
            self.assertEqual(overlap, set(),
                             f"served seeds must not reappear in new worklist: {overlap}")

    def test_zero_served_seeds_is_identical_to_no_argument(self) -> None:
        """Passing an empty served_seeds set must not change the result."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            items_default = build_eval_worklist(project_root=root)
            items_empty = build_eval_worklist(project_root=root, served_seeds=set())
            self.assertEqual(
                [(it.case, it.seed, it.completion_stage) for it in items_default],
                [(it.case, it.seed, it.completion_stage) for it in items_empty],
            )

    def test_train_three_fresh_sessions_distinct_first_seeds(self) -> None:
        """Same guarantee for train mode."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            # Provide a dummy history so the worklist does not fall back to
            # provisional weights (which logs a WARNING).
            _write_history(root, [{"per_scene_case": {"addition": _aggregate_metric(
                row_acc_macro=0.5, col_acc_macro=0.5)}}])
            served: set = set()
            first_seeds = []
            for _ in range(3):
                items = build_train_worklist(10, project_root=root, served_seeds=served)
                self.assertTrue(items)
                first_seed = items[0].seed
                first_seeds.append(first_seed)
                served.add(first_seed)
            self.assertEqual(len(set(first_seeds)), 3,
                             f"expected 3 distinct seeds, got {first_seeds}")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Error-case worklist tests
# ---------------------------------------------------------------------------

class TestErrorCaseDeficit(unittest.TestCase):
    """Error items are appended after normal items; deficit math matches quota."""

    def _quota(
        self,
        root: Path,
        *,
        error_cases: Optional[dict] = None,
    ) -> None:
        quota_path = root / "data" / "setmaker" / "quota.json"
        quota_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": 1,
            "eval": {
                "scene_cases": {"addition": 2},
                "error_cases": error_cases or {
                    "addition": 4,
                    "subtraction": 4,
                    "multiplication": 4,
                    "division": 4,
                },
            },
            "train": {"scene_cases": {"addition": 4}},
        }
        quota_path.write_text(json.dumps(doc), encoding="utf-8")

    def _write_error_sidecar(
        self,
        real_dir: Path,
        stem: str,
        equation_kind: str,
        error_kind: str,
    ) -> None:
        real_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": 1,
            "sample": stem,
            "equation_kind": equation_kind,
            "error_kind": error_kind,
        }
        (real_dir / f"{stem}.label.json").write_text(json.dumps(doc), encoding="utf-8")

    def test_error_items_appended_after_normal(self) -> None:
        """Error items must come after all normal deficit items in the list."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            items = build_eval_worklist(project_root=root)
            directives = [it.directive for it in items]
            # All "normal" items before any "error" item.
            saw_error = False
            for d in directives:
                if d == "error":
                    saw_error = True
                elif saw_error:
                    self.fail("normal item appeared after an error item")

    def test_error_items_carry_directive_error(self) -> None:
        """Every item appended for error coverage must have directive='error'."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            items = build_eval_worklist(project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertTrue(error_items, "expected at least one error item")
            for it in error_items:
                self.assertEqual(it.directive, "error")

    def test_error_quota_16_total_on_fresh_tree(self) -> None:
        """With 4 per kind x 4 kinds and no existing errors -> 16 error items."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            items = build_eval_worklist(project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertEqual(len(error_items), 16)

    def test_existing_errors_reduce_deficit(self) -> None:
        """Each existing error sidecar reduces the deficit for that equation_kind."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            real_dir = root / "data" / "eval" / "real"
            # 2 existing addition errors -> deficit for addition = 4 - 2 = 2.
            self._write_error_sidecar(real_dir, "err-001", "addition", "wrong_result")
            self._write_error_sidecar(real_dir, "err-002", "addition", "missing_carry")
            items = build_eval_worklist(project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            # addition: 2 deficit; subtraction/multiplication/division: 4 each -> 14 total
            self.assertEqual(len(error_items), 14)
            # Specifically: addition's representative case should appear 2 times
            from src.setmaker.worklist import _ERROR_KIND_SCENE_CASE
            addition_case = _ERROR_KIND_SCENE_CASE["addition"]
            addition_error_items = [it for it in error_items if it.case == addition_case]
            self.assertEqual(len(addition_error_items), 2)

    def test_met_error_quota_produces_no_error_items_for_that_kind(self) -> None:
        """A kind whose error quota is already met must not produce error items."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root, error_cases={"addition": 2, "subtraction": 4})
            real_dir = root / "data" / "eval" / "real"
            # Meet the addition error quota exactly.
            for i in range(2):
                self._write_error_sidecar(real_dir, f"err-a-{i}", "addition", "wrong_result")
            items = build_eval_worklist(project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            from src.setmaker.worklist import _ERROR_KIND_SCENE_CASE
            addition_case = _ERROR_KIND_SCENE_CASE["addition"]
            for it in error_items:
                self.assertNotEqual(it.case, addition_case,
                    "no error items expected for addition once quota is met")
            # subtraction should still appear (4 deficit)
            subtraction_case = _ERROR_KIND_SCENE_CASE["subtraction"]
            sub_items = [it for it in error_items if it.case == subtraction_case]
            self.assertEqual(len(sub_items), 4)

    def test_error_items_have_completion_stage_full(self) -> None:
        """Error items must pin completion_stage='full' (complete equation)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            items = build_eval_worklist(project_root=root)
            for it in items:
                if it.directive == "error":
                    self.assertEqual(it.completion_stage, "full",
                        f"error item {it.case} must have completion_stage='full'")

    def test_error_seeds_disjoint_from_eval_seeds(self) -> None:
        """Error item seeds must not overlap with normal eval item seeds."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            items = build_eval_worklist(project_root=root)
            normal_seeds = {it.seed for it in items if it.directive == "normal"}
            error_seeds = {it.seed for it in items if it.directive == "error"}
            overlap = normal_seeds & error_seeds
            self.assertEqual(overlap, set(), f"seed overlap: {overlap}")

    def test_no_error_cases_block_in_quota_produces_no_error_items(self) -> None:
        """A quota file without error_cases must not crash and must produce no error items."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Write quota without the error_cases block.
            _write_quota(root, {"addition": 2}, {"addition": 4})
            items = build_eval_worklist(project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertEqual(error_items, [])

    def test_bare_digit_grid_still_served(self) -> None:
        """bare_digit_grid in scene_cases must produce normal items as before."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            from src.setmaker.worklist import _ERROR_KIND_SCENE_CASE  # noqa: F401
            quota_path = root / "data" / "setmaker" / "quota.json"
            quota_path.parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "schema_version": 1,
                "eval": {
                    "scene_cases": {"bare_digit_grid": 3},
                    "error_cases": {},
                },
                "train": {"scene_cases": {"addition": 4}},
            }
            quota_path.write_text(json.dumps(doc), encoding="utf-8")
            items = build_eval_worklist(project_root=root)
            grid_items = [it for it in items if it.case == "bare_digit_grid"]
            self.assertEqual(len(grid_items), 3)
            for it in grid_items:
                self.assertEqual(it.directive, "normal")


class TestErrorItemsAppRoutes(unittest.TestCase):
    """API-level tests for error-directive enforcement via TestClient."""

    def _write_quota(self, root: Path) -> None:
        quota_path = root / "data" / "setmaker" / "quota.json"
        quota_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": 1,
            "eval": {
                "scene_cases": {"addition": 1},
                "error_cases": {"addition": 2},
            },
            "train": {"scene_cases": {"addition": 4}},
        }
        quota_path.write_text(json.dumps(doc), encoding="utf-8")

    def _fake_target(self, case, seed, completion_stage, **_kw):
        from src.setmaker.types import TargetScene, TargetSymbol
        symbols = [
            TargetSymbol(
                fine_label="main_2",
                glyph_key="main_2",
                yolo_class="digit_main",
                row_index=0,
                col_index=0,
                equation_idx=0,
                bbox=(100.0, 100.0, 150.0, 160.0),
            ),
        ]
        return TargetScene(
            case=case,
            equation_type="addition",
            completion_stage=completion_stage,
            seed=seed,
            reference="2",
            symbols=symbols,
        )

    def _png_b64(self) -> str:
        import base64, io
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.full((64, 64), 255, dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _draft(self) -> dict:
        return {
            "bbox_px": [100.0, 100.0, 150.0, 160.0],
            "fine_label": "main_2",
            "row_index": 0,
            "col_index": 0,
            "equation_idx": 0,
            "confidence": 0.9,
            "source": "matched",
            "flagged": False,
        }

    def _client(self, root):
        import tempfile
        from fastapi.testclient import TestClient
        from src.setmaker import app as app_mod
        application = app_mod.create_app(root)
        application.state.generate_target_fn = self._fake_target
        return TestClient(application)

    def test_next_returns_directive_in_item(self) -> None:
        """GET /api/next must include directive in the item dict."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_quota(root)
            client = self._client(root)
            client.post("/api/mode", json={"mode": "eval"})
            res = client.get("/api/next", params={"mode": "eval"}).json()
            self.assertIn("directive", res["item"])
            self.assertIn(res["item"]["directive"], ("normal", "error"))

    def test_save_normal_item_accepts_no_error_kind(self) -> None:
        """Normal items must save successfully without error_kind."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_quota(root)
            client = self._client(root)
            client.post("/api/mode", json={"mode": "eval"})
            # Advance until we find a normal item (first in the list).
            res = client.get("/api/next", params={"mode": "eval"}).json()
            if res["item"]["directive"] == "normal":
                resp = client.post(
                    "/api/save",
                    params={"mode": "eval"},
                    json={"image_png": self._png_b64(), "drafts": [self._draft()]},
                )
                self.assertEqual(resp.status_code, 200)

    def test_save_error_item_without_error_kind_is_422(self) -> None:
        """An error-directive item saved without error_kind must be rejected (422)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_quota(root)
            client = self._client(root)
            client.post("/api/mode", json={"mode": "eval"})
            # Drain the normal item first, then the next item is an error item.
            # Save the normal item.
            normal_res = client.get("/api/next", params={"mode": "eval"}).json()
            if normal_res["item"]["directive"] == "normal":
                client.post(
                    "/api/save",
                    params={"mode": "eval"},
                    json={"image_png": self._png_b64(), "drafts": [self._draft()]},
                )
            # Now get the error item.
            next_res = client.get("/api/next", params={"mode": "eval"}).json()
            if next_res.get("done"):
                return  # nothing to test if no error item served
            if next_res["item"]["directive"] != "error":
                return  # item not error directive, skip
            # Try to save without error_kind -> must be 422.
            resp = client.post(
                "/api/save",
                params={"mode": "eval"},
                json={"image_png": self._png_b64(), "drafts": [self._draft()]},
            )
            self.assertEqual(resp.status_code, 422)
            self.assertIn("error_kind", resp.json()["detail"])

    def test_save_error_item_with_valid_error_kind_is_200(self) -> None:
        """An error-directive item saved WITH a valid error_kind must be accepted."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_quota(root)
            client = self._client(root)
            client.post("/api/mode", json={"mode": "eval"})
            # Drain the normal item.
            normal_res = client.get("/api/next", params={"mode": "eval"}).json()
            if normal_res["item"]["directive"] == "normal":
                client.post(
                    "/api/save",
                    params={"mode": "eval"},
                    json={"image_png": self._png_b64(), "drafts": [self._draft()]},
                )
            next_res = client.get("/api/next", params={"mode": "eval"}).json()
            if next_res.get("done"):
                return
            if next_res["item"]["directive"] != "error":
                return
            # Save WITH error_kind -> must succeed.
            resp = client.post(
                "/api/save",
                params={"mode": "eval"},
                json={
                    "image_png": self._png_b64(),
                    "drafts": [self._draft()],
                    "error_kind": "wrong_result",
                },
            )
            self.assertEqual(resp.status_code, 200)

    def test_next_returns_complete_signal_when_worklist_exhausted(self) -> None:
        """When the worklist is fully drained, /api/next must return complete=true."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Minimal quota: 1 normal scene only (no error_cases).
            quota_path = root / "data" / "setmaker" / "quota.json"
            quota_path.parent.mkdir(parents=True, exist_ok=True)
            quota_path.write_text(json.dumps({
                "schema_version": 1,
                "eval": {"scene_cases": {"addition": 1}},
                "train": {"scene_cases": {"addition": 4}},
            }), encoding="utf-8")
            client = self._client(root)
            client.post("/api/mode", json={"mode": "eval"})
            # Drain the one scene.
            client.get("/api/next", params={"mode": "eval"})
            client.post(
                "/api/save",
                params={"mode": "eval"},
                json={"image_png": self._png_b64(), "drafts": [self._draft()]},
            )
            # Now the worklist is exhausted.
            res = client.get("/api/next", params={"mode": "eval"}).json()
            self.assertTrue(res.get("done") or res.get("complete"),
                            f"expected done/complete signal, got {res}")
            # The completion response must include progress.
            self.assertIn("progress", res)


# ===========================================================================
# Train error-case worklist tests (B2): operator/equation-kind gap coverage
# ===========================================================================

class TestTrainErrorCases(unittest.TestCase):
    """build_train_worklist appends deficit-driven directive='error' items.

    The measured real gap is the equation-kind/operator head, so the train set
    needs deliberately-wrong scenes. These are quota-driven (train.error_cases),
    counted against existing train GT files that carry an error marker, appended
    after the gap-allocated items, in the disjoint TRAIN_ERROR_SEED_BASE space.
    """

    def _quota(
        self,
        root: Path,
        *,
        train_error_cases=None,
        train_scene_cases=None,
        eval_error_cases=None,
    ) -> None:
        quota_path = root / "data" / "setmaker" / "quota.json"
        quota_path.parent.mkdir(parents=True, exist_ok=True)
        train_block = {"scene_cases": train_scene_cases or {"addition": 40}}
        if train_error_cases is not None:
            train_block["error_cases"] = train_error_cases
        else:
            train_block["error_cases"] = {
                "addition": 10,
                "subtraction": 10,
                "multiplication": 10,
                "division": 10,
            }
        eval_block: dict = {"scene_cases": {"addition": 1}}
        if eval_error_cases is not None:
            eval_block["error_cases"] = eval_error_cases
        doc = {
            "schema_version": 1,
            "eval": eval_block,
            "train": train_block,
        }
        quota_path.write_text(json.dumps(doc), encoding="utf-8")

    def _history(self, root: Path) -> None:
        # A measured history so the planner does not hit the provisional fallback
        # (which would log a WARNING and route by quota caps instead of metrics).
        _write_history(root, [{"per_scene_case": {
            "addition": _aggregate_metric(row_acc_macro=0.5, col_acc_macro=0.5),
        }}])

    def _write_train_gt(
        self,
        root: Path,
        stem: str,
        equation_type: str,
        *,
        error_kind=None,
    ) -> None:
        """Write a minimal train GT file (the set-maker train export sink)."""
        train_dir = root / "data" / "setmaker" / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        doc: dict = {
            "equation_type": equation_type,
            "completion_stage": "full",
            "split": "train",
            "case": equation_type,
            "symbols": [],
        }
        if error_kind is not None:
            doc["error_kind"] = error_kind
        (train_dir / f"{stem}.gt.json").write_text(json.dumps(doc), encoding="utf-8")

    def test_error_items_appended_after_normal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            self._history(root)
            items = build_train_worklist(20, project_root=root)
            saw_error = False
            for it in items:
                if it.directive == "error":
                    saw_error = True
                elif saw_error:
                    self.fail("normal train item appeared after an error item")

    def test_error_items_carry_directive_error_and_full_stage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            self._history(root)
            items = build_train_worklist(20, project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertTrue(error_items, "expected train error items")
            for it in error_items:
                self.assertEqual(it.directive, "error")
                self.assertEqual(it.completion_stage, "full")

    def test_error_quota_40_total_on_fresh_tree(self) -> None:
        # 10 per kind x 4 kinds, no existing train error GT -> 40 error items.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            self._history(root)
            items = build_train_worklist(20, project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertEqual(len(error_items), 40)

    def test_existing_train_error_gt_reduces_deficit(self) -> None:
        # Each existing train GT with an error_kind reduces the deficit for its kind.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root, train_error_cases={"addition": 4, "subtraction": 4})
            self._history(root)
            # 2 addition error GTs -> addition deficit 4-2 = 2; subtraction full 4.
            self._write_train_gt(root, "rt-add-001", "addition", error_kind="wrong_operator")
            self._write_train_gt(root, "rt-add-002", "addition", error_kind="missing_carry")
            items = build_train_worklist(10, project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertEqual(len(error_items), 6)  # 2 + 4
            from src.setmaker.worklist import _ERROR_KIND_SCENE_CASE
            add_case = _ERROR_KIND_SCENE_CASE["addition"]
            add_err = [it for it in error_items if it.case == add_case]
            self.assertEqual(len(add_err), 2)

    def test_normal_train_gt_without_marker_does_not_reduce_deficit(self) -> None:
        # A train GT with NO error_kind marker is a normal scene; it must not count.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root, train_error_cases={"addition": 3})
            self._history(root)
            self._write_train_gt(root, "rt-add-100", "addition")  # no error_kind
            self._write_train_gt(root, "rt-add-101", "addition")  # no error_kind
            items = build_train_worklist(10, project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertEqual(len(error_items), 3)  # full deficit; normals ignored

    def test_met_error_quota_yields_no_error_items_for_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root, train_error_cases={"addition": 2, "subtraction": 4})
            self._history(root)
            for i in range(2):
                self._write_train_gt(root, f"rt-a-{i}", "addition", error_kind="wrong_result")
            items = build_train_worklist(10, project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            from src.setmaker.worklist import _ERROR_KIND_SCENE_CASE
            add_case = _ERROR_KIND_SCENE_CASE["addition"]
            self.assertFalse([it for it in error_items if it.case == add_case])
            sub_case = _ERROR_KIND_SCENE_CASE["subtraction"]
            self.assertEqual(len([it for it in error_items if it.case == sub_case]), 4)

    def test_no_train_error_cases_block_yields_no_error_items(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            quota_path = root / "data" / "setmaker" / "quota.json"
            quota_path.parent.mkdir(parents=True, exist_ok=True)
            quota_path.write_text(json.dumps({
                "schema_version": 1,
                "eval": {"scene_cases": {"addition": 1}},
                "train": {"scene_cases": {"addition": 40}},  # no error_cases
            }), encoding="utf-8")
            self._history(root)
            items = build_train_worklist(20, project_root=root)
            self.assertEqual([it for it in items if it.directive == "error"], [])

    def test_zero_budget_yields_no_error_items(self) -> None:
        # Non-positive budget plans nothing at all, including error items.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            self._history(root)
            self.assertEqual(build_train_worklist(0, project_root=root), [])
            self.assertEqual(build_train_worklist(-3, project_root=root), [])

    def test_train_error_seeds_disjoint_from_train_normal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            self._history(root)
            items = build_train_worklist(20, project_root=root)
            normal_seeds = {it.seed for it in items if it.directive == "normal"}
            error_seeds = {it.seed for it in items if it.directive == "error"}
            self.assertTrue(normal_seeds)
            self.assertTrue(error_seeds)
            self.assertEqual(normal_seeds & error_seeds, set())
            # Train error seeds live in the dedicated base block.
            for s in error_seeds:
                self.assertGreaterEqual(s, TRAIN_ERROR_SEED_BASE)

    def test_train_error_seeds_disjoint_from_eval_error_seeds(self) -> None:
        # The same operation kind in eval-error and train-error must not share a
        # seed even though both use the flat per-kind counter scheme.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(
                root,
                eval_error_cases={"addition": 4, "subtraction": 4,
                                  "multiplication": 4, "division": 4},
            )
            self._history(root)
            eval_items = build_eval_worklist(project_root=root)
            train_items = build_train_worklist(20, project_root=root)
            eval_err = {it.seed for it in eval_items if it.directive == "error"}
            train_err = {it.seed for it in train_items if it.directive == "error"}
            self.assertTrue(eval_err)
            self.assertTrue(train_err)
            self.assertEqual(eval_err & train_err, set())
            # Bases are ordered: all eval-error seeds below ERROR_SEED_BASE ceiling,
            # all train-error seeds at/above their dedicated base.
            self.assertLess(max(eval_err), TRAIN_ERROR_SEED_BASE)
            self.assertGreaterEqual(min(train_err), TRAIN_ERROR_SEED_BASE)

    def test_malformed_train_gt_skipped_not_crashed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root, train_error_cases={"addition": 3})
            self._history(root)
            train_dir = root / "data" / "setmaker" / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "rt-bad.gt.json").write_text("{not valid json", encoding="utf-8")
            # Must not raise; the malformed file simply does not count.
            items = build_train_worklist(10, project_root=root)
            error_items = [it for it in items if it.directive == "error"]
            self.assertEqual(len(error_items), 3)

    def test_repeatable_with_error_items(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            self._history(root)
            first = build_train_worklist(20, project_root=root)
            second = build_train_worklist(20, project_root=root)
            self.assertEqual(first, second)

    def test_served_seeds_skip_train_error_items(self) -> None:
        # A train error seed already served in a prior session must not reappear.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root, train_error_cases={"addition": 4})
            self._history(root)
            first = build_train_worklist(10, project_root=root, served_seeds=set())
            first_err = [it.seed for it in first if it.directive == "error"]
            self.assertTrue(first_err)
            served = set(first_err)
            second = build_train_worklist(10, project_root=root, served_seeds=served)
            second_err = {it.seed for it in second if it.directive == "error"}
            self.assertEqual(served & second_err, set())


class TestTrainErrorDeficitClosureRoundtrip(unittest.TestCase):
    """Prove the full train-error loop closes end-to-end via export_train.

    Uses a temp tree so data/setmaker/train stays empty in the real repo.
    """

    def _quota(self, root: Path, *, train_error_cases=None) -> None:
        quota_path = root / "data" / "setmaker" / "quota.json"
        quota_path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": 1,
            "eval": {"scene_cases": {"addition": 1}},
            "train": {
                "scene_cases": {"addition": 10},
                "error_cases": train_error_cases or {
                    "addition": 3,
                    "subtraction": 3,
                },
            },
        }
        quota_path.write_text(json.dumps(doc), encoding="utf-8")

    def _history(self, root: Path) -> None:
        _write_history(root, [{"per_scene_case": {
            "addition": _aggregate_metric(row_acc_macro=0.5, col_acc_macro=0.5),
        }}])

    def _canvas_and_drafts(self):
        import numpy as np
        from PIL import Image
        from src.setmaker.types import AnnotationDraft
        canvas = Image.fromarray(np.full((512, 512), 255, dtype=np.uint8))
        drafts = [AnnotationDraft(
            bbox_px=[50.0, 50.0, 100.0, 100.0],
            fine_label="main_2",
            row_index=0,
            col_index=0,
            equation_idx=0,
            confidence=0.9,
            source="matched",
            flagged=False,
        )]
        return canvas, drafts

    def test_export_train_error_kind_and_deficit_closes(self) -> None:
        """export_train with error_kind writes the key and the worklist deficit drops."""
        import tempfile as _tf
        from src.setmaker.exporters import export_train
        from src.setmaker.worklist import (
            _count_existing_train_errors_by_kind,
            build_train_worklist,
        )
        canvas, drafts = self._canvas_and_drafts()

        with _tf.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root)
            self._history(root)

            from src.setmaker.worklist import _ERROR_KIND_SCENE_CASE
            add_case = _ERROR_KIND_SCENE_CASE["addition"]

            # Before export: full deficit for addition (quota=3, have=0).
            self.assertEqual(_count_existing_train_errors_by_kind(root).get("addition", 0), 0)
            items_before = build_train_worklist(20, project_root=root)
            err_before = [it for it in items_before
                          if it.directive == "error" and it.case == add_case]
            self.assertEqual(len(err_before), 3)

            # Export one train error scene with error_kind="wrong_operator".
            out = export_train(
                "rt-add-err-001",
                canvas,
                drafts,
                {
                    "equation_type": "addition",
                    "completion_stage": "full",
                    "case": "addition",
                    "split": "train",
                    "error_kind": "wrong_operator",
                },
                root,
            )
            gt_doc = json.loads(out["gt"].read_text())
            self.assertEqual(gt_doc["error_kind"], "wrong_operator")

            # Counter rises by 1; deficit drops to 2.
            self.assertEqual(_count_existing_train_errors_by_kind(root).get("addition", 0), 1)
            items_after = build_train_worklist(20, project_root=root)
            err_after = [it for it in items_after
                         if it.directive == "error" and it.case == add_case]
            self.assertEqual(len(err_after), 2,
                             "deficit must drop by 1 after one error GT is written")

    def test_normal_train_export_does_not_affect_error_deficit(self) -> None:
        """A normal train GT (no error_kind) is ignored by the error counter."""
        import tempfile as _tf
        from src.setmaker.exporters import export_train
        from src.setmaker.worklist import (
            _count_existing_train_errors_by_kind,
            build_train_worklist,
        )
        canvas, drafts = self._canvas_and_drafts()

        with _tf.TemporaryDirectory() as td:
            root = Path(td)
            self._quota(root, train_error_cases={"addition": 2})
            self._history(root)

            # Export a normal train sample (no error_kind).
            out = export_train(
                "rt-add-normal-001",
                canvas,
                drafts,
                {
                    "equation_type": "addition",
                    "completion_stage": "full",
                    "case": "addition",
                    "split": "train",
                },
                root,
            )

            # Error counter stays 0; deficit unchanged at 2.
            self.assertEqual(_count_existing_train_errors_by_kind(root).get("addition", 0), 0)
            items = build_train_worklist(10, project_root=root)
            err_items = [it for it in items if it.directive == "error"]
            self.assertEqual(len(err_items), 2,
                             "normal GT must not reduce the error deficit")

            # Confirm the normal GT has no error_kind key.
            gt_doc = json.loads(out["gt"].read_text())
            self.assertNotIn("error_kind", gt_doc)


if __name__ == "__main__":
    unittest.main()
