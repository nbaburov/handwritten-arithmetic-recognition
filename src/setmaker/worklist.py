"""Smart worklist planner for the set-maker (WS-C, the keystone).

Owns one job: decide the ordered list of ``(case, seed, completion_stage)`` scenes
to draw next, so the annotator never wastes effort on coverage that is already met
(eval) or on model strengths instead of model gaps (train). Two public builders:

* :func:`build_eval_worklist` is **deficit-driven**. It scans the existing real
  eval sidecars (``data/eval/real/`` plus the seed bank ``data/eval/bank/``) via
  the canonical ``load_label_sidecars`` loader, counts how many scenes already
  exist per ``scene_case``, subtracts those from the per-case quota in
  ``data/setmaker/quota.json`` (the machine-readable single source of truth; the
  quota is the single source of truth and is never derived from markdown),
  and emits exactly one item per still-needed scene, biggest deficit first. Cases
  whose quota is already met are skipped entirely.

* :func:`build_train_worklist` is **gap-driven**. It reads the most recent record
  from ``reports/eval/history.jsonl`` and takes its ``per_scene_case`` block (the
  schema-version-2 per-case ``AggregateMetric`` map). Each case gets a weakness
  score that weights row and column placement heavily (the client priority) over
  fine-label and equation-kind accuracy; ``round_budget`` scenes are then allocated
  across cases in proportion to weakness, with a per-case floor (so every case gets
  some coverage) and a per-case cap (diminishing returns past the quota value). If
  no history record carries a non-empty ``per_scene_case`` (true until the first
  schema-2 eval run), the planner logs a warning and falls back to the provisional
  per-case weights in the quota file's ``train`` block.

Seed namespaces are **disjoint** across modes and across each
``(case, completion_stage)`` bucket within a mode (risk R7): eval seeds live in one
numeric block, train seeds in a far-apart block, and each bucket carves out its own
contiguous sub-range. A generated equation can therefore never appear in both the
eval and the train set, and two different completion stages of the same case never
reuse a seed. Completion stages are pinned per item by cycling deterministically
through the stages valid for that case (never sampled randomly), so partial-scene
quotas stay satisfiable.

This module owns neither drawing, detection, matching, nor export; it only plans.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.config import resolve_project_root
from ..eval.labels import ERROR_KINDS, load_label_sidecars
from ..generation.completion_stages import valid_stages_for_case
from ..generation.layouts_types import SceneCase
from .types import WorklistItem

logger = logging.getLogger(__name__)

__all__ = ["build_eval_worklist", "build_train_worklist"]

# ---------------------------------------------------------------------------
# Module constants — documented, tunable in one place (no magic numbers inline)
# ---------------------------------------------------------------------------

# Disjoint seed namespaces per mode (risk R7). Eval and train each own a numeric
# block large enough that their per-bucket sub-ranges can never overlap. The two
# bases are far enough apart (one million) that the full eval block sits entirely
# below the full train block. Error-case items use further disjoint bases above
# both: eval error items live in ERROR_SEED_BASE, train error items in
# TRAIN_ERROR_SEED_BASE, so an eval error scene and a train error scene of the same
# operation kind can never share a seed even though both use the flat per-kind
# counter scheme.
EVAL_SEED_BASE: int = 1_000_000
TRAIN_SEED_BASE: int = 2_000_000
ERROR_SEED_BASE: int = 3_000_000
TRAIN_ERROR_SEED_BASE: int = 4_000_000

# Size of the seed sub-range reserved for each (case, completion_stage) bucket.
# Per-case quotas/caps are at most a few dozen, so 10_000 leaves vast headroom and
# guarantees no two buckets (and therefore no two stages of one case) collide.
SEED_BLOCK: int = 10_000

# Train weakness-score weights. Row and column placement dominate because correct
# spatial structure is the client headline. Equation kind is the measured real-data
# gap (addition eq_kind 0.19, subtraction 0.14 on the latest real eval), so it is
# weighted above the fine label: a case whose operator head is broken but whose
# row/col placement is already strong (e.g. subtraction_heavy_borrow eq_kind 0.10
# with perfect row/col) must still earn training budget instead of being starved.
# EQ_KIND=2.0 is the empirical sweet spot that lifts operator-weak cases without
# overtaking the row/col client priority (each still 3.0). A case with perfect
# metrics scores 0 (no weakness, deprioritised).
WEIGHT_ROW: float = 3.0
WEIGHT_COL: float = 3.0
WEIGHT_EQ_KIND: float = 2.0
WEIGHT_LABEL: float = 1.0

# Train allocation guards. Every case in the gap plan gets at least FLOOR scenes
# (coverage guarantee) and at most its quota value (diminishing-returns cap). The
# floor keeps a lopsided weakness distribution from starving any case (risk R4).
TRAIN_PER_CASE_FLOOR: int = 2

# A missing per-case metric (``None`` in the history record, e.g. a case with no
# spatial samples) is treated as fully weak on that axis so it is not silently
# dropped from the gap plan. 1.0 means "0 accuracy" for the ``(1 - acc)`` term.
_MISSING_METRIC_WEAKNESS: float = 1.0

# Filenames / relative locations (single source of truth for the planner's I/O).
_QUOTA_RELPATH = Path("data") / "setmaker" / "quota.json"
_HISTORY_RELPATH = Path("reports") / "eval" / "history.jsonl"
_EVAL_REAL_RELPATH = Path("data") / "eval" / "real"
_EVAL_BANK_RELPATH = Path("data") / "eval" / "bank"
_TRAIN_RELPATH = Path("data") / "setmaker" / "train"

# Keys under which a train GT file may record a deliberate as-drawn error tag.
# The set-maker train exporter writes ``equation_type`` (long form: addition /
# subtraction / multiplication / division), so the GT's operation kind is read
# from ``equation_type``; the error marker itself is ``error_kind`` (mirrors the
# eval sidecar field). Both are tolerated absent so a normal train GT counts as
# "no error" rather than raising.
_TRAIN_GT_ERROR_KEY: str = "error_kind"
_TRAIN_GT_KIND_KEY: str = "equation_type"

_QUOTA_SCHEMA_VERSION: int = 1
_SCHEMA_RECORD_TYPE: str = "schema"

# Representative scene_case and completion_stage for each error-case operation kind.
# Error items serve a normal equation target (so the human has something concrete to
# draw) but are tagged directive="error" so the frontend demands an error_kind tag.
# The map is the single source of truth for the error->scene_case projection; it is
# documented here rather than in quota.json so callers never need to parse it.
_ERROR_KIND_SCENE_CASE: Dict[str, str] = {
    "addition": "addition",
    "subtraction": "subtraction",
    "multiplication": "multiplication-simple",
    "division": "division-short",
}

# Error seeds use a flat per-operation-kind counter (not the full (case, stage)
# bucket scheme) to keep the namespace simple. Each kind gets SEED_BLOCK slots
# within ERROR_SEED_BASE, ordered by the sorted kind list for determinism.
_ERROR_KINDS_ORDERED: List[str] = sorted(_ERROR_KIND_SCENE_CASE)


def _error_seed(kind: str, counter: int, base: int = ERROR_SEED_BASE) -> int:
    """Compute a deterministic, namespace-disjoint seed for one error-case item.

    Seeds live in ``base`` (``ERROR_SEED_BASE`` for eval error items,
    ``TRAIN_ERROR_SEED_BASE`` for train error items), one SEED_BLOCK-sized
    sub-range per operation kind (sorted alphabetically for collision-free
    determinism). Because each base sits one million apart from the others, these
    are always disjoint from EVAL_SEED_BASE, TRAIN_SEED_BASE, and from each other.
    """
    idx = _ERROR_KINDS_ORDERED.index(kind)
    return base + idx * SEED_BLOCK + counter


# ---------------------------------------------------------------------------
# Internal helpers — project root, quota, seed namespaces
# ---------------------------------------------------------------------------

def _resolved_root(project_root: Optional[Path]) -> Path:
    """Return the supplied project root, or auto-resolve it from this file.

    Mirrors the sibling ``detect.py`` contract: callers may omit ``project_root``
    in production (it is resolved from the package location) and pass an explicit
    one in tests against a temporary tree.
    """
    if project_root is not None:
        return Path(project_root)
    return resolve_project_root(Path(__file__))


def _load_quota(project_root: Path) -> dict:
    """Load and lightly validate ``data/setmaker/quota.json``.

    The quota file is the machine-readable single source of truth for per-case
    targets. Validation is deliberately minimal (presence + schema version +
    required blocks) so callers fail loudly with the file path rather than hitting
    a confusing ``KeyError`` deep in the planner.

    Raises:
        FileNotFoundError: If the quota file is absent (with its path).
        ValueError: If the schema version is wrong or a required block is missing
            (message always includes the file path).
    """
    quota_path = project_root / _QUOTA_RELPATH
    if not quota_path.is_file():
        raise FileNotFoundError(
            f"{quota_path}: quota file not found; it is the worklist planner's "
            f"single source of truth and must exist (see the set-maker plan)."
        )
    with quota_path.open(encoding="utf-8") as fh:
        quota = json.load(fh)
    version = quota.get("schema_version")
    if version != _QUOTA_SCHEMA_VERSION:
        raise ValueError(
            f"{quota_path}: schema_version must be {_QUOTA_SCHEMA_VERSION}, "
            f"got {version!r}."
        )
    for block in ("eval", "train"):
        if block not in quota or "scene_cases" not in quota.get(block, {}):
            raise ValueError(
                f"{quota_path}: missing required '{block}.scene_cases' block."
            )
    return quota


def _bucket_index(case: str, completion_stage: str) -> int:
    """Return a stable, collision-free index for one ``(case, stage)`` bucket.

    The index is the position of ``(case, completion_stage)`` within the fully
    enumerated, sorted list of every valid ``(SceneCase, valid stage)`` pair. Using
    an enumeration (not a hash) makes the mapping provably one-to-one, so distinct
    buckets always map to distinct seed sub-ranges. Pairs are derived from the
    ontology, so the index is deterministic across processes and runs.
    """
    return _all_buckets().index((case, completion_stage))


@lru_cache(maxsize=None)
def _all_buckets() -> List[Tuple[str, str]]:
    """Enumerate every ``(scene_case, valid completion_stage)`` pair, sorted.

    The list is built once (memoised by ``lru_cache``); subsequent calls return
    the cached result without rebuilding or re-sorting the 78 pairs. The cache is
    safe because ``SceneCase`` and ``valid_stages_for_case`` are both fixed at
    import time (no runtime mutation). Behaviour is identical to the un-cached
    version; this is a pure performance optimisation that matters when many worklist
    items are planned in one call (each item calls ``_bucket_index`` which called
    this function per seed computation).
    """
    pairs: List[Tuple[str, str]] = []
    for case in SceneCase:
        for stage in valid_stages_for_case(case):
            pairs.append((case.value, stage))
    return sorted(pairs)


def _seed_for(mode_base: int, case: str, completion_stage: str, counter: int) -> int:
    """Compute a deterministic, namespace-disjoint seed for one worklist item.

    The seed is ``mode_base + bucket_index * SEED_BLOCK + counter``. Because each
    mode owns a far-apart base and each ``(case, stage)`` bucket owns a contiguous
    ``SEED_BLOCK``-sized sub-range, no seed produced for eval can equal a seed
    produced for train, and no two stages of the same case can collide (risk R7).

    Args:
        mode_base: ``EVAL_SEED_BASE`` or ``TRAIN_SEED_BASE``.
        case: Scene-case string.
        completion_stage: Pinned completion stage for this item.
        counter: Per-bucket monotonic index (``0`` for the first item in a bucket).

    Raises:
        ValueError: If ``counter`` would overflow the bucket's ``SEED_BLOCK``.
    """
    if not (0 <= counter < SEED_BLOCK):
        raise ValueError(
            f"seed counter {counter} out of range for bucket "
            f"({case}, {completion_stage}); max {SEED_BLOCK} items per bucket."
        )
    return mode_base + _bucket_index(case, completion_stage) * SEED_BLOCK + counter


def _items_for_case(
    case: str, count: int, mode_base: int, used_seeds: set, served_seeds: set = frozenset()
) -> List[WorklistItem]:
    """Build ``count`` pinned-stage worklist items for one case.

    Completion stages are pinned (never random) by cycling through the stages
    valid for the case, so partial-scene quotas are satisfiable and reproducible.
    Seeds come from :func:`_seed_for` and are recorded in ``used_seeds`` so a
    resume (or the other mode) never repeats one. Seeds in ``served_seeds``
    (the durable record of seeds already served to the annotator in prior
    sessions) are skipped so each new session receives a distinct equation
    (FIX 2 seed persistence).
    """
    if count <= 0:
        return []
    stages = valid_stages_for_case(SceneCase(case))
    items: List[WorklistItem] = []
    # Per-(case, stage) counters so each bucket fills its own seed sub-range.
    # We advance the counter past any served or used seeds to find the next fresh one.
    stage_counters: Dict[str, int] = {stage: 0 for stage in stages}
    i = 0
    while len(items) < count:
        stage = stages[i % len(stages)]
        counter = stage_counters[stage]
        # Advance past SEED_BLOCK boundary safety check is in _seed_for.
        # Skip any seed that was already served (durable) or used in this build.
        while True:
            if counter >= SEED_BLOCK:
                # Exhausted the namespace for this bucket; cannot satisfy quota.
                # Log and break out rather than crashing; caller gets a short list.
                logger.warning(
                    "seed namespace exhausted for (%s, %s): served seeds saturated "
                    "the %d-slot budget; reduce quota or clear served-seeds.",
                    case, stage, SEED_BLOCK,
                )
                return items
            seed = _seed_for(mode_base, case, stage, counter)
            counter += 1
            if seed not in served_seeds and seed not in used_seeds:
                break
        stage_counters[stage] = counter
        used_seeds.add(seed)
        items.append(WorklistItem(case=case, seed=seed, completion_stage=stage))
        i += 1
    return items


# ---------------------------------------------------------------------------
# Eval: deficit-driven planner
# ---------------------------------------------------------------------------

def _load_eval_sidecars(project_root: Path) -> Dict[str, object]:
    """Merge eval sidecars from real + bank directories (bank wins on collision).

    A missing directory contributes zero rather than raising, because either set
    may legitimately be empty on a fresh checkout.
    """
    merged: Dict[str, object] = {}
    for relpath in (_EVAL_REAL_RELPATH, _EVAL_BANK_RELPATH):
        samples_dir = project_root / relpath
        if not samples_dir.is_dir():
            continue
        merged.update(load_label_sidecars(samples_dir))
    return merged


def _count_existing_by_case(project_root: Path) -> Dict[str, int]:
    """Count existing real eval scenes per ``scene_case``.

    Scans both ``data/eval/real/`` and the seed bank ``data/eval/bank/`` through
    the canonical ``load_label_sidecars`` loader (so the deficit math always agrees
    with what the eval harness sees). The two stem-keyed maps are merged before
    counting (bank wins on a stem collision), so a scene present in both directories
    is counted once rather than twice. Sidecars whose ``scene_case`` is absent
    (``None``) cannot be attributed to a case and are not counted toward any quota.
    A missing directory contributes zero rather than raising, because either set
    may legitimately be empty on a fresh checkout.
    """
    counts: Dict[str, int] = {}
    for label in _load_eval_sidecars(project_root).values():
        if label.scene_case is None:
            continue
        counts[label.scene_case] = counts.get(label.scene_case, 0) + 1
    return counts


def _count_existing_errors_by_kind(project_root: Path) -> Dict[str, int]:
    """Count existing eval sidecars that carry a non-null ``error_kind`` per
    ``equation_kind`` (operation kind).

    Used to compute the per-kind deficit for the error-case worklist items.  Only
    sidecars from the real eval directory are counted (bank samples are curated by
    the research team, not drawn by the annotator for error coverage).
    """
    counts: Dict[str, int] = {}
    real_dir = project_root / _EVAL_REAL_RELPATH
    if not real_dir.is_dir():
        return counts
    for label in load_label_sidecars(real_dir).values():
        if label.error_kind is None:
            continue
        ek = label.equation_kind
        counts[ek] = counts.get(ek, 0) + 1
    return counts


def _count_existing_train_errors_by_kind(project_root: Path) -> Dict[str, int]:
    """Count existing train GT files that carry a non-null error marker, per kind.

    Scans ``data/setmaker/train/*.gt.json`` (the set-maker train export sink) and
    counts how many carry a non-null ``error_kind`` marker, grouped by the GT's
    ``equation_type`` (the long-form operation kind: addition / subtraction /
    multiplication / division). Used to compute the per-kind deficit for the train
    error-case worklist items so the planner stops requesting a kind once its
    deliberately-wrong corpus is complete.

    Robust to a missing directory (fresh checkout: zero), unreadable or malformed
    JSON (skipped, not raised), and GT files without the error key (counted as "no
    error"). A train GT that records its error tag under ``error_kind`` is the
    contract the train exporter must honour for this deficit loop to close; until
    it does, every train GT counts as zero and the planner serves the full quota.
    """
    counts: Dict[str, int] = {}
    train_dir = project_root / _TRAIN_RELPATH
    if not train_dir.is_dir():
        return counts
    for gt_path in sorted(train_dir.glob("*.gt.json")):
        try:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("train worklist: skipping unreadable GT %s", gt_path)
            continue
        if not isinstance(gt, dict):
            continue
        marker = gt.get(_TRAIN_GT_ERROR_KEY)
        if marker is None:
            continue
        kind = gt.get(_TRAIN_GT_KIND_KEY)
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _build_error_items(
    error_quotas: Dict[str, int],
    existing_errors: Dict[str, int],
    used_seeds: set,
    served_seeds: set,
    seed_base: int,
    log_label: str,
) -> List[WorklistItem]:
    """Build deficit-driven ``directive="error"`` items for one mode.

    Shared by the eval and train planners (DRY): for each operation kind in
    ``error_quotas`` (sorted for determinism), the deficit is the quota minus the
    count of existing error scenes for that kind. One item per still-needed scene
    is emitted with ``directive="error"``, ``completion_stage="full"`` (a complete
    equation is easier to draw deliberately wrong than a partial one), the kind's
    representative ``scene_case`` from :data:`_ERROR_KIND_SCENE_CASE`, and a seed
    from the supplied disjoint ``seed_base`` block. Seeds already in
    ``served_seeds`` or ``used_seeds`` are skipped (FIX 2 / no in-list repeats).
    """
    items: List[WorklistItem] = []
    for kind in sorted(error_quotas):
        target_count = int(error_quotas[kind])
        have = existing_errors.get(kind, 0)
        deficit = target_count - have
        if deficit <= 0:
            continue
        scene_case = _ERROR_KIND_SCENE_CASE.get(kind)
        if scene_case is None:
            logger.warning(
                "%s worklist: no representative scene_case for error kind %r; skipped",
                log_label, kind,
            )
            continue
        # Sequential seed search offset by `have` so resuming after some error
        # scenes already exist starts from the right counter.
        counter = have
        planned = 0
        while planned < deficit:
            if counter >= SEED_BLOCK:
                logger.warning(
                    "%s worklist: error seed namespace exhausted for kind %r; "
                    "reduce quota or clear served seeds.",
                    log_label, kind,
                )
                break
            seed = _error_seed(kind, counter, seed_base)
            counter += 1
            if seed in served_seeds or seed in used_seeds:
                continue
            used_seeds.add(seed)
            items.append(
                WorklistItem(
                    case=scene_case,
                    seed=seed,
                    completion_stage="full",
                    directive="error",
                )
            )
            planned += 1
    return items


def build_eval_worklist(
    *, project_root: Optional[Path] = None, served_seeds: Optional[set] = None
) -> List[WorklistItem]:
    """Plan the held-out real eval set as the per-case coverage deficit.

    Deficit-driven: for each ``scene_case`` in the quota's ``eval`` block, subtract
    the number of scenes that already exist (counted across ``data/eval/real/`` and
    ``data/eval/bank/`` via ``load_label_sidecars``) from the quota target. Cases
    that already meet or exceed their quota are skipped. Remaining cases are ordered
    biggest deficit first (ties broken by case name for determinism), and one
    worklist item is emitted per still-needed scene with a pinned completion stage
    and an eval-namespace seed.

    Seeds in ``served_seeds`` (the durable across-session record) are skipped so
    each new session receives a distinct equation even before any save occurs
    (FIX 2).

    Args:
        project_root: Project root containing ``data/`` and ``src/``. Auto-resolved
            from this module's location when omitted (matches the ``detect.py``
            convention); tests pass a temporary tree.
        served_seeds: Set of seeds already served in prior sessions (loaded by the
            app from the durable state file). Defaults to empty (no skipping) so
            the function works unchanged when called without this argument.

    Returns:
        Ordered ``List[WorklistItem]``; empty when every case meets its quota.

    Raises:
        FileNotFoundError: If the quota file is absent.
        ValueError: If the quota file fails schema validation.
    """
    root = _resolved_root(project_root)
    quota = _load_quota(root)
    targets: Dict[str, int] = quota["eval"]["scene_cases"]
    existing = _count_existing_by_case(root)
    _served: set = served_seeds if served_seeds is not None else set()

    # Compute per-case deficit; drop met cases.
    deficits: Dict[str, int] = {}
    for case, target in targets.items():
        deficit = int(target) - existing.get(case, 0)
        if deficit > 0:
            deficits[case] = deficit

    # Biggest deficit first; case name as a stable tie-breaker.
    ordered_cases = sorted(deficits, key=lambda c: (-deficits[c], c))

    used_seeds: set = set()
    worklist: List[WorklistItem] = []
    for case in ordered_cases:
        worklist.extend(
            _items_for_case(case, deficits[case], EVAL_SEED_BASE, used_seeds, _served)
        )

    logger.info(
        "eval worklist: %d cases below quota, %d scenes planned",
        len(ordered_cases),
        len(worklist),
    )

    # ------------------------------------------------------------------
    # Error-case items (appended after all normal + grid items).
    # For each operation kind in the quota's eval.error_cases block, count
    # existing real-eval sidecars that carry a non-null error_kind for that
    # equation_kind, compute the deficit, and append that many items with
    # directive="error" using the representative scene_case for the kind.
    # Seeds come from the disjoint ERROR_SEED_BASE namespace (one SEED_BLOCK
    # sub-range per kind, ordered alphabetically for determinism).
    # ------------------------------------------------------------------
    error_quotas: Dict[str, int] = quota.get("eval", {}).get("error_cases", {})
    if error_quotas:
        error_items = _build_error_items(
            error_quotas=error_quotas,
            existing_errors=_count_existing_errors_by_kind(root),
            used_seeds=used_seeds,
            served_seeds=_served,
            seed_base=ERROR_SEED_BASE,
            log_label="eval",
        )
        if error_items:
            logger.info(
                "eval worklist: appended %d error-case items",
                len(error_items),
            )
        worklist.extend(error_items)

    return worklist


# ---------------------------------------------------------------------------
# Train: gap-driven planner
# ---------------------------------------------------------------------------

def _latest_per_scene_case(project_root: Path) -> Optional[Dict[str, dict]]:
    """Return the ``per_scene_case`` map from the most recent eval run.

    Reads ``reports/eval/history.jsonl`` (a JSONL file whose first line is a schema
    descriptor and whose remaining lines are run records) and returns the
    ``per_scene_case`` block of the last run record. Returns ``None`` when the file
    is absent, holds no run records, or the latest record has no non-empty
    ``per_scene_case`` (the schema-version-1 case, true until the first schema-2
    eval run). Each value is an ``AggregateMetric`` serialised as a dict (keys such
    as ``row_acc_macro``, ``col_acc_macro``, ``equation_kind_acc``,
    ``mean_label_acc``).

    Malformed JSON lines are skipped rather than raising, so a partially written
    history never crashes the planner (the fallback path then applies).
    """
    history_path = project_root / _HISTORY_RELPATH
    if not history_path.is_file():
        return None

    last_record: Optional[dict] = None
    with history_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("record_type") == _SCHEMA_RECORD_TYPE:
                continue
            last_record = obj

    if last_record is None:
        return None
    per_scene_case = last_record.get("per_scene_case")
    if not per_scene_case:  # absent, None, or empty dict
        return None
    return per_scene_case


def _weakness_score(metric: dict) -> float:
    """Weakness composite for one case from its ``AggregateMetric`` dict.

    Higher is weaker (more deserving of training budget). Each accuracy term is
    converted to a ``(1 - acc)`` deficit and weighted; row and column placement
    dominate (client priority). A ``None`` metric (e.g. no spatial samples) counts
    as fully weak on that axis so the case is not silently dropped.
    """

    def deficit(value: object) -> float:
        if value is None:
            return _MISSING_METRIC_WEAKNESS
        return 1.0 - float(value)

    return (
        WEIGHT_ROW * deficit(metric.get("row_acc_macro"))
        + WEIGHT_COL * deficit(metric.get("col_acc_macro"))
        + WEIGHT_EQ_KIND * deficit(metric.get("equation_kind_acc"))
        + WEIGHT_LABEL * deficit(metric.get("mean_label_acc"))
    )


def _allocate(
    weakness: Dict[str, float], round_budget: int, caps: Dict[str, int]
) -> Dict[str, int]:
    """Split ``round_budget`` across cases proportional to weakness, floor+cap.

    Allocation is largest-remainder proportional to each case's weakness, clamped
    into ``[TRAIN_PER_CASE_FLOOR, cap]`` per case (the cap is the case's provisional
    quota, reflecting diminishing returns). The floor guarantees coverage for every
    case in the plan even when one case dominates the weakness distribution (risk
    R4). The returned counts may sum to slightly more than ``round_budget`` only
    when the per-case floors alone already exceed it; otherwise the total is held at
    ``round_budget`` (subject to caps).

    Cases with zero weakness (perfect metrics) are not allocated proportional share
    but still receive the floor so they are not abandoned between rounds.
    """
    cases = sorted(weakness)  # stable ordering for deterministic remainders
    if not cases:
        return {}

    # Start everyone at the floor (clamped to the cap), consuming budget.
    alloc: Dict[str, int] = {}
    for case in cases:
        cap = max(0, int(caps.get(case, 0)))
        alloc[case] = min(TRAIN_PER_CASE_FLOOR, cap)

    remaining = round_budget - sum(alloc.values())
    if remaining <= 0:
        return alloc

    total_weakness = sum(weakness[c] for c in cases)
    if total_weakness <= 0.0:
        # No signal: nothing weak. Leave everyone at the floor.
        return alloc

    # Proportional share of the remaining budget, by weakness, with headroom to
    # the cap. Largest-remainder rounding keeps the integer total exact.
    headroom = {c: max(0, int(caps.get(c, 0)) - alloc[c]) for c in cases}
    raw: Dict[str, float] = {
        c: remaining * (weakness[c] / total_weakness) for c in cases
    }
    floor_alloc: Dict[str, int] = {c: int(raw[c]) for c in cases}
    # Clamp to headroom before distributing remainders.
    for c in cases:
        floor_alloc[c] = min(floor_alloc[c], headroom[c])

    distributed = sum(floor_alloc.values())
    leftover = remaining - distributed
    # Hand out the leftover one unit at a time to the largest fractional parts that
    # still have headroom, so the total lands exactly on the budget when possible.
    # Each pass over the cases places at least one unit while any headroom remains,
    # so at most (total remaining headroom) iterations run before leftover hits 0.
    remainder_order = sorted(
        cases, key=lambda c: (-(raw[c] - int(raw[c])), c)
    )
    while leftover > 0 and any(floor_alloc[x] < headroom[x] for x in cases):
        for c in remainder_order:
            if leftover <= 0:
                break
            if floor_alloc[c] < headroom[c]:
                floor_alloc[c] += 1
                leftover -= 1

    for c in cases:
        alloc[c] += floor_alloc[c]
    return alloc


def build_train_worklist(
    round_budget: int, *, project_root: Optional[Path] = None, served_seeds: Optional[set] = None
) -> List[WorklistItem]:
    """Plan one round of real fine-tune scenes routed to model gaps.

    Gap-driven: read the latest ``per_scene_case`` block from
    ``reports/eval/history.jsonl``, score each case's weakness (row and column
    placement weighted heavy per client priority), and allocate ``round_budget``
    scenes across cases proportional to weakness with a per-case floor and a
    per-case cap drawn from the quota's ``train`` block. When no history record
    carries a non-empty ``per_scene_case`` (true until the first schema-2 eval run),
    the planner logs a warning and falls back to the provisional ``train`` weights
    in the quota file, treating each case's provisional target as its weakness
    signal so the round still routes sensibly.

    Items use the **train** seed namespace, disjoint from eval (risk R7), and pin a
    completion stage per item.

    Args:
        round_budget: Number of scenes to plan this round (must be positive).
        project_root: Project root; auto-resolved when omitted (tests pass a
            temporary tree).
        served_seeds: Set of seeds already served in prior sessions (FIX 2).
            Defaults to empty.

    After the gap-allocated items, deficit-driven ``directive="error"`` items are
    appended (one per still-needed deliberately-wrong scene) from the quota's
    ``train.error_cases`` block, counted against existing train GT files that carry
    an error marker. These exercise the operator/equation-kind head that the
    measured real gap exposes. They use the disjoint ``TRAIN_ERROR_SEED_BASE``
    namespace. When ``round_budget <= 0`` nothing is planned at all (no gap items
    and no error items), preserving the empty-on-non-positive-budget contract.

    Returns:
        Ordered ``List[WorklistItem]`` (weakest gap case first, then error items);
        empty when ``round_budget <= 0``.

    Raises:
        FileNotFoundError: If the quota file is absent.
        ValueError: If the quota file fails schema validation.
    """
    if round_budget <= 0:
        logger.info("train worklist: non-positive round_budget=%d, nothing planned", round_budget)
        return []

    root = _resolved_root(project_root)
    quota = _load_quota(root)
    caps: Dict[str, int] = {k: int(v) for k, v in quota["train"]["scene_cases"].items()}
    _served: set = served_seeds if served_seeds is not None else set()

    per_scene_case = _latest_per_scene_case(root)
    if per_scene_case is None:
        # Soft-gate fallback: no usable eval history. Route by the provisional
        # quota weights so the round is still useful, and surface the reason.
        logger.warning(
            "train worklist: no eval run with a non-empty per_scene_case in "
            "reports/eval/history.jsonl; falling back to provisional quota train "
            "weights. Run `eval` to route training to measured model gaps."
        )
        weakness: Dict[str, float] = {
            case: float(cap) for case, cap in caps.items() if cap > 0
        }
    else:
        # Score only cases that the quota knows about (the planner's universe).
        weakness = {}
        for case, cap in caps.items():
            if cap <= 0:
                continue
            metric = per_scene_case.get(case)
            if metric is None:
                # Case not measured this run: treat as maximally weak so it still
                # gets at least the floor and is not abandoned.
                weakness[case] = _weakness_score({})
            else:
                weakness[case] = _weakness_score(metric)

    allocation = _allocate(weakness, round_budget, caps)

    # Weakest (largest allocation, then largest weakness) first; case name breaks
    # ties so the order is fully deterministic.
    ordered_cases = sorted(
        (c for c in allocation if allocation[c] > 0),
        key=lambda c: (-allocation[c], -weakness.get(c, 0.0), c),
    )

    used_seeds: set = set()
    worklist: List[WorklistItem] = []
    for case in ordered_cases:
        worklist.extend(
            _items_for_case(case, allocation[case], TRAIN_SEED_BASE, used_seeds, _served)
        )

    logger.info(
        "train worklist: %d cases routed, %d scenes planned (budget %d)",
        len(ordered_cases),
        len(worklist),
        round_budget,
    )

    # ------------------------------------------------------------------
    # Train error-case items (appended after all gap-allocated items).
    # The measured real gap is the operator/equation-kind head, so the train set
    # needs deliberately-wrong scenes (wrong operator, wrong/missing carry or
    # borrow) the human draws and tags. For each operation kind in the quota's
    # train.error_cases block, count existing train GT files carrying an error
    # marker, compute the deficit, and append that many directive="error" items
    # using the kind's representative scene_case. Seeds come from the disjoint
    # TRAIN_ERROR_SEED_BASE block so they collide with neither the train gap items
    # nor the eval error items.
    # ------------------------------------------------------------------
    error_quotas: Dict[str, int] = quota.get("train", {}).get("error_cases", {})
    if error_quotas:
        error_items = _build_error_items(
            error_quotas=error_quotas,
            existing_errors=_count_existing_train_errors_by_kind(root),
            used_seeds=used_seeds,
            served_seeds=_served,
            seed_base=TRAIN_ERROR_SEED_BASE,
            log_label="train",
        )
        if error_items:
            logger.info(
                "train worklist: appended %d error-case items",
                len(error_items),
            )
        worklist.extend(error_items)

    return worklist
