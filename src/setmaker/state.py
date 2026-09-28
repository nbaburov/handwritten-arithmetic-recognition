"""Session persistence for the set-maker tool (WS-F).

Owns one responsibility: reading, writing, and summarising the per-mode
:class:`~src.setmaker.types.SessionState` that records the planned worklist and
the annotator's cursor. The state lives as JSON on disk at
``data/setmaker/state/<mode>.json`` (one file per mode) so the FastAPI app can
autosave after every ``/api/save`` and resume the worklist after a restart.

The module deliberately does not decide *what* to draw (that is the worklist
planner, WS-C) and does not read the quota single source of truth. It only
serialises the state the planner produced, advances the cursor as scenes are
completed, and reports done / pending / total counts. The project root is passed
in by the caller (the app resolves it once via
:func:`src.core.config.resolve_project_root`); this module never discovers paths
on its own, which keeps it a typed, injectable boundary that tests can point at a
temporary directory.

Public functions:
    state_dir(project_root)          -> directory holding the per-mode JSON files
    state_path(project_root, mode)   -> JSON path for one mode
    load(mode, project_root)         -> SessionState (fresh if no file on disk)
    save(state, project_root)        -> Path written (atomic temp-then-rename)
    advance(state)                   -> SessionState with the cursor item done
    progress(state)                  -> per-case + overall counts dict

``SessionState`` and ``WorklistItem`` are frozen, so ``advance`` returns a new
instance via :func:`dataclasses.replace` rather than mutating in place.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .types import Directive, ItemStatus, Mode, SessionState, WorklistItem

# Valid mode strings, kept in sync with the ``Mode`` literal in ``types``. Used to
# reject an unknown mode before it ever reaches the filesystem as a path segment.
VALID_MODES: frozenset[str] = frozenset(("eval", "train"))

# Valid per-item lifecycle states, kept in sync with the ``ItemStatus`` literal.
VALID_STATUSES: frozenset[str] = frozenset(("pending", "done"))

# Valid per-item directive values, kept in sync with the ``Directive`` literal.
VALID_DIRECTIVES: frozenset[str] = frozenset(("normal", "error"))

# Schema tag written into every state file so a future shape change can be
# detected on load rather than silently misread.
STATE_SCHEMA_VERSION: int = 1

# Schema tag for the served-seeds file.
SERVED_SEEDS_SCHEMA_VERSION: int = 1


def _utc_now_iso() -> str:
    """Return the current time as an ISO-8601 UTC string (repo timestamp idiom)."""
    return datetime.now(timezone.utc).isoformat()


def state_dir(project_root: Path) -> Path:
    """Return the directory that holds the per-mode state JSON files.

    This is ``<project_root>/data/setmaker/state``. The directory is not created
    here; :func:`save` creates it lazily so a pure :func:`load` never writes.
    """
    return Path(project_root) / "data" / "setmaker" / "state"


def state_path(project_root: Path, mode: Mode) -> Path:
    """Return the JSON path for one mode's session state.

    Raises ``ValueError`` (with the offending value) when ``mode`` is not a known
    mode, so an unexpected string can never become a filesystem path segment.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode '{mode}' is not valid; must be one of {sorted(VALID_MODES)}"
        )
    return state_dir(project_root) / f"{mode}.json"


def _worklist_item_from_dict(raw: Any, mode: Mode, file_path: Path) -> WorklistItem:
    """Build one :class:`WorklistItem` from a decoded JSON object.

    Validates the four fields against their contract and raises ``ValueError``
    naming ``file_path`` on any malformed entry, matching the project's
    fail-loud-with-path convention.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"{file_path}: each worklist entry must be a JSON object, got {type(raw).__name__}"
        )
    for key in ("case", "seed", "completion_stage"):
        if key not in raw:
            raise ValueError(f"{file_path}: worklist entry is missing required key '{key}'")

    case = raw["case"]
    if not isinstance(case, str) or not case:
        raise ValueError(f"{file_path}: worklist 'case' must be a non-empty string, got {case!r}")

    seed = raw["seed"]
    # bool is a subclass of int; reject it explicitly so a stray ``true`` is not read as 1.
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{file_path}: worklist 'seed' must be an int, got {seed!r}")

    completion_stage = raw["completion_stage"]
    if not isinstance(completion_stage, str) or not completion_stage:
        raise ValueError(
            f"{file_path}: worklist 'completion_stage' must be a non-empty string, got {completion_stage!r}"
        )

    status: ItemStatus = raw.get("status", "pending")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"{file_path}: worklist 'status' '{status}' is not valid; "
            f"must be one of {sorted(VALID_STATUSES)}"
        )

    # directive defaults to "normal" so older state files without the field load
    # cleanly (backward compatible).
    directive: Directive = raw.get("directive", "normal")
    if directive not in VALID_DIRECTIVES:
        raise ValueError(
            f"{file_path}: worklist 'directive' '{directive}' is not valid; "
            f"must be one of {sorted(VALID_DIRECTIVES)}"
        )

    return WorklistItem(
        case=case,
        seed=seed,
        completion_stage=completion_stage,
        status=status,
        directive=directive,
    )


def _trim_completed_prefix(
    worklist: List[WorklistItem], cursor: int
) -> "tuple[List[WorklistItem], int]":
    """Drop the leading run of already-done items and re-base the cursor.

    Resume semantics: the worklist is consumed front-to-back, so every item
    before the cursor has been drawn (``status == "done"``). On load those
    already-met items are trimmed so the resumed session lands on the first
    outstanding scene and the progress bar reflects only remaining work. Any
    ``done`` item that is not part of the leading run (an out-of-order
    completion) is preserved so its record is not lost. The returned cursor is
    clamped to the trimmed list bounds.
    """
    leading_done = 0
    for item in worklist:
        if item.status == "done":
            leading_done += 1
        else:
            break

    trimmed = worklist[leading_done:]
    new_cursor = max(0, cursor - leading_done)
    new_cursor = min(new_cursor, len(trimmed))
    return trimmed, new_cursor


def load(mode: Mode, project_root: Path) -> SessionState:
    """Load the persisted :class:`SessionState` for ``mode``.

    Returns the stored worklist and cursor so the annotator can resume where the
    last session stopped. When no state file exists yet (first run for this
    mode), returns a fresh empty state for the mode rather than raising, so the
    caller can treat "no prior session" and "empty session" identically.

    Resume trims already-met cases: the leading run of completed worklist items
    is dropped and the cursor re-based onto the first outstanding scene (see
    :func:`_trim_completed_prefix`). A malformed file raises ``ValueError``
    naming the path; a structurally valid file with a mismatched ``mode`` raises
    ``ValueError`` rather than silently trusting the filename.
    """
    path = state_path(project_root, mode)
    if not path.exists():
        return SessionState(mode=mode)

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc

    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top-level state must be a JSON object, got {type(doc).__name__}")

    stored_mode = doc.get("mode", mode)
    if stored_mode != mode:
        raise ValueError(
            f"{path}: stored mode '{stored_mode}' does not match requested mode '{mode}'"
        )

    raw_worklist = doc.get("worklist", [])
    if not isinstance(raw_worklist, list):
        raise ValueError(f"{path}: 'worklist' must be a JSON array, got {type(raw_worklist).__name__}")
    worklist = [_worklist_item_from_dict(entry, mode, path) for entry in raw_worklist]

    cursor = doc.get("cursor", 0)
    if not isinstance(cursor, int) or isinstance(cursor, bool):
        raise ValueError(f"{path}: 'cursor' must be an int, got {cursor!r}")
    if cursor < 0:
        raise ValueError(f"{path}: 'cursor' must be non-negative, got {cursor}")

    round_budget = doc.get("round_budget", None)
    if round_budget is not None and (not isinstance(round_budget, int) or isinstance(round_budget, bool)):
        raise ValueError(f"{path}: 'round_budget' must be an int or null, got {round_budget!r}")

    created_utc = doc.get("created_utc", "")
    last_saved_utc = doc.get("last_saved_utc", "")
    if not isinstance(created_utc, str):
        raise ValueError(f"{path}: 'created_utc' must be a string, got {created_utc!r}")
    if not isinstance(last_saved_utc, str):
        raise ValueError(f"{path}: 'last_saved_utc' must be a string, got {last_saved_utc!r}")

    trimmed_worklist, trimmed_cursor = _trim_completed_prefix(worklist, cursor)

    return SessionState(
        mode=mode,
        worklist=trimmed_worklist,
        cursor=trimmed_cursor,
        round_budget=round_budget,
        created_utc=created_utc,
        last_saved_utc=last_saved_utc,
    )


def _serialise(state: SessionState) -> Dict[str, Any]:
    """Build the JSON-ready dict for ``state`` (adds the schema tag)."""
    payload: Dict[str, Any] = {"schema_version": STATE_SCHEMA_VERSION}
    payload.update(asdict(state))
    return payload


def save(state: SessionState, project_root: Path) -> Path:
    """Persist ``state`` to ``data/setmaker/state/<mode>.json`` and return the path.

    Stamps ``last_saved_utc`` to now and sets ``created_utc`` on the first save
    (when it is still empty). The write is atomic: the JSON is written to a
    sibling temporary file and then renamed over the target, so a crash mid-write
    can never leave a half-written, unreadable state file. The state directory is
    created lazily here so a pure :func:`load` never writes to disk.

    ``state.mode`` is validated through :func:`state_path`, so an out-of-contract
    mode is rejected before any filesystem operation.
    """
    target = state_path(project_root, state.mode)
    target.parent.mkdir(parents=True, exist_ok=True)

    now = _utc_now_iso()
    stamped = replace(
        state,
        created_utc=state.created_utc or now,
        last_saved_utc=now,
    )

    payload = _serialise(stamped)
    serialised = json.dumps(payload, indent=2)

    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(serialised, encoding="utf-8")
    os.replace(tmp, target)
    return target


def advance(state: SessionState) -> SessionState:
    """Mark the scene at the cursor done and move the cursor to the next item.

    Returns a new :class:`SessionState` (the contract is frozen) in which the
    worklist item at ``cursor`` carries ``status == "done"`` and ``cursor`` has
    incremented by one. When the cursor is already at or past the end of the
    worklist there is nothing to complete, so the state is returned unchanged;
    the caller can detect completion via :func:`progress` (``pending == 0``).
    """
    if state.cursor < 0 or state.cursor >= len(state.worklist):
        return state

    new_worklist = list(state.worklist)
    new_worklist[state.cursor] = replace(new_worklist[state.cursor], status="done")
    return replace(state, worklist=new_worklist, cursor=state.cursor + 1)


def served_seeds_path(project_root: Path, mode: Mode) -> Path:
    """Return the path to the durable served-seeds file for ``mode``.

    Lives alongside the per-mode state JSON at
    ``data/setmaker/state/<mode>_served_seeds.json``. This file survives fresh
    worklist builds (``POST /api/mode``) so seeds are never re-served across
    sessions (FIX 2).
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode '{mode}' is not valid; must be one of {sorted(VALID_MODES)}"
        )
    return state_dir(project_root) / f"{mode}_served_seeds.json"


def load_served_seeds(mode: Mode, project_root: Path) -> set:
    """Load the durable set of already-served seeds for ``mode``.

    Returns an empty set when no file exists (first run). A malformed file is
    treated as empty and the file is silently ignored rather than raising, so a
    corrupted seeds file never blocks the annotator.
    """
    path = served_seeds_path(project_root, mode)
    if not path.exists():
        return set()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        seeds = doc.get("seeds", [])
        if isinstance(seeds, list):
            return {int(s) for s in seeds if isinstance(s, (int, float)) and not isinstance(s, bool)}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return set()


def save_served_seeds(mode: Mode, seeds: set, project_root: Path) -> None:
    """Persist the served-seeds set for ``mode`` atomically.

    The write is atomic (temp-then-rename). The state directory is created
    lazily here so this function can be called without a prior :func:`save`.
    """
    target = served_seeds_path(project_root, mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SERVED_SEEDS_SCHEMA_VERSION,
        "mode": mode,
        "seeds": sorted(seeds),
        "updated_utc": _utc_now_iso(),
    }
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def mark_seed_served(mode: Mode, seed: int, project_root: Path) -> None:
    """Atomically add ``seed`` to the durable served-seeds record for ``mode``.

    Called by ``/api/next`` on every serve (not only on save) so three
    consecutive fresh sessions each receive a distinct equation (FIX 2).
    """
    seeds = load_served_seeds(mode, project_root)
    seeds.add(seed)
    save_served_seeds(mode, seeds, project_root)


def progress(state: SessionState) -> Dict[str, Any]:
    """Summarise completion as per-case and overall counts.

    Returns a dict with an ``overall`` block (``total`` / ``done`` / ``pending``
    item counts and ``percent`` complete, 0.0 when the worklist is empty) and a
    ``per_case`` block mapping each scene_case present in the worklist to its own
    ``total`` / ``done`` / ``pending`` counts. Counts are derived purely from the
    in-memory worklist statuses; the route layer composes these with the quota
    single source of truth to show "quota / done / left / %" per the API
    contract. Cases are emitted in first-seen worklist order so the UI ordering is
    stable across calls.
    """
    per_case: "Dict[str, Dict[str, int]]" = {}
    total = 0
    done = 0
    for item in state.worklist:
        bucket = per_case.setdefault(item.case, {"total": 0, "done": 0, "pending": 0})
        bucket["total"] += 1
        total += 1
        if item.status == "done":
            bucket["done"] += 1
            done += 1
        else:
            bucket["pending"] += 1
    for bucket in per_case.values():
        bucket["pending"] = bucket["total"] - bucket["done"]

    pending = total - done
    percent = (done / total * 100.0) if total else 0.0

    return {
        "mode": state.mode,
        "overall": {
            "total": total,
            "done": done,
            "pending": pending,
            "percent": percent,
        },
        "per_case": per_case,
    }
