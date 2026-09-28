"""ShowcaseStore — pure I/O helper for the Showcase tab data-collection layer.

All methods are side-effect only (writes files/JSONL). No Gradio imports.
No pipeline internals — callers compute the outcome dict and pass it in.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# ── Canonical outcome status strings ─────────────────────────────────────────
STATUS_OK = "ok"
STATUS_BLANK = "blank_canvas"
STATUS_NO_MODELS = "no_models"
STATUS_NO_DETECTIONS = "no_detections"
STATUS_OOD = "ood"
STATUS_ERROR = "error"

# Statuses for which the feedback buttons should be enabled.
FEEDBACK_ENABLED_STATUSES = frozenset({STATUS_OK, STATUS_NO_DETECTIONS, STATUS_OOD})


from ..core.hashing import sha256_file_or_none as _sha256_file  # noqa: E402


@dataclass
class ShowcaseStore:
    """Manages per-sample directories under ``showcase_root``.

    Parameters
    ----------
    showcase_root:
        Base directory, e.g. ``<project_root>/data/generated/showcase/``.
    yolo_weights_path:
        Optional path to the YOLO .pt file currently loaded in the session.
    gnn_weights_path:
        Optional path to the GNN .pt file currently loaded in the session.
    """

    showcase_root: Path
    yolo_weights_path: Optional[Path] = None
    gnn_weights_path: Optional[Path] = None

    # Cached SHAs — computed once on first use.
    _yolo_sha: Optional[str] = field(default=None, init=False, repr=False)
    _gnn_sha: Optional[str] = field(default=None, init=False, repr=False)
    _shas_computed: bool = field(default=False, init=False, repr=False)

    # Cross-page synchronisation state.
    # Incremented inside record_prediction; ResultsPage polls this to detect new samples.
    latest_seq: int = field(default=0, init=False, repr=False)
    _latest_sample_dir: Optional[Path] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.showcase_root = Path(self.showcase_root)
        self.showcase_root.mkdir(parents=True, exist_ok=True)
        # Compute SHAs eagerly on construction (once per session load).
        self._yolo_sha = _sha256_file(self.yolo_weights_path)
        self._gnn_sha = _sha256_file(self.gnn_weights_path)
        self._shas_computed = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _next_seq(self) -> int:
        """Return the next sequential sample number (1-based)."""
        existing = sorted(self.showcase_root.glob("sample_*/"))
        return len(existing) + 1

    def _sample_dir(self, seq: int) -> Path:
        d = self.showcase_root / f"sample_{seq:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _index_path(self) -> Path:
        return self.showcase_root / "index.jsonl"

    def _append_index(self, record: Dict[str, Any]) -> None:
        with self._index_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def record_prediction(
        self,
        *,
        gray_png_bytes: Optional[bytes],
        outcome: Dict[str, Any],
    ) -> Path:
        """Persist a prediction record to a new sample directory.

        Parameters
        ----------
        gray_png_bytes:
            Raw PNG bytes of the input image (512x512 grayscale).
            May be None for ``blank_canvas`` — we still write a zero-byte
            placeholder to record the button-press event.
        outcome:
            Dict with at minimum ``status`` (one of the six canonical strings),
            plus whatever debug fields the handler computed.

        Returns
        -------
        Path
            The sample directory that was created.
        """
        seq = self._next_seq()
        sample_dir = self._sample_dir(seq)

        # Write input image (or empty file for blank).
        img_path = sample_dir / "input.png"
        if gray_png_bytes:
            img_path.write_bytes(gray_png_bytes)
        else:
            img_path.write_bytes(b"")

        # Enrich outcome with model provenance.
        full_outcome: Dict[str, Any] = {
            "schema_version": 1,
            "sample": f"sample_{seq:04d}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_provenance": {
                "yolo_weights": str(self.yolo_weights_path) if self.yolo_weights_path else None,
                "gnn_weights": str(self.gnn_weights_path) if self.gnn_weights_path else None,
                "yolo_sha": self._yolo_sha,
                "gnn_sha": self._gnn_sha,
            },
        }
        full_outcome.update(outcome)

        pred_path = sample_dir / "prediction.json"
        pred_path.write_text(json.dumps(full_outcome, indent=2, ensure_ascii=False))

        # Denormalised index line.
        index_record = {
            "event": "predict",
            "sample": full_outcome["sample"],
            "timestamp": full_outcome["timestamp"],
            "outcome_status": outcome.get("status"),
            "equation_kind": outcome.get("equation_kind"),
            "num_tokens": outcome.get("num_tokens"),
        }
        self._append_index(index_record)

        # Update cross-page sync state.
        self._latest_sample_dir = sample_dir
        self.latest_seq += 1

        return sample_dir

    def get_latest_sample(self) -> Optional[Path]:
        """Return the directory of the most recently recorded sample, or None."""
        return self._latest_sample_dir

    def record_feedback(
        self,
        *,
        sample_dir: Path,
        verdict: str,
        intended_text: str = "",
        per_token: Optional[list] = None,
    ) -> None:
        """Write (overwrite) feedback for a previously recorded sample.

        Parameters
        ----------
        sample_dir:
            The directory returned by a prior ``record_prediction`` call.
        verdict:
            ``"correct"`` or ``"wrong"``.
        intended_text:
            Optional free-text from the attendee describing what they intended.
        per_token:
            Optional list of per-token dicts. Keys: index, predicted_label,
            user_says_wrong. Stored as-is under ``per_token`` in feedback.json.
            None is omitted from the output (backward compatible).

        Notes
        -----
        This method **overwrites** feedback.json on each call — calling it
        twice for the same sample_dir is the intended update flow (first call
        captures the quick verdict; second call adds detail).
        """
        feedback: Dict[str, Any] = {
            "schema_version": 1,
            "verdict": verdict,
            "intended_text": intended_text,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if per_token is not None:
            feedback["per_token"] = per_token
        fb_path = sample_dir / "feedback.json"
        fb_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False))

        # Also append to the shared index.
        sample_name = sample_dir.name
        index_record = {
            "event": "feedback",
            "sample": sample_name,
            "timestamp": feedback["ts"],
            "verdict": verdict,
        }
        self._append_index(index_record)
