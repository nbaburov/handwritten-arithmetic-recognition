"""The single in-memory demo session.

One presenter, one active exercise: no persistence or multi-user management
needed. :class:`DemoSession` holds the exercise spec, the grading engine session,
the shared :class:`GridCalibration`, the derived :class:`BankExercise`, and
the loaded :class:`InferenceSession`. The app rebinds it on each
``/api/exercise/select`` and reads it on ``/api/evaluate``. Stored on
``app.state`` so tests can inject their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

from ..grading.exercises import BankExercise
from ..grading.grid_mapping import GridCalibration
from ..grading.types import ExerciseSpec, SessionCreated
from ..inference.run import InferenceSession


class _ActiveQuad(NamedTuple):
    """Groups the four fields that move together on each exercise selection.

    NamedTuple assignment is a single name binding in CPython, so a concurrent
    ``/api/evaluate`` sees either the old quad or the new one in full. Full
    thread-safety still requires uvicorn ``workers=1`` (single process); this
    only guards against intra-process interleaving.
    """

    exercise: ExerciseSpec
    session_created: SessionCreated
    calibration: GridCalibration
    bank_exercise: BankExercise


@dataclass
class DemoSession:
    """Mutable holder for the active exercise + its engine session.

    ``inference_session`` is shared across selections (weights load once). The
    active quad is ``None`` until the first selection; :meth:`require_active`
    fails loudly if ``/api/evaluate`` is called before ``/api/exercise/select``.

    Requires uvicorn ``workers=1``: the session lives in process memory and
    cannot be shared across worker processes.
    """

    inference_session: InferenceSession
    _active: Optional[_ActiveQuad] = None

    def select(
        self,
        exercise: ExerciseSpec,
        session_created: SessionCreated,
        calibration: GridCalibration,
        bank_exercise: BankExercise,
    ) -> None:
        """Bind the active quad atomically (single attribute assignment under the GIL)."""
        self._active = _ActiveQuad(
            exercise=exercise,
            session_created=session_created,
            calibration=calibration,
            bank_exercise=bank_exercise,
        )

    def is_active(self) -> bool:
        """True once an exercise has been selected (the quad is bound)."""
        return self._active is not None

    def require_active(self) -> tuple[ExerciseSpec, SessionCreated, GridCalibration, BankExercise]:
        """Return the active quad or raise ``ValueError`` (-> 422) if none selected."""
        active = self._active
        if active is None:
            raise ValueError(
                "no active exercise; POST /api/exercise/select before evaluating"
            )
        return active.exercise, active.session_created, active.calibration, active.bank_exercise
