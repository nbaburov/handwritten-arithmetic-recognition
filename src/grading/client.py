"""The swappable grading engine client Protocol and its factory.

``GradingClient`` is a structural Protocol: any object that exposes the four
methods (``create_session``, ``evaluate``, ``solution``, ``info``) satisfies it,
so the demo never imports a concrete client. ``make_client`` returns the
in-process ``MockGradingClient``; a live engine adapter would implement the
same Protocol. This module owns the interface and the selection logic, not
grading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .types import EvalResult, ScoringInfo, SessionCreated

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from ..core.run_config import GradingConfig
    from .types import ExerciseSpec, GridToken


class GradingError(RuntimeError):
    """Raised on any client transport or contract failure.

    Every client raises this type so the ``/api/evaluate`` route stays
    engine-agnostic (it maps this to a 502).
    """


@runtime_checkable
class GradingClient(Protocol):
    """The contract every grading client satisfies."""

    def create_session(self, spec: "ExerciseSpec") -> SessionCreated:
        """Start a session for one bank exercise; return the parsed create result."""

    def evaluate(
        self, session_id: str, ref_id: str, tokens: "list[GridToken]"
    ) -> EvalResult:
        """Grade the submitted grid tokens; return the engine-shaped evaluate result."""

    def solution(self, session_id: str) -> dict:
        """Return the full worked-solution element tree (raw dict)."""

    def info(self, session_id: str) -> ScoringInfo:
        """Return the session's score + history."""


def make_client(
    config: "GradingConfig", project_root: "Path"
) -> GradingClient:
    """Build the client selected by ``config.mode`` (only ``"mock"`` ships)."""
    mode = config.mode.lower()
    if mode == "mock":
        from .mock_client import MockGradingClient

        return MockGradingClient(config=config, project_root=project_root)
    raise ValueError(
        f"Unknown [grading] mode {config.mode!r}; expected 'mock'."
    )
