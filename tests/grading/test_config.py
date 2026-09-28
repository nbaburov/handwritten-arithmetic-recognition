"""Tests for grading engine config loading and client selection."""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

from src.grading.client import GradingClient, make_client
from src.grading.types import EvalResult, ScoringInfo, SessionCreated
from src.core.run_config import GradingConfig, load_grading_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MOCK_AVAILABLE = importlib.util.find_spec("src.grading.mock_client") is not None


class _FakeClient:
    """A minimal Protocol-satisfying client used to exercise the factory contract.

    WS1 owns the real ``MockGradingClient``; this fake lets WS0 verify the
    factory's structural contract (and the Protocol's ``runtime_checkable``
    behaviour) without depending on a file another workstream builds.
    """

    def create_session(self, spec: object) -> SessionCreated:  # noqa: D401
        return SessionCreated(session_id="s", ref_id="r", marks_total=1)

    def evaluate(self, session_id: str, ref_id: str, tokens: list) -> EvalResult:
        return EvalResult()

    def solution(self, session_id: str) -> dict:
        return {}

    def info(self, session_id: str) -> ScoringInfo:
        return ScoringInfo(finished=False, marks_total=1, marks_earned=0)


def test_load_returns_mock_config_from_shipped_config() -> None:
    """The shipped config.toml [grading] section selects the in-process mock."""
    cfg = load_grading_config(PROJECT_ROOT)
    assert cfg.mode == "mock"
    assert cfg.audience == "uk_KS2"


def test_load_returns_defaults_when_config_absent(tmp_path: Path) -> None:
    cfg = load_grading_config(tmp_path)
    assert cfg == GradingConfig()
    assert cfg.mode == "mock"


def test_load_reads_audience(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [grading]
            mode = "mock"
            audience = "nl_VO"
            """
        ),
        encoding="utf-8",
    )
    cfg = load_grading_config(tmp_path)
    assert cfg.mode == "mock"
    assert cfg.audience == "nl_VO"


def test_load_falls_back_on_invalid_mode(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[grading]\nmode = "bogus"\n', encoding="utf-8"
    )
    cfg = load_grading_config(tmp_path)
    assert cfg.mode == "mock"


def test_fake_client_satisfies_runtime_protocol() -> None:
    """The runtime_checkable Protocol accepts any object with the four methods."""
    fake = _FakeClient()
    assert isinstance(fake, GradingClient)
    for method in ("create_session", "evaluate", "solution", "info"):
        assert callable(getattr(fake, method))


def test_make_client_mock_uses_fake_when_real_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_client(mode=mock) imports its impl lazily and returns it.

    When WS1's real ``mock_client`` is present, the factory returns the real
    ``MockGradingClient``; otherwise this injects a Protocol-satisfying fake
    module so the selection logic is still proven in isolation.
    """
    if not _MOCK_AVAILABLE:
        import sys
        import types as _types

        fake_module = _types.ModuleType("src.grading.mock_client")

        def _factory(*, config: GradingConfig, project_root: Path) -> _FakeClient:
            return _FakeClient()

        fake_module.MockGradingClient = _factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.grading.mock_client", fake_module)

    client = make_client(GradingConfig(mode="mock"), PROJECT_ROOT)
    assert isinstance(client, GradingClient)
    for method in ("create_session", "evaluate", "solution", "info"):
        assert callable(getattr(client, method))


def test_make_client_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        make_client(GradingConfig(mode="carrier-pigeon"), PROJECT_ROOT)
