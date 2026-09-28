"""Grading-engine integration.

This package owns the swappable grading engine client behind a single Protocol so the
demo can grade recognized work against either a high-fidelity in-process mock
(no API key) or a live engine adapter, selected by ``[grading] mode`` in
``config.toml``. The frozen contract dataclasses in :mod:`.types` mirror the
captured real grading engine JSON exactly, and the contract tests round-trip every
captured fixture through them so the mock can never silently drift from the real
shape.
"""
