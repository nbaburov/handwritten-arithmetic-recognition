"""SHA-256 file hashing helpers (single source of truth)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_or_none(path: Optional[Path]) -> Optional[str]:
    """Return SHA-256 hex of a file, or None if path is None / unreadable."""
    if path is None:
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None
