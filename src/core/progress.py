from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class SimpleProgress:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.current = 0
        self.started_at = time.time()
        self.last_print = 0.0

    def update(self, n: int = 1) -> None:
        self.current += n
        now = time.time()
        if now - self.last_print < 0.2 and self.current < self.total:
            return
        elapsed = now - self.started_at
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.current) / rate if rate > 0 else 0.0
        pct = min(100.0, (self.current / self.total) * 100.0)
        print(
            f"\r{self.desc}: {self.current}/{self.total} ({pct:5.1f}%) "
            f"ETA {_format_eta(remaining)}",
            end="",
            flush=True,
        )
        self.last_print = now
        if self.current >= self.total:
            print("", flush=True)

    def close(self) -> None:
        if self.current < self.total:
            self.current = self.total
            self.update(0)


@contextmanager
def progress_bar(total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        bar = SimpleProgress(total=total, desc=desc)
        try:
            yield bar
        finally:
            bar.close()
        return

    bar = tqdm(total=total, desc=desc, unit="item", dynamic_ncols=True)
    try:
        yield bar
    finally:
        bar.close()


def progress_iter(items: Iterable[T], total: Optional[int], desc: str) -> Iterator[T]:
    inferred_total = total if total is not None else (len(items) if hasattr(items, "__len__") else 0)
    with progress_bar(total=inferred_total, desc=desc) as bar:
        for item in items:
            yield item
            bar.update(1)
