"""Small timing helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    """Holds elapsed milliseconds after a ``with timed() as t`` block."""

    def __init__(self) -> None:
        self.ms = 0.0


@contextmanager
def timed() -> Iterator[Timer]:
    t = Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.ms = (time.perf_counter() - start) * 1000.0
