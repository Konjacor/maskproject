from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed() -> Iterator[callable[[], float]]:
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    yield elapsed_ms
