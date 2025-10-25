from __future__ import annotations

import threading
import time
from typing import Optional


class RateLimiter:
    """Simple per-process rate limiter with blocking acquire semantics."""

    def __init__(self, requests_per_minute: Optional[int] = None) -> None:
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._last_acquire = 0.0

    def acquire(self) -> float:
        """Block until a slot is available. Returns wait time in seconds."""
        if self._min_interval <= 0:
            return 0.0
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last_acquire
            delay = self._min_interval - elapsed
            if delay > 0:
                time.sleep(delay)
            self._last_acquire = time.perf_counter()
            return max(delay, 0.0)
