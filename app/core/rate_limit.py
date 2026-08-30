"""Small in-process sliding-window limiter for expensive generation routes."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> int | None:
        """Record an allowed event or return seconds until the next slot."""
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return max(1, math.ceil(self.window_seconds - (current - events[0])))
            events.append(current)
            return None

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
