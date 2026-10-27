"""Guards for a *public, read-only* deployment (``ND_DEMO_MODE=readonly``).

A public endpoint backed by a free-tier LLM key must not be able to (a) burn the owner's quota, (b) store visitors' text, or
(c) be used as a free general-purpose model. This module supplies the in-memory pieces; it deliberately has no persistence, so a
restart resets the budgets (the free-tier provider's own limits and the process request cap in ``ND_LLM_MAX_REQUESTS`` remain
the hard stop).
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    code: str = ""
    message: str = ""
    retry_after_s: int = 0


MAX_TRACKED_CLIENTS = 10_000


def _wait(seconds: float) -> int:
    return max(1, math.ceil(seconds))


class DemoLimiter:
    """Per-client sliding windows (minute and day) plus one global daily budget of answers."""

    def __init__(
        self,
        per_minute: int,
        per_day: int,
        global_per_day: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.per_minute, self.per_day, self.global_per_day = per_minute, per_day, global_per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()  # least recently seen first
        self._global: deque[float] = deque()

    def check(self, client: str) -> Verdict:
        now = self._clock()
        with self._lock:
            day_ago, minute_ago = now - 86400, now - 60
            while self._global and self._global[0] < day_ago:
                self._global.popleft()
            if len(self._global) >= self.global_per_day:
                wait = _wait(self._global[0] + 86400 - now)
                return Verdict(
                    False,
                    "demo_budget_exhausted",
                    "This public demo has used today's answer budget. Recorded examples still work; try again tomorrow.",
                    wait,
                )
            hits = self._hits.setdefault(client, deque())
            self._hits.move_to_end(client)
            while hits and hits[0] < day_ago:
                hits.popleft()
            if len(hits) >= self.per_day:
                return Verdict(
                    False,
                    "demo_rate_limited",
                    "Daily limit for this demo reached.",
                    _wait(hits[0] + 86400 - now),
                )
            recent = [t for t in hits if t >= minute_ago]
            if len(recent) >= self.per_minute:
                return Verdict(
                    False,
                    "demo_rate_limited",
                    "Too many questions in a minute; please wait.",
                    _wait(recent[0] + 60 - now),
                )
            hits.append(now)
            self._global.append(now)
            # bound memory under a flood of distinct clients: evict the least recently seen (the global daily budget still
            # caps total answers, so evicting a client's counters cannot be used to exceed it)
            while len(self._hits) > MAX_TRACKED_CLIENTS:
                self._hits.popitem(last=False)
            return Verdict(True)
