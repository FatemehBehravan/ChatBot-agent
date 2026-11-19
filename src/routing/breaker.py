"""Skip a provider after repeated failures, then allow one trial when the cooldown ends."""

import time
from collections.abc import Callable


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_seconds: float, clock: Callable[[], float] | None = None) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock or time.monotonic
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def open_names(self) -> set[str]:
        now = self._clock()
        return {name for name, until in self._open_until.items() if now < until}

    def record_success(self, name: str) -> None:
        self._failures[name] = 0
        self._open_until.pop(name, None)

    def record_failure(self, name: str, *, force_open: bool = False) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if force_open or count >= self.threshold:
            self._open_until[name] = self._clock() + self.cooldown_seconds
            self._failures[name] = 0
