"""Retry timeout and rate-limit failures with exponential backoff and jitter."""

import random
import time
from collections.abc import Callable
from typing import TypeVar

from src.llm.errors import LLMError

T = TypeVar("T")


def call_with_retry(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.2,
    sleeper: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    random_source = rng or random.Random()
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except LLMError as exc:
            if not exc.retryable or attempt == attempts:
                raise
            sleeper(delay + random_source.uniform(0, delay))
            delay = min(delay * 2, 8.0)
    raise AssertionError("retry loop exited without a result")
