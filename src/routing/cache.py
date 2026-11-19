"""Remember a finished answer so the same hosted step is not paid for twice."""

import hashlib
import json
import time
from collections.abc import Callable

from src.llm.types import ChatMessage, Completion


class CompletionCache:
    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock or time.monotonic
        self._entries: dict[str, tuple[float, Completion]] = {}
        self._order: list[str] = []

    def get(self, key: str) -> Completion | None:
        item = self._entries.get(key)
        if item is None:
            return None
        expires, completion = item
        if self._clock() >= expires:
            self._drop(key)
            return None
        return completion

    def put(self, key: str, completion: Completion) -> None:
        if self._ttl <= 0:
            return
        if key in self._entries:
            self._order.remove(key)
        self._entries[key] = (self._clock() + self._ttl, completion)
        self._order.append(key)
        while len(self._order) > self._max:
            self._drop(self._order[0])

    def _drop(self, key: str) -> None:
        self._entries.pop(key, None)
        if key in self._order:
            self._order.remove(key)


def cache_key(provider: str, messages: list[ChatMessage], tools: list[dict] | None) -> str:
    payload = {
        "provider": provider,
        "messages": [
            {"role": message.role, "content": message.content, "name": message.name} for message in messages
        ],
        "tools": [tool.get("name") for tool in tools or []],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
