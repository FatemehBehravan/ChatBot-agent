"""Deterministic offline model used by a fresh clone and by the test suite."""

import re

_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_PREVIEW_LIMIT = 160


class StubModel:
    def __init__(self, model_name: str = "stub") -> None:
        self.model_name = model_name

    def complete(self, messages: list[dict[str, str]]) -> str:
        latest = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                latest = message.get("content", "")
                break
        preview = " ".join(latest.split())
        if len(preview) > _PREVIEW_LIMIT:
            preview = preview[: _PREVIEW_LIMIT - 3] + "..."
        if _PERSIAN.search(latest):
            return f"پاسخ stub: پیام شما دریافت شد. «{preview}»"
        return f'Stub reply: your message was received. "{preview}"'
