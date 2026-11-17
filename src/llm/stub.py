"""Deterministic offline model used by a fresh clone and by the test suite."""

import re
from collections.abc import Iterator

from src.llm.tokens import estimate_tokens
from src.llm.types import Capabilities, ChatMessage, Completion
from src.settings import ProviderConfig

_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_PREVIEW_LIMIT = 160


class StubClient:
    def __init__(self, provider: ProviderConfig | None = None, *, model_name: str = "stub") -> None:
        self.provider_name = "stub"
        self.model = model_name if provider is None else (provider.model or model_name)
        if provider is None:
            self.capabilities = Capabilities(4096, 512, True, True)
        else:
            self.capabilities = Capabilities(
                context_window=provider.context_window,
                max_output_tokens=provider.max_output_tokens,
                supports_tool_calling=provider.supports_tool_calling,
                supports_streaming=True,
            )

    def complete(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> Completion:
        content = self._reply(messages)
        return Completion(content=content, model=self.model, completion_tokens=self.count_tokens(content))

    def stream(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> Iterator[str]:
        yield self._reply(messages)

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def _reply(self, messages: list[ChatMessage]) -> str:
        latest = ""
        for message in reversed(messages):
            if message.role == "user":
                latest = message.content
                break
        preview = " ".join(latest.split())
        if len(preview) > _PREVIEW_LIMIT:
            preview = preview[: _PREVIEW_LIMIT - 3] + "..."
        if _PERSIAN.search(latest):
            return f"پاسخ stub: پیام شما دریافت شد. «{preview}»"
        return f'Stub reply: your message was received. "{preview}"'
