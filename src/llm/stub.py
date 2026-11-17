"""Deterministic offline model used by a fresh clone and by the test suite."""

from collections.abc import Iterator

from src.llm.stub_policy import plan
from src.llm.tokens import estimate_tokens
from src.llm.types import Capabilities, ChatMessage, Completion
from src.settings import ProviderConfig


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

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Completion:
        del max_tokens
        content, calls = plan(messages, tools)
        return Completion(
            content=content,
            model=self.model,
            completion_tokens=self.count_tokens(content),
            tool_calls=tuple(calls),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        completion = self.complete(messages, max_tokens=max_tokens, tools=tools)
        if completion.content:
            yield completion.content

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)
