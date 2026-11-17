"""Client operations every provider implements."""

from collections.abc import Iterator
from typing import Protocol

from src.llm.types import Capabilities, ChatMessage, Completion


class LLMClient(Protocol):
    provider_name: str
    model: str
    capabilities: Capabilities

    def complete(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> Completion:
        """Return one assistant message."""

    def stream(self, messages: list[ChatMessage], *, max_tokens: int | None = None) -> Iterator[str]:
        """Yield assistant text as it arrives. Optional for a provider to implement as one chunk."""

    def count_tokens(self, text: str) -> int:
        """Estimate how many tokens `text` will occupy."""
