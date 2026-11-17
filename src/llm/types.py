"""Messages, capabilities, and completions shared by every provider."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class Capabilities:
    context_window: int
    max_output_tokens: int
    supports_tool_calling: bool
    supports_streaming: bool


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
