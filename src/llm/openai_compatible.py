"""OpenAI-compatible chat client used for both the local model and the hosted API.

Which one is called is entirely configuration: base URL, key, model name, and timeout.
"""

import json
import random
from collections.abc import Callable, Iterator
from dataclasses import replace

import httpx

from src.llm.errors import (
    AuthError,
    ContextOverflowError,
    LLMTimeoutError,
    MalformedResponseError,
    RateLimitError,
)
from src.llm.retry import call_with_retry
from src.llm.tokens import estimate_tokens
from src.llm.tool_text import tool_call_from_text
from src.llm.types import Capabilities, ChatMessage, Completion, ToolCall
from src.settings import ProviderConfig

_CONTEXT_MARKERS = ("context length", "context_length", "maximum context", "too many tokens", "context window")


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: ProviderConfig,
        *,
        provider_name: str,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
        attempts: int = 3,
    ) -> None:
        if not provider.base_url or not provider.model or provider.timeout_seconds is None:
            raise ValueError("openai_compatible provider is missing base_url, model, or timeout_seconds")
        self.provider_name = provider_name
        self.model = provider.model
        self.capabilities = Capabilities(
            context_window=provider.context_window,
            max_output_tokens=provider.max_output_tokens,
            supports_tool_calling=provider.supports_tool_calling,
            supports_streaming=True,
        )
        self._base_url = provider.base_url.rstrip("/")
        self._api_key = provider.api_key or ""
        self._timeout = httpx.Timeout(provider.timeout_seconds)
        self._transport = transport
        self._sleeper = sleeper
        self._rng = rng
        self._attempts = attempts

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Completion:
        response = self._post(self._payload(messages, max_tokens=max_tokens, stream=False, tools=tools))
        completion = self._parse_completion(response)
        if tools and not completion.tool_calls:
            recovered = tool_call_from_text(completion.content or "", tools)
            if recovered is not None:
                completion = replace(completion, content="", tool_calls=(recovered,))
        return completion

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        response = self._post(self._payload(messages, max_tokens=max_tokens, stream=True, tools=tools))
        yield from self._parse_stream(response.text)

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def _payload(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None,
        stream: bool,
        tools: list[dict] | None,
    ) -> dict:
        limit = self.capabilities.max_output_tokens if max_tokens is None else max_tokens
        limit = min(limit, self.capabilities.max_output_tokens)
        payload: dict = {
            "model": self.model,
            "messages": [_serialize_message(message) for message in messages],
            "max_tokens": limit,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
                for tool in tools
            ]
        return payload

    def _post(self, payload: dict) -> httpx.Response:
        def operation() -> httpx.Response:
            try:
                with httpx.Client(timeout=self._timeout, transport=self._transport) as http:
                    response = http.post(
                        f"{self._base_url}/chat/completions",
                        json=payload,
                        headers=self._headers(),
                    )
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError("provider call timed out") from exc
            except httpx.TransportError as exc:
                raise LLMTimeoutError("provider unreachable") from exc
            self._raise_for_status(response)
            return response

        kwargs: dict = {"attempts": self._attempts, "rng": self._rng}
        if self._sleeper is not None:
            kwargs["sleeper"] = self._sleeper
        return call_with_retry(operation, **kwargs)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code == 429:
            raise RateLimitError("provider rate limit")
        if response.status_code in (401, 403):
            raise AuthError("provider rejected the API key")
        if response.status_code in (400, 413) and _is_context_overflow(response.text):
            raise ContextOverflowError("provider context window exceeded")
        if response.status_code >= 500:
            raise LLMTimeoutError(f"provider returned HTTP {response.status_code}")
        raise MalformedResponseError(f"provider returned HTTP {response.status_code}")

    def _parse_completion(self, response: httpx.Response) -> Completion:
        payload = _json_object(response.text)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise MalformedResponseError("provider response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise MalformedResponseError("provider response has no message content")
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        content = message.get("content")
        if not isinstance(content, str):
            if not tool_calls:
                raise MalformedResponseError("provider response has no message content")
            content = ""
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return Completion(
            content=content,
            model=str(payload.get("model") or self.model),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            tool_calls=tool_calls,
        )

    def _parse_stream(self, body: str) -> Iterator[str]:
        produced = False
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            data = stripped[5:].strip()
            if data == "[DONE]":
                break
            payload = _json_object(data)
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                produced = True
                yield content
        if not produced:
            raise MalformedResponseError("provider stream has no content")


def _serialize_message(message: ChatMessage) -> dict:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id or "", "content": message.content}
    payload: dict = {"role": message.role, "content": message.content}
    if message.tool_calls:
        if not message.content:
            payload["content"] = None
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    return payload


def _parse_tool_calls(raw: object) -> tuple[ToolCall, ...]:
    if not isinstance(raw, list):
        return ()
    calls = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = ""
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call-{index + 1}"),
                name=str(function.get("name") or ""),
                arguments=arguments,
            )
        )
    return tuple(calls)


def _json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError("provider response is not JSON") from exc
    if not isinstance(payload, dict):
        raise MalformedResponseError("provider response is not a JSON object")
    return payload


def _is_context_overflow(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _CONTEXT_MARKERS)


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None
