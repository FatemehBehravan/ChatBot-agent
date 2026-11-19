"""Recover a tool call that a model wrote into the message text.

Qwen sometimes prints the call as JSON instead of the API tool_calls field.
The OpenAI-compatible client reads it here, before the agent loop sees the turn.
"""

import json

from src.llm.types import ToolCall


def tool_call_from_text(content: str, tools: list[dict]) -> ToolCall | None:
    allowed = {str(tool.get("name")) for tool in tools}
    decoder = json.JSONDecoder()
    start = 0
    while start < len(content):
        index = content.find("{", start)
        if index < 0:
            return None
        try:
            payload, _end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        call = _tool_from_payload(payload, allowed)
        if call is not None:
            return call
        start = index + 1
    return None


def _tool_from_payload(payload: object, allowed: set[str]) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    function = payload.get("function") if isinstance(payload.get("function"), dict) else payload
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if name not in allowed:
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return ToolCall(id="call-embedded", name=str(name), arguments=json.dumps(arguments, ensure_ascii=False))
