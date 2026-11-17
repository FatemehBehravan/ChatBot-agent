"""Reason, call a tool, observe, and answer.

The step ceiling and the repeated-call stop come from config/limits.yaml.
A bad tool call becomes a structured observation the model can answer from.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.llm.base import LLMClient
from src.llm.types import ChatMessage, ToolCall

_PERSIAN = re.compile(r"[\u0600-\u06FF]")
Execute = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class AgentResult:
    reply: str
    new_messages: list[ChatMessage]
    tool_names: list[str]


def run_agent(
    client: LLMClient,
    messages: list[ChatMessage],
    *,
    tools: list[dict],
    execute: Execute,
    max_steps: int,
    max_identical: int,
) -> AgentResult:
    transcript = list(messages)
    names: list[str] = []
    seen: dict[str, int] = {}
    steps = 0
    while True:
        completion = client.complete(transcript, tools=tools)
        if not completion.tool_calls:
            return _finish(transcript, len(messages), completion.content or _empty(transcript), names)
        if steps >= max_steps:
            return _finish(transcript, len(messages), _stopped(transcript, "steps"), names)
        steps += 1
        transcript.append(
            ChatMessage(role="assistant", content=completion.content, tool_calls=completion.tool_calls)
        )
        repeated = False
        for call in completion.tool_calls:
            names.append(call.name)
            signature = f"{call.name}:{call.arguments}"
            seen[signature] = seen.get(signature, 0) + 1
            if seen[signature] > max_identical:
                repeated = True
                observation = {
                    "ok": False,
                    "error": "repeated_tool_call",
                    "message": "This same tool call already ran. Answer the user without calling it again.",
                }
            else:
                observation = _run_call(call, execute)
            transcript.append(
                ChatMessage(
                    role="tool",
                    content=json.dumps(observation, ensure_ascii=False),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
        if repeated:
            final = client.complete(transcript, tools=[])
            if final.tool_calls or not final.content:
                return _finish(transcript, len(messages), _stopped(transcript, "repeated"), names)
            return _finish(transcript, len(messages), final.content, names)


def message_to_dict(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments} for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.name:
        payload["name"] = message.name
    return payload


def message_from_dict(data: dict[str, Any]) -> ChatMessage:
    calls = tuple(
        ToolCall(id=item["id"], name=item["name"], arguments=item["arguments"])
        for item in data.get("tool_calls") or []
    )
    return ChatMessage(
        role=data["role"],
        content=data.get("content") or "",
        tool_calls=calls,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


def _run_call(call: ToolCall, execute: Execute) -> dict[str, Any]:
    if not call.name.strip():
        return {"ok": False, "error": "malformed_tool_call", "message": "The tool call had no name."}
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": "malformed_tool_call", "message": "Tool arguments were not valid JSON."}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "malformed_tool_call", "message": "Tool arguments must be a JSON object."}
    try:
        result = execute(call.name, arguments)
    except Exception:
        return {"ok": False, "error": "tool_failed", "message": "The tool failed before it could return a result."}
    if not isinstance(result, dict):
        return {"ok": False, "error": "tool_failed", "message": "The tool returned an unexpected result."}
    return result


def _finish(transcript: list[ChatMessage], start: int, reply: str, names: list[str]) -> AgentResult:
    transcript.append(ChatMessage(role="assistant", content=reply))
    return AgentResult(reply=reply, new_messages=transcript[start:], tool_names=names)


def _latest_user(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _persian(messages: list[ChatMessage]) -> bool:
    return bool(_PERSIAN.search(_latest_user(messages)))


def _empty(messages: list[ChatMessage]) -> str:
    if _persian(messages):
        return "پاسخی ساخته نشد."
    return "I could not produce an answer."


def _stopped(messages: list[ChatMessage], reason: str) -> str:
    if reason == "steps":
        if _persian(messages):
            return "این درخواست را متوقف کردم چون به سقف تعداد گام رسید."
        return "I stopped because the request reached the step limit."
    if _persian(messages):
        return "این درخواست را متوقف کردم چون یک ابزار مدام تکرار می‌شد."
    return "I stopped because the same tool call was repeating."
