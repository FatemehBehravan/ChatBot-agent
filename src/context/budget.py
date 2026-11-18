"""Fit one model call into that model's context window.

The full session stays on disk. Only the prompt is reduced, so a later call
on a larger window still sees the original turns. Order ids found anywhere in
the history are pinned into the prompt. Old turns are replaced by one summary
once history passes the configured trigger. Tool results are shortened, and
retrieved passages are dropped from the end of the ranked list.
"""

import json
import re
from datetime import datetime, timedelta, timezone

from src.llm.tokens import estimate_tokens
from src.llm.types import ChatMessage
from src.settings import LimitsConfig

_ORDER = re.compile(r"A-\d+", re.IGNORECASE)
_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_USER_WINDOW = timedelta(hours=24)


def allocate(window: int, limits: LimitsConfig) -> dict[str, int]:
    """Split one model window by the fractions in config/limits.yaml."""
    fractions = limits.context_budget
    output = max(1, int(window * fractions.output_reserve))
    usable = max(1, min(limits.token_limits.per_request.max_input_tokens, window - output))
    shares = {
        "system_prompt": fractions.system_prompt,
        "tool_schemas": fractions.tool_schemas,
        "history": fractions.history,
        "tool_results": fractions.tool_results,
        "retrieved_docs": fractions.retrieved_docs,
    }
    total = sum(shares.values())
    budgets = {name: max(1, int(usable * share / total)) for name, share in shares.items()}
    budgets["output_reserve"] = output
    budgets["input"] = usable
    return budgets


def fit_context(
    messages: list[ChatMessage],
    *,
    tools: list[dict],
    window: int,
    limits: LimitsConfig,
) -> tuple[list[ChatMessage], list[dict]]:
    budgets = allocate(window, limits)
    system = next((message for message in messages if message.role == "system"), None)
    rest = [message for message in messages if message.role != "system"]
    rest = _dedupe(rest)
    rest = [_trim_tool(message, budgets, limits.tool_output.max_tokens_per_result) for message in rest]
    turns = _turns(rest)
    history_tokens = sum(_turn_tokens(turn) for turn in turns)
    trigger = budgets["history"] * limits.summarization.trigger_ratio
    summary = None
    if len(turns) > limits.summarization.keep_last_turns and history_tokens > trigger:
        summary = ChatMessage(
            role="assistant",
            content="Older turns were summarized to fit the context window. " + _pin_text(_ids(turns[:-limits.summarization.keep_last_turns])),
        )
        turns = turns[-limits.summarization.keep_last_turns :]
    pin = _pin_message(_ids(rest))
    while len(turns) > 1 and _history_size(pin, summary, turns) > budgets["history"]:
        turns = turns[1:]
    fitted: list[ChatMessage] = []
    if system is not None:
        fitted.append(_with_content(system, _clip(system.content, budgets["system_prompt"])))
    if pin is not None:
        fitted.append(pin)
    if summary is not None and summary.content.strip():
        fitted.append(summary)
    for turn in turns:
        fitted.extend(turn)
    fitted = _clip_last_user(fitted, budgets["history"])
    tool_list = _trim_tools(tools, budgets["tool_schemas"])
    while _size(fitted, tool_list) > budgets["input"] and len(turns) > 1:
        turns = turns[1:]
        fitted = _assemble(system, pin, summary, turns, budgets)
        fitted = _clip_last_user(fitted, budgets["history"])
    if _size(fitted, tool_list) > budgets["input"]:
        tool_list = _names_only(tools)
    return fitted, tool_list


def message_tokens(message: ChatMessage) -> int:
    total = estimate_tokens(message.content)
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(call.arguments)
    return total


def blocked_message(user_text: str) -> str:
    if _PERSIAN.search(user_text):
        return "سقف توکن این گفتگو پر شده است."
    return "This conversation reached its token limit."


def over_limit(
    *,
    session_tokens: int,
    session_cost: float,
    user_tokens: int,
    user_cost: float,
    limits: LimitsConfig,
) -> bool:
    tokens = limits.token_limits
    cost = limits.cost_limits
    return (
        session_tokens >= tokens.per_session["max_total_tokens"]
        or user_tokens >= tokens.per_user["max_total_tokens_per_window"]
        or session_cost >= cost.per_session_usd
        or user_cost >= cost.per_user_usd_per_window
    )


def read_user_usage(path, now: datetime | None = None) -> tuple[int, float]:
    moment = now or datetime.now(timezone.utc)
    tokens, cost, _rows = _fresh_rows(path, moment)
    return tokens, cost


def add_user_usage(path, tokens: int, cost: float, now: datetime | None = None) -> None:
    moment = now or datetime.now(timezone.utc)
    _tokens, _cost, rows = _fresh_rows(path, moment)
    rows.append({"at": moment.isoformat(), "tokens": tokens, "cost": cost})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _fresh_rows(path, now: datetime) -> tuple[int, float, list[dict]]:
    if not path.is_file():
        return 0, 0.0, []
    raw = json.loads(path.read_text(encoding="utf-8"))
    kept = []
    tokens = 0
    cost = 0.0
    for row in raw:
        at = datetime.fromisoformat(row["at"])
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        if now - at > _USER_WINDOW:
            continue
        kept.append(row)
        tokens += int(row["tokens"])
        cost += float(row["cost"])
    return tokens, cost, kept


def _assemble(system, pin, summary, turns, budgets) -> list[ChatMessage]:
    fitted: list[ChatMessage] = []
    if system is not None:
        fitted.append(_with_content(system, _clip(system.content, budgets["system_prompt"])))
    if pin is not None:
        fitted.append(pin)
    if summary is not None and summary.content.strip():
        fitted.append(summary)
    for turn in turns:
        fitted.extend(turn)
    return fitted


def _dedupe(messages: list[ChatMessage]) -> list[ChatMessage]:
    kept: list[ChatMessage] = []
    for message in messages:
        if (
            message.role != "user"
            and kept
            and kept[-1].role == message.role
            and kept[-1].content == message.content
            and kept[-1].tool_calls == message.tool_calls
        ):
            continue
        kept.append(message)
    return kept


def _trim_tool(message: ChatMessage, budgets: dict[str, int], per_result: int) -> ChatMessage:
    if message.role != "tool":
        return message
    cap = max(1, min(per_result, budgets["tool_results"]))
    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError:
        return _with_content(message, _clip(message.content, cap))
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list) and results:
        kept_rows: list = []
        for row in results:
            trial = json.dumps({**payload, "results": kept_rows + [row]}, ensure_ascii=False)
            if estimate_tokens(trial) > min(cap, budgets["retrieved_docs"]) and kept_rows:
                break
            kept_rows.append(row)
        payload = {**payload, "results": kept_rows}
        return _with_content(message, _clip(json.dumps(payload, ensure_ascii=False), cap))
    return _with_content(message, _clip(json.dumps(payload, ensure_ascii=False), cap))


def _trim_tools(tools: list[dict], budget: int) -> list[dict]:
    if _json_tokens(tools) <= budget:
        return tools
    shortened = [{**tool, "description": str(tool.get("description") or "")[:80]} for tool in tools]
    if _json_tokens(shortened) <= budget:
        return shortened
    return _names_only(tools) if _json_tokens(_names_only(tools)) <= budget else shortened


def _names_only(tools: list[dict]) -> list[dict]:
    return [
        {"name": tool["name"], "description": str(tool.get("description") or "")[:80], "parameters": {"type": "object"}}
        for tool in tools
    ]


def _turns(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    turns: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    for message in messages:
        if message.role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _ids(groups) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    messages = groups if groups and isinstance(groups[0], ChatMessage) else [message for group in groups for message in group]
    for message in messages:
        for match in _ORDER.findall(message.content):
            order_id = match.upper()
            if order_id not in seen:
                seen.add(order_id)
                found.append(order_id)
        for call in message.tool_calls:
            for match in _ORDER.findall(call.arguments):
                order_id = match.upper()
                if order_id not in seen:
                    seen.add(order_id)
                    found.append(order_id)
    return found


def _pin_text(order_ids: list[str]) -> str:
    if not order_ids:
        return ""
    return "Pinned orders: " + ", ".join(order_ids[:20]) + "."


def _pin_message(order_ids: list[str]) -> ChatMessage | None:
    text = _pin_text(order_ids)
    if not text:
        return None
    return ChatMessage(role="assistant", content=text)


def _turn_tokens(turn: list[ChatMessage]) -> int:
    return sum(message_tokens(message) for message in turn)


def _history_size(pin, summary, turns) -> int:
    total = sum(_turn_tokens(turn) for turn in turns)
    if pin is not None:
        total += message_tokens(pin)
    if summary is not None:
        total += message_tokens(summary)
    return total


def _clip_last_user(messages: list[ChatMessage], budget: int) -> list[ChatMessage]:
    if sum(message_tokens(message) for message in messages) <= budget * 4:
        return messages
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            messages = list(messages)
            messages[index] = _with_content(messages[index], _clip(messages[index].content, max(1, budget // 2)))
            break
    return messages


def _size(messages: list[ChatMessage], tools: list[dict]) -> int:
    return sum(message_tokens(message) for message in messages) + _json_tokens(tools)


def _json_tokens(value: object) -> int:
    return estimate_tokens(json.dumps(value, ensure_ascii=False))


def _clip(text: str, token_budget: int) -> str:
    if estimate_tokens(text) <= token_budget:
        return text
    low = 0
    high = len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid].rstrip() + "…") <= token_budget:
            low = mid
        else:
            high = mid - 1
    clipped = text[:low].rstrip()
    if not clipped:
        return text[:1]
    marked = clipped + "…"
    if estimate_tokens(marked) <= token_budget:
        return marked
    return clipped


def _with_content(message: ChatMessage, content: str) -> ChatMessage:
    return ChatMessage(
        role=message.role,
        content=content,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
        name=message.name,
    )
