import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.context.budget import add_user_usage, allocate, fit_context, over_limit, read_user_usage
from src.llm.tokens import estimate_cost, estimate_tokens
from src.llm.types import ChatMessage


def _text(messages) -> str:
    return "\n".join(message.content for message in messages)


def _history(count: int) -> list[ChatMessage]:
    messages = [
        ChatMessage(role="system", content="You are the delivery assistant."),
        ChatMessage(role="user", content="Please check order A-246001."),
    ]
    messages.extend(ChatMessage(role="user", content="م" * 100) for _ in range(count))
    return messages


def test_persian_estimate_is_higher_than_ascii_and_cost_uses_the_price_list():
    assert estimate_tokens("م" * 100) > estimate_tokens("m" * 100)
    assert estimate_cost(1_000_000, 500_000, 2.0, 4.0) == 4.0
    assert estimate_cost(100, 100, None, None) == 0.0


def test_small_window_keeps_the_order_and_drops_old_turns(config):
    messages = _history(29)
    small, _tools = fit_context(messages, tools=[], window=4096, limits=config.limits)
    large, _tools = fit_context(messages, tools=[], window=128000, limits=config.limits)
    assert "A-246001" in _text(small)
    assert "A-246001" in _text(large)
    assert len(small) < len(large)
    assert "summarized" in _text(small)
    assert sum(estimate_tokens(message.content) for message in small) <= allocate(4096, config.limits)["input"]


def test_tool_results_are_shortened_and_duplicates_dropped(config):
    passages = [
        {"text": "FIRST " + ("آ" * 2000), "source_url": "fixture://a"},
        {"text": "SECOND " + ("ب" * 2000), "source_url": "fixture://b"},
        {"text": "THIRD " + ("پ" * 2000), "source_url": "fixture://c"},
    ]
    blob = json.dumps({"ok": True, "results": passages}, ensure_ascii=False)
    messages = [
        ChatMessage(role="user", content="help"),
        ChatMessage(role="tool", content=blob, name="search_knowledge_base", tool_call_id="1"),
        ChatMessage(role="tool", content=blob, name="search_knowledge_base", tool_call_id="1"),
        ChatMessage(role="tool", content="x" * 20000, name="get_order_status", tool_call_id="2"),
    ]
    fitted, _tools = fit_context(messages, tools=[], window=4096, limits=config.limits)
    tool_messages = [message for message in fitted if message.role == "tool"]
    assert len(tool_messages) == 2
    assert "FIRST" in tool_messages[0].content
    assert "THIRD" not in tool_messages[0].content
    assert estimate_tokens(tool_messages[1].content) <= config.limits.tool_output.max_tokens_per_result


def test_user_window_forgets_usage_older_than_a_day(tmp_path):
    path = tmp_path / "u-anon.json"
    now = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)
    add_user_usage(path, 50, 0.01, now=now - timedelta(hours=25))
    add_user_usage(path, 7, 0.02, now=now)
    tokens, cost = read_user_usage(path, now=now)
    assert tokens == 7
    assert cost == 0.02


def test_session_limit_stops_the_next_turn(config):
    limits = config.limits.model_copy(
        update={
            "token_limits": config.limits.token_limits.model_copy(
                update={"per_session": {"max_total_tokens": 1}}
            )
        }
    )
    assert over_limit(session_tokens=1, session_cost=0, user_tokens=0, user_cost=0, limits=limits)
    assert over_limit(session_tokens=0, session_cost=1, user_tokens=0, user_cost=0, limits=config.limits)
    client = TestClient(create_app(replace(config, limits=limits)))
    first = client.post("/v1/chat", json={"message": "hello", "user_id": "u-anon"})
    assert first.status_code == 200
    second = client.post(
        "/v1/chat",
        json={"message": "hello again", "user_id": "u-anon", "session_id": first.json()["session_id"]},
    )
    assert "limit" in second.json()["reply"]


def test_long_conversation_still_remembers_the_first_order(config):
    client = TestClient(create_app(config))
    first = client.post("/v1/chat", json={"message": "وضعیت سفارش A-246001 چیه؟", "user_id": "u-cust-1001"})
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    for index in range(40):
        filler = client.post(
            "/v1/chat",
            json={"message": f"یادداشت {index} " + ("م" * 80), "session_id": session_id, "user_id": "u-cust-1001"},
        )
        assert filler.status_code == 200
        assert "سقف" not in filler.json()["reply"]
    recall = client.post(
        "/v1/chat",
        json={
            "message": "what is the status of the order I asked about first?",
            "session_id": session_id,
            "user_id": "u-cust-1001",
        },
    )
    assert recall.status_code == 200
    assert "A-246001" in recall.json()["reply"]
    assert "get_order_status" in recall.json()["tools"]
    stored = json.loads((config.state_dir / "sessions" / f"{session_id}.json").read_text(encoding="utf-8"))
    assert sum(1 for message in stored["messages"] if message["role"] == "user") == 42
    assert stored["tokens_total"] < config.limits.token_limits.per_session["max_total_tokens"]
