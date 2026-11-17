import json
import uuid

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.llm.stub import StubClient
from src.llm.stub_policy import OUT_OF_SCOPE_EN, OUT_OF_SCOPE_FA
from src.llm.types import ChatMessage


def test_stub_replies_from_the_latest_user_message():
    reply = StubClient().complete(
        [
            ChatMessage(role="system", content="پاسخ فارسی بده"),
            ChatMessage(role="user", content="hello"),
        ]
    )
    assert reply.content == OUT_OF_SCOPE_EN


def test_stub_replies_in_persian_and_is_deterministic():
    messages = [ChatMessage(role="user", content="سلام")]
    first = StubClient().complete(messages)
    second = StubClient().complete(messages)
    assert first.content == second.content
    assert first.content == OUT_OF_SCOPE_FA


def test_health(config):
    response = TestClient(create_app(config)).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "provider": "stub", "model": "stub"}
    assert "replace-me" not in response.text
    assert "not-needed" not in response.text


def test_chat_keeps_a_session_and_answers_in_the_user_language(config):
    client = TestClient(create_app(config))
    first = client.post("/v1/chat", json={"message": "hello", "user_id": "u-anon"})
    assert first.status_code == 200
    body = first.json()
    assert body["reply"] == OUT_OF_SCOPE_EN
    assert body["provider"] == "stub"
    assert body["role"] == "anonymous"

    second = client.post(
        "/v1/chat",
        json={"message": "سلام", "session_id": body["session_id"], "user_id": "u-anon"},
    )
    assert second.status_code == 200
    assert second.json()["reply"] == OUT_OF_SCOPE_FA
    assert "charset=utf-8" in second.headers["content-type"].lower()

    stored = json.loads((config.state_dir / "sessions" / f"{body['session_id']}.json").read_text(encoding="utf-8"))
    assert [message["role"] for message in stored["messages"]] == ["user", "assistant", "user", "assistant"]
    assert all("system" != message["role"] for message in stored["messages"])


def test_unknown_user_is_rejected(config):
    response = TestClient(create_app(config)).post("/v1/chat", json={"message": "hello", "user_id": "u-missing"})
    assert response.status_code == 400
    assert "unknown user_id" in response.json()["detail"]


def test_missing_session_is_not_found(config):
    response = TestClient(create_app(config)).post(
        "/v1/chat",
        json={"message": "hello", "session_id": str(uuid.uuid4()), "user_id": "u-anon"},
    )
    assert response.status_code == 404


def test_session_cannot_be_continued_by_another_user(config):
    client = TestClient(create_app(config))
    created = client.post("/v1/chat", json={"message": "hello", "user_id": "u-anon"})
    response = client.post(
        "/v1/chat",
        json={"message": "hello", "session_id": created.json()["session_id"], "user_id": "u-cust-1001"},
    )
    assert response.status_code == 403


def test_blank_message_is_rejected(config):
    response = TestClient(create_app(config)).post("/v1/chat", json={"message": "   "})
    assert response.status_code == 422
