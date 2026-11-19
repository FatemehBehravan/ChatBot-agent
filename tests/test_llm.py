import json
import random

import httpx

from src.llm.errors import AuthError, ContextOverflowError, LLMTimeoutError, MalformedResponseError
from src.llm.factory import build_client
from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.stub import StubClient
from src.llm.stub_policy import greeting
from src.llm.types import ChatMessage
from src.settings import ProviderConfig


def _provider(**overrides) -> ProviderConfig:
    values = {
        "type": "openai_compatible",
        "base_url": "http://models.test/v1",
        "api_key": "test-key",
        "model": "unit-model",
        "context_window": 8192,
        "max_output_tokens": 128,
        "supports_tool_calling": True,
        "timeout_seconds": 5,
        "data_policy": "full",
    }
    values.update(overrides)
    return ProviderConfig.model_validate(values)


def _client(handler, *, sleeper=None, attempts=3) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        _provider(),
        provider_name="local",
        transport=httpx.MockTransport(handler),
        sleeper=sleeper or (lambda _seconds: None),
        rng=random.Random(1),
        attempts=attempts,
    )


def _completion(content="hello"):
    return httpx.Response(
        200,
        json={
            "model": "unit-model",
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    )


def test_stub_declares_capabilities_and_counts_tokens():
    client = StubClient()
    assert client.capabilities.context_window == 4096
    assert client.capabilities.max_output_tokens == 512
    assert client.capabilities.supports_tool_calling is True
    assert client.capabilities.supports_streaming is True
    assert client.count_tokens("hello") >= 1
    assert client.count_tokens("سلام") >= client.count_tokens("hi")
    assert list(client.stream([ChatMessage(role="user", content="hello")])) == [greeting(set(), False)]


def test_openai_client_completes_and_streams():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "unit-model"
        assert body["max_tokens"] == 128
        assert request.headers["authorization"] == "Bearer test-key"
        if body["stream"]:
            text = (
                'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=text)
        return _completion("Hello")

    client = _client(handler)
    completion = client.complete([ChatMessage(role="user", content="hi")])
    assert completion.content == "Hello"
    assert completion.prompt_tokens == 4
    assert list(client.stream([ChatMessage(role="user", content="hi")])) == ["Hel", "lo"]


def test_rate_limit_is_retried_with_jitter():
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return _completion()

    client = _client(handler, sleeper=sleeps.append)
    assert client.complete([ChatMessage(role="user", content="hi")]).content == "hello"
    assert calls["count"] == 2
    assert len(sleeps) == 1
    assert 0.2 < sleeps[0] <= 0.4


def test_timeout_is_normalized_and_stops_after_the_last_attempt():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    client = _client(handler, attempts=2)
    try:
        client.complete([ChatMessage(role="user", content="hi")])
    except LLMTimeoutError as exc:
        assert exc.retryable is True
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected LLMTimeoutError")


def test_auth_and_context_overflow_are_not_retried():
    sleeps: list[float] = []

    def auth(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    try:
        _client(auth, sleeper=sleeps.append).complete([ChatMessage(role="user", content="hi")])
    except AuthError as exc:
        assert exc.retryable is False
    else:
        raise AssertionError("expected AuthError")
    assert sleeps == []

    def overflow(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "maximum context length exceeded"}})

    try:
        _client(overflow, sleeper=sleeps.append).complete([ChatMessage(role="user", content="hi")])
    except ContextOverflowError:
        pass
    else:
        raise AssertionError("expected ContextOverflowError")
    assert sleeps == []


def test_malformed_response_is_normalized():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    try:
        _client(handler, attempts=1).complete([ChatMessage(role="user", content="hi")])
    except MalformedResponseError as exc:
        assert exc.retryable is False
    else:
        raise AssertionError("expected MalformedResponseError")


def test_openai_client_parses_a_tool_call():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "get_order_status"
        assert body["messages"][1]["role"] == "tool"
        return httpx.Response(
            200,
            json={
                "model": "unit-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_order_status",
                                        "arguments": '{"order_id":"A-246001"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    client = _client(handler)
    completion = client.complete(
        [
            ChatMessage(role="user", content="status"),
            ChatMessage(role="tool", content="{}", tool_call_id="call_0", name="get_order_status"),
        ],
        tools=[{"name": "get_order_status", "description": "Look up one order.", "parameters": {"type": "object"}}],
    )
    assert completion.content == ""
    assert completion.tool_calls[0].name == "get_order_status"
    assert completion.tool_calls[0].arguments == '{"order_id":"A-246001"}'


def test_factory_switches_local_and_remote_without_code_changes(config, monkeypatch):
    stub = build_client(config)
    assert isinstance(stub, StubClient)
    assert stub.capabilities.context_window == config.provider.context_window

    monkeypatch.setenv("APP_PROVIDER", "local")
    from src.settings import load_config

    local_config = load_config()
    local = build_client(local_config)
    assert isinstance(local, OpenAICompatibleClient)
    assert local.provider_name == "local"
    assert local.model == "qwen2.5:7b-instruct"
    assert local.capabilities.max_output_tokens == 1024
    assert local.capabilities.supports_tool_calling is True

    monkeypatch.setenv("APP_PROVIDER", "remote")
    remote = build_client(load_config())
    assert isinstance(remote, OpenAICompatibleClient)
    assert remote.provider_name == "remote"
    assert remote.capabilities.context_window == 128000
    assert remote.capabilities.supports_streaming is True
