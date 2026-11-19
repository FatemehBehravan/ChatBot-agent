"""Routing policy, both fallback directions, the circuit breaker, and the answer cache."""

import json
import logging

import pytest

from src.agent.chat import handle_chat
from src.agent.session import SessionStore
from src.errors import ConfigError
from src.llm.errors import LLMTimeoutError
from src.llm.factory import build_client
from src.llm.tokens import estimate_tokens
from src.llm.types import Capabilities, ChatMessage, Completion
from src.routing.client import DEGRADED_EN, DEGRADED_FA, RoutingClient
from src.settings import PricingConfig, ProviderConfig, load_config
from tests.support import apply_env, copy_config


class Fake:
    def __init__(self, name: str, errors: tuple[Exception, ...] = ()) -> None:
        self.provider_name = name
        self.model = f"{name}-model"
        self.errors = list(errors)
        self.calls = 0
        self.seen: list[list[ChatMessage]] = []
        self.capabilities = Capabilities(8192, 256, True, True)

    def complete(self, messages, max_tokens=None, tools=None):
        del max_tokens, tools
        self.calls += 1
        self.seen.append(list(messages))
        if self.errors:
            raise self.errors.pop(0)
        return Completion(content=f"ok-{self.provider_name}", model=self.model)

    def stream(self, messages, max_tokens=None, tools=None):
        yield self.complete(messages, max_tokens=max_tokens, tools=tools).content

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)


def _providers(*, local_window: int = 8192, local_tools: bool = True) -> dict[str, ProviderConfig]:
    local_output = 256 if local_window > 1000 else 16
    return {
        "local": ProviderConfig(
            type="stub",
            model="local-model",
            context_window=local_window,
            max_output_tokens=local_output,
            supports_tool_calling=local_tools,
            data_policy="full",
        ),
        "remote": ProviderConfig(
            type="openai_compatible",
            base_url="https://example.test/v1",
            api_key="test-key",
            model="remote-model",
            context_window=16000,
            max_output_tokens=256,
            supports_tool_calling=True,
            timeout_seconds=5,
            data_policy="masked_only",
            pricing=PricingConfig(input_per_1m_tokens=0.15, output_per_1m_tokens=0.60),
        ),
    }


def _client(config, local: Fake, remote: Fake, providers: dict[str, ProviderConfig], stub: Fake | None = None, **routing):
    policy = config.models.routing.model_copy(update=routing) if routing else config.models.routing
    return RoutingClient(
        providers,
        {"local": local, "remote": remote},
        policy,
        config.limits,
        clock=routing.get("clock"),
        stub=stub,
    )


def test_cheap_and_sensitive_turns_stay_off_the_hosted_model(config):
    providers = _providers()
    local, remote = Fake("local"), Fake("remote")
    client = _client(config, local, remote, providers, default_provider="remote")
    cheap = client.complete([ChatMessage(role="user", content="hello")])
    assert cheap.content == "ok-local"
    assert client.last_decision.reason == "cheap"
    assert remote.calls == 0

    private = client.complete([ChatMessage(role="user", content="my number is +98-555-0199")])
    assert private.content == "ok-local"
    assert client.last_decision.reason == "sensitive"
    assert remote.calls == 0

    client.note_turn(remaining_usd=0.0, sensitive=False)
    budget = client.complete(
        [ChatMessage(role="user", content="hello")],
        tools=[{"name": "get_order_status"}],
    )
    assert budget.content == "ok-local"
    assert client.last_decision.reason == "budget"
    assert remote.calls == 0


def test_a_step_that_does_not_fit_locally_uses_the_hosted_model(config):
    providers = _providers(local_window=80)
    local, remote = Fake("local"), Fake("remote")
    client = _client(config, local, remote, providers)
    reply = client.complete([ChatMessage(role="user", content="x" * 400)])
    assert reply.model == "remote-model"
    assert client.last_decision.reason == "context"
    assert local.calls == 0


def test_tool_steps_skip_a_model_that_cannot_call_tools(config):
    providers = _providers(local_tools=False)
    local, remote = Fake("local"), Fake("remote")
    client = _client(config, local, remote, providers)
    reply = client.complete(
        [ChatMessage(role="user", content="where is A-246001")],
        tools=[{"name": "get_order_status"}],
    )
    assert reply.model == "remote-model"
    assert client.last_decision.reason == "tools"
    assert local.calls == 0


def test_fallback_runs_in_both_directions_and_then_degrades(config):
    providers = _providers()
    timeout = LLMTimeoutError("timed out")
    local, remote = Fake("local", (timeout,)), Fake("remote")
    client = _client(config, local, remote, providers)
    hosted = client.complete(
        [ChatMessage(role="user", content="where is A-246001")],
        tools=[{"name": "get_order_status"}],
    )
    assert hosted.model == "remote-model"
    assert client.last_decision.reason == "fallback"
    assert client.last_decision.fallback_from == "local"

    small = _providers(local_window=80)
    remote_down, local_up = Fake("remote", (LLMTimeoutError("timed out"),)), Fake("local")
    other = _client(config, local_up, remote_down, small)
    back = other.complete([ChatMessage(role="user", content="x" * 400)])
    assert back.model == "local-model"
    assert other.last_decision.reason == "fallback"
    assert other.last_decision.fallback_from == "remote"

    local_down, remote_down, stub = Fake("local", (timeout,)), Fake("remote", (LLMTimeoutError("timed out"),)), Fake("stub")
    both = _client(config, local_down, remote_down, providers, stub)
    assert both.complete([ChatMessage(role="user", content="hello")]).content == "ok-stub"
    assert both.last_decision.provider == "stub"
    assert both.last_decision.model == "stub-model"
    assert both.last_decision.reason == "degraded"
    assert both.last_decision.fallback_from == "remote"
    assert stub.calls == 1

    persian_stub = Fake("stub")
    persian = _client(
        config,
        Fake("local", (LLMTimeoutError("timed out"),)),
        Fake("remote", (LLMTimeoutError("timed out"),)),
        providers,
        persian_stub,
    )
    assert persian.complete([ChatMessage(role="user", content="سلام")]).content == "ok-stub"
    assert persian_stub.seen[0][-1].content == "سلام"

    missing = _client(config, Fake("local", (timeout,)), Fake("remote", (LLMTimeoutError("timed out"),)), providers)
    assert missing.complete([ChatMessage(role="user", content="hello")]).content == DEGRADED_EN

    breaker = config.models.routing.circuit_breaker.model_copy(update={"failure_threshold": 1, "cooldown_seconds": 30})
    closed_local = Fake("local", (timeout,))
    closed_remote = Fake("remote", (LLMTimeoutError("timed out"),))
    closed_stub = Fake("stub")
    closed = RoutingClient(
        providers,
        {"local": closed_local, "remote": closed_remote},
        config.models.routing.model_copy(update={"circuit_breaker": breaker}),
        config.limits,
        clock=lambda: 0.0,
        stub=closed_stub,
    )
    assert closed.complete([ChatMessage(role="user", content="hello")]).content == "ok-stub"
    skipped = closed.complete([ChatMessage(role="user", content="hello again")])
    assert skipped.content == "ok-stub"
    assert closed_local.calls == 1
    assert closed_remote.calls == 1
    assert closed_stub.calls == 2


def test_circuit_opens_and_a_later_step_skips_the_failed_model(config):
    now = {"t": 1000.0}
    providers = _providers()
    local = Fake("local", (LLMTimeoutError("timed out"), LLMTimeoutError("timed out")))
    remote = Fake("remote")
    breaker = config.models.routing.circuit_breaker.model_copy(
        update={"failure_threshold": 2, "cooldown_seconds": 10}
    )
    client = RoutingClient(
        providers,
        {"local": local, "remote": remote},
        config.models.routing.model_copy(update={"circuit_breaker": breaker}),
        config.limits,
        clock=lambda: now["t"],
    )
    tools = [{"name": "get_order_status"}]
    message = [ChatMessage(role="user", content="where is A-246001")]
    client.complete(message, tools=tools)
    client.complete(message, tools=tools)
    assert local.calls == 2
    skipped = client.complete(message, tools=tools)
    assert skipped.model == "remote-model"
    assert client.last_decision.reason == "circuit"
    assert local.calls == 2

    now["t"] = 1011.0
    recovered = client.complete(message, tools=tools)
    assert recovered.model == "local-model"
    assert local.calls == 3


def test_a_repeated_hosted_answer_is_served_from_the_cache(config):
    providers = _providers(local_window=80)
    local, remote = Fake("local"), Fake("remote")
    client = _client(config, local, remote, providers)
    first = client.complete([ChatMessage(role="user", content="x" * 400)])
    second = client.complete([ChatMessage(role="user", content="x" * 400)])
    assert first.content == second.content == "ok-remote"
    assert remote.calls == 1
    assert client.last_decision.cached is True


def test_hosted_fallback_receives_masked_phone_numbers(config):
    providers = _providers()
    local = Fake("local", (LLMTimeoutError("timed out"),))
    remote = Fake("remote")
    client = _client(config, local, remote, providers)
    client.complete([ChatMessage(role="user", content="phone +98-555-0199")])
    sent = remote.seen[0][0].content
    assert "+98-555-0199" not in sent
    assert "+00-***-****" in sent
    assert client.last_decision.masked is True


def test_a_turn_records_the_serving_model(config, caplog):
    providers = _providers()
    local, remote = Fake("local"), Fake("remote")
    client = _client(config, local, remote, providers)
    caplog.set_level(logging.INFO, logger="src.routing.client")
    outcome = handle_chat(
        config,
        SessionStore(config.state_dir / "sessions"),
        client,
        message="hello",
        session_id=None,
        user_id="u-cust-1001",
    )
    assert outcome.provider == "local"
    assert outcome.model == "local-model"
    assert outcome.route == "sensitive"
    assert remote.calls == 0
    stored = json.loads((config.state_dir / "sessions" / f"{outcome.session_id}.json").read_text(encoding="utf-8"))
    assert stored["messages"][-1]["provider"] == "local"
    assert stored["messages"][-1]["route"] == "sensitive"
    assert "route provider=local model=local-model reason=sensitive" in caplog.text


def test_route_mode_loads_from_configuration(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    monkeypatch.setenv("APP_PROVIDER", "route")
    loaded = load_config()
    client = build_client(loaded)
    assert isinstance(client, RoutingClient)
    assert client.provider_name == "route"
    assert client.capabilities.context_window == 128000
    assert loaded.models.routing.circuit_breaker.failure_threshold == 3
    assert loaded.models.routing.cheap_input_tokens == 500


def test_invalid_circuit_threshold_fails_fast(tmp_path, monkeypatch):
    apply_env(monkeypatch, tmp_path)
    config_dir = tmp_path / "config"
    copy_config(config_dir)
    path = config_dir / "models.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("failure_threshold: 3", "failure_threshold: 0", 1),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    with pytest.raises(ConfigError, match="failure_threshold"):
        load_config()
