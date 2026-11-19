"""Build the client selected by APP_PROVIDER. No code change is required to switch."""

import httpx

from src.errors import ConfigError
from src.llm.base import LLMClient
from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.stub import StubClient
from src.routing.client import RoutingClient
from src.settings import AppConfig


def build_clients(
    config: AppConfig, *, transport: httpx.BaseTransport | None = None
) -> dict[str, LLMClient]:
    """One client per configured provider, plus the policy router used by the chat page."""
    clients = {
        name: _build(provider, name, transport) for name, provider in config.models.providers.items()
    }
    names = [name for name in config.models.routing.fallback_order if name in clients]
    routed = {
        name: clients[name]
        for name in names
        if config.models.providers[name].type != "stub"
    }
    if not routed:
        if config.app_provider == "route":
            raise ConfigError("APP_PROVIDER=route needs a local or remote provider in routing.fallback_order.")
        return clients
    clients["route"] = RoutingClient(
        {name: config.models.providers[name] for name in routed},
        routed,
        config.models.routing,
        config.limits,
        stub=clients.get("stub"),
    )
    return clients


def build_client(config: AppConfig, *, transport: httpx.BaseTransport | None = None) -> LLMClient:
    if config.app_provider == "route":
        return build_clients(config, transport=transport)["route"]
    return _build(config.provider, config.app_provider, transport)


def _build(provider, name: str, transport: httpx.BaseTransport | None) -> LLMClient:
    if provider.type == "stub":
        return StubClient(provider)
    if provider.type == "openai_compatible":
        return OpenAICompatibleClient(provider, provider_name=name, transport=transport)
    raise ConfigError(f"provider {name!r} has unknown type {provider.type!r}.")
