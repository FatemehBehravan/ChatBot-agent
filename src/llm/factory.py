"""Build the client selected by APP_PROVIDER. No code change is required to switch."""

import httpx

from src.errors import ConfigError
from src.llm.base import LLMClient
from src.llm.openai_compatible import OpenAICompatibleClient
from src.llm.stub import StubClient
from src.settings import AppConfig


def build_client(config: AppConfig, *, transport: httpx.BaseTransport | None = None) -> LLMClient:
    provider = config.provider
    if provider.type == "stub":
        return StubClient(provider)
    if provider.type == "openai_compatible":
        return OpenAICompatibleClient(provider, provider_name=config.app_provider, transport=transport)
    raise ConfigError(f"APP_PROVIDER={config.app_provider!r} has unknown type {provider.type!r}.")
