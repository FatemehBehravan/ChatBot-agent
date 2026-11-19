"""One client that serves each step from local or hosted, then the other if that call fails."""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

from src.context.budget import fit_context
from src.llm.base import LLMClient
from src.llm.errors import AuthError, LLMError
from src.llm.types import Capabilities, ChatMessage, Completion
from src.routing.breaker import CircuitBreaker
from src.routing.cache import CompletionCache, cache_key
from src.routing.policy import rank_providers
from src.security.redact import mask_phones
from src.settings import LimitsConfig, ProviderConfig, RoutingConfig

logger = logging.getLogger(__name__)

DEGRADED_EN = "Both models are unavailable right now. Ask again in a moment, or contact support."
DEGRADED_FA = "هر دو مدل الان در دسترس نیستند. کمی بعد دوباره بپرسید، یا با پشتیبانی تماس بگیرید."
_PERSIAN_START = "\u0600"
_PERSIAN_END = "\u06ff"


@dataclass(frozen=True)
class RouteDecision:
    provider: str
    model: str
    reason: str
    fallback_from: str = ""
    cached: bool = False
    masked: bool = False


class RoutingClient:
    """LLMClient that applies the routing policy on every completion."""

    def __init__(
        self,
        providers: dict[str, ProviderConfig],
        clients: dict[str, LLMClient],
        routing: RoutingConfig,
        limits: LimitsConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.provider_name = "route"
        self._providers = providers
        self._clients = clients
        self._routing = routing
        self._limits = limits
        self._clock = clock or time.monotonic
        default = providers[routing.default_provider]
        self.model = default.model or routing.default_provider
        windows = [providers[name].context_window for name in routing.fallback_order if name in providers]
        outputs = [providers[name].max_output_tokens for name in routing.fallback_order if name in providers]
        self.capabilities = Capabilities(
            context_window=max(windows),
            max_output_tokens=max(outputs),
            supports_tool_calling=any(
                providers[name].supports_tool_calling for name in routing.fallback_order if name in providers
            ),
            supports_streaming=True,
        )
        breaker = routing.circuit_breaker
        self._breaker = CircuitBreaker(breaker.failure_threshold, breaker.cooldown_seconds, self._clock)
        cache = routing.cache
        self._cache = CompletionCache(cache.ttl_seconds, cache.max_entries, self._clock)
        self._remaining_usd: float | None = None
        self._sensitive = False
        self.last_decision: RouteDecision | None = None

    def note_turn(self, *, remaining_usd: float, sensitive: bool) -> None:
        self._remaining_usd = remaining_usd
        self._sensitive = sensitive
        self.last_decision = None

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Completion:
        choices = rank_providers(
            providers=self._providers,
            routing=self._routing,
            messages=messages,
            tools=tools,
            open_circuits=self._breaker.open_names(),
            remaining_usd=self._remaining_usd,
            sensitive=self._sensitive,
        )
        if not choices:
            return self._degraded(messages, fallback_from="")
        fallback_from = ""
        for choice in choices:
            prepared, prepared_tools = self._prepare(choice.provider, messages, tools, masked=choice.masked)
            key = cache_key(choice.provider, prepared, prepared_tools)
            cached = None if prepared_tools else self._cache.get(key)
            try:
                if cached is not None:
                    completion = cached
                    from_cache = True
                else:
                    completion = self._clients[choice.provider].complete(
                        prepared, max_tokens=max_tokens, tools=prepared_tools or None
                    )
                    from_cache = False
                    if not prepared_tools and not completion.tool_calls:
                        self._cache.put(key, completion)
            except LLMError as exc:
                self._breaker.record_failure(choice.provider, force_open=isinstance(exc, AuthError))
                logger.info("route failed provider=%s error=%s", choice.provider, type(exc).__name__)
                fallback_from = choice.provider
                continue
            self._breaker.record_success(choice.provider)
            reason = "fallback" if fallback_from else choice.reason
            self._remember(
                RouteDecision(
                    provider=choice.provider,
                    model=completion.model,
                    reason=reason,
                    fallback_from=fallback_from,
                    cached=from_cache,
                    masked=choice.masked,
                )
            )
            return completion
        return self._degraded(messages, fallback_from=fallback_from)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        completion = self.complete(messages, max_tokens=max_tokens, tools=tools)
        if completion.content:
            yield completion.content

    def count_tokens(self, text: str) -> int:
        client = self._clients.get(self._routing.default_provider)
        if client is None:
            return 0
        return client.count_tokens(text)

    def _prepare(
        self,
        provider_name: str,
        messages: list[ChatMessage],
        tools: list[dict] | None,
        *,
        masked: bool,
    ) -> tuple[list[ChatMessage], list[dict]]:
        prepared = messages
        if masked:
            prepared = [replace(message, content=mask_phones(message.content)) for message in messages]
        provider = self._providers[provider_name]
        return fit_context(
            prepared,
            tools=list(tools or []),
            window=provider.context_window,
            limits=self._limits,
        )

    def _degraded(self, messages: list[ChatMessage], *, fallback_from: str) -> Completion:
        content = DEGRADED_FA if _persian(messages) else DEGRADED_EN
        completion = Completion(content=content, model="unavailable")
        self._remember(
            RouteDecision(
                provider="unavailable",
                model="unavailable",
                reason="degraded",
                fallback_from=fallback_from,
            )
        )
        return completion

    def _remember(self, decision: RouteDecision) -> None:
        self.last_decision = decision
        if decision.provider != "unavailable":
            self.model = decision.model
        logger.info(
            "route provider=%s model=%s reason=%s fallback_from=%s cached=%s masked=%s",
            decision.provider,
            decision.model,
            decision.reason,
            decision.fallback_from or "-",
            decision.cached,
            decision.masked,
        )


def _persian(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if message.role != "user":
            continue
        return any(_PERSIAN_START <= char <= _PERSIAN_END for char in message.content)
    return False
