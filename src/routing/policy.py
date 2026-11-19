"""Pick a provider for one model step.

The knobs are in config/models.yaml. Sensitive turns and cheap turns stay on an
unpriced full-data model. A step that does not fit that model, or a model that
cannot call tools, moves to the next provider in the fallback list.
"""

from dataclasses import dataclass

from src.llm.tokens import estimate_tokens
from src.llm.types import ChatMessage
from src.security.redact import contains_phone
from src.settings import ProviderConfig, RoutingConfig


@dataclass(frozen=True)
class RouteChoice:
    provider: str
    reason: str
    masked: bool


def rank_providers(
    *,
    providers: dict[str, ProviderConfig],
    routing: RoutingConfig,
    messages: list[ChatMessage],
    tools: list[dict] | None,
    open_circuits: set[str],
    remaining_usd: float | None,
    sensitive: bool,
) -> list[RouteChoice]:
    phone = sensitive or _has_phone(messages)
    tokens = _tokens(messages, include_system=True)
    plain = _tokens(messages, include_system=False)
    needs_tools = bool(tools)
    winner, _reason = _prefer(
        _pool(routing, providers, open_circuits=set(), needs_tools=needs_tools),
        providers,
        routing,
        tokens=tokens,
        non_system_tokens=plain,
        needs_tools=needs_tools,
        sensitive=phone,
        remaining_usd=remaining_usd,
    )
    pool = _pool(routing, providers, open_circuits=open_circuits, needs_tools=needs_tools)
    if not pool:
        return []
    chosen, reason = _prefer(
        pool,
        providers,
        routing,
        tokens=tokens,
        non_system_tokens=plain,
        needs_tools=needs_tools,
        sensitive=phone,
        remaining_usd=remaining_usd,
    )
    if chosen is None:
        return []
    if winner is not None and winner != chosen and winner in open_circuits:
        reason = "circuit"
    ordered = [chosen, *[name for name in pool if name != chosen]]
    return [
        RouteChoice(
            provider=name,
            reason=reason if index == 0 else "fallback",
            masked=phone and providers[name].data_policy != "full",
        )
        for index, name in enumerate(ordered)
    ]


def _pool(
    routing: RoutingConfig,
    providers: dict[str, ProviderConfig],
    *,
    open_circuits: set[str],
    needs_tools: bool,
) -> list[str]:
    names = [name for name in routing.fallback_order if name not in open_circuits and name in providers]
    if needs_tools:
        names = [name for name in names if providers[name].supports_tool_calling]
    return names


def _prefer(
    pool: list[str],
    providers: dict[str, ProviderConfig],
    routing: RoutingConfig,
    *,
    tokens: int,
    non_system_tokens: int,
    needs_tools: bool,
    sensitive: bool,
    remaining_usd: float | None,
) -> tuple[str | None, str]:
    if not pool:
        return None, ""
    fitting = [name for name in pool if _fits(providers[name], tokens)] or sorted(
        pool, key=lambda name: providers[name].context_window, reverse=True
    )
    cheap = (not needs_tools) and non_system_tokens <= routing.cheap_input_tokens
    low_budget = remaining_usd is not None and remaining_usd <= routing.hosted_budget_usd

    if sensitive:
        full = [name for name in fitting if providers[name].data_policy == "full"]
        if full:
            return full[0], "sensitive"
        return fitting[0], "sensitive_masked"
    if cheap:
        free = [name for name in fitting if not _priced(providers[name])]
        if free:
            return free[0], "cheap"
    if low_budget:
        free = [name for name in fitting if not _priced(providers[name])]
        if free:
            return free[0], "budget"
    if routing.default_provider in fitting and _fits(providers[routing.default_provider], tokens):
        if needs_tools and not providers[routing.default_provider].supports_tool_calling:
            return fitting[0], "tools"
        return routing.default_provider, "default"
    if needs_tools and routing.default_provider not in pool:
        return fitting[0], "tools"
    return max(fitting, key=lambda name: providers[name].context_window), "context"


def _fits(provider: ProviderConfig, tokens: int) -> bool:
    budget = provider.context_window - provider.max_output_tokens
    return tokens <= max(1, budget)


def _priced(provider: ProviderConfig) -> bool:
    pricing = provider.pricing
    if pricing is None:
        return False
    return (pricing.input_per_1m_tokens or 0) > 0 or (pricing.output_per_1m_tokens or 0) > 0


def _tokens(messages: list[ChatMessage], *, include_system: bool) -> int:
    total = 0
    for message in messages:
        if not include_system and message.role == "system":
            continue
        total += estimate_tokens(message.content)
    return total


def _has_phone(messages: list[ChatMessage]) -> bool:
    return any(contains_phone(message.content) for message in messages)
