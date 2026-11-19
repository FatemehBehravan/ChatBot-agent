"""Multi-turn chat. Each turn runs the tool loop and is stored on the session."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.agent.answers import finish_reply, prepare_turn, read_own_orders, reply_after_tools
from src.agent.loop import AgentResult, message_from_dict, message_to_dict, run_agent
from src.agent.prompt import assemble_prompt
from src.agent.session import SessionStore
from src.context.budget import add_user_usage, blocked_message, over_limit, read_user_usage
from src.llm.base import LLMClient
from src.llm.tokens import estimate_cost, estimate_tokens
from src.llm.types import ChatMessage
from src.security.audit import AUDITED_TOOLS, append_audit
from src.security.identity import UnknownUserError
from src.security.redact import mask_phones
from src.settings import AppConfig
from src.tools import execute_tool, tool_catalog


class SessionNotFoundError(Exception):
    pass


class SessionForbiddenError(Exception):
    pass


@dataclass(frozen=True)
class ChatOutcome:
    session_id: str
    user_id: str
    role: str
    reply: str
    provider: str
    model: str
    tools: tuple[str, ...] = ()
    route: str = ""


def handle_chat(
    config: AppConfig,
    store: SessionStore,
    client: LLMClient,
    *,
    message: str,
    session_id: str | None,
    user_id: str,
    provider_name: str | None = None,
) -> ChatOutcome:
    active = provider_name or config.app_provider
    identity = config.users.get(user_id)
    if identity is None:
        raise UnknownUserError(user_id)
    if session_id is None:
        session = store.create(identity.user_id)
    else:
        session = store.load(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        if session.user_id != identity.user_id:
            raise SessionForbiddenError(session_id)
    session.messages.append({"role": "user", "content": message})
    user_tokens, user_cost = read_user_usage(_usage_path(config, identity.user_id))
    if over_limit(
        session_tokens=session.tokens_total,
        session_cost=session.cost_usd,
        user_tokens=user_tokens,
        user_cost=user_cost,
        limits=config.limits,
    ):
        reply = blocked_message(message)
        session.messages.append({"role": "assistant", "content": reply})
        store.save(session)
        return ChatOutcome(
            session_id=session.session_id,
            user_id=identity.user_id,
            role=identity.role,
            reply=reply,
            provider=active,
            model=client.model,
        )
    role = config.roles.roles.get(identity.role)
    note_turn = getattr(client, "note_turn", None)
    if note_turn is not None:
        note_turn(
            remaining_usd=config.limits.cost_limits.per_session_usd - session.cost_usd,
            sensitive=role is not None and role.pii == "own_full",
        )
    history = [message_from_dict(item) for item in session.messages]

    def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = execute_tool(_database(config), config.roles, identity, name, arguments)
        if name in AUDITED_TOOLS or result.get("error") == "forbidden":
            outcome = result.get("error") or result.get("status")
            append_audit(
                _audit_path(config),
                user_id=identity.user_id,
                role=identity.role,
                action=name,
                model=client.model,
                provider=active,
                ok=bool(result.get("ok")),
                order_id=_order_id(arguments),
                outcome=outcome if isinstance(outcome, str) else None,
            )
        return result

    messages = [
        ChatMessage(
            role="system",
            content=assemble_prompt(
                config.project_root,
                config.system_prompt,
                role_name=identity.role,
                role=role,
                provider=active,
            ),
        ),
        *history,
    ]
    tools = tool_catalog(role) if role is not None else []
    owned = read_own_orders(messages, tools, execute) or prepare_turn(messages, tools, execute)
    if owned is None:
        result = run_agent(
            client,
            messages,
            tools=tools,
            execute=execute,
            max_steps=config.limits.agent_loop.max_steps,
            max_identical=config.limits.agent_loop.max_identical_tool_calls,
            context_window=client.capabilities.context_window,
            limits=config.limits,
            after_tools=reply_after_tools,
            finalize=finish_reply,
        )
    else:
        reply, new_messages, tool_names = owned
        result = AgentResult(reply, new_messages, tool_names)
    reply = result.reply
    new_messages = result.new_messages
    if role is not None and role.pii != "own_full":
        reply = mask_phones(reply)
        if new_messages and new_messages[-1].role == "assistant" and new_messages[-1].content == result.reply:
            new_messages = [*new_messages[:-1], replace(new_messages[-1], content=reply)]
    served_provider, served_model, route = _served(client, active)
    stored = [message_to_dict(item) for item in new_messages]
    if route and stored and stored[-1].get("role") == "assistant":
        stored[-1]["provider"] = served_provider
        stored[-1]["model"] = served_model
        stored[-1]["route"] = route
    session.messages.extend(stored)
    input_tokens = estimate_tokens(message) + sum(
        estimate_tokens(item.content) for item in result.new_messages if item.role == "tool"
    )
    output_tokens = sum(estimate_tokens(item.content) for item in result.new_messages if item.role == "assistant")
    pricing_provider = config.models.providers.get(served_provider)
    pricing = config.provider.pricing if pricing_provider is None else pricing_provider.pricing
    cost = estimate_cost(
        input_tokens,
        output_tokens,
        None if pricing is None else pricing.input_per_1m_tokens,
        None if pricing is None else pricing.output_per_1m_tokens,
    )
    session.tokens_total += input_tokens + output_tokens
    session.cost_usd += cost
    add_user_usage(_usage_path(config, identity.user_id), input_tokens + output_tokens, cost)
    store.save(session)
    return ChatOutcome(
        session_id=session.session_id,
        user_id=identity.user_id,
        role=identity.role,
        reply=reply,
        provider=served_provider,
        model=served_model,
        tools=tuple(result.tool_names),
        route=route,
    )


def _served(client: LLMClient, active: str) -> tuple[str, str, str]:
    decision = getattr(client, "last_decision", None)
    if decision is None:
        return active, client.model, ""
    return decision.provider, decision.model, decision.reason


def _database(config: AppConfig) -> Path:
    return config.state_dir / "db" / "app.db"


def _usage_path(config: AppConfig, user_id: str) -> Path:
    return config.state_dir / "usage" / f"{user_id}.json"


def _audit_path(config: AppConfig) -> Path:
    return config.state_dir / "audit" / "audit.jsonl"


def _order_id(arguments: dict[str, Any]) -> str | None:
    value = arguments.get("order_id")
    if isinstance(value, str) and value.startswith("A-"):
        return value
    return None
