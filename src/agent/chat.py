"""Multi-turn chat. Each turn runs the tool loop and is stored on the session."""

from dataclasses import dataclass
from pathlib import Path

from src.agent.loop import message_from_dict, message_to_dict, run_agent
from src.agent.session import SessionStore
from src.llm.base import LLMClient
from src.llm.types import ChatMessage
from src.security.identity import UnknownUserError
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


def handle_chat(
    config: AppConfig,
    store: SessionStore,
    client: LLMClient,
    *,
    message: str,
    session_id: str | None,
    user_id: str,
) -> ChatOutcome:
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
    role = config.roles.roles.get(identity.role)
    history = [message_from_dict(item) for item in session.messages]
    result = run_agent(
        client,
        [ChatMessage(role="system", content=config.system_prompt), *history],
        tools=tool_catalog(role) if role is not None else [],
        execute=lambda name, arguments: execute_tool(_database(config), config.roles, identity, name, arguments),
        max_steps=config.limits.agent_loop.max_steps,
        max_identical=config.limits.agent_loop.max_identical_tool_calls,
    )
    session.messages.extend(message_to_dict(item) for item in result.new_messages)
    store.save(session)
    return ChatOutcome(
        session_id=session.session_id,
        user_id=identity.user_id,
        role=identity.role,
        reply=result.reply,
        provider=config.app_provider,
        model=client.model,
    )


def _database(config: AppConfig) -> Path:
    return config.state_dir / "db" / "app.db"
