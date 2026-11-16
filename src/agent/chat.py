"""One-turn stub chat with a persisted session.

The tool loop, step limit, and model routing are later stages. This module
stores the conversation and asks the stub provider for a reply.
"""

from dataclasses import dataclass

from src.agent.session import SessionStore
from src.llm.stub import StubModel
from src.security.identity import UnknownUserError
from src.settings import AppConfig


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
    model = StubModel(config.provider.model or "stub")
    reply = model.complete([{"role": "system", "content": config.system_prompt}, *session.messages])
    session.messages.append({"role": "assistant", "content": reply})
    store.save(session)
    return ChatOutcome(
        session_id=session.session_id,
        user_id=identity.user_id,
        role=identity.role,
        reply=reply,
        provider=config.app_provider,
        model=model.model_name,
    )
