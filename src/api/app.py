"""HTTP API: health check and stub chat."""

import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.agent.chat import SessionForbiddenError, SessionNotFoundError, handle_chat
from src.agent.session import SessionStore
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.llm.errors import LLMError
from src.llm.factory import build_client
from src.security.identity import UnknownUserError
from src.settings import AppConfig
from src.stateprep import prepare_state

logger = logging.getLogger(__name__)


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: object) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


def create_app(config: AppConfig) -> FastAPI:
    prepare_state(config.state_dir)
    app = FastAPI(title="Alopeyk Delivery Assistant", version="0.1.0", default_response_class=UTF8JSONResponse)
    app.state.config = config
    app.state.sessions = SessionStore(config.state_dir / "sessions")
    app.state.client = build_client(config)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", provider=config.app_provider, model=config.provider.model or "stub")

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "alopeyk-agent", "health": "/health", "chat": "/v1/chat"}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        try:
            outcome = handle_chat(
                config,
                app.state.sessions,
                app.state.client,
                message=body.message,
                session_id=body.session_id,
                user_id=body.user_id,
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail={"error": type(exc).__name__, "message": str(exc)}) from exc
        except UnknownUserError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"unknown user_id {exc.user_id!r}. Known ids are listed in data/users.json.",
            ) from exc
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {exc}") from exc
        except SessionForbiddenError as exc:
            raise HTTPException(status_code=403, detail="session belongs to a different user") from exc
        logger.info(
            "chat user_id=%s session_id=%s provider=%s message_chars=%d",
            outcome.user_id,
            outcome.session_id,
            outcome.provider,
            len(body.message),
        )
        return ChatResponse(
            session_id=outcome.session_id,
            user_id=outcome.user_id,
            role=outcome.role,
            reply=outcome.reply,
            provider=outcome.provider,
            model=outcome.model,
        )

    return app
