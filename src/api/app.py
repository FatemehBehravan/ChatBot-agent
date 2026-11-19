"""HTTP API: health check, chat, and the single-page UI."""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from src.agent.chat import SessionForbiddenError, SessionNotFoundError, handle_chat
from src.agent.session import SessionStore
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.knowledge.index import index_seed
from src.data.load import load_snapshot
from src.llm.errors import LLMError
from src.llm.factory import build_clients
from src.security.identity import UnknownUserError
from src.settings import AppConfig
from src.stateprep import prepare_state

logger = logging.getLogger(__name__)
_PAGE = Path(__file__).resolve().parents[1] / "ui" / "index.html"


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content: object) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


def create_app(config: AppConfig) -> FastAPI:
    prepare_state(config.state_dir)
    db_path = config.state_dir / "db" / "app.db"
    orders = load_snapshot(config.data_dir, db_path)
    index_seed(config.data_dir / "knowledge" / "seed", db_path)
    logger.info("snapshot ready orders=%s", orders)
    app = FastAPI(title="Alopeyk Delivery Assistant", version="0.1.0", default_response_class=UTF8JSONResponse)
    app.state.config = config
    app.state.sessions = SessionStore(config.state_dir / "sessions")
    app.state.clients = build_clients(config)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", provider=config.app_provider, model=config.provider.model or "stub")

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(_PAGE.read_text(encoding="utf-8"))

    @app.get("/v1/users")
    def list_users() -> list[dict[str, str]]:
        return [
            {"user_id": user.user_id, "role": user.role, "display_name": user.display_name}
            for user in config.users.values()
        ]

    @app.get("/v1/providers")
    def list_providers() -> list[dict[str, str]]:
        listed = [
            {"provider": name, "model": provider.model or name}
            for name in ("stub", "local", "remote")
            if (provider := config.models.providers.get(name)) is not None
        ]
        if "route" in app.state.clients:
            listed.insert(0, {"provider": "route", "model": "policy"})
        return listed

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        selected = body.provider or config.app_provider
        client = app.state.clients.get(selected)
        if client is None:
            raise HTTPException(status_code=400, detail=f"unknown provider {selected!r}")
        try:
            outcome = handle_chat(
                config,
                app.state.sessions,
                client,
                message=body.message,
                session_id=body.session_id,
                user_id=body.user_id,
                provider_name=selected,
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
            "chat user_id=%s session_id=%s provider=%s model=%s route=%s message_chars=%d",
            outcome.user_id,
            outcome.session_id,
            outcome.provider,
            outcome.model,
            outcome.route or "-",
            len(body.message),
        )
        return ChatResponse(
            session_id=outcome.session_id,
            user_id=outcome.user_id,
            role=outcome.role,
            reply=outcome.reply,
            provider=outcome.provider,
            model=outcome.model,
            tools=list(outcome.tools),
            route=outcome.route,
        )

    return app
