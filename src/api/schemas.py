"""HTTP request and response bodies for the stage 1 chat API."""

import re
import uuid

from pydantic import BaseModel, Field, field_validator

_USER_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    user_id: str = "u-anon"

    @field_validator("message")
    @classmethod
    def message_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("session_id")
    @classmethod
    def session_id_is_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            uuid.UUID(value)
        except ValueError as exc:
            raise ValueError("session_id must be a UUID") from exc
        return value

    @field_validator("user_id")
    @classmethod
    def user_id_is_safe(cls, value: str) -> str:
        if not _USER_ID.fullmatch(value):
            raise ValueError("user_id must be 1-64 characters of letters, digits, '_' or '-'")
        return value


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    role: str
    reply: str
    provider: str
    model: str
    tools: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
