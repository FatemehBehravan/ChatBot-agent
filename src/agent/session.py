"""Session files on the state volume. One JSON file per conversation."""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Session:
    session_id: str
    user_id: str
    messages: list[dict[str, str]] = field(default_factory=list)


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def create(self, user_id: str) -> Session:
        session = Session(session_id=str(uuid.uuid4()), user_id=user_id)
        self.save(session)
        return session

    def load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session(
            session_id=data["session_id"],
            user_id=data["user_id"],
            messages=list(data["messages"]),
        )

    def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        payload = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "messages": session.messages,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _path(self, session_id: str) -> Path:
        # Callers pass a UUID. Reject anything that could escape the directory.
        uuid.UUID(session_id)
        return self.directory / f"{session_id}.json"
