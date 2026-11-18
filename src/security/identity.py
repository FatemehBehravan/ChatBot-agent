"""Load identities from the synthetic users file.

The request user_id only selects a row. Role and customer_id come from that
row and stay on the session. Chat text and tool arguments cannot replace them.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from src.errors import ConfigError


class UnknownUserError(Exception):
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        super().__init__(f"unknown user_id {user_id!r}")


@dataclass(frozen=True)
class Identity:
    user_id: str
    role: str
    display_name: str
    customer_id: str | None


def load_users(path: Path) -> dict[str, Identity]:
    if not path.is_file():
        raise ConfigError(f"missing users file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: expected a list of users")

    users: dict[str, Identity] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ConfigError(f"{path}: user at index {index} must be an object")
        user_id = row.get("user_id")
        role = row.get("role")
        display_name = row.get("display_name")
        if not isinstance(user_id, str) or not user_id:
            raise ConfigError(f"{path}: user at index {index} is missing user_id")
        if not isinstance(role, str) or not role:
            raise ConfigError(f"{path}: user {user_id} is missing role")
        if not isinstance(display_name, str) or not display_name:
            raise ConfigError(f"{path}: user {user_id} is missing display_name")
        customer_id = row.get("customer_id")
        if customer_id is not None and not isinstance(customer_id, str):
            raise ConfigError(f"{path}: user {user_id} has a non-string customer_id")
        if user_id in users:
            raise ConfigError(f"{path}: duplicate user_id {user_id}")
        users[user_id] = Identity(
            user_id=user_id,
            role=role,
            display_name=display_name,
            customer_id=customer_id,
        )
    if not users:
        raise ConfigError(f"{path}: no users defined")
    return users
