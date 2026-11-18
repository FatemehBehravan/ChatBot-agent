"""Append-only audit log for sensitive actions.

Each line is one JSON object: who, what, when, and which model.
Message text and phone numbers are not written.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

AUDITED_TOOLS = {
    "get_order_status",
    "list_my_orders",
    "search_orders",
    "create_support_ticket",
    "request_order_cancellation",
}


def append_audit(
    path: Path,
    *,
    user_id: str,
    role: str,
    action: str,
    model: str,
    provider: str,
    ok: bool,
    order_id: str | None = None,
    outcome: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "user_id": user_id,
        "role": role,
        "action": action,
        "model": model,
        "provider": provider,
        "ok": ok,
    }
    if order_id:
        record["order_id"] = order_id
    if outcome:
        record["outcome"] = outcome
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
