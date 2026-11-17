"""Domain tools over the orders snapshot and the knowledge index.

Reads use parameterized SQL on a read-only connection. The two write tools
do nothing until the caller comes back with the confirmation id and
confirmed=true. The same confirmation id returns the same result.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.data.db import connect_ro, connect_rw
from src.knowledge.index import search
from src.security.identity import Identity
from src.settings import RoleConfig, RolesConfig
from src.tools.schemas import (
    CreateSupportTicketInput,
    EstimateInput,
    GetOrderStatusInput,
    ListMyOrdersInput,
    RequestCancellationInput,
    SearchKnowledgeInput,
    SearchOrdersInput,
)

# Reference "now" is 2026-11-14T10:00:00+03:30. Duration stats ignore orders
# created in the last hour before that instant.
_CREATED_BEFORE = "2026-11-14T09:00:00+03:30"
_NOW = "2026-11-14T10:00:00+03:30"
_MIN_SAMPLE = 30
_HALF_WIDTHS_KM = (1.0, 2.0)
_STATS_ROW_LIMIT = 10000
_NOTE_LIMIT = 400
_PASSAGE_LIMIT = 500

_STATS_SQL = """
SELECT
    (julianday(delivered_at) - julianday(picked_up_at)) * 1440.0 AS transit_minutes,
    (julianday(delivered_at) - julianday(created_at)) * 1440.0 AS total_minutes,
    price_toman
FROM orders
WHERE status = 'delivered'
  AND picked_up_at IS NOT NULL
  AND delivered_at IS NOT NULL
  AND created_at < ?
  AND distance_km BETWEEN ? AND ?
  AND (? IS NULL OR city = ?)
  AND (? IS NULL OR vehicle_type = ?)
  AND (? IS NULL OR CAST(substr(created_at, 12, 2) AS INTEGER) BETWEEN ? AND ?)
LIMIT ?
"""

_WRITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_confirmations (
  confirmation_id TEXT PRIMARY KEY,
  tool_name TEXT NOT NULL,
  user_id TEXT NOT NULL,
  payload TEXT NOT NULL,
  state TEXT NOT NULL,
  result_json TEXT
);
CREATE TABLE IF NOT EXISTS support_tickets (
  ticket_id TEXT PRIMARY KEY,
  confirmation_id TEXT NOT NULL UNIQUE,
  user_id TEXT NOT NULL,
  order_id TEXT,
  message TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cancellation_requests (
  request_id TEXT PRIMARY KEY,
  confirmation_id TEXT NOT NULL UNIQUE,
  user_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

ToolHandler = Callable[[Path, Identity, RoleConfig, BaseModel], dict[str, Any]]


def _invalid(exc: ValidationError) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "invalid_arguments",
        "message": "The arguments did not match the tool schema.",
        "details": [
            {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ],
    }


def _missing_database() -> dict[str, Any]:
    return {
        "ok": False,
        "error": "database_unavailable",
        "message": "The order database is not loaded.",
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return round(ordered[index], 1)


def _customer_id(role: RoleConfig, identity: Identity) -> str | None:
    if role.scope == "own_customer":
        return identity.customer_id
    return None


def _order_brief(row: Any) -> dict[str, Any]:
    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "city": row["city"],
        "vehicle_type": row["vehicle_type"],
        "distance_km": row["distance_km"],
        "created_at": row["created_at"],
        "price_toman": row["price_toman"],
    }


def get_order_status(db_path: Path, identity: Identity, role: RoleConfig, args: GetOrderStatusInput) -> dict[str, Any]:
    if role.scope == "own_customer" and not identity.customer_id:
        return {"ok": False, "error": "not_found", "message": "Order not found."}
    if not db_path.is_file():
        return _missing_database()
    customer_id = _customer_id(role, identity)
    connection = connect_ro(db_path)
    try:
        row = connection.execute(
            """
            SELECT order_id, status, city, vehicle_type, package_type, distance_km,
                   created_at, accepted_at, picked_up_at, delivered_at, price_toman,
                   courier_id, notes
            FROM orders
            WHERE order_id = ? AND (? IS NULL OR customer_id = ?)
            """,
            (args.order_id, customer_id, customer_id),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return {"ok": False, "error": "not_found", "message": "Order not found."}
    notes = row["notes"] or ""
    return {
        "ok": True,
        "order": {
            **_order_brief(row),
            "package_type": row["package_type"],
            "accepted_at": row["accepted_at"],
            "picked_up_at": row["picked_up_at"],
            "delivered_at": row["delivered_at"],
            "courier_id": row["courier_id"],
            "notes": notes[:_NOTE_LIMIT],
        },
    }


def list_my_orders(db_path: Path, identity: Identity, role: RoleConfig, args: ListMyOrdersInput) -> dict[str, Any]:
    if not identity.customer_id:
        return {"ok": False, "error": "not_found", "message": "This user has no orders."}
    if not db_path.is_file():
        return _missing_database()
    connection = connect_ro(db_path)
    try:
        rows = connection.execute(
            """
            SELECT order_id, status, city, vehicle_type, distance_km, created_at, price_toman
            FROM orders
            WHERE customer_id = ? AND (? IS NULL OR status = ?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (identity.customer_id, args.status, args.status, args.limit),
        ).fetchall()
    finally:
        connection.close()
    return {"ok": True, "orders": [_order_brief(row) for row in rows], "limit": args.limit}


def search_orders(db_path: Path, identity: Identity, role: RoleConfig, args: SearchOrdersInput) -> dict[str, Any]:
    del identity, role
    if not db_path.is_file():
        return _missing_database()
    connection = connect_ro(db_path)
    try:
        rows = connection.execute(
            """
            SELECT order_id, status, city, vehicle_type, distance_km, created_at, price_toman
            FROM orders
            WHERE (? IS NULL OR city = ?)
              AND (? IS NULL OR status = ?)
              AND (? IS NULL OR order_id = ?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (args.city, args.city, args.status, args.status, args.order_id, args.order_id, args.limit),
        ).fetchall()
    finally:
        connection.close()
    return {"ok": True, "orders": [_order_brief(row) for row in rows], "limit": args.limit}


def _sample(db_path: Path, args: EstimateInput, half_width: float) -> list[Any]:
    low = args.distance_km - half_width
    high = args.distance_km + half_width
    connection = connect_ro(db_path)
    try:
        return connection.execute(
            _STATS_SQL,
            (
                _CREATED_BEFORE,
                low,
                high,
                args.city,
                args.city,
                args.vehicle_type,
                args.vehicle_type,
                args.hour_from,
                args.hour_from,
                args.hour_to,
                _STATS_ROW_LIMIT,
            ),
        ).fetchall()
    finally:
        connection.close()


def _filters(args: EstimateInput, half_width: float, widened: bool) -> dict[str, Any]:
    return {
        "filters": {
            "city": args.city,
            "vehicle_type": args.vehicle_type,
            "hour_from": args.hour_from,
            "hour_to": args.hour_to,
            "created_before": _CREATED_BEFORE,
            "status": "delivered",
        },
        "distance_window": {
            "center_km": args.distance_km,
            "half_width_km": half_width,
            "min_km": round(args.distance_km - half_width, 3),
            "max_km": round(args.distance_km + half_width, 3),
            "widened": widened,
        },
    }


def estimate_delivery_time(db_path: Path, identity: Identity, role: RoleConfig, args: EstimateInput) -> dict[str, Any]:
    del identity, role
    if not db_path.is_file():
        return _missing_database()
    rows: list[Any] = []
    half_width = _HALF_WIDTHS_KM[0]
    for half_width in _HALF_WIDTHS_KM:
        rows = _sample(db_path, args, half_width)
        if len(rows) >= _MIN_SAMPLE:
            break
    else:
        return {
            "ok": True,
            "insufficient_data": True,
            "sample_size": len(rows),
            "minimum_sample_size": _MIN_SAMPLE,
            "message": "insufficient data",
            **_filters(args, half_width, widened=half_width != _HALF_WIDTHS_KM[0]),
        }
    transit = [row["transit_minutes"] for row in rows if row["transit_minutes"] is not None]
    total = [row["total_minutes"] for row in rows if row["total_minutes"] is not None]
    if len(transit) < _MIN_SAMPLE or len(total) < _MIN_SAMPLE:
        return {
            "ok": True,
            "insufficient_data": True,
            "sample_size": min(len(transit), len(total)),
            "minimum_sample_size": _MIN_SAMPLE,
            "message": "insufficient data",
            **_filters(args, half_width, widened=half_width != _HALF_WIDTHS_KM[0]),
        }
    return {
        "ok": True,
        "insufficient_data": False,
        "sample_size": len(rows),
        "minimum_sample_size": _MIN_SAMPLE,
        "in_transit_minutes": {"median": _percentile(transit, 0.5), "p80": _percentile(transit, 0.8)},
        "total_minutes": {"median": _percentile(total, 0.5), "p80": _percentile(total, 0.8)},
        "note": "Historical sample, not a guarantee.",
        **_filters(args, half_width, widened=half_width != _HALF_WIDTHS_KM[0]),
    }


def estimate_price_range(db_path: Path, identity: Identity, role: RoleConfig, args: EstimateInput) -> dict[str, Any]:
    del identity, role
    if not db_path.is_file():
        return _missing_database()
    rows: list[Any] = []
    half_width = _HALF_WIDTHS_KM[0]
    for half_width in _HALF_WIDTHS_KM:
        rows = _sample(db_path, args, half_width)
        if len(rows) >= _MIN_SAMPLE:
            break
    else:
        return {
            "ok": True,
            "insufficient_data": True,
            "sample_size": len(rows),
            "minimum_sample_size": _MIN_SAMPLE,
            "message": "insufficient data",
            **_filters(args, half_width, widened=half_width != _HALF_WIDTHS_KM[0]),
        }
    prices = [float(row["price_toman"]) for row in rows if row["price_toman"] is not None]
    if len(prices) < _MIN_SAMPLE:
        return {
            "ok": True,
            "insufficient_data": True,
            "sample_size": len(prices),
            "minimum_sample_size": _MIN_SAMPLE,
            "message": "insufficient data",
            **_filters(args, half_width, widened=half_width != _HALF_WIDTHS_KM[0]),
        }
    return {
        "ok": True,
        "insufficient_data": False,
        "sample_size": len(prices),
        "minimum_sample_size": _MIN_SAMPLE,
        "price_toman": {"median": int(_percentile(prices, 0.5)), "p80": int(_percentile(prices, 0.8))},
        "note": "Historical sample, not a guarantee.",
        **_filters(args, half_width, widened=half_width != _HALF_WIDTHS_KM[0]),
    }


def search_knowledge_base(
    db_path: Path, identity: Identity, role: RoleConfig, args: SearchKnowledgeInput
) -> dict[str, Any]:
    del identity, role
    if not db_path.is_file():
        return _missing_database()
    try:
        passages = search(db_path, args.query, limit=args.limit)
    except sqlite3.Error:
        return {"ok": False, "error": "knowledge_unavailable", "message": "The knowledge index is not loaded."}
    return {
        "ok": True,
        "results": [
            {
                "source_url": passage.source_url,
                "title": passage.title,
                "text": passage.text[:_PASSAGE_LIMIT],
                "score": round(passage.score, 3),
            }
            for passage in passages
        ],
    }


def _ensure_writes(connection: Any) -> None:
    connection.executescript(_WRITE_SCHEMA)


def _summary(tool_name: str, payload: dict[str, Any]) -> str:
    if tool_name == "create_support_ticket":
        return f"Create a support ticket for order {payload.get('order_id') or '(no order)'}."
    return f"Cancel order {payload.get('order_id')} before pickup."


def _confirmed_write(
    db_path: Path,
    identity: Identity,
    tool_name: str,
    args: CreateSupportTicketInput | RequestCancellationInput,
    perform: Callable[[Any, dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    connection = connect_rw(db_path)
    try:
        _ensure_writes(connection)
        if not args.confirmation_id:
            confirmation_id = str(uuid.uuid4())
            payload = args.model_dump(exclude={"confirmed", "confirmation_id"})
            connection.execute(
                """
                INSERT INTO tool_confirmations
                    (confirmation_id, tool_name, user_id, payload, state, result_json)
                VALUES (?, ?, ?, ?, 'pending', NULL)
                """,
                (confirmation_id, tool_name, identity.user_id, json.dumps(payload)),
            )
            connection.commit()
            return {
                "ok": True,
                "status": "confirmation_required",
                "confirmation_id": confirmation_id,
                "summary": _summary(tool_name, payload),
            }
        row = connection.execute(
            """
            SELECT payload, state, result_json
            FROM tool_confirmations
            WHERE confirmation_id = ? AND user_id = ? AND tool_name = ?
            """,
            (args.confirmation_id, identity.user_id, tool_name),
        ).fetchone()
        if row is None:
            return {
                "ok": False,
                "error": "confirmation_not_found",
                "message": "No confirmation with this id belongs to you.",
            }
        if row["state"] != "pending":
            return json.loads(row["result_json"])
        payload = json.loads(row["payload"])
        if not args.confirmed:
            return {
                "ok": True,
                "status": "confirmation_required",
                "confirmation_id": args.confirmation_id,
                "summary": _summary(tool_name, payload),
            }
        result = perform(connection, payload, args.confirmation_id)
        connection.execute(
            "UPDATE tool_confirmations SET state = ?, result_json = ? WHERE confirmation_id = ?",
            ("completed" if result.get("ok") else "rejected", json.dumps(result), args.confirmation_id),
        )
        connection.commit()
        return result
    finally:
        connection.close()


def _order_in_scope(connection: Any, order_id: str, customer_id: str | None) -> Any:
    return connection.execute(
        "SELECT status, picked_up_at, customer_id FROM orders WHERE order_id = ? AND (? IS NULL OR customer_id = ?)",
        (order_id, customer_id, customer_id),
    ).fetchone()


def create_support_ticket(
    db_path: Path, identity: Identity, role: RoleConfig, args: CreateSupportTicketInput
) -> dict[str, Any]:
    if role.scope == "own_customer" and not identity.customer_id:
        return {"ok": False, "error": "not_found", "message": "Order not found."}

    def perform(connection: Any, payload: dict[str, Any], confirmation_id: str) -> dict[str, Any]:
        order_id = payload.get("order_id")
        if order_id is not None and _order_in_scope(connection, order_id, _customer_id(role, identity)) is None:
            return {"ok": False, "error": "not_found", "message": "Order not found."}
        ticket_id = "T-" + uuid.uuid4().hex[:8]
        connection.execute(
            """
            INSERT INTO support_tickets
                (ticket_id, confirmation_id, user_id, order_id, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, confirmation_id, identity.user_id, order_id, payload["message"], _NOW),
        )
        return {"ok": True, "status": "created", "ticket_id": ticket_id, "order_id": order_id}

    return _confirmed_write(db_path, identity, "create_support_ticket", args, perform)


def request_order_cancellation(
    db_path: Path, identity: Identity, role: RoleConfig, args: RequestCancellationInput
) -> dict[str, Any]:
    if role.scope == "own_customer" and not identity.customer_id:
        return {"ok": False, "error": "not_found", "message": "Order not found."}

    def perform(connection: Any, payload: dict[str, Any], confirmation_id: str) -> dict[str, Any]:
        customer_id = _customer_id(role, identity)
        order = _order_in_scope(connection, payload["order_id"], customer_id)
        if order is None:
            return {"ok": False, "error": "not_found", "message": "Order not found."}
        if order["picked_up_at"] or order["status"] not in ("searching_courier", "courier_assigned"):
            return {
                "ok": False,
                "error": "cancellation_not_allowed",
                "message": "Cancellation is only allowed before pickup.",
            }
        updated = connection.execute(
            """
            UPDATE orders
            SET status = 'cancelled_by_customer'
            WHERE order_id = ?
              AND picked_up_at IS NULL
              AND status IN ('searching_courier', 'courier_assigned')
              AND (? IS NULL OR customer_id = ?)
            """,
            (payload["order_id"], customer_id, customer_id),
        )
        if updated.rowcount != 1:
            return {
                "ok": False,
                "error": "cancellation_not_allowed",
                "message": "Cancellation is only allowed before pickup.",
            }
        request_id = "X-" + uuid.uuid4().hex[:8]
        connection.execute(
            """
            INSERT INTO cancellation_requests
                (request_id, confirmation_id, user_id, order_id, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (request_id, confirmation_id, identity.user_id, payload["order_id"], payload["reason"], _NOW),
        )
        return {
            "ok": True,
            "status": "cancelled",
            "request_id": request_id,
            "order_id": payload["order_id"],
        }

    return _confirmed_write(db_path, identity, "request_order_cancellation", args, perform)


_TOOLS: dict[str, tuple[str, type[BaseModel], ToolHandler]] = {
    "get_order_status": ("Look up one order by id.", GetOrderStatusInput, get_order_status),
    "list_my_orders": ("List the signed-in customer's newest orders.", ListMyOrdersInput, list_my_orders),
    "search_orders": ("Search orders. Staff only.", SearchOrdersInput, search_orders),
    "estimate_delivery_time": (
        "Median and P80 delivery time from delivered orders near a distance.",
        EstimateInput,
        estimate_delivery_time,
    ),
    "estimate_price_range": (
        "Median and P80 price from delivered orders near a distance.",
        EstimateInput,
        estimate_price_range,
    ),
    "search_knowledge_base": ("Search the help pages.", SearchKnowledgeInput, search_knowledge_base),
    "create_support_ticket": (
        "Open a support ticket. Requires a confirmation id and confirmed=true.",
        CreateSupportTicketInput,
        create_support_ticket,
    ),
    "request_order_cancellation": (
        "Cancel an order before pickup. Requires a confirmation id and confirmed=true.",
        RequestCancellationInput,
        request_order_cancellation,
    ),
}


def tool_catalog(role: RoleConfig) -> list[dict[str, Any]]:
    catalog = []
    for name in role.tools:
        spec = _TOOLS.get(name)
        if spec is None:
            continue
        description, model, _handler = spec
        catalog.append(
            {
                "name": name,
                "description": description,
                "parameters": model.model_json_schema(),
                "confirmation_required": name in role.confirmation_required,
            }
        )
    return catalog


def execute_tool(
    db_path: Path,
    roles: RolesConfig,
    identity: Identity,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _TOOLS.get(name)
    if spec is None:
        return {"ok": False, "error": "unknown_tool", "message": f"Unknown tool {name}."}
    role = roles.roles.get(identity.role)
    if role is None or name not in role.tools:
        return {"ok": False, "error": "forbidden", "message": f"{name} is not available for this role."}
    description, model, handler = spec
    del description
    try:
        parsed = model.model_validate(arguments or {})
    except ValidationError as exc:
        return _invalid(exc)
    return handler(db_path, identity, role, parsed)
