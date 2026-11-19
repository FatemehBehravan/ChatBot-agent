from pathlib import Path

from src.data.db import connect_ro
from src.data.load import load_snapshot
from src.knowledge.index import index_seed
from src.security.identity import load_users
from src.tools import execute_tool, tool_catalog

ROOT = Path(__file__).resolve().parents[1]


def _ready(tmp_path, *, knowledge: bool = False):
    db_path = tmp_path / "app.db"
    load_snapshot(ROOT / "data", db_path)
    if knowledge:
        index_seed(ROOT / "data" / "knowledge" / "seed", db_path)
    users = load_users(ROOT / "data" / "users.json")
    return db_path, users


def test_delivery_time_uses_the_snapshot_and_refuses_to_invent(config, tmp_path):
    db_path, users = _ready(tmp_path)
    near = execute_tool(
        db_path,
        config.roles,
        users["u-anon"],
        "estimate_delivery_time",
        {"distance_km": 5, "vehicle_type": "motorcycle", "city": "Tehran"},
    )
    assert near["insufficient_data"] is False
    assert 17 <= near["in_transit_minutes"]["median"] <= 20
    assert near["total_minutes"]["median"] > near["in_transit_minutes"]["median"]
    assert near["sample_size"] >= 30
    assert near["distance_window"]["half_width_km"] == 1.0
    assert near["filters"]["vehicle_type"] == "motorcycle"
    assert near["filters"]["city"] == "Tehran"

    persian_city = execute_tool(
        db_path,
        config.roles,
        users["u-anon"],
        "estimate_delivery_time",
        {"distance_km": 5, "vehicle_type": "موتور", "city": "تهران"},
    )
    assert persian_city["insufficient_data"] is False
    assert persian_city["filters"]["city"] == "Tehran"
    assert persian_city["filters"]["vehicle_type"] == "motorcycle"
    assert persian_city["in_transit_minutes"]["median"] == near["in_transit_minutes"]["median"]

    car = execute_tool(
        db_path,
        config.roles,
        users["u-cust-1004"],
        "estimate_delivery_time",
        {"distance_km": 10, "vehicle_type": "car", "city": "Tehran"},
    )
    assert 40 <= car["in_transit_minutes"]["median"] <= 45

    evening = execute_tool(
        db_path,
        config.roles,
        users["u-anon"],
        "estimate_delivery_time",
        {"distance_km": 5, "vehicle_type": "motorcycle", "city": "Tehran", "hour_from": 17, "hour_to": 20},
    )
    night = execute_tool(
        db_path,
        config.roles,
        users["u-anon"],
        "estimate_delivery_time",
        {"distance_km": 5, "vehicle_type": "motorcycle", "city": "Tehran", "hour_from": 0, "hour_to": 5},
    )
    assert evening["in_transit_minutes"]["median"] > night["in_transit_minutes"]["median"]

    far = execute_tool(
        db_path,
        config.roles,
        users["u-anon"],
        "estimate_delivery_time",
        {"distance_km": 80, "vehicle_type": "motorcycle", "city": "Tehran"},
    )
    assert far["insufficient_data"] is True
    assert far["message"] == "insufficient data"
    assert "in_transit_minutes" not in far

    hostile = execute_tool(
        db_path,
        config.roles,
        users["u-ops-01"],
        "estimate_delivery_time",
        {"distance_km": "5; DROP TABLE orders;--", "vehicle_type": "motorcycle"},
    )
    assert hostile["ok"] is False
    assert hostile["error"] == "invalid_arguments"
    readonly = connect_ro(db_path)
    count = readonly.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    readonly.close()
    assert count > 5000


def test_order_tools_respect_role_and_ownership(config, tmp_path):
    db_path, users = _ready(tmp_path)
    own = execute_tool(
        db_path, config.roles, users["u-cust-1001"], "get_order_status", {"order_id": "A-246001"}
    )
    assert own["ok"] is True
    assert own["order"]["status"] == "in_transit"

    other = execute_tool(
        db_path, config.roles, users["u-cust-1004"], "get_order_status", {"order_id": "A-246001"}
    )
    assert other["error"] == "not_found"

    denied = execute_tool(
        db_path, config.roles, users["u-cust-1001"], "search_orders", {"status": "in_transit"}
    )
    assert denied["error"] == "forbidden"
    names = [item["name"] for item in tool_catalog(config.roles.roles["customer"])]
    assert "search_orders" not in names

    found = execute_tool(
        db_path,
        config.roles,
        users["u-support-01"],
        "search_orders",
        {"city": "Tehran", "status": "in_transit"},
    )
    assert any(row["order_id"] == "A-246001" for row in found["orders"])
    assert len(found["orders"]) <= 20

    listed = execute_tool(db_path, config.roles, users["u-cust-1001"], "list_my_orders", {"limit": 20})
    assert listed["ok"] is True
    assert listed["orders"]
    assert all(len(row) <= 7 for row in listed["orders"])


def test_knowledge_search_returns_a_source(config, tmp_path):
    db_path, users = _ready(tmp_path, knowledge=True)
    result = execute_tool(
        db_path,
        config.roles,
        users["u-anon"],
        "search_knowledge_base",
        {"query": "Can I cancel my order after the courier already picked up the package?"},
    )
    assert result["results"][0]["source_url"] == "fixture://cancellation"


def test_writes_wait_for_confirmation_and_cancel_only_before_pickup(config, tmp_path):
    db_path, users = _ready(tmp_path)
    customer = users["u-cust-1001"]
    pending = execute_tool(
        db_path,
        config.roles,
        customer,
        "create_support_ticket",
        {"order_id": "A-246001", "message": "The courier has not moved.", "confirmed": True},
    )
    assert pending["status"] == "confirmation_required"
    created = execute_tool(
        db_path,
        config.roles,
        customer,
        "create_support_ticket",
        {"order_id": "A-246001", "message": "ignored on confirm", "confirmed": True, "confirmation_id": pending["confirmation_id"]},
    )
    again = execute_tool(
        db_path,
        config.roles,
        customer,
        "create_support_ticket",
        {"order_id": "A-246001", "message": "again", "confirmed": True, "confirmation_id": pending["confirmation_id"]},
    )
    assert created["status"] == "created"
    assert again["ticket_id"] == created["ticket_id"]

    blocked = execute_tool(
        db_path,
        config.roles,
        customer,
        "request_order_cancellation",
        {"order_id": "A-246001", "reason": "changed my mind"},
    )
    refused = execute_tool(
        db_path,
        config.roles,
        customer,
        "request_order_cancellation",
        {"order_id": "A-246001", "reason": "changed my mind", "confirmed": True, "confirmation_id": blocked["confirmation_id"]},
    )
    assert refused["error"] == "cancellation_not_allowed"

    readonly = connect_ro(db_path)
    open_order = readonly.execute(
        """
        SELECT order_id FROM orders
        WHERE picked_up_at IS NULL AND status IN ('searching_courier', 'courier_assigned')
        LIMIT 1
        """
    ).fetchone()["order_id"]
    readonly.close()

    staff_denied = execute_tool(
        db_path,
        config.roles,
        users["u-ops-01"],
        "request_order_cancellation",
        {"order_id": open_order, "reason": "ops cannot cancel"},
    )
    assert staff_denied["error"] == "forbidden"

    not_theirs = execute_tool(
        db_path,
        config.roles,
        customer,
        "request_order_cancellation",
        {"order_id": open_order, "reason": "not my order"},
    )
    refused_other = execute_tool(
        db_path,
        config.roles,
        customer,
        "request_order_cancellation",
        {
            "order_id": open_order,
            "reason": "not my order",
            "confirmed": True,
            "confirmation_id": not_theirs["confirmation_id"],
        },
    )
    assert refused_other["error"] == "not_found"

    support = users["u-support-01"]
    started = execute_tool(
        db_path,
        config.roles,
        support,
        "request_order_cancellation",
        {"order_id": open_order, "reason": "customer asked by phone"},
    )
    done = execute_tool(
        db_path,
        config.roles,
        support,
        "request_order_cancellation",
        {
            "order_id": open_order,
            "reason": "customer asked by phone",
            "confirmed": True,
            "confirmation_id": started["confirmation_id"],
        },
    )
    repeated = execute_tool(
        db_path,
        config.roles,
        support,
        "request_order_cancellation",
        {
            "order_id": open_order,
            "reason": "customer asked by phone",
            "confirmed": True,
            "confirmation_id": started["confirmation_id"],
        },
    )
    assert done["status"] == "cancelled"
    assert repeated["request_id"] == done["request_id"]
    status = execute_tool(db_path, config.roles, support, "get_order_status", {"order_id": open_order})
    assert status["order"]["status"] == "cancelled_by_customer"
