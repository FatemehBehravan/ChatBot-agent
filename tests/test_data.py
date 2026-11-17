import sqlite3
from pathlib import Path

import pytest

from src.data.db import connect_ro, connect_rw
from src.data.load import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_loads_once_and_lookups_are_read_only(tmp_path):
    db_path = tmp_path / "db" / "app.db"
    first = load_snapshot(ROOT / "data", db_path)
    assert first > 5000

    writable = connect_rw(db_path)
    writable.execute("UPDATE orders SET notes = 'touched' WHERE order_id = 'A-240001'")
    writable.commit()
    writable.close()

    second = load_snapshot(ROOT / "data", db_path)
    assert second == first

    readonly = connect_ro(db_path)
    note = readonly.execute("SELECT notes FROM orders WHERE order_id = 'A-240001'").fetchone()["notes"]
    assert note == "touched"
    plan = readonly.execute(
        "EXPLAIN QUERY PLAN SELECT order_id FROM orders WHERE customer_id = ?",
        ("C-1001",),
    ).fetchall()
    assert any("idx_orders_customer" in " ".join(str(cell) for cell in row) for row in plan)
    with pytest.raises(sqlite3.OperationalError):
        readonly.execute("DELETE FROM orders")
    readonly.close()
