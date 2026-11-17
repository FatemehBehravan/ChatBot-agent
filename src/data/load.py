"""Idempotent load of the synthetic snapshot into SQLite."""

import csv
import hashlib
import sqlite3
from pathlib import Path

from src.data.db import connect_rw

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
  customer_id   TEXT PRIMARY KEY,
  full_name     TEXT NOT NULL,
  phone         TEXT NOT NULL,
  city          TEXT NOT NULL,
  customer_type TEXT NOT NULL,
  registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS couriers (
  courier_id       TEXT PRIMARY KEY,
  city             TEXT NOT NULL,
  vehicle_type     TEXT NOT NULL,
  rating_avg       REAL,
  completed_orders INTEGER
);
CREATE TABLE IF NOT EXISTS orders (
  order_id      TEXT PRIMARY KEY,
  customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
  courier_id    TEXT REFERENCES couriers(courier_id),
  city          TEXT NOT NULL,
  vehicle_type  TEXT NOT NULL,
  package_type  TEXT NOT NULL,
  distance_km   REAL NOT NULL,
  created_at    TEXT NOT NULL,
  accepted_at   TEXT,
  picked_up_at  TEXT,
  delivered_at  TEXT,
  status        TEXT NOT NULL,
  price_toman   INTEGER NOT NULL,
  rating        INTEGER,
  notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_stats ON orders(city, vehicle_type, distance_km);
CREATE TABLE IF NOT EXISTS load_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

_SNAPSHOTS = ("customers.csv", "couriers.csv", "orders_snapshot.csv")


def load_snapshot(data_dir: Path, db_path: Path) -> int:
    """Load the CSVs. A second run with the same files does not rewrite the rows."""
    digest = _digest(data_dir)
    connection = connect_rw(db_path)
    try:
        connection.executescript(_SCHEMA)
        current = connection.execute(
            "SELECT value FROM load_meta WHERE key = 'snapshot_hash'"
        ).fetchone()
        if current is not None and current["value"] == digest:
            count = connection.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
            return int(count)
        with connection:
            connection.execute("DELETE FROM orders")
            connection.execute("DELETE FROM couriers")
            connection.execute("DELETE FROM customers")
            _insert_customers(connection, data_dir / "customers.csv")
            _insert_couriers(connection, data_dir / "couriers.csv")
            count = _insert_orders(connection, data_dir / "orders_snapshot.csv")
            connection.execute(
                """
                INSERT INTO load_meta (key, value) VALUES ('snapshot_hash', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (digest,),
            )
        return count
    finally:
        connection.close()


def _digest(data_dir: Path) -> str:
    hasher = hashlib.sha256()
    for name in _SNAPSHOTS:
        hasher.update((data_dir / name).read_bytes())
    return hasher.hexdigest()


def _insert_customers(connection: sqlite3.Connection, path: Path) -> None:
    rows = [
        (
            row["customer_id"],
            row["full_name"],
            row["phone"],
            row["city"],
            row["customer_type"],
            row["registered_at"],
        )
        for row in _rows(path)
    ]
    connection.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", rows)


def _insert_couriers(connection: sqlite3.Connection, path: Path) -> None:
    rows = [
        (
            row["courier_id"],
            row["city"],
            row["vehicle_type"],
            float(row["rating_avg"]) if row["rating_avg"] else None,
            int(row["completed_orders"]) if row["completed_orders"] else None,
        )
        for row in _rows(path)
    ]
    connection.executemany("INSERT INTO couriers VALUES (?,?,?,?,?)", rows)


def _insert_orders(connection: sqlite3.Connection, path: Path) -> int:
    rows = [
        (
            row["order_id"],
            row["customer_id"],
            row["courier_id"] or None,
            row["city"],
            row["vehicle_type"],
            row["package_type"],
            float(row["distance_km"]),
            row["created_at"],
            row["accepted_at"] or None,
            row["picked_up_at"] or None,
            row["delivered_at"] or None,
            row["status"],
            int(row["price_toman"]),
            int(row["rating"]) if row["rating"] else None,
            row["notes"] or None,
        )
        for row in _rows(path)
    ]
    connection.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield {key: (value or "").strip() for key, value in row.items()}
