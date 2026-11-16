-- Suggested schema (portable SQL; adapt types to your database).
-- Timestamps are stored as ISO-8601 text or native timestamptz — your choice.

CREATE TABLE customers (
  customer_id   TEXT PRIMARY KEY,
  full_name     TEXT NOT NULL,          -- PII
  phone         TEXT NOT NULL,          -- PII
  city          TEXT NOT NULL,
  customer_type TEXT NOT NULL CHECK (customer_type IN ('individual','business')),
  registered_at DATE NOT NULL
);

CREATE TABLE couriers (
  courier_id       TEXT PRIMARY KEY,
  city             TEXT NOT NULL,
  vehicle_type     TEXT NOT NULL CHECK (vehicle_type IN ('motorcycle','car','van')),
  rating_avg       REAL,
  completed_orders INTEGER
);

CREATE TABLE orders (
  order_id      TEXT PRIMARY KEY,
  customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
  courier_id    TEXT REFERENCES couriers(courier_id),
  city          TEXT NOT NULL,
  vehicle_type  TEXT NOT NULL CHECK (vehicle_type IN ('motorcycle','car','van')),
  package_type  TEXT NOT NULL CHECK (package_type IN ('document','food','parcel','fragile')),
  distance_km   REAL NOT NULL,
  created_at    TEXT NOT NULL,
  accepted_at   TEXT,
  picked_up_at  TEXT,
  delivered_at  TEXT,
  status        TEXT NOT NULL,
  price_toman   INTEGER NOT NULL,
  rating        INTEGER,
  notes         TEXT                    -- untrusted free text
);

-- Suggested indexes for the required queries.
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status   ON orders(status);
CREATE INDEX idx_orders_stats    ON orders(city, vehicle_type, distance_km);

-- Application-owned tables (yours to design): sessions, messages, tickets, cancellation_requests, audit_log, usage_ledger.
