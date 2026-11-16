# Data Dictionary

All data in this folder is **synthetic**, generated for this assessment. It is not real company data. Reference "now": `2026-11-14T10:00:00+03:30`. All timestamps are ISO-8601 with the `+03:30` offset.

## `orders_snapshot.csv`

| Column | Meaning |
|---|---|
| `order_id` | e.g. `A-240001` |
| `customer_id` | FK → `customers.csv` |
| `courier_id` | FK → `couriers.csv`; empty if no courier was ever assigned |
| `city` | Tehran, Karaj, Isfahan, Mashhad, Shiraz, Tabriz |
| `vehicle_type` | `motorcycle`, `car`, `van` |
| `package_type` | `document`, `food`, `parcel`, `fragile` |
| `distance_km` | Route distance of the delivery leg (km) |
| `created_at` | Order placed |
| `accepted_at` | A courier accepted (empty if none) |
| `picked_up_at` | Courier picked up the package (empty if not yet) |
| `delivered_at` | Delivered (empty if not delivered) |
| `status` | `searching_courier`, `courier_assigned`, `in_transit`, `delivered`, `cancelled_by_customer`, `cancelled_no_courier` |
| `price_toman` | Synthetic price in Toman |
| `rating` | 1–5, empty if not rated |
| `notes` | Free text written by customers or staff — **untrusted input** |

## `customers.csv`
`customer_id, full_name, phone, city, customer_type (individual|business), registered_at`. `full_name` and `phone` are PII (fake values).

## `couriers.csv`
`courier_id, city, vehicle_type, rating_avg, completed_orders`. No personal data.

## `users.json`
Authenticated identities used by the app and by evals: `user_id`, `role`, optional `customer_id`. Identity must come from here (the session), never from chat text.

## Metric definitions (use these in tools and tests)

| Metric | Definition |
|---|---|
| `assign_minutes` | `accepted_at − created_at` |
| `pickup_minutes` | `picked_up_at − accepted_at` |
| `transit_minutes` | `delivered_at − picked_up_at` — the in-transit delivery duration |
| `total_minutes` | `delivered_at − created_at` — what the customer experiences end to end |

Rules for delivery-time statistics:

- Use only orders with `status = delivered` (all four timestamps present).
- Report median and a high percentile (e.g. P80), always with the sample size `n`.
- "For a distance of X km" means a **distance window** around X. Choose and document your window (e.g. ±1 km) and how you widen it (or refuse) when `n` is small. Never extrapolate beyond the data: if too few comparable orders exist, say so.
- Traffic and demand vary with order-creation time, city and vehicle; derive such effects from the data rather than assuming them.
- Orders created in the last hour before "now" are still in progress and must not enter duration statistics.

## `schema.sql`
A suggested relational schema. Adapt to your database; keep primary keys and the metric definitions intact.

## `knowledge/`
See `knowledge/README.md`.
