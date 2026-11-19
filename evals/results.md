# Stage 10 comparison

Same scenarios in `evals/scenarios.yaml` (26, including two added: own orders without an id, and a 5 km motorcycle price). Reference time for the snapshot is 2026-11-14. Re-run on 2026-11-19 from the host, after the hosted addendum in `prompts/remote.md`. Local model at `http://127.0.0.1:11434/v1`. Hosted model is `gemini-2.5-flash-lite`.

| Metric | Stub | Local `qwen2.5:7b-instruct` | Hosted `gemini-2.5-flash-lite` |
|---|---:|---:|---:|
| Pass rate | 26/26 (100%) | 18/26 (69.2%) | 18/26 (69.2%) |
| Tokens | 7676 | 7006 | 5062 |
| Total latency | 1.66 s | 2537.75 s | 96.85 s |
| Cost | $0 | $0 (local model is unpriced) | $0.001689 |

Hosted cost uses the rates in `config/models.yaml` (0.15 / 0.60 USD per 1M tokens), not a provider invoice.

Previous hosted run, before that addendum, was 16/26. This run passes `order-status-own-fa` and `cancel-in-transit-refused` as well. Local stayed 18/26. It now passes `write-after-confirmation` and misses `model-switch-midconversation`.

Local failures: `staff-search-in-transit`, `eta-car-10km-tehran-en` (number outside 40–45), `eta-peak-vs-night` (no slower/longer), `kb-cancel-after-pickup`, `kb-fragile-packaging-fa` (timeout), `kb-not-found`, `long-conversation-keeps-facts`, `model-switch-midconversation`.

Hosted failures: `staff-search-in-transit`, `eta-car-10km-tehran-en`, `eta-peak-vs-night`, `kb-cancel-after-pickup`, `kb-not-found`, `write-needs-confirmation`, `long-conversation-keeps-facts`, `model-switch-midconversation`.

Copy this table into `docs/DELIVERY.md` in stage 11.
