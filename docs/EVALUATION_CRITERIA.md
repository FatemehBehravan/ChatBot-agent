# Evaluation Criteria / معیارهای ارزیابی

Submissions are scored out of 100 across the areas below. Reasoning and evidence in `docs/DELIVERY.md` count in every area.

| Area | Weight | What reviewers look for |
|---|---:|---|
| Architecture & code quality | 12 | Clear module boundaries, typed config, readable code, sensible abstractions (not over-engineered), error handling |
| Hybrid model use & routing | 14 | Local and hosted models genuinely used together; config-driven policy; fallback both ways; circuit breaker; per-turn model attribution; hosted-usage reduction |
| Context & token management | 12 | Correct counting (incl. Persian text), per-model budgets, trimming/summarization that preserves critical facts, safe model switching, enforced limits, long-conversation proof |
| Access control & security | 14 | RBAC at exposure/execution/data layers, identity not taken from model or chat, PII policy for the hosted model, injection resilience (order notes, retrieved pages), SQL safety, audit log |
| Prompt engineering | 10 | Versioned prompts, role/model-aware assembly, instruction/data separation, grounding rules, structured outputs with repair, refusal/escalation, evidence-driven iteration |
| Tools, data grounding & knowledge base | 14 | Correct, verifiable delivery-time statistics (definitions, windows, sample size, refusal on thin data); sound DB usage; polite ingestion; Persian-aware retrieval with citations; validated tool schemas; idempotent confirmed writes |
| Containerization & reproducibility | 6 | `docker compose up --build` works from a clean clone; multi-stage, non-root, pinned images; healthchecks; volumes; no secrets in images; tests run in-container offline |
| Testing & evaluation | 10 | Offline tests that matter, eval runner, scenario quality, honest model comparison |
| Documentation & decision reasoning | 8 | Decision log with alternatives and evidence, trade-offs acknowledged, accurate README, clear limitations |

## Bonus (up to +5, never compensates for missing core items)
Streaming responses, a UI that shows model/tokens/citations with proper RTL, semantic caching, guarded read-only analytics tool, observability dashboard, load/latency benchmark, thoughtful extra evals, CI pipeline.

## Red flags (heavy deductions)
- Secrets committed to the repository, history or images.
- Access control enforced only through the prompt.
- String-built SQL with model-controlled fragments.
- Statistics presented without sample size, or invented when data is thin.
- Tests that require network or real keys; crawling that ignores `robots.txt`.
- Only one model type actually used while claiming a hybrid setup.
- Fabricated numbers in the delivery document.
- AI usage not disclosed, or inability to explain submitted code.

## How reviewers will verify
1. Clone, follow README, `docker compose up --build`, chat with the stub.
2. Run tests inside the container with networking disabled.
3. Run with a local model and a hosted model; run the eval runner.
4. Independently recompute a few delivery-time statistics from `orders_snapshot.csv` and compare.
5. Try adversarial chats: role-escalation claims, injection via order notes, hostile tool inputs, oversized inputs, long conversations, stopping the local model server or using a bad API key mid-session.
6. Read `docs/DELIVERY.md` and ask follow-up questions about the decisions.
