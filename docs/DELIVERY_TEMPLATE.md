# Delivery Document / سند تحویل

> Copy this file to `docs/DELIVERY.md` and fill it in. You may write in English or Persian (or both).
> این فایل را به `docs/DELIVERY.md` کپی و تکمیل کنید. نوشتن به فارسی یا انگلیسی (یا هر دو) مجاز است.
>
> Rule of thumb / قاعده‌ی کلی: every choice needs **why**, **alternatives rejected**, and **evidence** (numbers, logs, eval results). A choice without evidence is an opinion.

---

## 0. Summary / خلاصه
- What was built, in 5–8 lines. / آنچه ساخته‌اید، در ۵ تا ۸ خط.
- What works, what is partial, what is missing. / چه چیزی کامل، ناقص یا انجام‌نشده است.
- How to run (link to README section). / نحوه‌ی اجرا (لینک به بخش README).

## 1. Architecture / معماری
- Diagram (Mermaid, image, or ASCII) showing: client → API → agent loop → routing → local/remote LLM → tools → data; plus where context management, RBAC and audit sit.
- Request lifecycle for one chat turn (step-by-step).
- Deviations from the required structure and why.

## 2. Decision Log / سیاهه‌ی تصمیم‌ها

Fill one row per decision. Add rows as needed.

| # | Decision area | Chosen | Alternatives considered | Why chosen | Trade-offs / risks | What would change my mind |
|---|---|---|---|---|---|---|
| 1 | Language & runtime | | | | | |
| 2 | Agent framework (or none) | | | | | |
| 3 | Local model (name, size, quantization) | | | | | |
| 4 | Local serving tool | | | | | |
| 5 | Hosted API provider & model | | | | | |
| 6 | HTTP/UI layer | | | | | |
| 7 | Token counting method | | | | | |
| 8 | Session / state storage | | | | | |
| 9 | Database (and read-only access approach) | | | | | |
| 10 | Knowledge retrieval method (keyword / BM25 / embeddings / hybrid) and embedding model | | | | | |
| 11 | Persian text handling (normalization, tokenization, chunking) | | | | | |
| 12 | Validation / schema library | | | | | |
| 13 | UI choice (or none) | | | | | |
| 14 | Containerization approach (base images, compose layout, local-model access) | | | | | |
| 15 | Test & eval tooling | | | | | |
| 16 | Other | | | | | |

## 3. Model Selection Evidence / شواهد انتخاب مدل

### 3.1 Hardware and setup
- Machine specs used for the local model; quantization; context size configured.

### 3.2 Comparison on the same scenarios
| Metric | Local model | Hosted model |
|---|---|---|
| Scenario pass rate | | |
| Tool-call validity rate (well-formed, correct tool) | | |
| Instruction-following on format constraints | | |
| Avg / p95 latency per turn | | |
| Avg tokens in / out per turn | | |
| Estimated cost per 100 turns | | |
| Failure modes observed | | |

### 3.3 Conclusion
- Which model handles which kind of work, and why.

## 4. Context Window & Token Management / مدیریت context و توکن

- Context window of each model as configured vs. as verified.
- **Budget allocation table** per model (tokens for: system, tool schemas, history, tool results, retrieved docs, reserved output).
- Strategies used (truncation, sliding window, summarization, pinned facts, dedupe) and the trigger thresholds.
- What is guaranteed never to be dropped, and how you enforce it.
- Behavior when switching models mid-conversation.
- Limits (request / session / user): values, rationale, user-facing messages.
- Evidence: long-conversation test results (tokens before/after, quality impact).

## 5. Access Control & Security / کنترل دسترسی و امنیت

- Roles × tools × data-scope matrix (as implemented).
- How the three enforcement layers work (exposure, execution, data) with file references.
- How identity flows from session to tool call.
- PII policy: what can reach the hosted model, masking method, log redaction.
- Prompt-injection: threats considered, defenses, test evidence (including the poisoned record).
- Audit log format and sample entries.
- Known gaps and residual risk.

## 6. Prompt Engineering / مهندسی پرامپت

- Prompt inventory (files, purpose, version).
- Assembly logic (role-aware, model-aware).
- Local vs hosted variants and why they differ.
- Structured-output handling and repair path.
- Refusal / escalation rules.
- **Iteration log**: at least one failure observed → prompt change → measured result.

## 7. Hybrid Routing & Resilience / مسیریابی ترکیبی

- Routing policy: inputs, rules, defaults (link to config).
- Fallback behavior in both directions; circuit breaker parameters.
- Cost/usage reduction mechanisms and measured effect.
- Degraded-mode behavior shown to the user.
- Sample routing decisions with reasons (from logs).

## 7b. Data, Statistics & Knowledge Base / داده، آمار و پایگاه دانش

- Database choice, schema changes, indexes, loader idempotency, read-only enforcement.
- `estimate_delivery_time`: distance-window choice, widening/refusal rules and minimum sample size (with justification), percentile choice, how in-transit time and total time are separated, how in-progress orders are excluded. Include a verification of your numbers against an independent calculation.
- Knowledge ingestion: crawl scope, politeness settings, extraction method, documents/chunks count, de-duplication, refresh strategy.
- Retrieval: method, parameters, Persian normalization, citation format.
- Retrieval quality: your question set, hit rate, failure analysis.
- Grounding behavior: how "not found" and "insufficient data" are produced and tested.

## 8. Tools & Agent Behavior / ابزارها و رفتار ایجنت

- Tool list with schema summary, permissions, side effects, idempotency approach.
- Confirmation flow for write actions.
- Loop control: step limit, repetition detection, error handling.

## 9. Testing & Evaluation / تست و ارزیابی

- Test inventory and how to run.
- Eval scenarios added by you and why.
- Latest eval report (table) and notable failures with analysis.

## 9b. Containerization & Operations / داکرایز و عملیات

- Compose services diagram; image sizes; build and cold-start time.
- Volumes and what persists; how data load and index build run (idempotency evidence).
- Healthchecks and startup order.
- How the local model is reached (host vs. `local-llm` profile) and how the hosted key is injected.
- Evidence: `docker compose up --build` from a clean clone; tests run in the container with networking disabled.
- Optional UI: framework, what it shows (model per turn, tokens, citations), RTL handling.

## 10. Cost Estimate / برآورد هزینه

- Token usage per typical conversation (local vs hosted).
- Estimated hosted-API cost at 1k conversations (state assumptions and the pricing source you used).

## 11. Failure Modes & Mitigations / حالت‌های شکست و راهکارها

| Failure mode | Likelihood | Impact | Mitigation implemented | Remaining risk |
|---|---|---|---|---|
| Local model emits malformed tool call | | | | |
| Hosted API rate-limited / down | | | | |
| Context overflow | | | | |
| Injection via tool output | | | | |
| Budget exhausted mid-conversation | | | | |
| (add your own) | | | | |

## 12. Limitations & Next Steps / محدودیت‌ها و گام‌های بعدی
- What you would do with more information or resources, in priority order, with reasons.

## 13. AI-Assistance Disclosure / اعلام استفاده از هوش مصنوعی
- Which parts were produced with AI assistance, which tools, and how you verified correctness.
- One piece of AI-generated output you rejected or corrected, and why.

## 14. Self-Review Checklist / چک‌لیست خودارزیابی
- [ ] No secrets in repo or history
- [ ] `docker compose up --build` works from a clean clone; no secrets in images
- [ ] Tests pass offline (inside the container, networking disabled) from a clean checkout
- [ ] Statistical answers state sample size and basis; insufficient data is refused, not guessed
- [ ] Knowledge answers cite sources; "not found" is honest
- [ ] README commands verified on a clean checkout
- [ ] RBAC enforced in code at all three layers
- [ ] Hosted-model PII policy enforced and tested
- [ ] Context budgets enforced for every model, including after model switch
- [ ] Routing and fallback tested in both directions
- [ ] Prompts versioned with changelog and evidence
- [ ] Eval report and model comparison included
- [ ] Every decision in section 2 has evidence or a stated reason
