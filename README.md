# Alopeyk Delivery Assistant Agent — Engineering Assessment

> 🇮🇷 نسخه فارسی: [README.fa.md](README.fa.md)

## Run the stage 1 stub

Stage 1 is a Python HTTP API. `docker compose up --build` starts it with the deterministic stub provider. No API key and no local model are required, and a `.env` file is optional. Running the image from Docker Desktop uses the same defaults, which are baked into the image; Compose can still override them.

```bash
docker compose up --build
```

The API listens on port 8000. On PowerShell, call `curl.exe` so the request is not rewritten as `Invoke-WebRequest`.

```bash
curl http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"سلام\"}"
```

```powershell
curl.exe http://localhost:8000/health
curl.exe -s -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" --data-raw '{"message":"سلام"}'
```

Send the `session_id` from the first response to continue. `user_id` defaults to `u-anon`; other ids are in `data/users.json`. Stage 1 checks that the id exists. It does not authenticate the caller.

State directories (sessions, database, retrieval index, audit log) are created on the `app_state` volume at startup. The same step is safe to run again:

```bash
docker compose run --rm --no-deps app python /app/scripts/prepare_state.py
```

Offline tests run in the built image with networking disabled:

```bash
docker run --rm --network none -v "./data:/app/data:ro" alopeyk-agent-app python -m pytest -q
```

The container starts as root only long enough to create the state volume, then the server process runs as uid 10001. `docker compose exec` opens a separate root shell, so read the server process instead:

```bash
docker compose exec app python -c "print(open('/proc/1/status').read().split('Uid:')[1].split()[0])"
```

An optional Ollama service is not started by the command above. Stage 1 does not call it.

```bash
docker compose --profile local-llm up --build
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

Copy `.env.example` to `.env` when you want to override a setting. `APP_PROVIDER` chooses the model: `stub` (default, no network), `local`, or `remote`. Restart the container after changing it.

On startup the snapshot CSVs are loaded into SQLite if they changed, and the fixture knowledge pages are indexed. Crawling the public site is separate and uses the network:

```bash
python scripts/ingest_knowledge.py
```

## 1. Background

[Alopeyk](https://alopeyk.com/) is an on-demand delivery platform: customers request a courier (motorcycle, car or van) to pick up and deliver parcels, documents, food and more within and between cities. The company publishes a lot of educational and help content (blog and help/education pages) and keeps a large operational database of orders.

Your job is to design and build the backend of a **chat assistant agent** for this domain. It must answer from two kinds of grounding — a **knowledge base** built from the company's public blog/education content, and an **orders database snapshot** — and perform a few controlled actions through tools, while behaving correctly under real-world constraints: limited context windows, token budgets, permission tiers, mixed local/hosted models, and untrusted data.

Example questions the assistant must handle:

- "For a 5 km trip by motorcycle in Tehran, how long does delivery usually take?" → answered **statistically from the orders snapshot**, with sample size, typical range and caveats (not a made-up promise).
- "Where is my order A-246001?" → answered from the orders table, only if it belongs to the signed-in customer.
- "How should I package fragile items?" → answered from the **knowledge base with source citations**, or an honest "not found".
- "Open a support ticket for my late order." → a write action that requires explicit confirmation.

## 2. What you will build

A containerized chat agent (HTTP API and/or CLI, plus an optional UI) that:

1. Holds multi-turn conversations in Persian.
2. Uses tools over an orders database and a retrieval index of knowledge-base documents.
3. Works with **a local model and a hosted API model at the same time**, driven by an explicit, configurable routing policy.
4. Enforces role-based access control **in code**, not only in the prompt.
5. Manages the context window and token limits deliberately.
6. Is measurable: it ships with offline tests and an evaluation harness.
7. Runs end-to-end with `docker compose up`.

## 3. Ground rules

- **Language, platform, framework and database are your choice** (SQLite, PostgreSQL, DuckDB, … and any vector/keyword retrieval approach). Agent frameworks (LangGraph, LlamaIndex, a raw SDK, or none) are all acceptable — justify the choices in your delivery document.
- You may use AI coding assistants. Disclose where and how, and be ready to explain every line you ship.
- No secrets in the repository or in Docker images. Provide `.env.example`; real keys stay in environment variables.
- The test suite must pass **offline** (no network, no real API keys) using a stub model and the fixture corpus in `data/knowledge/seed/`.
- The orders snapshot in `data/` is **synthetic** and was generated for this assessment; it is not real company data.
- Knowledge-base content from the website is gathered by **your ingestion code**, not committed wholesale. Respect the site's `robots.txt`, terms of use and reasonable request rates; store the source URL and fetch date for every document. Commit the ingestion code and, at most, a small sample.
- Reference "now" for all data and evaluations: `2026-11-14T10:00:00+03:30`.

## 4. Models

### Local model
Any model served on your machine through an OpenAI-compatible endpoint (Ollama, llama.cpp server, LM Studio, vLLM, …). Choose a model that fits your hardware, supports tool/function calling and handles Persian acceptably; families worth trying: Qwen2.5-Instruct, Llama 3.1/3.2 Instruct, Mistral/Mistral-Nemo, Gemma, Phi. Ollama's OpenAI-compatible URL is typically `http://localhost:11434/v1` (from inside a container: `http://host.docker.internal:11434/v1`, or `http://ollama:11434/v1` with the compose profile in section 8).

### Hosted API model
Any OpenAI-compatible endpoint. To keep cost near zero you may use:

- the free/trial credit of a direct provider account, if one is available to you (e.g. OpenAI), or
- an OpenAI-compatible gateway with starter credit such as **AvalAI** or **BoxAPI**, or any equivalent service.

Free-credit offers and terms change over time — check the provider's current page. Keep base URL, key and model name in environment variables so the provider can be swapped without code changes. Pick a low-cost model tier and use the budget guards you build in Stage 6.

### Embeddings (if you use semantic retrieval)
A local embedding model (e.g. via Ollama) or a hosted one — your choice. Persian support matters; justify the choice. Keyword/BM25 retrieval is acceptable if you show it works on the evaluation questions.

## 5. Data

| File | Content |
|---|---|
| `data/orders_snapshot.csv` | ~6,000 synthetic orders: city, vehicle type, package type, distance, lifecycle timestamps, status, price, rating, free-text notes |
| `data/customers.csv` | Synthetic customers (fake contact data, treated as PII) |
| `data/couriers.csv` | Synthetic couriers (IDs and aggregate stats only) |
| `data/users.json` | Authenticated identities with role and customer scope |
| `data/schema.sql` | Suggested relational schema (adapt to your DB) |
| `data/README.md` | Data dictionary and metric definitions |
| `data/knowledge/` | Knowledge-base guidance, source list, and a small **fixture corpus** for offline tests |
| `config/roles.yaml`, `config/models.yaml`, `config/limits.yaml` | Starter configuration |
| `evals/scenarios.yaml` | Starter evaluation scenarios (extend them) |

Some records intentionally contain tricky content (e.g. free-text notes). Treat all stored text as untrusted input.

## 6. Required project structure

Any language is fine; keep the same responsibilities and boundaries. Document any deviation in your delivery document.

```
alopeyk-agent/
├── README.md                  # English task description (this file)
├── README.fa.md               # Persian task description
├── Dockerfile                 # yours (multi-stage, non-root)
├── docker-compose.yml         # starter provided; complete it
├── .dockerignore
├── .env.example               # all env vars, no real secrets
├── .gitignore
├── config/                    # models.yaml, roles.yaml, limits.yaml
├── data/                      # provided synthetic data + knowledge/ guidance and fixtures
├── prompts/                   # versioned prompts + CHANGELOG.md
├── src/
│   ├── api/                   # chat entrypoint (HTTP / CLI)
│   ├── ui/                    # optional UI
│   ├── agent/                 # orchestration loop, session state, step limits
│   ├── llm/                   # provider abstraction: local, remote, stub; errors, retries
│   ├── routing/               # local-vs-remote policy, fallback, circuit breaker
│   ├── context/               # token counting, budget allocator, trimming, summarization
│   ├── tools/                 # tool registry, schemas, domain tools
│   ├── data/                  # DB access layer, snapshot loader, stats queries
│   ├── knowledge/             # ingestion, normalization, chunking, index, retrieval
│   ├── security/              # RBAC, PII masking, injection defenses, audit log
│   └── observability/         # structured logs, usage / cost metrics
├── scripts/                   # ingest knowledge, load data, run evals
├── tests/                     # offline unit + integration tests
├── evals/
│   ├── scenarios.yaml         # provided + your scenarios
│   └── run_evals.*            # eval runner (pass rate, tokens, latency, cost)
└── docs/
    ├── DELIVERY.md            # your filled-in delivery document (from the template)
    ├── DELIVERY_TEMPLATE.md   # template provided
    ├── EVALUATION_CRITERIA.md # how submissions are assessed
    └── demo/                  # transcripts or recordings of real runs
```

## 7. Optional UI

A UI is **not required** — a CLI or HTTP API is enough. If you build one, some quick options:

- **Streamlit** — fastest in Python (`st.chat_message`, `st.chat_input`, streaming with `st.write_stream`); good for a demo with a sidebar to switch the signed-in user.
- **Chainlit** — chat-native, shows tool-call steps and streaming out of the box.
- **Gradio** `ChatInterface` — very small amount of code.
- A small web page calling your HTTP API (any framework).

Worth showing in a UI (all optional): which model served each turn, token usage, tool calls, source citations, and a dev-only user switcher. For Persian, set `direction: rtl` and right alignment on chat bubbles and use a Persian-friendly font bundled locally (the UI should not need internet to render).

## 8. Containerization (required)

- `docker compose up --build` starts the whole system (agent API, database if separate, UI if any) and is the primary documented way to run the project.
- A `Dockerfile` with a multi-stage build, pinned base-image versions, a non-root runtime user and a `.dockerignore`.
- Configuration only through environment variables / `.env`; **no secrets baked into images**; state (DB, sessions, retrieval index) on named volumes.
- Healthchecks and correct startup ordering (`depends_on` with health conditions); data loading and index building are **idempotent** and run automatically or through a documented one-line command.
- The local model must be reachable from the container: either the host's server via `host.docker.internal` (the starter compose adds the Linux `host-gateway` mapping), or the `local-llm` compose profile with an Ollama container plus a documented model-pull step.
- The tests must run **inside the container** with a documented command and must pass with networking disabled.
- A starter `docker-compose.yml` is provided; complete and adapt it.

## 9. Stages

Work through the stages in order. Deliver each stage as a separate pull request or a tagged commit (`stage-1` … `stage-11`) so the progression is visible in history. A stage is done when all its acceptance items are checked.

### Stage 1 — Skeleton, configuration and containers
- [x] Project layout from section 6 (or a documented variant).
- [x] Typed configuration loaded from `config/*.yaml` and environment variables; invalid config fails fast with a clear message.
- [x] `Dockerfile`, `.dockerignore` and a completed `docker-compose.yml` meeting section 8.
- [x] A fresh clone can chat with the stub model using only `docker compose up --build` and the README.

### Stage 2 — LLM provider layer
- [x] A common client interface: chat completion, optional streaming, token counting, declared capabilities (context window, max output, tool-calling support).
- [x] Adapters: local model, hosted API model, and a deterministic stub for tests.
- [x] Normalized errors (timeout, rate limit, auth, context overflow, malformed response) with retry/backoff (with jitter) and per-call timeouts.
- [x] Switching provider requires configuration changes only.

### Stage 3 — Data layer and knowledge base
- [x] The snapshot CSVs are loaded into your chosen database by an idempotent loader; indexes support the required queries efficiently.
- [x] The application connects with a **read-only** database role/mode for analytics and lookup paths.
- [x] Knowledge ingestion pipeline: fetches public blog/help/education pages from the company website (polite crawling, `robots.txt` respected), extracts clean text, stores source URL, title, language and fetch date, de-duplicates by content hash.
- [x] Persian-aware preprocessing: normalization (Arabic/Persian letter variants, digits, ZWNJ) before indexing and querying.
- [x] Chunking and retrieval (keyword, BM25, embeddings or hybrid — justified) that return passages **with source URLs**; works offline against `data/knowledge/seed/`.
- [x] Retrieval quality checked on a small question set you write (hit rate reported in your delivery document).

### Stage 4 — Domain tools
- [x] Tools covering at least: `get_order_status`, `list_my_orders`, `search_orders` (staff only), `estimate_delivery_time`, `estimate_price_range`, `search_knowledge_base`, `create_support_ticket`, `request_order_cancellation`.
- [x] `estimate_delivery_time` computes **statistics from the orders snapshot** (see `data/README.md` for metric definitions): inputs distance, optional vehicle type, city and order-creation time window; returns median, typical range (e.g. P80), sample size, the distance window and filters actually used, and separately the in-transit time and total time since order creation. It must report insufficient data instead of extrapolating.
- [x] Inputs/outputs defined by validated schemas (parameterized queries only — no string-built SQL); outputs are size-bounded; errors come back as structured results the model can act on.
- [x] Write actions are idempotent and require an explicit confirmation step from the user; cancellation is only allowed before pickup.
- [ ] Optional bonus: a read-only, guarded analytics tool for operations managers (query allow-listing, row/time limits, no writes).

### Stage 5 — Agent loop, chat and (optional) UI
- [ ] Loop of reason → tool call → observe → answer, with a maximum step count and repetition/loop detection.
- [ ] Multi-turn sessions with IDs and persistence.
- [ ] Graceful handling of: tool failure, malformed tool calls from a weak model, ambiguous requests (ask a clarifying question), out-of-scope requests.
- [ ] Responds in the user's language (Persian or English). Answers grounded in data state their basis (sample size, source) and never present statistics as guarantees.
- [ ] Optional: UI per section 7.

### Stage 6 — Context window and token management
- [ ] Token counting per model (exact tokenizer, or a documented approximation with a safety margin that accounts for Persian text).
- [ ] A budget allocator that splits each model's window between: system prompt, tool schemas, history, tool results, retrieved passages, reserved output.
- [ ] Strategies implemented and justified: tool-output truncation, sliding window, summarization of old turns, pinned facts (e.g. active order IDs), de-duplication, retrieved-passage selection under budget.
- [ ] A conversation can move between models with different windows (small local window ↔ large hosted window) without errors or silent loss of critical facts.
- [ ] Per-request, per-session and per-user token/cost limits with clear user-facing behavior when a limit is reached.
- [ ] A long-conversation test proving the above.

### Stage 7 — Access control and security
- [ ] Roles and permissions defined in `config/roles.yaml` (extend the starter).
- [ ] Enforced at three layers: (1) tool exposure — the model only sees tools its role may use, (2) tool execution — authorize again at call time, (3) data — per-row scoping and per-field redaction.
- [ ] Identity comes from the authenticated session context, never from model output or claims typed in chat.
- [ ] PII policy: what may reach the hosted model, masking before sending, redaction in logs.
- [ ] Prompt-injection resilience: order notes, retrieved web content and documents are treated as data, not instructions. Demonstrate against the poisoned note in `data/orders_snapshot.csv` (order `A-240001`).
- [ ] SQL safety: no model-controlled SQL fragments reach the database unvalidated; test with hostile inputs.
- [ ] Audit log for sensitive actions (who, what, when, which model).

### Stage 8 — Prompt engineering
- [ ] Prompts live in `prompts/`, versioned, with a `CHANGELOG.md` explaining each change and the evidence behind it.
- [ ] Role-aware prompt assembly; clear separation between instructions and untrusted data (notes, retrieved passages).
- [ ] Prompt variants for local vs hosted models where behavior differs (few-shot examples, stricter formats for weaker models), with justification.
- [ ] Grounding rules: cite sources for knowledge answers, state the basis of statistical answers, say "not found / insufficient data" instead of guessing.
- [ ] Structured outputs (e.g. ticket fields) validated, with a repair/retry path.
- [ ] Refusal and escalation behavior defined (when to say no, when to hand off to a human).
- [ ] At least one documented before/after prompt iteration backed by eval results.

### Stage 9 — Hybrid routing and resilience
- [ ] Config-driven routing policy that chooses local vs hosted per request or per step. Consider: data sensitivity, task complexity, required context length, tool-calling reliability, remaining budget, latency, availability.
- [ ] Fallback in both directions, circuit breaker, and a graceful degraded-mode message.
- [ ] The serving model is recorded per turn in logs/metadata.
- [ ] At least one mechanism that cuts hosted-API usage (e.g. local model for cheap steps, caching, history compression).

### Stage 10 — Testing and evaluation
- [ ] Offline unit/integration tests for budgeting, RBAC, tool validation, statistics correctness, retrieval, routing and fallback — runnable inside the container.
- [ ] An eval runner over `evals/scenarios.yaml` plus your own scenarios, reporting pass rate, token usage, latency and cost per model.
- [ ] The local and hosted models compared on the same scenarios; results included in your delivery document.

### Stage 11 — Documentation and delivery
- [ ] `docs/DELIVERY.md` completed from `docs/DELIVERY_TEMPLATE.md`.
- [ ] README run instructions verified on a clean checkout, including `docker compose up --build`.
- [ ] Demo transcripts or a short recording in `docs/demo/`.
- [ ] Final self-review against `docs/EVALUATION_CRITERIA.md`.

## 10. Deliverables

1. A GitHub repository (public, or private with access granted to the reviewers) following the structure above, with a commit history showing the stages.
2. Working instructions to run the agent with (a) the stub, (b) your local model, and (c) a hosted API model — all through Docker Compose.
3. `docs/DELIVERY.md` — your decisions, evidence and trade-offs.
4. Eval results and demo evidence.

## 11. Assessment

See [`docs/EVALUATION_CRITERIA.md`](docs/EVALUATION_CRITERIA.md). Reasoning and evidence matter as much as working code: be explicit about what you chose, what you rejected, and what you would change with more information.

## Wishing you success.