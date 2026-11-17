# Prompt changelog

## 1.1.0 — 2026-11-17

Stage 5 loop. The prompt now tells the model to call tools, cite sample size and source, keep statistics separate from guarantees, ask when the order id is missing, and wait for confirmation before a write. The offline stub follows the same rules in code so tests do not need a live model.

## 1.0.0 — 2026-11-14

Initial system prompt for the stage 1 stub. It names the assistant and forbids invented order facts. No eval comparison yet; prompt iteration belongs to stage 8.
