# Prompt changelog

## 1.2.3 — 2026-11-19

The hosted model is `gemini-2.5-flash-lite` through the gateway in `.env`. Its addendum now repeats the tool, sample-size, source, and confirmation rules. The evaluation showed it answering those turns from its own text.

## 1.2.2 — 2026-11-19

A 5 km motorcycle delivery in Tehran was answered as if the snapshot had no rows. The local model called `estimate_delivery_time` with `city=تهران`, and the snapshot stores `Tehran`, so the tool returned an empty sample. Tool arguments now map Persian city and vehicle names onto the snapshot values. The local addendum tells the model to quote a returned median instead of sending the user to support.

## 1.2.1 — 2026-11-18

A customer who asks to track their order without an id was told to type the id, including after they said they did not know it. The system prompt and the local addendum now say to call `list_my_orders` and answer from that list. Asking for an id stays only for roles that do not have that tool.

## 1.2.0 — 2026-11-18

Stage 8. The shared rules stay in `system.md`. `assemble_prompt` adds the session role from `config/roles.yaml`, then `prompts/local.md` or `prompts/remote.md`.

Local and stub get two short examples because a smaller model misses tool-call shape more often. The hosted model does not get those examples; its schema is already reliable, and the system-prompt budget is the same 10% cap. Instructions stay in the system message. Notes and retrieved pages stay in tool fields that start with `[data]`.

Refusal: no phone numbers, no role claimed in chat, no topic outside delivery support. Escalation: the user asks for a person, cancellation is past pickup, or the request still cannot be resolved. The reply tells them to contact support.

Before this change, `CreateSupportTicketInput` rejected a ticket whose message was over 500 characters, whose order id had surrounding spaces, and which carried an unknown key. After, one repair pass trims, strips, and drops that key, and the tool returns `confirmation_required`. An order id that is still not `A-<digits>` stays `invalid_arguments` so the model can correct it on the next step. Evidence: `tests/test_prompts.py`.

## 1.1.0 — 2026-11-17

Stage 5 loop. The prompt now tells the model to call tools, cite sample size and source, keep statistics separate from guarantees, ask when the order id is missing, and wait for confirmation before a write. The offline stub follows the same rules in code so tests do not need a live model.

## 1.0.0 — 2026-11-14

Initial system prompt for the stage 1 stub. It names the assistant and forbids invented order facts. No eval comparison yet; prompt iteration belongs to stage 8.
