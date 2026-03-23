# CLAUDE.md

## Project Overview

Companion Framework — an AI companion engine with persistent memory, autonomous behavior, and emergent personality. Python 3.11, Flask + Socket.IO, Celery, PostgreSQL (pgvector), Neo4j.

## Key Files

- `src/core/conversation/pipeline.py` — Heart of response generation (ConversationPipeline)
- `src/handlers/message_handler.py` — Message entry point
- `src/config/persona_config.py` — Identity and pronoun configuration
- `src/autonomy/always_on_service.py` — Proactive behavior loop
- `src/llm/provider_factory.py` — LLM provider chain with failover
- `src/celery_app.py` — Task registration and beat schedule
- `data/persona.yaml` — Companion and user identity config

## Conventions

- Pronouns and user names in prompts must come from `get_persona_config()`, never hardcoded.
- LLM-facing prompt strings use f-string interpolation with config variables (e.g., `{user_name}`, `{c_possessive}`).
- Module-level lazy imports are used throughout to avoid circular dependencies.
- Singletons use `get_*()` factory functions (e.g., `get_persona_config()`, `get_db()`).

## Bug Fixes

When fixing a bug, always write a test that reproduces the bug and verifies the fix. This ensures regressions don't reoccur.

## Testing

- Tests live in `tests/`.
- Bug fix PRs must include a regression test.
- Run tests with `pytest` from project root.
