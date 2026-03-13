# Companion Framework — Hardening Plan

## Priorities
1. Security (injection, secrets, auth)
2. Multi-agent isolation (data bleed prevention)
3. Test coverage (from 3 tests to comprehensive)
4. Observability & admin tooling (error visibility, cost tracking per agent)
5. Infrastructure reliability (connection pooling, silent failures)

---

## Phase 1: Security Hardening

### 1.1 SQL Injection & XSS
**Files:** `scripts/simulate_relationship.py`, `scripts/generate_report.py`

- [ ] `rollback_checkpoint()` uses f-string for `DROP DATABASE IF EXISTS {pg_db}` — validate `pg_db` against allowlist or use `psycopg2.sql.Identifier`
- [ ] `generate_report.py` injects LLM-generated content into HTML without escaping — use `html.escape()` on all user/LLM content in opinion_html, curiosity_html, episode_html, reflection_html
- [ ] Audit all `db.execute()` calls across the codebase for any remaining f-string SQL (grep for `f".*SELECT|INSERT|UPDATE|DELETE.*{`)

### 1.2 Secrets Management
**Files:** `src/database/db.py`, `scripts/simulate_relationship.py`, `.env.example`

- [ ] Remove hardcoded default password `companion_change_me` from `db.py:91`, `simulate_relationship.py:462`
- [ ] Make DB password required (raise on missing) rather than falling back to a known default
- [ ] Audit `.env.example` — ensure no real keys leaked, only placeholder format
- [ ] Add `.env` to `.gitignore` (verify it's there and correct)

### 1.3 Auth Consolidation
**Current state:** Two parallel auth systems (Firebase JWT + legacy session tokens). Both work but the dual-path creates confusion and potential bypass.

- [ ] Decide on one canonical auth path for the web frontend (Firebase is the right choice)
- [ ] Keep legacy session auth for CLI and admin API only, clearly labeled
- [ ] Add rate limiting to `/api/auth/login` and `/api/auth/register` (prevent brute force)
- [ ] Fix Firebase fallback: currently allows unverified claims when public keys unavailable — this should reject, not allow

### 1.4 WebSocket Security
**File:** `src/routes/chat_routes.py`

- [ ] Validate that Socket.IO `connect` handler rejects connections without valid auth (currently has email-only fallback)
- [ ] Add per-connection rate limiting (messages per minute) to prevent abuse
- [ ] Ensure `disconnect` properly cleans up Redis presence keys

---

## Phase 2: Multi-Agent Data Isolation

### 2.1 Database-Level Isolation
**Current state:** Data keyed by `companion_id` column, but no enforced isolation. A bug in any query could leak data across companions.

- [ ] Add composite unique constraints where missing: `(companion_id, email)` on messages, facts, opinions, curiosity_threads, etc.
- [ ] Create a `CompanionScope` context manager that wraps DB queries and automatically injects `companion_id` filter — prevents forgetting the WHERE clause
- [ ] Audit every raw SQL query in `db.py` for missing `companion_id` filters (the file is 2000+ lines — methodical grep required)
- [ ] Add a DB-level row security policy (Postgres RLS) as defense-in-depth if feasible without major refactor

### 2.2 Redis Key Namespacing
**Current state:** Redis keys like `ws_connected:{email}` are not companion-scoped.

- [ ] Prefix all Redis keys with `{companion_id}:` — e.g., `kai:ws_connected:{email}`, `mira:action_budget:...`
- [ ] Audit `reach_out_pressure.py` persistence file — currently a single file, needs per-companion file or DB storage
- [ ] Audit `action_budget.py` Redis keys for companion scoping

### 2.3 File-Level Isolation
**Current state:** Some subsystems write to shared files (e.g., `autonomous_findings.json`, daily logs).

- [ ] Move file-based state to DB or namespace by companion_id: `data/autonomous_findings_{companion_id}.json`
- [ ] Ensure `COMPANION_MEMORY.md` (core memory) is per-companion (verify path includes companion_id)
- [ ] Verify `mood_persistence.py` stores per-companion mood state

### 2.4 LLM Context Isolation
- [ ] Verify that `context_builder.py` never loads entity profiles or facts from the wrong companion
- [ ] Add assertion in pipeline: `assert context.companion_id == self.companion_id` before LLM call

---

## Phase 3: Test Coverage

### 3.1 Test Infrastructure
- [ ] Set up pytest with fixtures for:
  - In-memory or test-scoped PostgreSQL (use `testing.postgresql` or Docker fixture)
  - Mock Redis (use `fakeredis`)
  - Mock LLM provider (returns canned responses, tracks calls)
  - Test companion config (minimal persona.yaml)
- [ ] Add `pytest.ini` or `pyproject.toml` with test config
- [ ] Add `requirements-test.txt` with test dependencies
- [ ] Set up CI (GitHub Actions) to run tests on PR

### 3.2 Unit Tests (Priority Order)

**Tier 1 — Data integrity (most critical):**
- [ ] `test_fact_store.py` — add/dedup/contradict/archive facts, confidence decay math, semantic dedup
- [ ] `test_relationship_store.py` — add/update/contradict relationships, type validation
- [ ] `test_db.py` — CursorWrapper behavior, connection lifecycle, environment validation
- [ ] `test_confidence_decay.py` — decay formula correctness, mention boost, clamping
- [ ] `test_importance_scorer.py` — scoring boundaries, category validation

**Tier 2 — Pipeline correctness:**
- [ ] `test_message_analyzer.py` — mode detection, emotional read, departure detection, fallback behavior
- [ ] `test_context_builder.py` — all 20+ sources assembled, graceful degradation when sources fail
- [ ] `test_pipeline.py` — end-to-end with mock LLM: message in → response out, proper stage ordering
- [ ] `test_message_validator.py` — contradiction detection, regeneration logic
- [ ] `test_checkpoint_detector.py` — trigger conditions (turns, tokens, duration), cooldown

**Tier 3 — Autonomy correctness:**
- [ ] `test_reach_out_pressure.py` — accumulation, decay math, release, persistence
- [ ] `test_action_budget.py` — daily limits, modifiers, high-urgency bypass, reset
- [ ] `test_reach_out_engine.py` — guard evaluation order, suppression classification
- [ ] `test_interjection_engine.py` — session caps, quiet hours, intimate scene detection
- [ ] `test_goal_formation.py` — duplicate detection, mode classification, limit enforcement
- [ ] `test_opinion_store.py` — UPSERT, evidence counting, decay

**Tier 4 — Memory retrieval:**
- [ ] `test_semantic_search.py` — embedding search, reranking, fallback to BM25
- [ ] `test_fact_network.py` — spreading activation math, relationship bridging, decay
- [ ] `test_temporal_context.py` — recent facts, keyword matching, time windows
- [ ] `test_retrieval_agent.py` — plan generation, entity detection, fallback to heuristics
- [ ] `test_memory_retriever.py` — multi-source dedup, ranking

**Tier 5 — Tasks:**
- [ ] `test_fact_extraction_task.py` — extraction with mock LLM, entity validation, approval routing
- [ ] `test_curiosity_extraction_task.py` — thread creation, deflection detection, urgency aging
- [ ] `test_daily_summary_task.py` — summary generation, file + DB storage
- [ ] `test_memory_pruning_task.py` — retention scoring, safety rails (never prune high importance)

### 3.3 Integration Tests
- [ ] `test_multi_agent_isolation.py` — two companions share DB, verify zero data leakage
- [ ] `test_pipeline_integration.py` — full pipeline with real DB, mock LLM, verify all stages fire
- [ ] `test_celery_tasks_integration.py` — message → fact extraction → storage → retrieval roundtrip
- [ ] `test_auth_flow.py` — login → token → authenticated request → logout

### 3.4 Coverage Target
- Phase 3 target: 60%+ line coverage on `src/` (from ~0% today)
- Long-term target: 80%+ on core pipeline and memory subsystems

---

## Phase 4: Admin Tooling & Observability

### 4.1 Admin CLI (`scripts/admin.py`)
A single unified admin CLI replacing scattered scripts. Uses argparse subcommands.

```
python scripts/admin.py status              # System health (12 checks)
python scripts/admin.py costs               # Cost summary (today, week, by model, by companion)
python scripts/admin.py costs --companion kai --days 30
python scripts/admin.py errors              # Recent errors from logs + DB
python scripts/admin.py errors --tail       # Stream errors in real-time
python scripts/admin.py companions          # List companions, enabled status, message counts
python scripts/admin.py companion kai       # Deep status: messages, facts, opinions, goals, memory stats
python scripts/admin.py tasks               # Celery task status (running, failed, queued)
python scripts/admin.py tasks --failed      # Show recent failures with tracebacks
python scripts/admin.py memory kai          # Memory stats: facts, episodes, observations, biographies
python scripts/admin.py memory kai --search "jesse"  # Search memories
python scripts/admin.py goals kai           # Active goals, steps, progress
python scripts/admin.py audit               # Security audit (open ports, auth config, secrets check)
python scripts/admin.py db                  # DB stats (table sizes, index health, connection count)
```

Implementation:
- [ ] Create `scripts/admin.py` with subcommand architecture
- [ ] `status` subcommand — wraps existing `SystemHealth.run_diagnostics()`, adds per-companion message/fact counts
- [ ] `costs` subcommand — wraps existing cost trackers, adds `--companion` filter by querying per-companion LLM call logs
- [ ] `errors` subcommand — parse `logs/` files for ERROR/CRITICAL, query DB for failed task records
- [ ] `companions` subcommand — list from `data/companions.yaml`, query DB for per-companion stats
- [ ] `tasks` subcommand — query Celery inspect for active/reserved/scheduled, query DB for task failure records
- [ ] `memory` subcommand — count facts/episodes/observations per companion, optional semantic search
- [ ] `goals` subcommand — query companion_goals and goal_steps tables
- [ ] `db` subcommand — `pg_stat_user_tables` for sizes, `pg_stat_activity` for connections

### 4.2 Per-Agent Cost Tracking
**Current gap:** Cost tracking exists but doesn't break down by companion_id.

- [ ] Add `companion_id` parameter to `CostTracker.record_call()` in `src/core/cost_tracker.py`
- [ ] Thread `companion_id` through `provider_factory.py` → LLM calls → cost recording
- [ ] Add `companion_id` column to SQLite cost tracking tables in `src/services/cost_tracker.py`
- [ ] Admin CLI `costs --companion kai` filters and aggregates by companion
- [ ] Add daily cost-per-companion summary to maintenance report

### 4.3 Error Visibility
**Current gap:** Silent `except Exception: pass` everywhere makes debugging impossible.

- [ ] Create `src/utils/error_tracker.py` — lightweight error recording to DB or structured log
- [ ] Replace bare `except Exception: pass` with `except Exception as e: error_tracker.record(e, context)` across:
  - `src/memory/` (all modules)
  - `src/tasks/` (all tasks)
  - `src/autonomy/` (all modules)
  - `src/core/conversation/context_builder.py` (parallel fetchers)
- [ ] Error tracker stores: timestamp, module, function, exception type, message, companion_id, stack trace
- [ ] Admin CLI `errors` reads from this store
- [ ] Add Sentry integration for CRITICAL errors (optional, behind env var)
- [ ] Keep graceful degradation behavior — still catch and continue, but now record

### 4.4 Structured Logging
**Current state:** Mix of `logger.info/debug/error` with unstructured messages.

- [ ] Add structured JSON logging option (behind `LOG_FORMAT=json` env var) for machine parsing
- [ ] Include `companion_id` in all log records via a logging filter or contextvars
- [ ] Add request_id to pipeline logs for tracing a message through all stages
- [ ] Ensure task logs include task name + companion_id

### 4.5 Ops Bot Enhancements
- [ ] Add `/errors [companion]` command to ops bot — surfaces recent errors from error tracker
- [ ] Add `/costs [companion]` breakdown to existing `/costs` command
- [ ] Add `/goals [companion]` to show active goals and progress
- [ ] Daily automated cost report via ops bot (morning summary of yesterday's spend per companion)

---

## Phase 5: Infrastructure Reliability

### 5.1 Connection Pooling
**Current state:** `CompanionDB` creates a new connection per query via context manager. No pooling.

- [ ] Replace per-request `psycopg2.connect()` with `psycopg2.pool.ThreadedConnectionPool` (min=2, max=10)
- [ ] Add connection health checking (test before returning from pool)
- [ ] Add pool exhaustion metric to system health diagnostics
- [ ] Keep the `_CursorWrapper` pattern — it solves a real problem

### 5.2 Database Migrations
**Current state:** Schema via `CREATE TABLE IF NOT EXISTS` — no way to alter columns, add indexes, or evolve schema.

- [ ] Set up Alembic (already have `alembic.ini` in repo root)
- [ ] Generate initial migration from current `_init_db()` schema
- [ ] Add migration for new columns needed by this plan (companion_id on cost tables, error_log table)
- [ ] Add `alembic upgrade head` to startup script
- [ ] Document migration workflow in README

### 5.3 Monolithic Process Splitting
**Current state:** Flask + SocketIO + APScheduler + Redis listeners all in one process.

This is not urgent for single-user but matters for reliability:
- [ ] Document which components could be split (scheduler → separate process, Redis listeners → separate process)
- [ ] For now: add watchdog thread that monitors main process health and restarts stuck components
- [ ] Long-term: consider splitting scheduler into Celery Beat (already partially done) and removing APScheduler

### 5.4 Neo4j Deprecation Cleanup
**Current state:** Graphiti Neo4j references still active but being phased out.

- [ ] Identify all Graphiti-only code paths (grep for `graphiti`, `neo4j`)
- [ ] Add feature flag `GRAPHITI_ENABLED=false` (default off) to gate all Neo4j calls
- [ ] Remove Neo4j from default docker-compose.yml (keep in docker-compose.dev.yml for optional use)
- [ ] Mark Graphiti modules as deprecated in docstrings

### 5.5 Monolithic db.py Decomposition
**Current state:** `db.py` is 2000+ lines handling everything.

- [ ] Split into focused modules:
  - `src/database/connection.py` — pool, CursorWrapper, execute()
  - `src/database/messages.py` — message CRUD
  - `src/database/users.py` — user/profile CRUD
  - `src/database/schema.py` — table creation, migrations
- [ ] Keep `get_db()` singleton in `db.py` as facade for backward compatibility
- [ ] Do this incrementally — move one domain at a time with re-exports

---

## Execution Order

**Sprint 1 (Security + Test Infra):**
- Phase 1.1 (SQL injection, XSS)
- Phase 1.2 (secrets)
- Phase 3.1 (test infrastructure)
- Phase 3.2 Tier 1 (data integrity tests)

**Sprint 2 (Isolation + Error Visibility):**
- Phase 2.1 (DB isolation)
- Phase 2.2 (Redis namespacing)
- Phase 4.3 (error tracker, replace silent exceptions)
- Phase 3.2 Tier 2 (pipeline tests)

**Sprint 3 (Admin Tooling + Cost Tracking):**
- Phase 4.1 (admin CLI)
- Phase 4.2 (per-agent cost tracking)
- Phase 4.5 (ops bot enhancements)
- Phase 3.2 Tier 3 (autonomy tests)

**Sprint 4 (Infrastructure + Cleanup):**
- Phase 5.1 (connection pooling)
- Phase 5.2 (Alembic migrations)
- Phase 5.4 (Neo4j deprecation)
- Phase 3.2 Tiers 4-5 (memory + task tests)
- Phase 3.3 (integration tests)

**Sprint 5 (Polish):**
- Phase 1.3 (auth consolidation)
- Phase 1.4 (WebSocket security)
- Phase 2.3 (file isolation)
- Phase 2.4 (LLM context isolation)
- Phase 4.4 (structured logging)
- Phase 5.5 (db.py decomposition)
