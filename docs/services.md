# Services Reference

Detailed reference for all services, background processes, Celery tasks, and external integrations in the companion framework.

---

## Entry Points

### Web Backend (`src/web_chat.py`)

Flask + Socket.IO server on port 5000. This is the primary entry point for the entire companion system in production.

**Startup sequence:**

1. Eventlet monkey-patch (must be first -- enables cooperative I/O for Socket.IO)
2. Create Flask app via `create_app()` factory
3. Initialize autonomy LLM (dedicated Fireworks model for background operations)
4. Create Socket.IO with WebSocket event handlers
5. Register static routes (web UI, images, observe dashboard)
6. Start background services via `init_scheduled_jobs()`:
   - Unified scheduler (APScheduler)
   - Always-On Service (Telegram + proactive messaging)
   - Ops Bot (admin commands via Telegram)
   - Scheduled jobs (semantic memory extraction, value inference, diagnostics, daily summary, core memory refresh)
7. Start Redis pub/sub listeners:
   - `image_generated` channel -- forwards completed image events to Socket.IO rooms
   - `companion_interjection` channel -- forwards unprompted messages to active web sessions
   - Observe dashboard subscriber
8. Run server on `0.0.0.0:5000`

**Route blueprints:** `auth_routes`, `chat_routes`, `settings_routes`, `cost_routes`, `integrations_routes`, `approval_routes`, `observe_routes`

### Celery Worker (`src/celery_app.py`)

Redis-backed async task queue. Processes operations too slow for the request/response cycle (image generation, fact extraction, memory consolidation).

- **Broker/backend:** Redis (`redis://localhost:6379/0` default)
- **Timezone:** `America/Los_Angeles`
- **Task time limit:** 600s hard / 540s soft
- **Concurrency:** 4 workers, solo pool
- **Safety:** `task_acks_late=True`, `task_reject_on_worker_lost=True` (prevents message loss during deploys)
- **Worker recycling:** Restarts after 50 tasks to prevent memory leaks
- **Beat scheduler:** Built-in crontab-based scheduling for periodic tasks (see Celery Tasks section below)
- **Sentry integration:** Optional error tracking for worker failures

### Code Executor (`executor/app.py`)

Sandboxed Python execution service on port 5001. Used by the companion for autonomous code execution tasks (research, data analysis, tool use).

- **Timeout:** 30s per execution
- **Memory limit:** 512 MB (container capped at 1 GB)
- **CPU limit:** 1 core
- **Capabilities dropped:** `NET_ADMIN`, `SYS_ADMIN`
- **Access:** PostgreSQL (for memory/reminders), Tavily (web search), RunComfy (image generation), Google Gemini

---

## Background Services

### AlwaysOnService (`src/autonomy/always_on_service.py`)

Background thread started at app init via `start_always_on()`. Makes the companion feel present by running a continuous dual-cadence loop.

**Two loop paths:**

| Path | Interval | Purpose |
|------|----------|---------|
| Offline / Telegram | 3-25 min (dynamic) | Check if companion should reach out via Telegram, driven by reach-out pressure |
| WebSocket | 2-5 min | Check if companion should interject during an active web conversation |

**Loop cycle:**
1. Sleep for interval
2. `_check_user_schedule()` -- autopilot wake/sleep
3. `_check_activity_changes()` -- companion schedule transitions
4. `_check_calendar_transitions()` -- user's calendar event endings
5. If WebSocket active: `_check_interjection()` via InterjectionEngine
6. If offline: `_check_reach_out_with_pressure()` via ReachOutEngine
7. `_check_goal_actions()` -- budget-gated autonomous goal execution

**Delegates to:**
- `ReachOutEngine` -- decides when/what to send via Telegram
- `InterjectionEngine` -- decides when/what to interject in web chat
- `ReachOutPressure` -- dynamic interval timing
- `autonomous_action_task` -- budget-gated goal actions
- Publishes interjections to Redis pub/sub for WebSocket delivery

### UnifiedScheduler (`src/scheduling/unified_scheduler.py`)

Thread-safe singleton wrapping APScheduler's `BackgroundScheduler`. Consolidates all timed tasks into one scheduler to avoid thread-leak bugs.

- **Job types:** interval, cron, one-shot (via `add_interval_job`, `add_cron_job`, `add_one_time_job`)
- **Timezone:** Pacific (`America/Los_Angeles`)
- **Stats tracking:** Execution counts, errors, durations per job
- **Singleton access:** `get_unified_scheduler()`
- **Ops integration:** Jobs can be listed/cancelled from the ops bot

**Registered jobs (via `init_scheduled_jobs`):**

| Job ID | Schedule | Description |
|--------|----------|-------------|
| `semantic_memory_extraction` | Hourly | Extract biographical memories from recent conversations |
| `value_inference` | Daily 3:00 AM | Analyze companion messages to infer personality |
| `system_diagnostics` | Every 6 hours | Check all subsystems, report failures to Sentry |
| `daily_summary` | Daily 12:05 AM | Generate daily conversation summary (dispatches Celery task) |
| `core_memory_refresh` | Sundays 4:00 AM | Regenerate companion's narrative memory (dispatches Celery task) |

### TelegramOpsBot (`src/ops/telegram_ops_bot.py`)

One-way notification sender for system alerts, separate from the companion's conversation channel.

- **Purpose:** Health-check failures, maintenance summaries, cost alerts, deployment status, error spikes
- **Interface:** Uses `python-telegram-bot` Bot class directly (not Updater)
- **Methods:** `send_sync()` (blocking) and `send()` (async)
- **Config:** `TELEGRAM_OPS_BOT_TOKEN` + `TELEGRAM_OPS_CHAT_ID`
- **Singleton:** `get_ops_telegram_bot()`
- **Command handling:** Interactive command side lives in `src/ops/ops_bot_commands.py` (`start_ops_bot()`)

---

## Celery Tasks

All times are Pacific (`America/Los_Angeles`).

### Post-Message Tasks

Dispatched asynchronously after each user message:

| Task | Module | Description |
|------|--------|-------------|
| `episodic_embedding_task` | `src.tasks.episodic_embedding_task` | Embed messages for semantic search |
| `fact_extraction_task` | `src.tasks.fact_extraction_task` | Extract S-P-O facts from conversation |
| `relationship_extraction_task` | `src.tasks.relationship_extraction_task` | Extract entity relationships |
| `graphiti_extraction_task` | `src.tasks.graphiti_extraction_task` | Temporal knowledge graph updates |
| `scene_extraction_task` | `src.tasks.scene_extraction_task` | Track roleplay scene state |
| `internal_state_task` | `src.tasks.internal_state_task` | Update companion mood/energy/needs |
| `relationship_dynamics_task` | `src.tasks.relationship_dynamics_task` | Gottman-informed relationship tracking |
| `episode_tracking_task` | `src.tasks.episode_tracking_task` | Group messages into conversation episodes |
| `curiosity_extraction_task` | `src.tasks.curiosity_extraction_task` | Extract unresolved curiosity threads |
| `correction_task` | `src.tasks.correction_task` | Learn from user corrections |
| `interaction_outcome_task` | `src.tasks.interaction_outcome_task` | Per-message interaction outcome tracking |

### Daily Tasks

| Beat Key | Celery Task | Schedule | Description |
|----------|-------------|----------|-------------|
| `daily-reflection` | `tasks.reflection_task.reflect_on_day` | 12:30 AM | Daily reflection with insights (after daily summary) |
| `daily-opinion-formation` | `tasks.opinion_formation.form_opinions` | 1:00 AM | Form opinions from reflections |
| `daily-value-refresh` | `tasks.value_inference.run_incremental` | 1:00 AM | Incremental value refresh (volatile categories) |
| `daily-event-consolidation` | `tasks.consolidate_events` | 3:00 AM | Consolidate event narratives |
| `daily-entity-profile-fact-validation` | `tasks.entity_profile_fact_validator.validate_facts_against_profiles` | 3:30 AM | Archive facts contradicting YAML profiles |
| `daily-episode-learning` | `tasks.learn_from_episodes` | 4:00 AM | Extract lessons from conversation episodes |
| `daily-curiosity-urgency` | `tasks.curiosity_extraction.update_curiosity_urgency` | 5:00 AM | Increase urgency for unexplored topics |
| `daily-calendar-schedule` | `tasks.calendar_schedule.generate_daily_plan` | 5:30 AM | Generate companion's daily schedule |
| `daily-memory-gap-analysis` | `tasks.memory_gap_analysis.analyze_memory_gaps` | 6:00 AM | Detect knowledge gaps, generate curiosity |
| `daily-biography-refresh` | `tasks.refresh_biographies` | Every 24h | Refresh synthesized biographies |
| `goal-planning-decompose` | `tasks.goal_planning.decompose_unplanned_goals` | Every 6h (0:30, 6:30, 12:30, 18:30) | Break goals into actionable steps |
| `check-due-reminders` | `tasks.reminder_check.check_due_reminders` | Every 2h (8 AM - 8 PM) | Surface due reminders to queued thoughts |
| `check-gmail-inbox` | `tasks.gmail_check.check_inbox` | Every 30 min (8 AM - 9 PM) | Monitor Gmail inbox for interesting emails |
| `goal-formation-signals` | `tasks.goal_signals.form_from_signals` | 9:00 AM, 9:00 PM | Form goals from curiosities, opinions, findings |
| `image-orphan-recovery` | `src.tasks.image_generation_task.recover_orphaned_images` | Every 5 min | Recover stuck image generation requests |

### Weekly Tasks

| Beat Key | Celery Task | Schedule | Description |
|----------|-------------|----------|-------------|
| `weekly-reflection` | `tasks.reflection_task.reflect_on_week` | Sunday 3:00 AM | Analyze patterns across the week |
| `weekly-opinion-decay` | `tasks.opinion_formation.decay_weak_opinions` | Sunday 2:00 AM | Prune low-confidence opinions |
| `weekly-memory-pruning` | `tasks.memory_pruning.prune_stale_facts` | Sunday 5:30 AM | Archive stale low-retention facts |
| `weekly-relationship-evaluation` | `tasks.relationship_evaluation.evaluate_weekly` | Tuesday 2:30 AM | Companion self-assesses relationship meaning |
| `weekly-value-inference` | `tasks.value_inference.run_weekly` | Wednesday 3:30 AM | Infer personality from recent messages |

### Monthly Tasks

| Beat Key | Celery Task | Schedule | Description |
|----------|-------------|----------|-------------|
| `monthly-reflection` | `tasks.reflection_task.reflect_on_month` | 1st of month 4:00 AM | Deep monthly relationship evolution reflection |
| `monthly-value-inference-full` | `tasks.value_inference.run_full_history` | 1st of month 4:00 AM | Full-history personality inference |

### On-Demand Tasks

Not scheduled via beat; triggered by application logic:

| Task | Module | Trigger |
|------|--------|---------|
| `image_generation_task` | `src.tasks.image_generation_task` | User requests image or companion decides to send one |
| `autonomous_action_task` | `src.tasks.autonomous_action_task` | AlwaysOnService `_check_goal_actions()` (budget-gated) |
| `event_synthesis_task` | `src.tasks.event_synthesis_task` | Event-based narrative synthesis |
| `goal_signal_task` | `src.tasks.goal_signal_task` | Signal-based goal formation |
| `core_memory_task` | `src.tasks.core_memory_task` | Also triggered via UnifiedScheduler (Sundays 4 AM) |

---

## Docker Services

Defined in `docker-compose.yml`:

| Service | Container Name | Port | Role |
|---------|---------------|------|------|
| `agent-service` | `companion-backend` | 5000 | Flask + Socket.IO web backend (runs `python -m src.web_chat`) |
| `celery-worker` | `companion-celery-worker` | -- | Async task processing (concurrency=4, solo pool) |
| `postgres` | `companion-postgres` | 5432 | Primary database (pgvector/pg16, max 200 connections) |
| `redis` | `companion-redis` | 6379 | Message broker for Celery + pub/sub for real-time events |
| `graphiti-neo4j` | `companion-graphiti-neo4j` | 7475 (HTTP), 7688 (Bolt) | Neo4j 5.26 for Graphiti temporal knowledge graph |
| `code-executor` | `companion-code-executor` | 5001 | Sandboxed Python execution (mem limit 1 GB, 1 CPU) |
| `caddy` | `companion-caddy` | 80, 443 | SSL termination + reverse proxy |
| `n8n` | `companion-n8n` | 5678 (internal) | Workflow automation (accessed via Caddy proxy) |
| `companion-cli` | `companion-cli` | -- | Terminal chat interface (interactive TTY) |

**Notes:**
- Original Neo4j service is disabled (data migrated to PostgreSQL). `graphiti-neo4j` is a dedicated instance for Graphiti only.
- `agent-service` depends on `graphiti-neo4j`, `redis`, `postgres`.
- `celery-worker` depends on `redis`, `graphiti-neo4j`, `postgres`.
- All services use `restart: "no"` except `code-executor` (`unless-stopped`).

---

## External Integrations

| Provider | Purpose | Required Env Vars | Optional? |
|----------|---------|-------------------|-----------|
| **Fireworks AI** | Primary LLM provider (DeepSeek-v3p2, Kimi-k2 fallback) + autonomy LLM | `FIREWORKS_API_KEY`, `FIREWORKS_MODEL` | No |
| **OpenAI** | Entity extraction, embeddings | `OPENAI_API_KEY` | No (for entity/embedding features) |
| **Anthropic** | Alternative LLM provider | `ANTHROPIC_API_KEY` | Yes |
| **DeepSeek** | Alternative LLM provider (direct API) | `DEEPSEEK_API_KEY` | Yes |
| **Google Gemini** | LLM access from code executor | `GOOGLE_GEMINI_API_KEY` | Yes |
| **Ollama** | Local model inference (dev) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Yes |
| **PostgreSQL** | Messages, users, sessions, facts, episodes | `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | No |
| **Redis** | Celery broker, result backend, pub/sub | `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | No |
| **Neo4j (Graphiti)** | Temporal knowledge graph memory | `NEO4J_GRAPHITI_URI`, `NEO4J_GRAPHITI_USER`, `NEO4J_GRAPHITI_PASSWORD` | No (for knowledge graph) |
| **Telegram** | Proactive companion messaging | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Yes |
| **Telegram Ops** | System alerts and admin commands | `TELEGRAM_OPS_BOT_TOKEN`, `TELEGRAM_OPS_CHAT_ID` | Yes |
| **Google OAuth** | Calendar and Drive integration | `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | Yes |
| **Gmail** | Inbox monitoring (via `gmail_check_task`) | Google OAuth credentials | Yes |
| **Deepgram** | Voice transcription and TTS | `DEEPGRAM_API_KEY`, `DEEPGRAM_VOICE_ID`, `DEEPGRAM_TTS_MODEL` | Yes |
| **RunComfy** | Image generation (ComfyUI cloud) | `RUNCOMFY_API_KEY`, `RUNCOMFY_PERSONAL_DEPLOYMENT_ID`, `RUNCOMFY_GENERAL_DEPLOYMENT_ID`, `RUNCOMFY_SKETCH_DEPLOYMENT_ID` | Yes |
| **Cloudinary** | Image hosting/backup | `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` | Yes |
| **Tavily** | Web search (used by code executor) | `TAVILY_API_KEY` | Yes |
| **Sentry** | Error tracking and monitoring | `SENTRY_DSN` | Yes |
| **Zep** | External memory service | `ZEP_API_KEY`, `ZEP_PROJECT_ID`, `ZEP_BASE_URL` | Yes |
| **Statsig** | Feature flags | `STATSIG_SERVER_KEY` | Yes |
| **n8n** | Workflow automation | `N8N_HOST`, `N8N_API_KEY`, `N8N_USER`, `N8N_PASSWORD` | Yes |
