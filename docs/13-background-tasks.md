# Background Tasks & Scheduling

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Celery Configuration (`celery_app.py`)

- **Broker**: Redis (localhost:6379/0)
- **Serializer**: JSON
- **Timezone**: America/Los_Angeles
- **Time limits**: 10m max, 9m soft
- **Worker prefetch**: 1 task at a time
- **Auto-restart**: After 50 tasks (memory leak prevention)
- **Task files**: 34 in `src/tasks/`

## Task Catalog

### After Every Message

| Task | Trigger | What It Does |
|------|---------|-------------|
| `fact_extraction_task` | Async | Extract SPO triples, validate, route sensitive to approval |
| `episode_tracking_task` | Async | Detect boundaries, link messages, update topic/emotion |
| `relationship_extraction_task` | Async | Extract typed relationships from messages |
| `correction_task` | Async | Detect and store user corrections |
| `graphiti_extraction_task` | Async | Populate Neo4j knowledge graph |
| `internal_state_task` | Async | Update energy, mood, needs |
| Episodic embedding | Async | Generate and store message embedding |
| Scene extraction | Async | Extract physical scene state |

### Nightly Tasks

| Task | Schedule | What It Does |
|------|----------|-------------|
| `daily_summary_task` | 12:05 AM PT | Synthesize yesterday's messages into journal |
| `reflection_task` | 12:30 AM PT | Analyze emotional arc, extract insights |
| `biography_refresh_task` | Daily | Re-synthesize theme paragraphs from facts |

### Weekly/Periodic Tasks

| Task | Schedule | What It Does |
|------|----------|-------------|
| `goal_planning_task` | Weekly (Sunday) or after emotional days | Form new goals from reflections/signals |
| `value_inference_task` | Weekly | Extract hidden values from companion messages |
| `relationship_evaluation_task` | Weekly | Private self-assessment of relationship |
| `observation_task` | When thresholds met | Compress conversations (50 msgs or 30K tokens) |
| `observation_scheduler` | Daily | Consolidate old observations into reflections |
| `memory_pruning_task` | Periodic | Clean up low-value memories |

### On-Demand Tasks

| Task | Trigger | What It Does |
|------|---------|-------------|
| `image_generation_task` | Pipeline Stage 5 | Generate image via NanoBanana/RunComfy |
| `autonomous_action_task` | AlwaysOnService | Execute ready goal steps |
| `event_synthesis_task` | After fact extraction | Create life event timelines |

---

## Scheduling System

**Source**: `src/scheduling/`

### Unified Scheduler (`unified_scheduler.py`)

APScheduler-based system started at app startup. Manages cron and interval triggers for all periodic tasks.

### Companion Schedule (`companion_schedule.py`)

The companion's own daily schedule (wake/sleep, activity blocks). Generates activity milestones used by the interjection engine for natural conversation hooks.

### Calendar Integration

| Module | Purpose |
|--------|---------|
| `calendar_schedule_generator.py` | Analyzes user's Google Calendar, generates companion's proactive check schedule |
| `calendar_schedule_service.py` | Caches computed schedule, provides `is_user_available()`, `time_until_next_free_slot()` |

### User Context (`core/user_context.py`)

Hardcoded probable activity schedule:
- **Weekday**: Wake 7:30 -> job hunting 8:30-15:00 -> kids 15:00-21:00 -> evening -> sleep 23:00
- **Saturday**: Wake 8:00 -> kids all day -> sleep 22:00
- **Sunday**: Wake 8:30 -> relaxed day -> sleep 23:00

Manual wake override: Messages after 3 AM set user as awake. Autopilot toggle disables schedule assumptions during crisis mode.
