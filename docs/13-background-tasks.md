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
- **Task files**: 40 in `src/tasks/`

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
| `episodic_embedding_task` | Async | Generate and store message embedding (pgvector) |
| `scene_extraction_task` | Async | Extract physical scene state |
| `curiosity_extraction_task` | Async | Extract curiosity triggers from the conversation |
| `interaction_outcome_task` | Async | Track how user engaged with companion's message |
| `conversation_batch_task` | Async (debounced) | Batched per-conversation tasks (episode, event, outcome) |
| `relationship_dynamics_task` | Async | Update Gottman-informed relationship state |

### Nightly Tasks

| Task | Schedule | What It Does |
|------|----------|-------------|
| `daily_summary_task` | 12:05 AM PT (APScheduler) | Synthesize yesterday's messages into journal |
| `reflection_task` | 12:30 AM PT | Analyze emotional arc, extract insights |
| `biography_refresh_task` | Every 24 hours | Re-synthesize biography paragraphs from facts |
| `episode_learning_task` | 4:00 AM PT | Extract lessons and patterns from recent episodes |
| `event_consolidation_task` | 3:00 AM PT | Consolidate synthesized events |
| `episodic_consolidation_task` | 2:00 AM PT | Promote recurring episodes to semantic facts |
| `memory_gap_analysis_task` | 6:00 AM PT | Detect knowledge gaps, generate curiosity threads |
| `entity_profile_fact_validator` | 3:30 AM PT | Archive facts that contradict entity profiles |
| `sleep_time_consolidation_task` | Every 3 hours | Merge, promote, and detect stale facts |
| `core_memory_task` | On-demand | Refresh `COMPANION_MEMORY.md` narrative |

### Weekly/Periodic Tasks

| Task | Schedule | What It Does |
|------|----------|-------------|
| `goal_planning_task` | Every 6h (6:30AM, 12:30PM, 6:30PM, 12:30AM) | Decompose unplanned goals into actionable steps |
| `goal_signal_task` | 9 AM and 9 PM | Form new goals from curiosities, opinions, research findings |
| `value_inference_task` | Weekly (Wednesday 3:30 AM) + monthly full history | Extract hidden values from companion messages |
| `weekly_reflection_task` | Sunday 3:00 AM | Analyze patterns across the week |
| `monthly_reflection_task` | 1st of month, 4:00 AM | Relationship evolution analysis |
| `relationship_evaluation_task` | Tuesday 2:30 AM | Private self-assessment of relationship |
| `observation_task` | When thresholds met | Compress conversations (50 msgs or 30K tokens) |
| `observation_scheduler` | Daily | Consolidate old observations into reflections |
| `memory_pruning_task` | Sunday 5:30 AM | Archive stale low-importance facts |
| `embedding_pruning_task` | Sunday 6:00 AM | Archive stale pgvector message embeddings |
| `graphiti_pruning_task` | 1st of month, 5:00 AM | Archive old Neo4j episodes and expired edges |
| `calendar_schedule_task` | 5:30 AM PT daily | Generate companion's daily calendar plan |
| `reminder_check_task` | Every 2h (8 AM–8 PM) | Surface due reminders as queued_thoughts |
| `gmail_check_task` | Every 30 min (8 AM–9 PM) | Surface notable emails as queued_thoughts |
| `opinion_formation_task` | 1:00 AM PT daily | Form opinions from reflections; Sunday also decays weak ones |

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
