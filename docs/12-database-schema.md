# Database Schema

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Design Principles

### Per-User Schema Isolation

Each user gets their own PostgreSQL schema (e.g., `user_james`, `user_alex`). Shared tables (auth, user registry) live in the `public` schema. Data isolation is physical — no `WHERE user_email = X` filtering needed for user-scoped tables. Within a user's schema, companion data is filtered by `companion_id` (a user can own multiple companions).

### Table Name Constants

All table names are defined in `src/database/tables.py`. Never hardcode table names in SQL queries — always import from `tables.py` and use f-strings: `f"SELECT * FROM {T.MESSAGES} WHERE ..."`.

### Provider-Agnostic PostgreSQL

The connection layer (`src/database/connection.py`) must stay provider-agnostic. Self-hosted PostgreSQL is the default, but the system should work with any managed provider (Neon, Supabase, Railway, etc.) by changing env vars. Do not use self-hosted-only features (direct filesystem access to PG data dirs, etc.). Stick to standard SQL + pgvector.

---

## PostgreSQL Tables

### Core Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `messages` | id, user_email, role, content, embedding_vec, created_at | Conversation history with pgvector embeddings |
| `facts` | subject, predicate, object, confidence, importance, embedding, search_vector, mention_count, archived_at | SPO fact storage with hybrid search |
| `fact_links` | source_fact_id, target_fact_id, link_type, strength | Associative links between facts |
| `detected_contradictions` | subject, old_value, new_value, detection_method, string_similarity | Contradiction audit trail |
| `corrections` | subject, wrong_claim, correct_info, correction_type, importance | User-reported corrections |
| `relationships` | source_entity, relationship_type, target_entity, confidence, valid_from, valid_until | Typed entity relationships |

### Episode & Observation Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `episodes` | episode_id, user_email, topic, emotional_state, satisfaction, approach_summary | Conversation episode records |
| `episode_messages` | episode_id, message_id, turn_number | Links messages to episodes |
| `observations` | observation_date, content, compression_ratio, topics, emotional_tone | Compressed conversation observations |
| `observation_reflections` | period_type, period_start, period_end, content, themes | Weekly/monthly consolidations |

### Autonomy Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `companion_goals` | goal, motivation, category, progress, status, goal_mode, energy_cost | Companion's personal goals |
| `companion_goal_steps` | goal_id, description, action_type, parameters, status, depends_on, wait_for | Decomposed action steps |
| `companion_opinions` | user_email, topic, text, confidence, evidence_count, category | Formed opinions |
| `companion_journal` | user_email, date, content | Daily reflection entries |
| `daily_summaries` | user_email, date, content | Daily journal synthesis |
| `internal_state` | user_email, state (JSONB) | Energy, mood, needs, queued thoughts |

### Synthesis Tables

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `synthesized_paragraphs` | theme, subject, content, fact_ids, base_importance, embedding_vec | Theme-grouped biography paragraphs |
| `synthesized_events` | event_type, subject, title, timeline (JSON), narrative, outcome, impact | Life event timelines and narratives |

### Hidden Tables (Privacy)

| Table | Purpose |
|-------|---------|
| `_companion_internal_state_v` | Inferred values (SHA-256 hashed keys) |
| `_companion_relationship_eval` | Private relationship self-assessment |

## External Storage

| Storage | Location | Purpose |
|---------|----------|---------|
| Entity profiles | `/app/data/entity_profiles/*.yaml` | Ground truth (git-versioned) |
| Core memory | `/app/data/COMPANION_MEMORY.md` | Personal narrative |
| Activities | `/app/data/companion_activities.json` | Background life |
| Reach-out pressure | `/app/data/reach_out_pressure.json` | Proactive messaging state |
| User state | `/app/data/james_state.json` | Awake/asleep tracking |
| Daily logs | `/app/data/daily_logs/YYYY-MM-DD.md` | Daily summaries |
