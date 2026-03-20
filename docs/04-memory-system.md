# Memory System

[Back to Architecture Index](../ARCHITECTURE.md)

---

The memory system comprises 24 modules organized into 5 categories. Together they form a comprehensive memory architecture that prevents hallucination, enables rich context retrieval, and maintains temporal awareness.

## 4.1 Core Memory Systems

### Core Memory (`memory/core_memory.py`)

The companion's personal narrative about the user, written in first person. A 500-800 word document stored at `/app/data/COMPANION_MEMORY.md` and cached in-memory with a configurable TTL (default 3600s).

Synthesized by LLM from entity profiles, facts, biographies, and relationship state. Refreshed weekly via `biography_refresh_task` or manually.

### Fact Store (`memory/fact_store.py`)

Subject-Predicate-Object triple storage with hybrid search capabilities.

**Storage**: Each fact stores `(subject, predicate, object)` with confidence, importance, temporal metadata, embedding vector, and full-text search vector.

**Deduplication**: Three-tier system:
1. Exact match on (subject, predicate, object)
2. Semantic pattern matching
3. SequenceMatcher similarity (0.85 threshold)

**Contradiction handling**: Detects contradictions via string similarity (<0.4) or embedding cosine (<0.7), archives the old fact with reason.

**Search modes**:
- `search_facts_hybrid()` — BM25 + pgvector weighted combination (default: 0.6 vector, 0.4 text)
- `search_with_spreading_activation()` — Brain-like retrieval through `fact_links` and relationship bridges (see Fact Network below)

**Reinforcement**: `reinforce_fact()` increments mention_count, boosts confidence by +0.02 (max 0.99).

### Episodic Memory (`memory/episodic_memory.py`)

Stores 1536-dim OpenAI embeddings for each message in pgvector. Prefixes messages with sender name for contextual embedding. Used by semantic search at query time.

### Episodic Episodes (`memory/episodic_episodes.py`)

Groups sequential messages into coherent episodes with topic, emotional state, and satisfaction scoring.

**Boundary detection**: Time gaps (4h default) and explicit topic shifts.

**Episode fields**: `topic`, `trigger`, `emotional_state`, `resolution`, `satisfaction` (0.0-1.0, LLM-scored), `approach_summary`, `message_count`.

### Observational Memory (`memory/observational_memory.py`)

Mastra-inspired conversation compression pipeline achieving 5-10x compression.

**Two-stage process**:
1. **Observer**: When thresholds are met (50 messages OR 30K tokens), groups messages by conversation gaps (4h), compresses each group into a dated narrative observation via `gpt-4o-mini` (temp=0.1, max_tokens=1500). Prompt: "preserve facts, decisions, emotional arc; 20-30% original length"
2. **Reflector**: When >=3 observations are >7 days old, consolidates into weekly reflection summaries via `gpt-4o-mini` (temp=0.3, max_tokens=800)

**Feature flag**: `OBSERVATIONAL_MEMORY_ENABLED` (default: false)

## 4.2 Relationship & Entity Systems

### Relationship Store (`memory/relationship_store.py`)

Typed relationship storage with 21 relationship types:

```
PARENT_OF, CHILD_OF, SIBLING_OF, MARRIED_TO, DIVORCED_FROM, SEPARATED_FROM,
PARTNER_OF, DATING, EX_PARTNER_OF, WORKS_AT, COWORKER_OF, MANAGER_OF,
REPORTS_TO, FRIEND_OF, ACQUAINTANCE_OF, NEIGHBOR_OF, OWNS_PET, PET_OF,
LIVES_WITH, ROOMMATE_OF, KNOWS
```

Supports bidirectional queries, confidence scoring, temporal validity (`valid_from`/`valid_until`), and exclusive relationship violation detection (e.g., multiple marriages).

### Relationship Extractor (`memory/relationship_extractor.py`)

Scans messages for explicit relationship statements ("my son", "married to", "works at"). Uses fast keyword pre-filter before LLM extraction (Fireworks Kimi K2, temp=0.1).

### Entity Profile Manager (`memory/entity_profile_manager.py`)

YAML-based ground truth profiles stored in `/app/data/entity_profiles/`. Every write auto-commits to git with a descriptive message, providing a full audit trail.

Profile sections: `name`, `role`, `summary`, `family`, `employment`, `preferences`, `relationships`, `hard_facts`, `aliases`.

Provides `get_grounding_context()` — a "KNOWN ENTITIES" block injected into the LLM prompt to prevent hallucination about known people and facts.

## 4.3 Biography & Event Synthesis

### Synthesized Biographies (`memory/synthesized_biographies.py`)

Groups facts by theme (work, family, preferences, crisis, etc.), synthesizes each theme into a 2-4 sentence paragraph via LLM, stores with temporal decay.

**Decay half-lives by category**:

| Category | Half-Life | Example |
|----------|-----------|---------|
| Crisis | 7 days | Medical emergency, breakup |
| Event | 14 days | Party, trip, interview |
| Plan | 21 days | Upcoming move, job application |
| Relationship/Work | 30 days | New coworker, promotion |
| Detail | 60 days | Casual observation |
| Preference | 90 days | Favorite food, hobby |
| Identity | 365 days | Name, age, core traits |

**Effective importance** = base_importance * temporal_decay_factor

### Synthesized Events (`memory/synthesized_events.py`)

Detects significant life events (crisis, career, milestone, health, school), groups related facts into chronological timelines, synthesizes each into a narrative paragraph.

Each event type has an importance boost (crisis: +3, career/health: +2, milestone/relationship: +1) and decay window (crisis: 14d, career: 30d, milestone: 60d).

## 4.4 Neural Search & Linking

### Fact Network (`memory/fact_network.py`)

Directed graph of typed links between facts enabling brain-like associative retrieval.

**Link types**: `EXPLAINS`, `CAUSES`, `CONTRADICTS`, `TEMPORAL_BEFORE`, `SAME_EVENT`, `SIMILAR`, `SUPPORTS`

**Spreading activation algorithm**:
1. Seed facts initialized at 1.0 activation
2. Spread through `fact_links`: `spread = source_activation * link_strength * decay_factor^depth`
3. Relationship bridges: facts about entity B activate via entity relationships (parent_of, etc.)
4. Accumulate activation per fact (max, not sum)
5. Return facts with activation >= `min_activation` (0.2), sorted descending

Parameters: `max_depth` (1-3), `decay_factor` (0.6), `bridge_decay` (0.5), `relationship_bridge` (bool).

**Link detection**: LLM-based (Fireworks Kimi K2, optional via `COMPANION_FACT_LINKS_LLM`) or heuristic fallback.

### Semantic Search (`memory/semantic_search.py`)

Two-stage retrieval over message embeddings:
1. pgvector cosine similarity (fast, gets 3x candidates)
2. Fireworks Qwen3 reranking (reorders by true relevance, optional via `ENABLE_RERANKING`)

### Embeddings Service (`memory/embeddings.py`)

Centralized wrapper around OpenAI `text-embedding-3-small` (1536 dimensions, $0.02/1M tokens). Supports single and batch embedding generation (batch_size=100).

## 4.5 Supporting Memory Systems

### Confidence Decay (`memory/confidence_decay.py`)

Computes `effective_confidence = base * recency_factor * mention_boost`:
- `recency_factor`: Linear decay from 1.0 to 0.3 over 90 days
- `mention_boost`: 1.0 + (mention_count - 1) * 0.1, capped at 1.5
- Result clamped to [0.1, 0.99]

Provides hedging qualifiers: >0.8 = none, 0.6-0.8 = "I think", 0.4-0.6 = "I'm not sure but", <0.4 = "I might be wrong but"

### Importance Scorer (`memory/importance_scorer.py`)

LLM-scored importance (1-10) via Fireworks Kimi K2:
- 9-10: Core identity moments (trauma, major life changes)
- 7-8: Significant milestones (moves, job changes)
- 5-6: Notable preferences, habits, recurring patterns
- 3-4: Minor details, casual observations
- 1-2: Mundane daily activities

### Other Supporting Modules

| Module | Purpose |
|--------|---------|
| `companion_journal.py` | Database storage for daily reflection entries |
| `fact_approval.py` | Approval queue for sensitive facts (medical, legal, relationship) |
| `graphiti_search.py` | Integration with Graphiti knowledge graph API |
| `fireworks_reranker.py` | Wrapper for Fireworks Qwen3 reranker |
| `message_condenser.py` | Token-efficient message summarization |
| `retrieval_agent.py` | Multi-modal retrieval coordinator (facts + semantics + episodes) |
| `temporal_context.py` | Time-aware fact/event filtering |
| `benchmark_evaluator.py` | Memory quality evaluation |

## Memory Data Flow

**During message handling** (write path):
```
Message received
  -> episodic_memory.embed_and_store_message()     # Embedding stored
  -> episode_tracking_task                          # Episode boundary + linking
  -> fact_extraction_task                           # SPO extraction + approval routing
  -> relationship_extraction_task                   # Typed relationship extraction
  -> observation_task (background)                  # Threshold-based compression
  -> daily_summary_task (nightly)                   # Daily journal synthesis
  -> reflection_task (nightly)                      # Emotional arc + insights
```

**During response generation** (read path):
```
context_builder.build()
  -> core_memory.get_formatted_for_prompt()         # Personal narrative
  -> entity_profile_manager.get_grounding_context() # Known entities
  -> synthesized_biographies.get_biography_context() # Theme paragraphs
  -> synthesized_events.get_recent_events()         # Ongoing events
  -> semantic_search.search_memory()                # Similar past messages
  -> fact_store.search_with_spreading_activation()  # Relevant facts
  -> episodic_episodes.format_episodes_for_prompt() # Similar episodes
  -> observational_memory.get_relevant_observations() # Compressed history
```
