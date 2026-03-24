# Conversation Pipeline

[Back to Architecture Index](../ARCHITECTURE.md)

---

**Source**: `src/core/conversation/pipeline.py`

The pipeline is the central nervous system — it orchestrates how user messages flow through the system and produce responses. Every message passes through 9 stages.

## Pipeline Stages

```
[1] Context Building          — 20+ parallel data sources via ContextBuilder
[1.4] Clear Addressed Thoughts — Remove queued thoughts the user just addressed
[1.5] Memory Validation       — Detect memory queries, retrieve verified records
[1.6+1.7] Message Analysis    — Merged mode detection + inner monologue (1 LLM call)
[1.8] Departure Detection     — Track if user announced they're leaving
[1.9] Activity Tracking       — Track if user is briefly busy (making coffee, etc.)
[2] Prompt Assembly           — Identity + reference data + instructions
[3] LLM Generation            — Multi-turn chat format with regeneration loop
[4] Post-Generation Validation — Claim extraction + contradiction checking
[4b] Quality Critique         — Score response, regenerate if below threshold
[5] Image Intent Detection    — Trigger image generation if appropriate
```

## Stage Details

### Stage 1: Context Building

**Source**: `src/core/conversation/context_builder.py`

Uses a `ThreadPoolExecutor` (6 workers) to fetch 20+ context sources in parallel, reducing wall-clock time from 2-5s (sequential) to 0.5-2s.

**`ConversationContext` dataclass fields** (25+):

| Field | Source | Description |
|-------|--------|-------------|
| `entity_profiles` | YAML files (git-versioned) | Ground truth about known entities |
| `personality` | `personality.md` + evolved traits | Companion's voice and character |
| `memories` | pgvector semantic search | Similar past messages |
| `conversation_history` | PostgreSQL messages | Recent message history |
| `continuity_context` | Episode summaries | What happened recently |
| `biographies` | Synthesized paragraphs | Theme-grouped knowledge about people |
| `synthesized_events` | Event narratives | Ongoing life events with timelines |
| `episode_context` | Episode records | Current/recent conversation episodes |
| `core_memory` | `COMPANION_MEMORY.md` | Companion's personal narrative |
| `observations_context` | Compressed observations | Mastra-style conversation compression |
| `reflections_context` | Companion journal | Daily/weekly reflections |
| `opinions_context` | Opinion store | Companion's formed opinions |
| `curiosity_context` | Curiosity threads | Topics companion wants to follow up on |
| `goals_context` | Active goals | Companion's personal goals |
| `internal_state` | Energy, mood, needs | Companion's subjective inner world |
| `scene_state` | Scene tracking | Physical location, activity, presence |
| `values_context` | Inferred values | Companion's hidden values and desires |
| `temporal_context` | Time awareness | Current time, calendar, user schedule |
| `graphiti_context` | Neo4j knowledge graph | Entity relationships |
| `relationship_dynamics` | Relationship store | Typed relationships between entities |
| `relationship_evaluation` | Private self-assessment | Companion's view of the relationship |
| `fertility_context` | Cycle tracking | If applicable to companion persona |

If any source fails, the pipeline continues with the remaining sources.

### Stage 1.5: Memory Validation

**Source**: `src/core/memory_validation_agent.py`

Pre-generation anti-confabulation gate. Classifies the user's message as a memory query (`specific_event`, `emotional_vague`, `factual`, or `none`), retrieves verified records, and injects instructions:

| Query Type | No Records Found | Records Found |
|------------|------------------|---------------|
| `specific_event` | "Say you don't remember, don't invent" | "Reference ONLY these details" |
| `emotional_vague` | "Say you don't remember, don't invent" | "Choose from these verified memories" |
| `factual` | "Use these facts or say you don't know" | "Use these facts" |

### Stages 1.6+1.7: Message Analysis

**Source**: `src/core/conversation/message_analyzer.py`

A merged LLM call that replaces two separate calls (mode detection + inner monologue), saving 100-300ms latency.

**Single LLM call** (temp=0.3, max_tokens=500, 5s timeout) produces:

```
MODE: <conversation_mode> <confidence>
EMOTIONAL_READ: What's beneath the surface of the user's message
BOUNDARY_CHECK: Does this touch a value or boundary?
STRATEGY: How to respond
WEAVE: Curiosity threads to naturally include (or "nothing")
REACTION: Honest gut reaction
DEPARTURE: yes/no (is user leaving?)
ACTIVITY: description | duration_minutes (or "none")
```

**Conversation modes**: `casual`, `emotional_support`, `intellectual`, `playful`, `intimate`, `conflict`, `planning`, `catching_up`, `deep_conversation`, `creative`.

Fallback: If the merged analyzer fails, falls back to separate `ModeDetector` and `InnerMonologueGenerator` calls.

### Stage 2: Prompt Assembly

**Source**: `pipeline.py:_assemble_prompt()`

Uses a deliberate **attention pattern** optimized for LLM positional bias:

```
[TOP — highest positional attention]
  IDENTITY: "You are [companion]" + boundaries + personality

[MIDDLE — steady attention]
  REFERENCE DATA: Entity profiles, memories, personality, relationships,
  scene state, internal state, values, activities, temporal context,
  knowledge graph, events, episodes, observations, opinions, curiosity,
  biographies, fertility context

[BOTTOM — highest recency attention]
  INSTRUCTIONS: Continuity context, memory checkpoint, conversation mode
  hints, inner monologue injection, cognitive prompt, mode-specific
  instructions (proactive/texting/voice)

[VERY END — recency bias enforcement]
  FINAL REMINDER: Current time, identity, reference guidelines, scene
```

### Stage 3: LLM Generation

**Source**: `pipeline.py:_call_llm()`

- **Format**: Multi-turn chat (system + conversation history turns + user message)
- **Provider**: `ResilientProviderChain` (Fireworks -> Anthropic failover)
- **Temperature**: Dynamic based on emotional context:
  - Base: 0.7 (natural, varied)
  - Elevated to 0.8 for genuine emotional moments (vulnerability, grief, deep connection)
  - Physical intimacy stays 0.7

If `CODE_EXECUTION_ENABLED`, uses `_call_llm_with_tools()` — an agentic loop (up to 5 iterations) where the LLM can call `execute_code` to use tool modules (web search, browser, Google APIs, etc.).

### Stage 4: Post-Generation Validation

**Source**: `src/core/message_validator_agent.py`

1. `ClaimExtractor` (`src/core/claim_extractor.py`) — Extracts verifiable claims from the response
   - Pre-filter: keyword check for claim indicators ("called", "went", "works", "drinks", "remember", etc.)
   - Claims typed as: `memory`, `fact`, `action`, `embellishment`
   - Severity: `high` (preferences, family, employment), `medium` (events, timing), `low` (minor details)
   - Preserves negation: "No coffee for you" becomes "you do not drink coffee"

2. `ClaimVerifier` — Checks claims against entity profiles + stored corrections

3. **Regeneration**: High-severity contradictions trigger prompt injection with correction hints, then re-call LLM (up to 2 regenerations)

### Stage 4b: Quality Critique

**Source**: `src/core/conversation/response_critic.py`

**Feature flag**: `COMPANION_RESPONSE_CRITIC_ENABLED` (default: false)

Scores response on 5 dimensions via LLM (temp=0.2, max_tokens=150, 3s timeout):

| Dimension | Check |
|-----------|-------|
| SCORE | 1-10 overall quality |
| GENERIC | Does it sound like a chatbot? |
| ADDRESSES | Does it respond to the user's message? |
| REGISTER | Is emotional tone appropriate? |
| REPETITIVE | Does it reuse phrases from recent messages? |

Regeneration triggered if score < `COMPANION_QUALITY_THRESHOLD` (default: 4). Returns passing defaults (score=7) on failure to avoid blocking.

### Stage 5: Image Intent Detection

**Source**: `src/core/image_intent_detector.py`

Two-pass system:
1. Fast LLM classifies intent type: `NONE`, `SELFIE`, `OBSERVING`, `SHOW_OUTFIT`, `INTIMATE`, `GENERAL`
2. Secondary guardrails in pipeline:
   - Intimate: requires confidence > 0.95 AND intimate tone in conversation
   - Couple scenes: blocks "together", "both of us", "with [user]"
   - Rate limiting: 5 images/day, 1 sketch/day (Redis counters)

## Pipeline Result

```python
@dataclass
class PipelineResult:
    response: str                       # Generated response text
    context_used: ConversationContext    # All context that was built
    processing_time_ms: int             # Wall-clock processing time
    llm_model: str                      # Which model was used
    success: bool                       # Whether pipeline succeeded
    image_task_id: Optional[str]        # If image generation triggered
    conversation_mode: Optional[str]    # Detected mode
    inner_monologue: Optional[str]      # Private pre-response reasoning
    quality_score: Optional[float]      # Post-response quality score
```

## Pipeline Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `COMPANION_MESSAGE_ANALYZER_ENABLED` | `true` | Merged mode+monologue analysis |
| `COMPANION_MODE_DETECTION_ENABLED` | `true` | Conversation mode detection (fallback) |
| `COMPANION_INNER_MONOLOGUE_ENABLED` | `true` | Inner monologue generation (fallback) |
| `COMPANION_RESPONSE_CRITIC_ENABLED` | `false` | Post-generation quality critique |
| `CODE_EXECUTION_ENABLED` | `false` | Agentic tool execution loop |
