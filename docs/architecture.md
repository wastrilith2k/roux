# Architecture

## System Overview

```
+-----------------------------------------------------------------------+
|                          CLIENT LAYER                                  |
|   Browser (Socket.IO)  |  Telegram Bot  |  Voice Client               |
+---------------------------+-------+-----------------------------------+
                            |       |
+---------------------------v-------v-----------------------------------+
|                        WEB LAYER                                       |
|   Flask + Socket.IO (port 5000)                                        |
|   routes/chat_routes.py  |  routes/api_routes.py                       |
+---------------------------+-------+-----------------------------------+
                            |       |
+---------------------------v-------v-----------------------------------+
|                   REQUEST HANDLER LAYER                                 |
|   handlers/message_handler.py  ->  MessageProcessor                    |
+-------------------------------+---------------------------------------+
                                |
+-------------------------------v---------------------------------------+
|                  RESPONSE GENERATION                                   |
|   core/conversation/pipeline.py  ->  ConversationPipeline              |
|                                                                        |
|   Stages:                                                              |
|     ContextBuilder -----> MessageAnalyzer -----> Prompt Assembly        |
|     (20+ parallel          (mode detection,      (modular .md           |
|      sources via            inner monologue)       prompts)             |
|      ThreadPoolExecutor)          |                    |               |
|                                   v                    v               |
|                            LLM Call (ResilientProviderChain)           |
|                                   |                                    |
|                    MessageValidator -> ResponseCritic                   |
|                                   |                                    |
|                         ImageIntentDetector                            |
|                                   |                                    |
|                            PipelineResult                              |
+---+-----------+-----------+-------+----------+------------------------+
    |           |           |                  |
+---v---+  +---v---+  +---v-----------+  +---v-----------------------+
| Post- |  | DATA  |  |  BACKGROUND   |  | EXTERNAL INTEGRATIONS     |
| grSQL |  | LAYER |  |  SERVICES     |  |                           |
| +pg-  |  | Neo4j |  | AlwaysOn-     |  | LLM Providers (Fireworks, |
| vector|  | (Gra- |  |  Service      |  |   DeepSeek, OpenAI,       |
|       |  |  phiti)|  |  (thread)     |  |   Anthropic)              |
| Redis |  |       |  | Celery        |  | Google Workspace          |
| (bro- |  |       |  |  Workers      |  | Telegram                  |
| ker + |  |       |  |  (35+ tasks)  |  | Deepgram / ElevenLabs     |
| pub/  |  |       |  | APScheduler   |  | RunComfy / Cloudinary     |
| sub)  |  |       |  |               |  | Tavily / Sentry           |
+-------+  +-------+  +---------------+  +---------------------------+
```

---

## Module Map

| Module | Purpose | Key Exports | Type |
|--------|---------|-------------|------|
| `agents/` | Temporal reasoning | `TemporalContextAgent` | Library |
| `autonomy/` | Proactive messaging: reach-out, interjection, goals, values, opinions | `AlwaysOnService`, `ReachOutEngine`, `InterjectionEngine` | Service + Tasks |
| `core/` | Core reasoning and conversation pipeline | `ConversationPipeline`, `ContextBuilder` | Pipeline |
| `core/conversation/` | Modular conversation components | `Pipeline`, `ContextBuilder`, `InnerMonologue`, `MessageAnalyzer`, `ModeDetector`, `ResponseCritic` | Library |
| `database/` | PostgreSQL access layer | `CompanionDB` (singleton) | Library |
| `diagnostics/` | System health monitoring | `SystemHealth` | Library |
| `handlers/` | Message entry points | `MessageProcessor` | Library |
| `integrations/` | External service APIs | `GoogleService`, `CalendarService` | Library |
| `llm/` | LLM provider abstraction with failover | `ResilientProviderChain` | Library |
| `memory/` | Knowledge extraction and retrieval (facts, episodes, graphs, embeddings, biographies, observations) | `FactStore`, `EpisodicMemory`, `GraphitiSearch`, `SemanticSearch` | Library |
| `middleware/` | Flask auth and error handling | Auth decorators | Library |
| `ops/` | Admin operations and Telegram ops bot | `TelegramOpsBot` | Service |
| `routes/` | HTTP/WebSocket endpoints | Flask blueprints | Web API |
| `scheduling/` | Task scheduling (APScheduler) | `UnifiedScheduler`, `CompanionSchedule` | Service |
| `services/` | Service utilities | `CostTracker` | Library |
| `tasks/` | 35+ Celery async tasks | Fact extraction, memory synthesis, etc. | Async Tasks |
| `user/` | User profile management | `UserProfile` | Library |
| `utils/` | Utilities (timezone, paths, error tracking) | -- | Library |
| `voice/` | STT (Deepgram) and TTS (ElevenLabs/Edge) | `VoiceService` | Library |

All modules live under `src/`.

---

## Message Flow

```
User Input
    |
    v
1.  WebSocket event --> chat_routes.py --> authenticate session
    |
    v
2.  MessageProcessor.process()
    |
    v
3.  ConversationPipeline (multi-stage)
    |
    +-- 3a. ContextBuilder
    |        Spawns 20+ parallel fetches via ThreadPoolExecutor:
    |        facts, episodes, graph, semantic memory, core memory,
    |        biographies, observations, internal state, calendar,
    |        relationship dynamics, curiosity, opinions, projects, ...
    |
    +-- 3b. MessageAnalyzer
    |        Mode detection (chat / voice / deep-talk / ...)
    |        Inner monologue (WEAVE: world-model, emotions, associations,
    |        values, experiences including shared memories)
    |
    +-- 3c. Prompt Assembly
    |        Combines persona, context, conversation history,
    |        modular prompts from prompts/core/*.md
    |
    +-- 3d. LLM Call via ResilientProviderChain
    |        Fireworks -> DeepSeek -> OpenAI -> Anthropic (failover)
    |
    +-- 3e. MessageValidator
    |        Structural checks on response
    |
    +-- 3f. ResponseCritic
    |        Quality gate: tone, accuracy, persona fidelity
    |
    +-- 3g. ImageIntentDetector
    |        Detects if response warrants image generation
    |
    +-- 3h. PipelineResult returned
    |
    v
4.  Response stored in PostgreSQL (messages table)
    Response sent to client via WebSocket
    |
    v
5.  Async post-processing (10+ Celery tasks dispatched):
    - Fact extraction (S-P-O triples)
    - Embedding generation (pgvector)
    - Relationship/entity extraction
    - Scene analysis
    - Internal state update
    - Episodic memory formation
    - Graphiti knowledge graph update
    - Observation compression
    - Curiosity queue update
    - Tone drift detection
```

---

## Proactive Message Flow

The `AlwaysOnService` runs as a background thread with two loops:

```
AlwaysOnService (daemon thread)
    |
    +-- OFFLINE LOOP (user not on WebSocket)
    |   |
    |   +-- Interval: 3-25 minutes (varies by time of day, context)
    |   +-- ReachOutEngine evaluates:
    |   |     - Time since last interaction
    |   |     - Internal state (mood, energy, needs)
    |   |     - Pending curiosity topics
    |   |     - Calendar awareness
    |   |     - Relationship dynamics
    |   +-- If decision = reach out:
    |         Generate message via ConversationPipeline
    |         Send via Telegram API
    |
    +-- WEBSOCKET LOOP (user connected)
    |   |
    |   +-- Interval: 2-5 minutes
    |   +-- InterjectionEngine evaluates:
    |   |     - Conversation flow and silence duration
    |   |     - Internal state
    |   |     - Topic relevance
    |   +-- If decision = interject:
    |         Generate message via ConversationPipeline
    |         Publish via Redis pub/sub
    |         Deliver via Socket.IO
    |
    +-- GOAL ACTIONS
        |
        +-- Budget-gated autonomous research/actions
        +-- Web search, synthesis, and reporting
```

---

## Memory Subsystems

| Subsystem | Backend | Purpose | Retrieval Method |
|-----------|---------|---------|-----------------|
| Fact Store | PostgreSQL | S-P-O facts with confidence decay | Keyword match + spreading activation |
| Episodic Memory | PostgreSQL | Episodes grouped by topic and time window | pgvector semantic search |
| Graphiti | Neo4j | Temporal knowledge graph with entity relationships | Graph queries + semantic search |
| Semantic Search | pgvector | Embedding-based retrieval across memory types | Cosine similarity |
| Core Memory | PostgreSQL | Curated narrative (MEMORY.md style) | Full-text retrieval |
| Synthesized Biographies | PostgreSQL | Character arcs with temporal decay weighting | Direct fetch + decay scoring |
| Synthesized Events | PostgreSQL | Crisis, career, and milestone narratives | Topic/type fetch |
| Observations | PostgreSQL | Compressed conversation observations | Latest N retrieval |
| Curiosity Queue | PostgreSQL | Unresolved follow-up topics | Urgency/recency sort |
| Opinion Store | PostgreSQL | Companion's formed views and stances | Subject lookup |
| Internal State | PostgreSQL | Mood, energy, needs tracking | Latest per category |

---

## External Integrations

| Provider | Purpose | Notes |
|----------|---------|-------|
| Fireworks | Primary LLM (DeepSeek-V3p2) | First in failover chain |
| DeepSeek | LLM failover #2 | Direct API |
| OpenAI | LLM failover #3 | GPT models |
| Anthropic | LLM failover #4 | Claude models |
| PostgreSQL + pgvector | Core persistence + vector embeddings | All structured data |
| Neo4j | Knowledge graph | Via Graphiti library |
| Redis | Celery broker + pub/sub messaging | Interjections, image results |
| Telegram | Autonomous outbound messaging + ops bot | Bot API |
| Google Workspace | Gmail and Calendar access | OAuth2 |
| Deepgram | Speech-to-text | Streaming + batch |
| ElevenLabs | Text-to-speech (primary) | Voice cloning |
| Edge TTS | Text-to-speech (fallback) | Microsoft Edge voices |
| RunComfy | Image generation | ComfyUI workflows |
| Cloudinary | Image CDN and storage | Upload + transform |
| Tavily | Web search | For autonomous research |
| Sentry | Error tracking and alerting | SDK integration |

---

## Database Schema

Key tables in PostgreSQL:

| Table | Purpose |
|-------|---------|
| `profiles` | User and companion profile data |
| `messages` | Full conversation history |
| `facts` | S-P-O knowledge triples with confidence scores |
| `episodes` | Episodic memory entries |
| `observations` | Compressed conversation observations |
| `relationships` | Entity relationship mappings |
| `internal_state` | Mood, energy, needs per category |
| `relationship_dynamics` | Evolving relationship metrics |
| `goals` | Autonomous goal tracking |
| `opinions` | Companion's formed opinions |
| `curiosity_queue` | Unresolved topics for follow-up |
| `core_memory` | Curated narrative memory blocks |
| `reflections` | Journal-style self-reflections |
| Embedding tables | pgvector columns for semantic search |

---

## Configuration

Configuration is resolved with the following priority: **env vars > YAML > defaults**.

| Source | Path | Purpose |
|--------|------|---------|
| Persona YAML | `data/persona.yaml` | Companion and user identity, pronouns, personality |
| Entity Profiles | `data/entity_profiles/*.yaml` | Known entity facts and metadata |
| Conversation Prompts | `prompts/core/*.md` | Modular prompt templates for pipeline stages |
| App Config | `src/config/app_config.py` | Application-level Python configuration |
| Persona Config | `src/config/persona_config.py` | Persona loading and access (singleton) |
| Environment | `.env` | API keys, credentials, connection strings (git-ignored) |
| Docker Compose | `docker-compose.yml` | Service orchestration (Postgres, Redis, Neo4j, workers) |

---

## Architectural Patterns

| Pattern | Where Used | Description |
|---------|-----------|-------------|
| Factory | `ResilientProviderChain` | Constructs LLM provider chain with ordered failover |
| Singleton | `CompanionDB`, `UnifiedScheduler`, `PersonaConfig` | Single shared instance across the application |
| Pipeline | `ConversationPipeline` | Multi-stage processing: context, analysis, generation, validation, critique |
| Thread Pool | `ContextBuilder` | 20+ context sources fetched in parallel via `ThreadPoolExecutor` |
| Pub/Sub | Redis channels | Interjection delivery and image generation results |
| Async Tasks | Celery workers | 35+ tasks for memory synthesis, extraction, embeddings, and maintenance |
| Graceful Degradation | Throughout | Missing context sources fail independently without blocking the pipeline; LLM providers fail over transparently |
