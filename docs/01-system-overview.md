# System Overview

[Back to Architecture Index](../ARCHITECTURE.md)

---

## High-Level Architecture

```
User Message (WebSocket / Telegram / CLI)
        |
   Flask-SocketIO (eventlet)
        |
   Message Handler
        |
   Multi-Stage Conversation Pipeline
   (complexity classification, context building, memory validation, message analysis,
    inner monologue, LLM generation, claim validation, response critique, image intent)
        |
   Response + Store to PostgreSQL
        |
   40 Celery Background Tasks
   (fact extraction, episode learning, opinion formation,
    relationship evaluation, biography synthesis, goal planning, etc.)
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Runtime** | Python 3.12, eventlet |
| **Web Framework** | Flask + Flask-SocketIO |
| **Task Queue** | Celery (Redis broker) |
| **Primary Database** | PostgreSQL + pgvector |
| **Knowledge Graph** | Neo4j 5.26 (Graphiti) |
| **Cache/Pub-Sub** | Redis |
| **Reverse Proxy** | Caddy (automatic SSL) |
| **Containers** | Docker Compose (9 services) |
| **LLM Providers** | Fireworks AI (primary), OpenAI, Anthropic, DeepSeek |
| **Embeddings** | OpenAI text-embedding-3-small (1536 dims) |
| **Voice** | Deepgram (STT), ElevenLabs/Edge TTS (TTS) |
| **Image Generation** | NanoBanana (Gemini), RunComfy (ComfyUI) |
| **Messaging** | Telegram Bot API |
| **Search** | Tavily API |
| **Browser** | Playwright (headless Chromium) |

## Core Design Principles

- **Graceful degradation**: Every optional subsystem has a feature flag; source failures in context building don't block the pipeline
- **Anti-confabulation**: Two-gate system with pre-generation memory validation and post-generation claim verification
- **Latency optimization**: Parallel context fetching (ThreadPoolExecutor), merged LLM calls, strict timeouts on ancillary steps
- **Provider resilience**: Automatic failover chain across LLM providers with exponential backoff
- **Companion-as-person**: Internal state (energy, mood, needs), personal goals, opinions, values, and curiosities

## Project Structure

```
companion-framework/
  src/
    core/                    # Conversation pipeline, internal state, mood, memory retrieval
      conversation/          # Pipeline stages (context builder, analyzer, critic, checkpoint)
      commands/              # Slash command handlers
    tasks/                   # 40 Celery background tasks
    autonomy/                # Proactive reach-out, goals, opinions, reflection, interjection (19 modules)
    memory/                  # 29 memory modules (facts, episodes, semantic, graph, biographies)
    integrations/            # Google (Gmail + Calendar), Cloudinary
    scheduling/              # Calendar-aware scheduling, companion schedule
    llm/                     # LLM provider abstraction (Fireworks, OpenAI, Anthropic, DeepSeek)
    routes/                  # Flask endpoints (chat, observe, auth, settings, costs, approval)
    handlers/                # Message entry point
    voice/                   # TTS/STT integration (Deepgram, ElevenLabs, Edge TTS)
    image/                   # Image generation (NanoBanana, RunComfy)
    config/                  # App config, persona config, logging, socketio
    database/                # PostgreSQL client, auth
    middleware/              # Auth middleware, error handlers
    services/                # Cost tracking
    agents/                  # Temporal context agent
    user/                    # User profile management
    diagnostics/             # System health checks
    ops/                     # Ops bot, maintenance runner
    auth/                    # Firebase auth
    utils/                   # Name normalization
  executor/                  # Sandboxed code execution container
    tools/                   # Browser, web, search, Google, image, memory tools
  instances/                 # Companion definitions (6 companions)
    settings/                # World/setting files
  scripts/                   # Admin CLI, simulation, backup, OAuth, seeding
  prompts/core/              # Modular prompt templates (5 files)
  public/                    # Web interfaces (chat + observation dashboard)
  cli/                       # Terminal chat client
  tests/                     # 70+ test files
  migrations/                # Alembic database migrations
```
