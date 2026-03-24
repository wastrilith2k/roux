# Companion Framework: Architecture & Technical Documentation

An autonomous AI companion framework that creates persistent, evolving relationships between AI characters. Each companion has its own personality, memories, opinions, goals, and internal state. They remember past conversations, form opinions, develop curiosities, and reach out proactively when they have something to say.

---

## Documentation Index

| # | Document | Description |
|---|----------|-------------|
| 1 | [System Overview](docs/01-system-overview.md) | High-level architecture, technology stack, design principles, project structure |
| 2 | [Infrastructure & Deployment](docs/02-infrastructure.md) | Docker services, Caddy routing, startup sequence, dev vs production |
| 3 | [Conversation Pipeline](docs/03-conversation-pipeline.md) | 9-stage pipeline: context building, message analysis, prompt assembly, LLM generation, claim validation, quality critique, image intent |
| 4 | [Memory System](docs/04-memory-system.md) | 24 memory modules: fact store, episodes, observations, relationships, entity profiles, biographies, events, neural search, confidence decay |
| 5 | [Autonomy System](docs/05-autonomy-system.md) | Reach-out pressure, reach-out engine, interjection engine, goal system (formation, planning, budget, routing, tracking), internal state, mood, curiosity, opinions, values, personality evolution |
| 6 | [LLM Infrastructure](docs/06-llm-infrastructure.md) | Provider chain with failover (Fireworks, OpenAI, Anthropic, DeepSeek), vision, LLM usage by subsystem |
| 7 | [Code Execution & Web Browsing](docs/07-code-execution.md) | Sandboxed executor, 8 tool modules: Playwright browser, web content, Tavily search, Google Workspace, memory/facts, reminders, image generation, weather |
| 8 | [Integrations, Voice & Image](docs/08-integrations.md) | Google APIs (Gmail, Calendar, Docs), Telegram bridge, Cloudinary, n8n, Deepgram/ElevenLabs voice, NanoBanana/RunComfy image generation |
| 9 | [Companions & Prompts](docs/09-companions-and-prompts.md) | Instance structure, 6 included companions (modern + D&D fantasy), persona config, entity profiles, prompt template hierarchy |
| 10 | [Simulation Engine](docs/10-simulation-engine.md) | Multi-agent relationship simulation, configuration, running simulations, observation dashboard |
| 11 | [Web & CLI Interfaces](docs/11-interfaces.md) | Web chat (typewriter, mood emoji, settings), CLI client (slash commands), SocketIO event reference |
| 12 | [Database Schema](docs/12-database-schema.md) | PostgreSQL tables (core, episodes, autonomy, synthesis, hidden), external file storage |
| 13 | [Background Tasks & Scheduling](docs/13-background-tasks.md) | Celery config, 30+ task catalog (per-message, nightly, weekly, on-demand), APScheduler, calendar integration |
| 14 | [Admin, Operations & Testing](docs/14-operations.md) | Admin CLI, ops bot, report generator, database seeding, test suite (13 files), test patterns |
| 15 | [Configuration Reference](docs/15-configuration.md) | All environment variables: required, LLM providers, feature flags, integrations, database, tuning |

---

## Quick Reference

### Architecture Diagram

```
User Message (WebSocket / Telegram / CLI)
        |
   Flask-SocketIO (eventlet)
        |
   Message Handler
        |
   9-Stage Conversation Pipeline
   (context building, memory retrieval, message analysis,
    inner monologue, LLM generation, claim validation, response critique)
        |
   Response + Store to PostgreSQL
        |
   30+ Celery Background Tasks
   (fact extraction, episode learning, opinion formation,
    relationship evaluation, biography synthesis, goal planning, etc.)
```

### Stack

**Python 3.12** | **Flask + SocketIO** | **PostgreSQL + pgvector** | **Redis** | **Celery** | **Neo4j (Graphiti)** | **Docker** | **Caddy**

**LLM Routing:** Fireworks AI (primary) -> DeepSeek -> OpenAI -> Anthropic (automatic failover)

### Key Subsystem Counts

| Subsystem | Count |
|-----------|-------|
| Pipeline stages | 9 |
| Memory modules | 24 |
| Autonomy modules | 16 |
| Celery background tasks | 30+ |
| Docker services | 9 |
| LLM providers | 5 |
| Feature flags | 14 |
| Companion instances | 6 |
| Test files | 13 |
