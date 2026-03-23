# Companion Framework

An AI companion engine with persistent memory, autonomous behavior, and emergent personality. The companion maintains long-term relationships through conversation, proactive outreach, and self-directed goals — backed by a multi-layered memory system and resilient LLM pipeline.

## Quick Start

```bash
# Configure environment
cp .env.example .env   # Add API keys and database credentials

# Start all services
docker-compose up
```

The web UI is available at `https://localhost` (via Caddy reverse proxy) or `http://localhost:5000` (direct Flask).

## Tech Stack

- **Runtime:** Python 3.11, Flask, Socket.IO, Eventlet
- **Task Queue:** Celery + Redis
- **Databases:** PostgreSQL (pgvector), Neo4j (Graphiti knowledge graph)
- **LLM Providers:** Fireworks (primary) with failover to DeepSeek, OpenAI, Anthropic
- **Voice:** Deepgram (STT), ElevenLabs/Edge TTS
- **Channels:** Web UI (Socket.IO), Telegram, CLI

## Project Structure

```
src/                  Core framework
  autonomy/           Proactive messaging, goals, values, opinions
  core/               Conversation pipeline and reasoning
  core/conversation/  Modular pipeline components
  database/           PostgreSQL access layer
  handlers/           Message entry points
  llm/                LLM provider abstraction with failover
  memory/             Knowledge extraction and retrieval (12 subsystems)
  routes/             HTTP and WebSocket endpoints
  scheduling/         Background task scheduling
  tasks/              35+ Celery async tasks
  voice/              Speech-to-text and text-to-speech
data/                 Persona config, entity profiles
prompts/              Modular conversation prompts
instances/            Multi-agent persona configurations
cli/                  Terminal chat client
executor/             Sandboxed code execution service
migrations/           Alembic database migrations
docs/                 Documentation
```

## Documentation

- [Architecture](docs/architecture.md) — System overview, module map, data flows, and design patterns
- [Onboarding](docs/onboarding.md) — Getting started guide for new developers
- [Services](docs/services.md) — Entry points, background services, Celery tasks, and integrations
- [CLI](cli/README.md) — Terminal chat client usage
- [Hardening Plan](PLAN.md) — Security, testing, and infrastructure roadmap
