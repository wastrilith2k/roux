# Onboarding Guide

A practical getting-started guide for developers working on the companion-framework.

---

## Prerequisites

### Required Software

- **Docker and Docker Compose** -- the primary way to run the full stack.
- **Python 3.11+** -- needed for local development outside Docker. The Docker image uses Python 3.12.
- **Node.js and npm** -- installed automatically in Docker; needed locally only if running MCP servers.

### Required API Keys

At minimum, you need these set in your `.env` file:

| Variable | Purpose |
|---|---|
| `FIREWORKS_API_KEY` | Primary LLM provider (DeepSeek-v3 via Fireworks) |
| `OPENAI_API_KEY` | Entity extraction, embeddings, and tool-calling with gpt-4o-mini |
| `POSTGRES_PASSWORD` | PostgreSQL database password |
| `REDIS_URL` | Redis connection string (defaults work in Docker) |

You should also have at least one of `ANTHROPIC_API_KEY` or `DEEPSEEK_API_KEY` configured as a fallback LLM provider.

### Optional Integration Keys

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram bot for proactive messaging |
| `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | Google Calendar/Drive integration |
| `DEEPGRAM_API_KEY` | Speech-to-text (voice notes) |
| `ELEVENLABS_API_KEY` | Expressive text-to-speech |
| `RUNCOMFY_API_KEY` | Image generation via ComfyUI |
| `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` | Image hosting/CDN |
| `TAVILY_API_KEY` | Web search from code executor |
| `SENTRY_DSN` | Error tracking |
| `STATSIG_SERVER_KEY` | Feature flags |
| `ZEP_API_KEY` | Zep memory service |

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url> companion-framework
cd companion-framework
cp .env.example .env
```

Edit `.env` and fill in at least the required API keys listed above. The `.env.example` file documents every variable with sensible defaults for ports and service names.

### 2. Start all services

```bash
docker-compose up
```

This launches the following services:

| Service | Container | Port | Description |
|---|---|---|---|
| **agent-service** | `companion-backend` | `5000` | Flask + Socket.IO web chat server (main entry point) |
| **celery-worker** | `companion-celery-worker` | -- | Async task processor (35+ background tasks) |
| **postgres** | `companion-postgres` | `5432` | PostgreSQL 16 with pgvector extension |
| **redis** | `companion-redis` | `6379` | Message broker for Celery, pub/sub |
| **graphiti-neo4j** | `companion-graphiti-neo4j` | `7475` (HTTP), `7688` (Bolt) | Neo4j for Graphiti temporal knowledge graph |
| **code-executor** | `companion-code-executor` | `5001` | Sandboxed Python execution environment |
| **companion-cli** | `companion-cli` | -- | Terminal chat client |
| **n8n** | `companion-n8n` | `5678` (internal) | Workflow automation |
| **caddy** | `companion-caddy` | `80`, `443` | Reverse proxy with automatic SSL |

### 3. Access the web UI

Open [http://localhost:5000](http://localhost:5000) in your browser. The Flask app serves both the HTTP API and WebSocket-based chat interface.

### 4. Run database migrations

```bash
docker-compose exec agent-service alembic upgrade head
```

---

## Project Structure

```
src/                  -- Core framework code
  autonomy/           -- Proactive messaging, goals, values, opinions
  core/               -- Conversation pipeline and reasoning
  config/             -- Persona and system configuration
  database/           -- PostgreSQL access layer
  handlers/           -- Message entry points
  image/              -- Image generation prompts and logic
  llm/                -- LLM provider abstraction (Fireworks, OpenAI, Anthropic)
  memory/             -- Knowledge extraction and retrieval
  routes/             -- HTTP/WebSocket endpoint blueprints
  scheduling/         -- Background task scheduling (APScheduler)
  services/           -- Shared service utilities
  tasks/              -- 35+ Celery async tasks
  user/               -- User profile and session management
  utils/              -- Shared utilities
  voice/              -- Speech-to-text (Deepgram) and text-to-speech (ElevenLabs, Edge TTS)
  web_chat.py         -- Main application entry point
  celery_app.py       -- Celery configuration, task includes, beat schedule

data/                 -- Persona config, entity profiles
  persona.yaml        -- Companion identity configuration
  entity_profiles/    -- YAML profiles for companion and user entities

prompts/              -- Modular conversation prompts
  core/               -- personality.md, cognitive_flow.md

instances/            -- Multi-agent persona configurations (e.g., instances/kai/, instances/mira/)
cli/                  -- Terminal chat client (separate Dockerfile)
executor/             -- Sandboxed code execution service
migrations/           -- Alembic database migrations
scripts/              -- Utility scripts
tests/                -- Test suite (pytest)
docs/                 -- Documentation
```

---

## Creating a New Companion Persona

1. **Copy the base persona config.** Start from `data/persona.yaml` or create an instance directory under `instances/<name>/` (see `instances/kai/` or `instances/mira/` for examples).

2. **Set identity fields.** In your `persona.yaml`, configure:
   - `companion.name` and `companion.short_name`
   - `companion.pronouns` (subject, object, possessive, reflexive)
   - `companion.entity_profile` -- references a file in `data/entity_profiles/`
   - `primary_user.name`, `primary_user.email`, `primary_user.timezone`

3. **Create a personality prompt.** Write `prompts/core/personality.md` (or an instance-specific override). This is the core prompt that shapes conversational tone and behavior.

4. **Create an entity profile.** Add a YAML file in `data/entity_profiles/` matching the `entity_profile` value. This holds structured facts about the companion's identity, background, and boundaries.

5. **Configure voice (optional).** In the `voice` section of `persona.yaml`, set:
   - `elevenlabs_voice_id` -- for expressive TTS
   - `edge_tts_fallback` -- free fallback voice (e.g., `en-US-AvaNeural`)

6. **Configure channels.** Enable or disable `telegram` and `web` under the `channels` section.

---

## Key Files to Read First

These files form the critical path for understanding how the system works:

| File | What It Does |
|---|---|
| `src/web_chat.py` | Main entry point. Flask + Socket.IO startup, background service initialization. |
| `src/core/conversation/pipeline.py` | The heart of response generation -- the full conversation processing pipeline. |
| `src/handlers/message_handler.py` | Message entry point. Where inbound messages are received and dispatched. |
| `src/config/persona_config.py` | Identity configuration loader. Reads `persona.yaml` and wires it into the system. |
| `src/autonomy/always_on_service.py` | Proactive behavior loop. Drives autonomous messaging via Telegram. |
| `src/llm/provider_factory.py` | LLM provider chain. Manages Fireworks, OpenAI, Anthropic, and fallback routing. |
| `src/celery_app.py` | Celery configuration. All 35+ task includes and the beat schedule for periodic jobs. |
| `data/persona.yaml` | Companion identity config. Name, pronouns, entity profile references, voice settings. |

---

## Adding a Scheduled Task

1. **Create the task file** in `src/tasks/`. Follow the naming convention `<name>_task.py`. See existing tasks for patterns.

2. **Add the Celery task decorator** to your function:
   ```python
   from src.celery_app import celery_app

   @celery_app.task(name='tasks.my_new_task')
   def my_new_task():
       # task logic here
       pass
   ```

3. **Register the module** in `src/celery_app.py` by adding it to the `include` list:
   ```python
   'src.tasks.my_new_task',
   ```

4. **Add a beat schedule entry** (if periodic) in the `beat_schedule` dict in `src/celery_app.py`:
   ```python
   'my-new-task': {
       'task': 'tasks.my_new_task',
       'schedule': crontab(hour=5, minute=0),  # 5:00 AM daily
   },
   ```

---

## Development Workflow

### Running locally (without Docker)

```bash
# Install dependencies
pip install -r requirements-base.txt

# Set environment variables (or source your .env)
export FIREWORKS_API_KEY=...
export OPENAI_API_KEY=...
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=companion
export POSTGRES_USER=companion
export POSTGRES_PASSWORD=...
export REDIS_URL=redis://localhost:6379/0
export CELERY_BROKER_URL=redis://localhost:6379/0

# You still need PostgreSQL, Redis, and Neo4j running locally or via Docker:
docker-compose up postgres redis graphiti-neo4j

# Run the web server
python -m src.web_chat

# In a separate terminal, run the Celery worker
celery -A src.celery_app worker --loglevel=info --concurrency=4 --pool=solo
```

### Running with Docker (full stack)

```bash
docker-compose up              # all services
docker-compose up -d           # detached mode
docker-compose logs -f agent-service  # follow backend logs
```

### Database migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Or inside Docker
docker-compose exec agent-service alembic upgrade head

# Create a new migration after changing models
alembic revision --autogenerate -m "describe the change"
```

### Running tests

Tests live in the `tests/` directory and use pytest:

```bash
pytest tests/
pytest tests/test_persona_config.py -v   # run a specific test file
```

---

## Environment Variables Reference

### Required -- LLM Providers

| Variable | Default | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | -- | Primary LLM provider API key |
| `FIREWORKS_MODEL` | `accounts/fireworks/models/deepseek-v3p2` | Primary model |
| `FIREWORKS_FALLBACK_MODEL` | `accounts/fireworks/models/kimi-k2-instruct-0905` | Fallback model |
| `OPENAI_API_KEY` | -- | Used for entity extraction and embeddings |
| `ANTHROPIC_API_KEY` | -- | Claude API (used for autonomous task narratives) |
| `DEEPSEEK_API_KEY` | -- | Direct DeepSeek API access |

### Required -- Database and Redis

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `postgres` | PostgreSQL hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `companion` | Database name |
| `POSTGRES_USER` | `companion` | Database user |
| `POSTGRES_PASSWORD` | `companion_change_me` | Database password |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Celery broker (usually same as REDIS_URL) |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Celery result backend |

### Required -- Knowledge Graph

| Variable | Default | Description |
|---|---|---|
| `NEO4J_GRAPHITI_URI` | `bolt://graphiti-neo4j:7687` | Graphiti Neo4j Bolt endpoint |
| `NEO4J_GRAPHITI_USER` | `neo4j` | Neo4j username |
| `NEO4J_GRAPHITI_PASSWORD` | `graphiti_password` | Neo4j password |
| `ENTITY_EXTRACTOR` | `openai` | Entity extraction provider |
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider (`openai` or `ollama`) |

### Optional -- Telegram

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | -- | Telegram bot token for companion messaging |
| `TELEGRAM_CHAT_ID` | -- | Target chat for proactive messages |
| `TELEGRAM_OPS_BOT_TOKEN` | -- | Separate bot for system/ops notifications |
| `TELEGRAM_OPS_CHAT_ID` | -- | Chat ID for ops notifications |

### Optional -- Google Integration

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_OAUTH_CLIENT_ID` | -- | Google OAuth client ID |
| `GOOGLE_OAUTH_CLIENT_SECRET` | -- | Google OAuth client secret |

### Optional -- Voice

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | -- | Deepgram speech-to-text |
| `DEEPGRAM_VOICE_ID` | `aura-asteria-en` | Deepgram TTS voice |

### Optional -- Image Generation

| Variable | Default | Description |
|---|---|---|
| `RUNCOMFY_API_KEY` | -- | RunComfy image generation API |
| `RUNCOMFY_PERSONAL_DEPLOYMENT_ID` | -- | Personal model deployment |
| `RUNCOMFY_GENERAL_DEPLOYMENT_ID` | -- | General model deployment |
| `RUNCOMFY_SKETCH_DEPLOYMENT_ID` | -- | Sketch model deployment |
| `CLOUDINARY_CLOUD_NAME` | -- | Cloudinary CDN account |
| `CLOUDINARY_API_KEY` | -- | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | -- | Cloudinary API secret |

### Optional -- Monitoring and Feature Flags

| Variable | Default | Description |
|---|---|---|
| `SENTRY_DSN` | -- | Sentry error tracking DSN |
| `STATSIG_SERVER_KEY` | -- | Statsig feature flag server key |

### Feature Flags (Runtime)

| Variable | Default | Description |
|---|---|---|
| `COMPANION_AUTONOMY_ENABLED` | `true` | Enable proactive/autonomous messaging |
| `COMPANION_TELEGRAM_ENABLED` | `true` | Enable Telegram bot integration |
| `COMPANION_CODE_EXECUTION_ENABLED` | `true` | Enable sandboxed code execution |
| `COMPANION_REACH_OUT_INTERVAL` | `15` | Minutes between autonomous reach-out attempts |
| `USE_MODULAR_PROMPTS` | `modular` | Prompt system mode |
| `USE_CONDENSED_HISTORY` | `true` | Condense verbose messages in conversation history |
| `SKIP_CONVERSATION_HISTORY` | `false` | Disable conversation history entirely |
| `STRIP_EMOTIONAL_FLUFF` | `true` | Remove filler from LLM responses |
