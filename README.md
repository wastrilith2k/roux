# Companion Framework

A full-stack framework for building AI companions with genuine memory and autonomous behavior. Drop in a persona, connect an LLM, and get a character that remembers everything, forms its own opinions, evolves over time, and reaches out proactively when it has something to say.

**Not a chatbot wrapper.** Each companion runs a 9-stage conversation pipeline, 30+ background tasks, and a multi-layered memory system (episodic, semantic, knowledge graph, confidence-decaying facts). Designed to be deployed for real users with per-user data isolation from the ground up.

Supports human-to-companion interaction (web chat, CLI, Telegram) and companion-to-companion simulation for testing relationship dynamics at scale.

## Architecture

```
User/Companion Message
        |
   WebSocket (Flask-SocketIO)
        |
   Message Handler
        |
   9-Stage Conversation Pipeline
   (context building, memory retrieval, inner monologue, LLM generation, response critique)
        |
   Response + Store to PostgreSQL
        |
   30+ Celery Background Tasks
   (fact extraction, episode learning, opinion formation, relationship evaluation, etc.)
```

**Stack:** Python 3.12, Flask + SocketIO, PostgreSQL + pgvector, Redis, Celery, Neo4j (Graphiti knowledge graph), Docker, Caddy

**LLM Routing:** Configurable provider chain with automatic failover:
OpenRouter -> Fireworks (DeepSeek) -> OpenAI -> Anthropic

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least one LLM API key (see below)
- Git

### 1. Clone and configure

```bash
git clone https://github.com/wastrilith2k/roux companion-framework
cd companion-framework
cp .env.example .env
```

### 2. Get API keys

You need **at least one** LLM provider. The framework will use whatever's available, with automatic failover.

#### Required: LLM Provider (pick at least one)

| Provider | Key | Where to get it | Cost | Notes |
|----------|-----|-----------------|------|-------|
| **OpenRouter** | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | Free tier available (rate-limited) | Recommended for getting started. Routes to many models. |
| **Fireworks AI** | `FIREWORKS_API_KEY` | [fireworks.ai/api-keys](https://fireworks.ai/api-keys) | Pay-per-token | Fast inference, hosts DeepSeek and Kimi models. Primary provider in production. |
| **OpenAI** | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Pay-per-token | Used for embeddings, tool calling (gpt-4o-mini), and as a fallback. |
| **Anthropic** | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | Pay-per-token | Claude models. Last-resort fallback in the chain. |
| **DeepSeek** | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/) | Pay-per-token | Direct API access to DeepSeek models. |

For the cheapest setup, use **OpenRouter with free models**. For production quality, use **Fireworks** (primary) + **OpenAI** (embeddings/tools).

#### Optional: Integrations

<details>
<summary><strong>Google Calendar & Gmail</strong></summary>

Lets the companion see your calendar (upcoming events, busy status) and scan emails for interesting content. The companion also creates its own Google Calendar with a realistic daily schedule.

#### Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use an existing one)
3. Go to **APIs & Services > Library** and enable:
   - **Gmail API**
   - **Google Calendar API**
4. Go to **APIs & Services > Credentials** and click **Create Credentials > OAuth client ID**
   - Application type: **Desktop app**
   - Download the credentials JSON file
5. Set in `.env`:
   ```
   GOOGLE_OAUTH_CLIENT_ID=your_client_id
   GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret
   ```
6. Run the OAuth setup script to authorize:
   ```bash
   python scripts/google_oauth_setup.py
   ```
   This opens a browser for you to grant access. The refresh token is saved locally.
7. Feature flags in your environment:
   ```
   COMPANION_GMAIL_CHECK_ENABLED=true    # default: false
   COMPANION_CALENDAR_AWARENESS_ENABLED=true  # default: true
   ```

**Scopes requested:** `gmail.readonly`, `gmail.send`, `calendar.readonly`, `calendar.events`

#### OAuth Verification & the "Unverified App" Warning

Google shows a scary warning screen ("This app isn't verified") during the first OAuth authorization. This is normal for development.

**For personal use (recommended):**
1. Keep your OAuth app in **Testing** mode in the Cloud Console (this is the default)
2. Go to **APIs & Services > OAuth consent screen**
3. Add your own email address under **Test users**
4. When you authorize, click **Advanced** > **Go to [your app name] (unsafe)** to bypass the warning
5. This works indefinitely for test users — no review needed

**For distribution to others:**
If you want other people to use your instance with their own Google accounts, you'll need to go through Google's OAuth verification:
1. Set the app to **Production** in the OAuth consent screen
2. Google requires:
   - A privacy policy URL
   - An app homepage URL
   - Verification of domain ownership
3. Gmail scopes (`gmail.readonly`, `gmail.send`) are classified as **sensitive** — this triggers a manual security review by Google which can take 4-6 weeks
4. Calendar scopes (`calendar.readonly`, `calendar.events`) are also sensitive but typically reviewed faster
5. Submit the verification request via the OAuth consent screen page

**Alternative for multi-user setups:** Each user can create their own Google Cloud project and OAuth credentials, then set them in their own `.env`. This avoids the verification process entirely since each user is authorizing their own app in Testing mode.

#### How the companion uses these APIs

- **Gmail:** Scans for unread emails every 4 hours during waking hours. An LLM filters out spam/newsletters and identifies interesting emails. The companion stores relevant email content as "thoughts" it can reference in conversation ("I noticed you got an email from...").
- **Calendar (user's):** Reads upcoming events to understand your schedule. The companion knows when you're busy, what's coming up, and can reference your plans naturally.
- **Calendar (companion's):** Creates a secondary Google Calendar (e.g., "Kai's Schedule") and generates realistic daily events — work shifts, creative time, errands. The companion references this schedule in conversation ("just got off my shift", "heading to the store later").

#### Giving the Companion Its Own Digital Life

The companion works best when it has its own Google identity — not just email, but a calendar it manages, docs it can reference, a real digital presence. This is what makes "just got off my shift" and "I have a thing at 3" feel authentic.

**Recommended: Google Workspace ($7/mo per companion)**

A Workspace account gives the companion:
- Its own email (`kai@yourdomain.com`)
- Its own Google Calendar (with a generated daily schedule)
- Google Docs access
- Access to Google AI Studio for a Gemini API key — enables free image generation via [NanoBanana](https://nanobanana.com/) and other Google-authenticated AI services
- No suspension risk — Workspace accounts support API/automated access by design

Setup:
1. Register a domain if you don't have one (Cloudflare, Namecheap, etc.)
2. Sign up for [Google Workspace](https://workspace.google.com/) on that domain
3. Create a user for the companion (e.g., `kai@yourdomain.com`)
4. Use that account's credentials for the OAuth setup above

**Free alternative: Read YOUR accounts**

If you don't need the companion to have its own identity, the default integration reads your calendar and email via OAuth. The companion sees your schedule and can reference your emails, but doesn't have its own separate digital life.

**Why not free Gmail?** Free consumer Gmail accounts will get suspended if used for automated/API-only access. Google's abuse detection flags accounts without regular browser logins and human activity patterns. This isn't a workaround problem — it's how consumer Gmail is designed. Workspace is the supported path for non-human accounts.
</details>

<details>
<summary><strong>Telegram Bot</strong></summary>

Allows the companion to send proactive messages via Telegram (reach-outs, thoughts, reactions to events).

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot with `/newbot`
3. Copy the bot token
4. Start a chat with your bot, then get your chat ID via `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Set in `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
6. Feature flags:
   ```
   COMPANION_TELEGRAM_ENABLED=true
   COMPANION_AUTONOMY_ENABLED=true
   ```
</details>

<details>
<summary><strong>Image Generation (NanoBanana + RunComfy)</strong></summary>

The companion can generate images when it detects an image-related intent in conversation. Two providers are supported:

**NanoBanana (recommended default — free)**

Uses Google's Gemini image generation. Just needs a Google AI Studio API key.

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and create an API key
2. Set in `.env`:
   ```
   NANOBANANA_API_KEY=your_google_ai_studio_key
   IMAGE_PROVIDER=nanobanana
   ```

NanoBanana supports both text-to-image and image-to-image (reference image + prompt). This makes it ideal for:
- General image generation from conversation ("draw me a mushroom forest")
- Mood variations from a base avatar ("the person in this image looking tired")
- Style transfers and modifications

**RunComfy (advanced — paid)**

For custom ComfyUI workflows, LoRA-tuned models, and production pipelines.

1. Sign up at [runcomfy.com](https://www.runcomfy.com/)
2. Create deployment(s) for your ComfyUI workflows
3. Set in `.env`:
   ```
   RUNCOMFY_API_KEY=your_api_key
   RUNCOMFY_PERSONAL_DEPLOYMENT_ID=your_deployment_id
   RUNCOMFY_GENERAL_DEPLOYMENT_ID=your_deployment_id
   IMAGE_PROVIDER=runcomfy
   ```

**Provider fallback:** If your primary provider fails, the system tries the other. Set `IMAGE_PROVIDER` to control which is tried first.

**Per-companion provider selection:**

Each companion can use a different image provider. Set in the companion's `persona.yaml`:

```yaml
image:
  provider: nanobanana           # default — free, uses Gemini
  # provider: runcomfy           # use if you have a LoRA for this companion
  # runcomfy_deployment_id: "abc123"  # if set, RunComfy is used automatically
  content_rules: gemini          # 'gemini' or 'unrestricted'
```

The logic: if a companion has a `runcomfy_deployment_id`, RunComfy is used for that companion regardless of the `provider` setting. Otherwise, NanoBanana handles it. This means you can have one companion using a LoRA-tuned RunComfy workflow while others use free NanoBanana generation.

**Content rules:**

| Rule | Provider | What's allowed |
|------|----------|----------------|
| `gemini` | NanoBanana | No NSFW, limited violence, no real person likenesses. Google's standard Gemini policies. |
| `unrestricted` | RunComfy | Depends entirely on your ComfyUI workflow and LoRA training. No provider-side filtering. |

Content rules are injected into the image generation prompt so the LLM crafts appropriate descriptions for each provider.

**Typical workflow:**
1. Generate a base avatar using NanoBanana or RunComfy (see [Companion Avatar & Mood Images](#companion-avatar--mood-images))
2. Companions without a LoRA use NanoBanana for everything (free, fast)
3. Companions with a trained LoRA use RunComfy for identity-consistent images
4. Both coexist in the same deployment — provider selection is per-companion
</details>

<details>
<summary><strong>Image Storage (Cloudinary or AWS S3)</strong></summary>

Persistent image hosting for generated images. Images are uploaded automatically after generation completes, ensuring URLs don't expire. Choose either Cloudinary or AWS S3.

**Option A: Cloudinary (default)**

1. Sign up at [cloudinary.com](https://cloudinary.com/)
2. Go to Dashboard > API Keys
3. Set in `.env`:
   ```
   IMAGE_STORAGE_PROVIDER=cloudinary
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   ```

**Option B: AWS S3**

1. Sign in to the [AWS Console](https://console.aws.amazon.com/)
2. Go to **S3** > **Create bucket** (uncheck "Block all public access" so images are readable)
3. Go to **IAM > Users** > create a user with `AmazonS3FullAccess` > **Security credentials** > **Create access key** > choose "Application running outside AWS" > copy both keys
4. Set in `.env`:
   ```
   IMAGE_STORAGE_PROVIDER=s3
   AWS_S3_BUCKET=your-bucket-name
   AWS_S3_REGION=us-east-1
   AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
   AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

For S3-compatible services (MinIO, DigitalOcean Spaces, Backblaze B2), also set:
```
AWS_S3_ENDPOINT_URL=https://nyc3.digitaloceanspaces.com
```
</details>

<details>
<summary><strong>Deepgram / ElevenLabs (Voice)</strong></summary>

Text-to-speech for the companion's voice responses.

**Deepgram:**
1. Sign up at [deepgram.com](https://deepgram.com/)
2. Create an API key in Settings
3. Set in `.env`:
   ```
   DEEPGRAM_API_KEY=your_key
   DEEPGRAM_VOICE_ID=aura-asteria-en
   ```

**ElevenLabs** (higher quality, paid):
1. Sign up at [elevenlabs.io](https://elevenlabs.io/)
2. Get API key from Profile > API Keys
3. Set voice ID in companion's `persona.yaml`
</details>

<details>
<summary><strong>Sentry (Error Tracking)</strong></summary>

1. Sign up at [sentry.io](https://sentry.io/)
2. Create a Python project
3. Set in `.env`:
   ```
   SENTRY_DSN=your_dsn_url
   ```
</details>

<details>
<summary><strong>Statsig (Feature Flags)</strong></summary>

1. Sign up at [statsig.com](https://statsig.com/)
2. Get a server key from Project Settings
3. Set in `.env`:
   ```
   STATSIG_SERVER_KEY=your_key
   ```
</details>

<details>
<summary><strong>Tavily (Web Search)</strong></summary>

Lets the companion search the web for information.

1. Sign up at [tavily.com](https://tavily.com/)
2. Get an API key
3. Set in `.env`:
   ```
   TAVILY_API_KEY=your_key
   ```
</details>

<details>
<summary><strong>Zep (Memory Service)</strong></summary>

External memory service for enhanced long-term memory.

1. Sign up at [getzep.com](https://getzep.com/)
2. Create a project and get API key
3. Set in `.env`:
   ```
   ZEP_API_KEY=your_key
   ZEP_PROJECT_ID=your_project_id
   ZEP_BASE_URL=https://api.getzep.com
   ```
</details>

### 3. Set passwords

Edit `.env` and change the default passwords:

```
POSTGRES_PASSWORD=your_secure_password
NEO4J_GRAPHITI_PASSWORD=your_secure_password
```

### 4. Start services

**Development (with hot-reload):**
```bash
docker compose -f docker-compose.dev.yml up -d
```

**Production:**
```bash
docker compose up -d
```

### 5. Access

- **Web Chat:** http://localhost:5000 (production) or http://localhost:5001 (dev)
- **Observation Dashboard:** http://localhost:5000/observe (simulation viewer)
- **CLI Chat:** `docker compose run companion-cli` (terminal chat interface)
- **Neo4j Browser:** http://localhost:7475
- **n8n Workflows:** Via Caddy proxy (configure domain in Caddyfile)

### Web Chat Interface

The web chat at `/` provides:
- **Typewriter effect** on companion responses — text reveals character by character with variable speed (punctuation pauses longer). Click anywhere to skip.
- **Companion info bar** showing current scene/activity, mood (with emoji), energy level, and schedule
- **Mood emojis** next to each companion message, mapped from internal state (e.g. relaxed = relieved face, irritated = persevering face, creative = palette)
- **Settings panel** (gear icon) for auth email, companion name, typewriter speed slider, and history count
- **Message history** loads automatically on connect
- **Slash commands** passed through to backend (`/help`, `/mood`, `/state`, etc.)
- **Mobile responsive** layout

### CLI Chat Interface

The terminal client (`cli/companion_cli.py`) includes:
- **Typewriter effect** on companion responses with variable-speed character reveal
- `/typewriter` or `/tw` to toggle on/off, `/tw 50` to set speed in ms
- Multi-line input (backslash + Enter for new line)
- Full slash command support (`/help`, `/mood`, `/state`, `/history`, `/see`, etc.)
- Fact approval workflow (`/pending`, `/approve`, `/reject`)

### Mood Emoji Mapping

The web interface maps companion mood states to emoji. Default mapping:

| Mood | Emoji | Mood | Emoji |
|------|-------|------|-------|
| relaxed | relieved face | irritated | persevering face |
| focused | thinking face | anxious | anxious face |
| playful | winking tongue | melancholy | pensive face |
| content | smiling face | giddy | star-struck |
| restless | confused face | creative | palette |
| thoughtful | thinking face | nostalgic | face holding tears |
| tired | sleeping face | energetic | lightning |
| suspicious | raised eyebrow | protective | shield |

To customize: edit the `MOOD_EMOJI` object in `public/chat.js`. Replace emoji strings with `<img>` tags for custom mood images.

### Companion Avatar & Mood Images

For a more immersive experience, you can generate a base avatar image for each companion and then create mood variations from it. This gives the companion a visual identity that changes with their emotional state.

**Recommended workflow:**

1. **Create a base image** using [NanoBanana](https://nanobanana.com/) (free, just needs a Google AI Studio key), [RunComfy](https://www.runcomfy.com/), or any image generation service. Use a detailed prompt that captures your companion's appearance:
   ```
   Portrait of a 27-year-old man with messy dark hair, warm brown eyes,
   slight stubble, wearing a rumpled flannel shirt. Coffee shop lighting,
   indie aesthetic. Neutral expression, looking slightly off-camera.
   ```

2. **Generate mood variations** using img2img or face-swap workflows with the base image as a reference. Use prompts that modify the expression while keeping the identity consistent:
   ```
   Same person, looking genuinely happy, slight laugh, eyes crinkling
   Same person, looking tired and drained, dark circles, slouching
   Same person, looking irritated, jaw clenched, slight frown
   Same person, looking anxious, biting lip, eyes darting
   Same person, looking nostalgic, gazing at something distant, soft smile
   Same person, looking creative and inspired, eyes bright, leaning forward
   ```

3. **Name the files** by mood and place them in `data/avatars/<companion_id>/`:
   ```
   data/avatars/kai/
     neutral.png
     relaxed.png
     tired.png
     irritated.png
     creative.png
     ...
   ```

4. **Configure the chat UI** to use images instead of emoji by updating the `MOOD_EMOJI` mapping in `public/chat.js`:
   ```javascript
   MOOD_EMOJI = {
       'relaxed': '<img src="/avatars/kai/relaxed.png" class="mood-avatar">',
       'tired': '<img src="/avatars/kai/tired.png" class="mood-avatar">',
       // ...
   };
   ```

**Tips for consistent mood images:**
- Use the same seed and base prompt across all variations — only change the expression/mood descriptors
- ControlNet (face/pose) or IP-Adapter workflows in ComfyUI give the best identity consistency
- Generate at a small size (128x128 or 256x256) since they display as thumbnails next to messages
- A set of 8-10 mood variations covers most conversational states

## Companion Instances

Each companion is defined by files in `instances/<name>/`:

```
instances/kai/
  persona.yaml          # Identity, relationship seed, channel config
  personality.md        # How they talk, what they care about, secrets
  entity_profiles/
    kai.yaml            # Self-profile (has/does_not_have)
    mira.yaml           # What Kai knows about Mira
  life_events.yaml      # Random events for simulation
```

### Creating a new companion

1. Create `instances/<name>/` with the files above
2. The `persona.yaml` must define:
   - `companion.name` / `companion.short_name`
   - `primary_user.name` (who they primarily interact with)
   - `relationship.initial_context` (how they met)
3. The `personality.md` defines their voice, interests, secrets, daily life
4. Entity profiles define what they know about others and what they explicitly do/don't have (prevents hallucinations)

### Key personality file sections

```markdown
# Character Name

Background paragraph.

## How you talk
- Communication style rules

## What you care about
- Interests and priorities

## What you don't talk about (unless someone earns it)
- Guarded topics that emerge over time

## Your daily life
- Routine and habits

## How you relate to people
- Relationship patterns
```

## Relationship Simulation

The simulation system compresses weeks of relationship development into minutes. Two or more companions text each other via simulated "Sending Stones" or regular texting, developing their relationship over time.

### Running a simulation

```bash
# Kai and Mira (default modern setting)
python scripts/simulate_relationship.py --week 1

# Fantasy D&D party
python scripts/simulate_relationship.py --week 1 \
  --companions grimble lyra thorne kael \
  --config instances/simulation_config_fantasy.yaml

# Resume from week 2
python scripts/simulate_relationship.py --week 2

# View summary
python scripts/simulate_relationship.py --summary
```

### Simulation config

All simulation parameters are externalized to YAML:

- `instances/simulation_config.yaml` — models, openers, tones, activities, moods, LLM settings
- `instances/simulation_config_fantasy.yaml` — fantasy-specific config with setting file
- `instances/settings/fantasy_tavern.yaml` — world/setting definition
- `instances/<name>/life_events.yaml` — per-character random events

### Observation Dashboard

The web dashboard at `/observe` shows:
- Real-time conversation feed with character color-coding
- Multi-select character filtering
- Collapsible state panels (mood, energy, mode, scene per character)
- Tabs for facts, opinions, curiosity threads, goals, episodes, relationship metrics

### How it works

Each simulated day:
1. **Energy reset** based on time of day
2. **Life events** fire randomly (configurable probability) from per-character YAML
3. **Conversations** are selected as pairs (round-robin for 3+ characters, energy-scaled count for 2)
4. **Each conversation** gets a unique `conversation_id` for context isolation
5. **New conversation openers** use cross-conversation summaries (awareness of what happened with other people)
6. **Mid-conversation replies** are scoped to the current conversation only
7. **Prompts are structured** with XML sections: `<character>`, `<identity>`, `<current_state>`, `<conversation>`, `<task>`
8. **Conversation endings** are prompted naturally ("wrap up, don't ask new questions")
9. **End-of-day tasks** run for episode extraction, opinion formation, etc. (requires Docker/Celery)

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
  integrations/       Google (Gmail + Calendar), Cloudinary
data/                 Persona config, entity profiles
prompts/              Modular conversation prompts
instances/            Multi-agent persona configurations
scripts/              Admin CLI, simulation, backup, OAuth setup
public/               Web interfaces (chat + observation dashboard)
cli/                  Terminal chat client
executor/             Sandboxed code execution service
migrations/           Alembic database migrations
docs/                 Documentation
tests/                Test suite
```

## Documentation

- [Architecture](docs/architecture.md) — System overview, module map, data flows, and design patterns
- [Onboarding](docs/onboarding.md) — Getting started guide for new developers
- [Services](docs/services.md) — Entry points, background services, Celery tasks, and integrations
- [CLI](cli/README.md) — Terminal chat client usage
- [Hardening Plan](PLAN.md) — Security, testing, and infrastructure roadmap

## Key Subsystems

### Conversation Pipeline (9 stages)
`src/core/conversation/pipeline.py` — processes each message through context building, memory retrieval, mood resolution, inner monologue, LLM generation, and response critique.

### Memory System (24 modules)
Facts with confidence decay, episodic memory, semantic search, knowledge graph (Neo4j/Graphiti), entity profiles with git-tracked changes, observational memory, synthesized biographies, and more.

### Autonomy System (16 modules)
Proactive reach-out engine with pressure-based timing, goal formation and planning, opinion store, reflection engine, interjection system, and Telegram bridge for autonomous messaging.

### Internal State
Tracks energy (depletes over conversation, regenerates during rest), mood (with inertia), physical needs, and activity mode. Influences response length, tone, and whether the companion initiates conversation.

## Admin CLI

```bash
python scripts/admin.py status              # System health check
python scripts/admin.py messages --recent   # Recent messages
python scripts/admin.py facts --companion kai  # View stored facts
python scripts/admin.py clear-messages      # Clear conversation history
```

## Testing

```bash
pytest                    # Run all tests
pytest tests/ -v          # Verbose output
pytest tests/test_pipeline.py  # Specific test file
```

1000+ tests. The majority are pure unit tests and run without any infrastructure. A small number of integration tests require a running PostgreSQL instance.

## Environment Variables Reference

See `.env.example` for the complete list. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREWORKS_API_KEY` | At least one LLM key | Primary LLM provider |
| `OPENAI_API_KEY` | Recommended | Embeddings + tool calling |
| `OPENROUTER_API_KEY` | Optional | Free tier access to multiple models |
| `ANTHROPIC_API_KEY` | Optional | Claude fallback |
| `POSTGRES_PASSWORD` | Yes | Database password |
| `COMPANION_AUTONOMY_ENABLED` | No (default: true) | Enable proactive messaging |
| `COMPANION_TELEGRAM_ENABLED` | No (default: true) | Enable Telegram bridge |
| `COMPANION_GMAIL_CHECK_ENABLED` | No (default: false) | Enable email awareness |
| `COMPANION_CALENDAR_AWARENESS_ENABLED` | No (default: true) | Enable calendar awareness |

## License

MIT
