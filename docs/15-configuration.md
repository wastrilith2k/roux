# Configuration Reference

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | Database password |
| At least one LLM key | See LLM Providers below |

### LLM Providers (at least one required)

| Variable | Provider | Notes |
|----------|----------|-------|
| `FIREWORKS_API_KEY` | Fireworks AI | Primary production. Fast, cheap. |
| `OPENAI_API_KEY` | OpenAI | Recommended for embeddings + tool calling |
| `OPENROUTER_API_KEY` | OpenRouter | Free tier for development |
| `ANTHROPIC_API_KEY` | Anthropic | Claude fallback |
| `DEEPSEEK_API_KEY` | DeepSeek | Direct API access |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `COMPANION_AUTONOMY_ENABLED` | `true` | Proactive messaging via Telegram/WebSocket |
| `COMPANION_TELEGRAM_ENABLED` | `true` | Telegram bridge |
| `COMPANION_VOICE_ENABLED` | `true` | Voice message handling |
| `COMPANION_INNER_MONOLOGUE_ENABLED` | `true` | Pre-response private reasoning |
| `COMPANION_MESSAGE_ANALYZER_ENABLED` | `true` | Merged mode+monologue analysis |
| `COMPANION_RESPONSE_CRITIC_ENABLED` | `false` | Post-generation quality critique |
| `CODE_EXECUTION_ENABLED` | `false` | Agentic tool execution |
| `COMPANION_CALENDAR_AWARENESS_ENABLED` | `true` | Google Calendar integration |
| `COMPANION_GMAIL_CHECK_ENABLED` | `false` | Email scanning |
| `COMPANION_TOPIC_NOVELTY_CHECK_ENABLED` | `true` | Prevent duplicate proactive messages |
| `COMPANION_ACTIVITY_MILESTONES` | `false` | Activity-based interjection triggers |
| `OBSERVATIONAL_MEMORY_ENABLED` | `false` | Mastra-style conversation compression |
| `ENABLE_RERANKING` | `true` | Fireworks Qwen3 reranking for semantic search |
| `COMPANION_FACT_LINKS_LLM` | `false` | LLM-based fact link detection (vs heuristic) |

### Authentication Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_PROVIDER` | `firebase` | `firebase` or `cognito` |
| `FIREBASE_PROJECT_ID` | | Firebase project ID (Firebase mode) |
| `AWS_COGNITO_USER_POOL_ID` | | Cognito user pool ID (Cognito mode) |
| `AWS_COGNITO_CLIENT_ID` | | Cognito app client ID (Cognito mode) |
| `AWS_COGNITO_REGION` | `us-east-1` | AWS region for Cognito |

### Image Storage Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_STORAGE_PROVIDER` | `cloudinary` | `cloudinary` or `s3` |
| `CLOUDINARY_CLOUD_NAME` | | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | | Cloudinary API secret |
| `AWS_S3_BUCKET` | | S3 bucket name |
| `AWS_S3_REGION` | `us-east-1` | S3 bucket region |
| `AWS_ACCESS_KEY_ID` | | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | | AWS secret key |
| `AWS_S3_ENDPOINT_URL` | | For S3-compatible services (MinIO, DO Spaces) |
| `AWS_S3_KEY_PREFIX` | `companion` | Prefix for all S3 keys |

### Integrations

| Variable | Integration |
|----------|------------|
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram companion bot |
| `TELEGRAM_OPS_BOT_TOKEN`, `TELEGRAM_OPS_CHAT_ID` | Telegram ops/admin bot |
| `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` | Google APIs |
| `TAVILY_API_KEY` | Web search |
| `NANOBANANA_API_KEY` | Image generation (Gemini) |
| `RUNCOMFY_API_KEY`, `RUNCOMFY_PERSONAL_DEPLOYMENT_ID` | Image generation (ComfyUI) |
| `DEEPGRAM_API_KEY` | Speech-to-text |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | Text-to-speech |
| `SENTRY_DSN` | Error tracking |
| `NEO4J_GRAPHITI_PASSWORD`, `NEO4J_GRAPHITI_URI` | Knowledge graph |
| `ZEP_API_KEY`, `ZEP_PROJECT_ID` | External memory service |
| `N8N_HOST`, `N8N_API_KEY` | Workflow automation |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `postgres` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `companion` | Database name |
| `POSTGRES_USER` | `companion` | Database user |
| `POSTGRES_PASSWORD` | (required) | Database password |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |

### Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPANION_REACH_OUT_INTERVAL` | `15` | Base proactive messaging interval (minutes) |
| `COMPANION_QUALITY_THRESHOLD` | `4` | Minimum quality score before regeneration |
| `FIREWORKS_MODEL` | `kimi-k2-instruct-0905` | Primary LLM model |
| `FIREWORKS_FALLBACK_MODEL` | `deepseek-v3p2` | Fallback model |
| `IMAGE_PROVIDER` | `nanobanana` | Default image provider |
| `IMAGE_GENERATION_DAILY_LIMIT` | `5` | Max images per day |
