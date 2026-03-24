# Infrastructure & Deployment

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Docker Services

```yaml
# docker-compose.yml — 9 services
graphiti-neo4j:    # Neo4j 5.26 — knowledge graph (port 7688 Bolt)
postgres:          # PostgreSQL + pgvector — primary data (port 5432)
redis:             # Redis — Celery broker + pub/sub (port 6379)
agent-service:     # Flask+SocketIO backend (port 5000)
celery-worker:     # Async task processing (pool=solo, 4 workers)
companion-cli:     # Terminal chat client (stdin/tty)
n8n:               # Workflow automation (port 5678)
caddy:             # Reverse proxy + SSL (ports 80/443)
code-executor:     # Sandboxed Python execution (port 5001, 1GB mem limit)
```

## Caddy Routing

| Path | Target | Notes |
|------|--------|-------|
| `/socket.io/*` | agent-service:5010 | 5-minute read/write timeout for WebSocket |
| `/api/*` | agent-service:5010 | REST endpoints |
| `/oauth/*` | agent-service:5010 | Google OAuth callbacks |
| `/webhook/*` | agent-service:5010 | n8n and external webhooks |
| `n8n.{domain}` | n8n:5678 | Workflow automation UI |

Security headers: `X-Frame-Options: SAMEORIGIN`, `X-Content-Type-Options: nosniff`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy: strict-origin-when-cross-origin`. SSL via automatic Let's Encrypt.

## Application Startup Sequence (`web_chat.py`)

1. Eventlet monkey-patch (must be first import)
2. Flask app factory (`create_app`)
3. Sentry initialization (optional)
4. Middleware + error handler registration
5. Route blueprint registration (auth, chat, settings, cost, integrations, approval, observe)
6. Background service launch:
   - `UnifiedScheduler` — APScheduler for cron/interval jobs
   - `AlwaysOnService` — Telegram bridge + proactive messaging loop
   - Ops Bot — admin Telegram commands
   - Celery Beat — scheduled task dispatch
7. Redis pub/sub listeners (image generation events, interjections)
8. Flask+SocketIO run on port 5000

## Development vs Production

| Aspect | Development (`docker-compose.dev.yml`) | Production (`docker-compose.prod.yml`) |
|--------|----------------------------------------|----------------------------------------|
| Ports | Offset (5433, 6380, 5001) | Standard (5432, 6379, 5000) |
| Database | `companion_dev` | `companion` |
| Restart | `no` | `unless-stopped` |
| Celery Beat | Disabled | Enabled (scheduled tasks) |
| Sentry | Disabled | Enabled |
| Gmail/Calendar | Disabled | Enabled |
