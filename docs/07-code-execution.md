# Code Execution & Web Browsing

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Architecture

The companion can write and execute Python code in a sandboxed container. This gives it access to web search, browsing, Google APIs, and other tools.

```
Conversation Pipeline
  -> _call_llm_with_tools()          # Agentic loop (up to 5 iterations)
  -> CodeExecutor.execute(code)      # HTTP POST to executor container
  -> executor/sandbox.py             # Subprocess with resource limits
  -> tools/ modules                  # Available tool libraries
  -> Result returned to pipeline
```

**Feature flag**: `CODE_EXECUTION_ENABLED` (default: false)

## Executor Service (`executor/app.py`)

Flask app running in isolated container:
- `POST /execute`: Run Python code (timeout 1-120s, max memory 512 MB)
- `GET /health`: Liveness check
- `GET /tools`: List available tool modules

## Sandbox (`executor/sandbox.py`)

Subprocess isolation with:
- Memory limit: 512 MB (configurable)
- Execution timeout: 30s default
- Pre-imported tool modules
- Env var passthrough for API keys

## Available Tool Modules

### Browser (`executor/tools/browser.py` + `executor/browser_manager.py`)

Persistent Playwright (headless Chromium) session with thread-safe access:

| Function | Description |
|----------|-------------|
| `go(url)` | Navigate to URL (auto-adds https://) |
| `click(text, selector)` | Click by visible text or CSS selector |
| `fill(value, name, selector)` | Fill form field |
| `submit()` | Find and click submit button |
| `type_text(text, press_enter)` | Keystroke typing (for search boxes) |
| `back()` | Browser back button |
| `scroll(direction)` | Scroll one viewport up/down |
| `read()` | Extract current page info |
| `close()` | Close session |

Every function returns: `{title, url, text (first 8000 chars), links [{text, href}], forms [{action, method, inputs}]}`

The `BrowserManager` runs a dedicated Playwright worker thread. Operations dispatch via a command queue for thread safety. Session (cookies, history, page state) persists across code execution calls.

### Web Content (`executor/tools/web.py`)

URL content extraction:
- Primary: `trafilatura.fetch_url()` + `trafilatura.extract()` for clean article body
- Fallback: `requests.get()` + regex HTML tag stripping
- Output truncated to `max_length` (default 5000 chars)

### Web Search (`executor/tools/search.py`)

Tavily-powered search:
- `web_search(query, max_results=5)` — General web search
- `search_news(query, max_results=5)` — News-focused search with published dates
- `get_answer(question)` — Direct Q&A via Tavily QNA

### Google Workspace (`executor/tools/google.py`)

Gmail, Calendar, and Docs access via OAuth2 credentials (bind-mounted read-only from `/app/credentials/google_token.json`):
- Send/search email
- Create calendar events
- Create/read Google Docs

### Memory & Fact Search (`executor/tools/memory.py`)

PostgreSQL-backed memory access:
- Full-text search over conversation history (tsvector/tsquery with ts_rank)
- Retrieve facts about a person
- List recent conversation topics
- Store new observations

### Reminders (`executor/tools/reminders.py`)

Companion's private to-do list and follow-up tracker (separate from Google Calendar):
- CRUD for reminders with due dates and notes
- Get due/overdue items
- Drives the proactive-curiosity system and daily check-in topics
- Stored in PostgreSQL `companion_reminders` table

### Image Generation (`executor/tools/image.py`)

Two-path image generation:
- `generate_companion_image()` — RunComfy with the companion's LoRA model (for images featuring the companion)
- `generate_image()` — Google Gemini/Imagen (for scenery, objects, other people — cheaper and faster)
- `check_image_status()` — Poll async RunComfy jobs

### Weather (`executor/tools/weather.py`)

Current conditions and forecast via Open-Meteo (free, no API key required):
- `get_current_weather()` — Temperature (F), feels-like, humidity, wind, conditions
- `get_forecast()` — Daily high/low, precipitation chance, sunrise/sunset
- Location by name via geocoder; WMO weather codes mapped to readable strings
