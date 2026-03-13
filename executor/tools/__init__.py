"""
Companion Tool Modules -- Python APIs the companion calls via code execution.

WHAT: Seven tool modules the companion can import when writing Python code:
      memory, reminders, image, search, web, google, weather.

WHY:  The companion's code-executor sandbox needs a clean, well-documented
      API surface. Each module wraps an external service (Tavily, RunComfy,
      Google APIs, Open-Meteo, PostgreSQL) into simple functions the LLM can
      call without knowing implementation details.

HOW:  The companion writes Python code like `from tools import memory` and
      calls functions that return plain dicts. The code-executor container
      has these modules on its PYTHONPATH. Each module handles its own auth
      (env vars) and error handling (returns error dicts, never raises).

Available modules:
  memory    -- Search memories and facts (PostgreSQL full-text search)
  reminders -- Companion's internal calendar/reminder CRUD
  image     -- Generate images (RunComfy for companion portraits, Gemini for general)
  search    -- Web search and Q&A via Tavily
  web       -- Fetch and extract text from URLs (trafilatura)
  google    -- Google Workspace: Gmail send/search, Calendar events, Docs CRUD
  weather   -- Current conditions and forecast via Open-Meteo (free, no API key)

Usage:
    from tools import memory
    results = memory.search_memories("coffee shops we've discussed")

    from tools import search
    results = search.web_search("latest AI news")

    from tools import google
    google.send_email("someone@example.com", "Hello", "Hi there!")
"""

__version__ = "0.1.0"

# Make modules importable
from . import memory
from . import reminders
from . import image
from . import search
from . import web
from . import google
from . import weather
