"""
Code Executor Client -- HTTP interface to the sandboxed code-execution container.

WHAT: Sends Python code to the companion-code-executor container for execution
      and returns stdout/stderr/return_code. Also defines EXECUTE_CODE_TOOL
      with usage examples for the LLM's tool-calling interface.

WHY:  The companion can write and run Python code (web searches, Google API
      calls, calculations, image generation) but that code must run in a
      sandboxed container with resource limits, not in the main Flask process.

HOW:  Simple HTTP POST to `http://companion-code-executor:8080/execute` with
      a JSON body containing the code string and optional timeout. The executor
      container (executor/app.py) runs the code in a subprocess sandbox with
      memory and time limits.

Singleton: `get_code_executor()` at module bottom.
"""
import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Execute Python code in the sandboxed executor container."""

    # How long to cache a negative availability check before retrying (seconds)
    AVAILABILITY_TTL = 60

    def __init__(self, base_url: Optional[str] = None, default_timeout: int = 30):
        """Initialize the code executor client.

        Args:
            base_url: URL of the code executor service (default from env)
            default_timeout: Default execution timeout in seconds
        """
        self.base_url = base_url or os.environ.get(
            "CODE_EXECUTOR_URL",
            "http://companion-code-executor:5001"
        )
        self.default_timeout = default_timeout
        self._available = None  # Cached availability status
        self._available_checked_at = 0.0  # Timestamp of last check

    def is_available(self) -> bool:
        """Check if the code executor service is available.

        Caches positive results indefinitely (service is up).
        Caches negative results for AVAILABILITY_TTL seconds, then retries.
        This prevents a single failed health check from permanently disabling tools.
        """
        import time as _time

        # If previously available, trust the cache
        if self._available is True:
            return True

        # If previously unavailable, retry after TTL expires
        if self._available is False:
            elapsed = _time.time() - self._available_checked_at
            if elapsed < self.AVAILABILITY_TTL:
                return False
            logger.info("Code executor availability TTL expired, rechecking...")

        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            self._available = response.status_code == 200
        except Exception as e:
            logger.warning(f"Code executor not available: {e}")
            self._available = False

        self._available_checked_at = _time.time()

        if self._available:
            logger.info("Code executor is available")
        else:
            logger.warning(
                f"Code executor unavailable (will retry in {self.AVAILABILITY_TTL}s)"
            )

        return self._available

    def execute(self, code: str, timeout: Optional[int] = None) -> str:
        """Execute Python code in the sandbox and return results.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default 30)

        Returns:
            String output from execution (stdout + any errors)
        """
        timeout = timeout or self.default_timeout

        try:
            response = requests.post(
                f"{self.base_url}/execute",
                json={"code": code, "timeout": timeout},
                timeout=timeout + 10  # Extra time for HTTP overhead
            )

            data = response.json()

            if data.get("success"):
                output = data.get("output", "").strip()
                # Include stderr as warning if present
                if data.get("error"):
                    output += f"\n[Warning: {data['error']}]"
                return output if output else "(No output)"
            else:
                error = data.get("error", "Unknown error")
                return f"Execution failed: {error}"

        except requests.Timeout:
            return f"Execution timed out after {timeout}s"
        except requests.ConnectionError:
            return "Code executor service unavailable"
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"Execution error: {str(e)}"

    def list_tools(self) -> dict:
        """Get list of available tool modules."""
        try:
            response = requests.get(
                f"{self.base_url}/tools",
                timeout=10
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return {"error": str(e)}

    def test(self) -> str:
        """Run a basic test of the executor."""
        try:
            response = requests.get(
                f"{self.base_url}/execute/test",
                timeout=30
            )
            data = response.json()
            if data.get("success"):
                return data.get("output", "Test passed")
            else:
                return f"Test failed: {data.get('error')}"
        except Exception as e:
            return f"Test failed: {str(e)}"


# Singleton instance
_executor: Optional[CodeExecutor] = None


def get_code_executor() -> CodeExecutor:
    """Get or create the code executor singleton."""
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor


# Tool definition for LLM
EXECUTE_CODE_TOOL = {
    "name": "execute_code",
    "description": """Execute Python code. ALWAYS start with: from tools import <module>

Available modules: weather, search, web, google, memory, reminders, image, browser

WEATHER:
  from tools import weather
  print(weather.get_current_weather("Portland"))
  print(weather.get_forecast("Portland", days=3))

WEB SEARCH:
  from tools import search
  print(search.web_search("query", max_results=5))
  print(search.search_news("query"))
  print(search.get_answer("question"))

WEB PAGE READING:
  from tools import web
  print(web.fetch_url("https://example.com"))
  print(web.fetch_url("https://example.com", max_length=10000, include_links=True))

GOOGLE (Gmail/Calendar/Docs):
  from tools import google
  google.send_email(to, subject, body)
  print(google.search_emails("is:unread", max_results=3))
  google.create_calendar_event(summary, start_iso, end_iso, description)
  google.create_document(title, content)

MEMORY:
  from tools import memory
  print(memory.search_memories("query"))
  print(memory.get_facts_about("User"))

REMINDERS:
  from tools import reminders
  print(reminders.get_reminders())
  reminders.add_reminder(title, due_date, notes)

IMAGE:
  from tools import image
  print(image.generate_companion_image("prompt"))
  print(image.generate_image("prompt"))

BROWSER (persistent session — navigate, click, fill forms, search):
  from tools.browser import go, click, fill, submit, type_text, back, scroll, read, close
  page = go("https://news.ycombinator.com")
  print(page['title'], page['links'][:5])
  page = click("Show HN")                          # Click by link text
  page = go("https://google.com")
  page = type_text("python tutorial", press_enter=True)  # Search
  page = fill("user@email.com", name="email")       # Fill form field
  page = submit()                                    # Submit form
  page = scroll()                                    # Scroll down
  page = back()                                      # Go back
  page = read()                                      # Read current page
  close()                                            # End session
  # Session persists between calls — cookies, history, page state maintained.
  # Every function returns: {title, url, text, links [{text,href}], forms [{action,method,inputs}]}

ALWAYS print() results so output is captured.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code. Modules are pre-imported. Use print() to show results."
            }
        },
        "required": ["code"]
    }
}
