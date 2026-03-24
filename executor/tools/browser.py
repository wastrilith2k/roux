"""
Browser — persistent web browsing within a single session.

Browse the web like a real person: navigate pages, click links, fill forms,
search, scroll, and go back. The session persists across code executions
(cookies, history, page state all maintained).

Usage:
    from tools.browser import go, click, fill, submit, type_text, back, scroll, read, close

    # Navigate to a page
    page = go("https://news.ycombinator.com")
    print(page['title'])      # "Hacker News"
    print(page['links'][:5])  # First 5 links

    # Click a link by text
    page = click("Show HN")

    # Search
    page = go("https://google.com")
    page = type_text("python playwright tutorial", press_enter=True)

    # Fill a form
    page = fill("user@example.com", name="email")
    page = fill("mypassword", name="password")
    page = submit()

    # Read current page
    page = read()
    print(page['text'][:500])

    # Scroll down
    page = scroll()

    # Go back
    page = back()

    # Close session (frees resources)
    close()

Every function returns a dict with:
    title  — page title
    url    — current URL
    text   — visible page text (first 8000 chars)
    links  — list of {text, href}
    forms  — list of {action, method, inputs: [{tag, type, name, placeholder}]}
"""

import requests
import json

_BASE = "http://localhost:5001/browser"
_TIMEOUT = 35


def _post(endpoint, data=None):
    """Send a POST request to the browser API."""
    try:
        resp = requests.post(f"{_BASE}/{endpoint}", json=data or {}, timeout=_TIMEOUT)
        result = resp.json()
        if 'error' in result and resp.status_code >= 400:
            print(f"[browser error] {result['error']}")
        return result
    except requests.exceptions.ConnectionError:
        print("[browser error] Cannot connect to browser service. Is the executor running?")
        return {'error': 'Connection failed', 'title': '', 'url': '', 'text': '', 'links': [], 'forms': []}
    except Exception as e:
        print(f"[browser error] {e}")
        return {'error': str(e), 'title': '', 'url': '', 'text': '', 'links': [], 'forms': []}


def _get(endpoint):
    """Send a GET request to the browser API."""
    try:
        resp = requests.get(f"{_BASE}/{endpoint}", timeout=_TIMEOUT)
        return resp.json()
    except Exception as e:
        print(f"[browser error] {e}")
        return {'error': str(e), 'title': '', 'url': '', 'text': '', 'links': [], 'forms': []}


def go(url: str) -> dict:
    """Navigate to a URL. Auto-adds https:// if missing."""
    return _post("go", {"url": url})


def click(text: str = "", selector: str = "") -> dict:
    """Click an element by its visible text or CSS selector.

    Examples:
        click("Sign Up")           # Click by link/button text
        click(selector="#submit")   # Click by CSS selector
    """
    return _post("click", {"text": text, "selector": selector})


def fill(value: str, name: str = "", selector: str = "") -> dict:
    """Fill a form field with a value.

    Args:
        value: The text to enter
        name: Input field name attribute (preferred)
        selector: CSS selector (fallback)

    Examples:
        fill("user@example.com", name="email")
        fill("password123", name="password")
    """
    return _post("fill", {"value": value, "name": name, "selector": selector})


def submit() -> dict:
    """Find and click the submit button on the current page."""
    return _post("submit")


def type_text(text: str, name: str = "", selector: str = "", press_enter: bool = False) -> dict:
    """Type text keystroke by keystroke (for search boxes, etc).

    Args:
        text: Text to type
        name: Input field name (optional)
        selector: CSS selector (optional)
        press_enter: Press Enter after typing (for search)

    Examples:
        type_text("python tutorial", press_enter=True)  # Search
        type_text("hello", name="message")               # Type in a specific field
    """
    return _post("type", {"text": text, "name": name, "selector": selector, "press_enter": press_enter})


def back() -> dict:
    """Go back to the previous page."""
    return _post("back")


def scroll(direction: str = "down") -> dict:
    """Scroll one viewport down (or up).

    Args:
        direction: 'down' (default) or 'up'
    """
    return _post("scroll", {"direction": direction})


def read() -> dict:
    """Read the current page without navigating. Returns page info."""
    return _get("read")


def close() -> dict:
    """Close the browser session and free resources."""
    return _post("close")
