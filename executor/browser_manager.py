"""
Browser Manager — persistent Playwright browser session with thread-safe access.

WHAT: Manages a single Playwright browser instance that persists across multiple
      code execution calls. Provides methods for navigation, clicking, form filling,
      scrolling, and reading page content.

WHY:  LLM tool calls run in sandboxed subprocesses that die after each call. A
      browser session (cookies, history, page state) can't survive between calls.
      This manager runs in the executor's main process with a dedicated thread for
      Playwright (which has thread affinity — can only be used from the thread
      that created it).

HOW:  A single worker thread runs the Playwright event loop. All browser operations
      are dispatched to this thread via a command queue. Flask endpoints call the
      public methods, which enqueue commands and wait for results.
"""

import logging
import threading
import queue
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Page info extraction script — runs in the browser context
PAGE_INFO_SCRIPT = """
() => {
    // Visible text (cleaned up)
    const text = document.body ? document.body.innerText : '';

    // Links (deduplicated, visible only)
    const links = [];
    const seen = new Set();
    document.querySelectorAll('a[href]').forEach(a => {
        const href = a.href;
        const linkText = (a.innerText || a.textContent || '').trim().substring(0, 100);
        if (href && linkText && !seen.has(href) && linkText.length > 0) {
            seen.add(href);
            links.push({text: linkText, href: href});
        }
    });

    // Forms
    const forms = [];
    document.querySelectorAll('form').forEach(form => {
        const inputs = [];
        form.querySelectorAll('input, textarea, select').forEach(el => {
            if (el.type === 'hidden') return;
            inputs.push({
                tag: el.tagName.toLowerCase(),
                type: el.type || '',
                name: el.name || '',
                placeholder: el.placeholder || '',
                value: el.value || '',
                id: el.id || '',
            });
        });
        if (inputs.length > 0) {
            forms.push({
                action: form.action || '',
                method: (form.method || 'GET').toUpperCase(),
                inputs: inputs,
            });
        }
    });

    return {
        title: document.title || '',
        url: window.location.href,
        text: text.substring(0, 8000),
        links: links.slice(0, 50),
        forms: forms.slice(0, 10),
    };
}
"""


class BrowserManager:
    """Thread-safe persistent browser session via Playwright."""

    def __init__(self):
        self._thread = None
        self._queue = queue.Queue()
        self._browser = None
        self._page = None
        self._context = None
        self._playwright = None
        self._lock = threading.Lock()
        self._started = False

    def _ensure_thread(self):
        """Start the Playwright worker thread if not running."""
        with self._lock:
            if not self._started:
                self._thread = threading.Thread(target=self._worker, daemon=True, name='playwright-worker')
                self._thread.start()
                self._started = True

    def _worker(self):
        """Worker thread — owns the Playwright instance."""
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        logger.info("Playwright browser launched")

        while True:
            try:
                cmd, args, result_queue = self._queue.get()

                if cmd == 'SHUTDOWN':
                    break

                try:
                    result = self._dispatch(cmd, args)
                    result_queue.put(('ok', result))
                except Exception as e:
                    logger.error(f"Browser command {cmd} failed: {e}")
                    result_queue.put(('error', str(e)))

            except Exception as e:
                logger.error(f"Browser worker error: {e}")

        # Cleanup
        if self._page:
            self._page.close()
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def _dispatch(self, cmd: str, args: dict) -> dict:
        """Dispatch a command to the appropriate handler. Runs in worker thread."""
        handlers = {
            'go': self._handle_go,
            'click': self._handle_click,
            'fill': self._handle_fill,
            'submit': self._handle_submit,
            'type': self._handle_type,
            'back': self._handle_back,
            'scroll': self._handle_scroll,
            'read': self._handle_read,
            'close': self._handle_close,
        }
        handler = handlers.get(cmd)
        if not handler:
            raise ValueError(f"Unknown browser command: {cmd}")
        return handler(args)

    def _ensure_page(self):
        """Create a page if none exists."""
        if not self._context:
            self._context = self._browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        if not self._page:
            self._page = self._context.new_page()
            self._page.set_default_timeout(30000)

    def _get_page_info(self) -> dict:
        """Extract structured page info."""
        self._ensure_page()
        try:
            return self._page.evaluate(PAGE_INFO_SCRIPT)
        except Exception as e:
            return {
                'title': '',
                'url': self._page.url if self._page else '',
                'text': f'[Error reading page: {e}]',
                'links': [],
                'forms': [],
            }

    def _handle_go(self, args: dict) -> dict:
        self._ensure_page()
        url = args.get('url', '')
        if not url.startswith('http'):
            url = 'https://' + url
        self._page.goto(url, wait_until='domcontentloaded', timeout=30000)
        self._page.wait_for_load_state('networkidle', timeout=10000)
        return self._get_page_info()

    def _handle_click(self, args: dict) -> dict:
        self._ensure_page()
        text = args.get('text', '')
        selector = args.get('selector', '')

        if text:
            # Try exact text match first, then partial
            try:
                self._page.click(f'text="{text}"', timeout=5000)
            except Exception:
                self._page.click(f'text={text}', timeout=10000)
        elif selector:
            self._page.click(selector, timeout=10000)
        else:
            raise ValueError("click requires 'text' or 'selector'")

        self._page.wait_for_load_state('networkidle', timeout=10000)
        return self._get_page_info()

    def _handle_fill(self, args: dict) -> dict:
        self._ensure_page()
        selector = args.get('selector', '')
        name = args.get('name', '')
        value = args.get('value', '')

        if name:
            selector = f'[name="{name}"], #{name}, [placeholder*="{name}" i]'

        if not selector:
            raise ValueError("fill requires 'selector' or 'name'")

        self._page.fill(selector, value, timeout=10000)
        return self._get_page_info()

    def _handle_submit(self, args: dict) -> dict:
        self._ensure_page()
        # Try common submit patterns
        submit_selectors = [
            'button[type="submit"]',
            'input[type="submit"]',
            'button:has-text("Submit")',
            'button:has-text("Search")',
            'button:has-text("Sign")',
            'button:has-text("Log")',
            'button:has-text("Send")',
            'form button',
        ]

        for sel in submit_selectors:
            try:
                self._page.click(sel, timeout=3000)
                self._page.wait_for_load_state('networkidle', timeout=10000)
                return self._get_page_info()
            except Exception:
                continue

        raise ValueError("Could not find a submit button")

    def _handle_type(self, args: dict) -> dict:
        self._ensure_page()
        selector = args.get('selector', '')
        name = args.get('name', '')
        text = args.get('text', '')
        press_enter = args.get('press_enter', False)

        if name:
            selector = f'[name="{name}"], #{name}, [placeholder*="{name}" i]'

        if not selector:
            # Try to find the focused element or first visible input
            selector = 'input:visible, textarea:visible'

        self._page.type(selector, text, timeout=10000)

        if press_enter:
            self._page.keyboard.press('Enter')
            self._page.wait_for_load_state('networkidle', timeout=10000)

        return self._get_page_info()

    def _handle_back(self, args: dict) -> dict:
        self._ensure_page()
        self._page.go_back(wait_until='domcontentloaded', timeout=15000)
        return self._get_page_info()

    def _handle_scroll(self, args: dict) -> dict:
        self._ensure_page()
        direction = args.get('direction', 'down')
        if direction == 'down':
            self._page.evaluate('window.scrollBy(0, window.innerHeight)')
        elif direction == 'up':
            self._page.evaluate('window.scrollBy(0, -window.innerHeight)')
        # Small wait for any lazy-loaded content
        self._page.wait_for_timeout(500)
        return self._get_page_info()

    def _handle_read(self, args: dict) -> dict:
        return self._get_page_info()

    def _handle_close(self, args: dict) -> dict:
        if self._page:
            self._page.close()
            self._page = None
        if self._context:
            self._context.close()
            self._context = None
        return {'title': '', 'url': '', 'text': 'Session closed.', 'links': [], 'forms': []}

    # --- Public API (thread-safe, called from Flask request threads) ---

    def _send_command(self, cmd: str, args: dict = None, timeout: float = 35) -> dict:
        """Send a command to the worker thread and wait for result."""
        self._ensure_thread()
        result_queue = queue.Queue()
        self._queue.put((cmd, args or {}, result_queue))

        try:
            status, result = result_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"Browser command '{cmd}' timed out after {timeout}s")

        if status == 'error':
            raise RuntimeError(result)

        return result

    def go(self, url: str) -> dict:
        return self._send_command('go', {'url': url})

    def click(self, text: str = '', selector: str = '') -> dict:
        return self._send_command('click', {'text': text, 'selector': selector})

    def fill(self, value: str, selector: str = '', name: str = '') -> dict:
        return self._send_command('fill', {'selector': selector, 'name': name, 'value': value})

    def submit(self) -> dict:
        return self._send_command('submit')

    def type_text(self, text: str, selector: str = '', name: str = '', press_enter: bool = False) -> dict:
        return self._send_command('type', {'selector': selector, 'name': name, 'text': text, 'press_enter': press_enter})

    def back(self) -> dict:
        return self._send_command('back')

    def scroll(self, direction: str = 'down') -> dict:
        return self._send_command('scroll', {'direction': direction})

    def read(self) -> dict:
        return self._send_command('read')

    def close(self) -> dict:
        return self._send_command('close')

    def shutdown(self):
        """Shut down the worker thread."""
        if self._started:
            self._queue.put(('SHUTDOWN', {}, queue.Queue()))


# Singleton
_browser_manager = None

def get_browser_manager() -> BrowserManager:
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
    return _browser_manager
