"""
Web Page Fetching Tools -- URL content extraction for the code executor.

WHAT: fetch_url() downloads a web page and extracts clean readable text using
      trafilatura (primary) or basic HTML tag stripping (fallback). Returns
      title, content, URL, and content length with configurable truncation.

WHY:  The companion needs to read web pages the user shares or that search
      results link to. Raw HTML is unusable in a prompt; trafilatura extracts
      the article body cleanly.

HOW:  trafilatura.fetch_url() + trafilatura.extract() for clean extraction.
      Falls back to requests + regex tag stripping if trafilatura fails or
      isn't installed. Content is truncated to max_length (default 5000 chars)
      with a "[Truncated]" marker.
"""
import re
from typing import Dict, Optional

try:
    import trafilatura
except ImportError:
    trafilatura = None

import requests


def fetch_url(url: str, max_length: int = 5000, include_links: bool = False) -> Dict:
    """Fetch a web page and extract its readable text content.

    Args:
        url: The URL to fetch (must start with http:// or https://)
        max_length: Maximum characters of content to return (default 5000)
        include_links: Whether to include hyperlinks in the output (default False)

    Returns:
        Dict with:
        - title: Page title (if found)
        - content: Extracted text content
        - url: The fetched URL
        - length: Length of full extracted content before truncation

    Example:
        >>> from tools import web
        >>> page = web.fetch_url("https://en.wikipedia.org/wiki/Portland,_Oregon")
        >>> print(page['title'])
        >>> print(page['content'][:200])
    """
    if not url or not isinstance(url, str):
        return {"error": "URL is required"}

    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    max_length = min(max(100, max_length), 20000)

    if trafilatura is None:
        return _fallback_fetch(url, max_length)

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"error": f"Failed to download page: {url}"}

        content = trafilatura.extract(
            downloaded,
            include_links=include_links,
            include_formatting=True,
            favor_recall=True,
        )

        if not content:
            return _fallback_fetch(url, max_length)

        title = _extract_title(downloaded)
        full_length = len(content)
        if len(content) > max_length:
            content = content[:max_length] + f"\n\n[Truncated — {full_length} chars total]"

        return {
            "title": title,
            "content": content,
            "url": url,
            "length": full_length,
        }

    except Exception as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}


def _extract_title(html: str) -> str:
    """Extract title from HTML."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        import html as html_mod
        return html_mod.unescape(match.group(1)).strip()
    return ""


def _fallback_fetch(url: str, max_length: int) -> Dict:
    """Fallback: fetch with requests and do basic HTML stripping."""
    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CompanionBot/1.0)"},
    )
    resp.raise_for_status()

    html = resp.text
    title = ""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        import html as html_mod
        title = html_mod.unescape(match.group(1)).strip()

    # Strip tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    import html as html_mod
    text = html_mod.unescape(text)

    full_length = len(text)
    if len(text) > max_length:
        text = text[:max_length] + f"\n\n[Truncated — {full_length} chars total]"

    return {
        "title": title,
        "content": text,
        "url": url,
        "length": full_length,
    }
