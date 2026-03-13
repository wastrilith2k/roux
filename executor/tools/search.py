"""
Web Search Tools -- Tavily-powered search, news, and Q&A for the code executor.

WHAT: Three functions: web_search (general search), search_news (news-focused),
      and get_answer (direct Q&A). Returns structured result dicts with title,
      URL, and content snippet.

WHY:  The companion needs real-time information (weather, news, factual
      questions) that isn't in its training data or memory. Tavily provides
      fast, LLM-optimized search results.

HOW:  Each function creates a TavilyClient with the API key from env vars,
      calls the appropriate search method, and normalizes results into a
      consistent list-of-dicts format. Content is truncated to 500 chars
      to keep results concise.
"""
import os
from typing import List, Dict, Optional


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


def web_search(query: str, max_results: int = 5, include_domains: List[str] = None) -> List[Dict]:
    """Search the web for information.

    Args:
        query: Search query (e.g., "latest AI news", "weather in San Francisco")
        max_results: Maximum number of results to return (default 5, max 10)
        include_domains: Optional list of domains to restrict search to

    Returns:
        List of search results, each with:
        - title: Page title
        - url: Page URL
        - content: Snippet/summary of the content

    Example:
        >>> from tools import search
        >>> results = search.web_search("best coffee shops in SF", max_results=3)
        >>> for r in results:
        ...     print(f"{r['title']}: {r['url']}")
    """
    if not TAVILY_API_KEY:
        return [{"error": "Tavily API not configured"}]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)

        # Limit max_results to reasonable range
        max_results = min(max(1, max_results), 10)

        search_params = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic"  # or "advanced" for more thorough search
        }

        if include_domains:
            search_params["include_domains"] = include_domains

        response = client.search(**search_params)

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500]  # Truncate long content
            })

        return results

    except ImportError:
        return [{"error": "tavily-python not installed"}]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def search_news(query: str, max_results: int = 5) -> List[Dict]:
    """Search for recent news articles.

    Args:
        query: News topic to search for
        max_results: Maximum results (default 5)

    Returns:
        List of news articles with title, url, content, and published_date

    Example:
        >>> from tools import search
        >>> news = search.search_news("AI developments", max_results=3)
        >>> for article in news:
        ...     print(f"{article['title']}")
    """
    if not TAVILY_API_KEY:
        return [{"error": "Tavily API not configured"}]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)

        max_results = min(max(1, max_results), 10)

        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            topic="news"
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500],
                "published_date": item.get("published_date", "")
            })

        return results

    except ImportError:
        return [{"error": "tavily-python not installed"}]
    except Exception as e:
        return [{"error": f"News search failed: {str(e)}"}]


def get_answer(question: str) -> Dict:
    """Get a direct answer to a question using Tavily's QA feature.

    Args:
        question: A question to answer (e.g., "What is the capital of France?")

    Returns:
        {"answer": "...", "sources": [...]}

    Example:
        >>> from tools import search
        >>> result = search.get_answer("What time does the Super Bowl start?")
        >>> print(result['answer'])
    """
    if not TAVILY_API_KEY:
        return {"error": "Tavily API not configured"}

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.qna_search(query=question)

        return {
            "answer": response if isinstance(response, str) else response.get("answer", ""),
            "sources": []  # QnA doesn't always return sources
        }

    except ImportError:
        return {"error": "tavily-python not installed"}
    except Exception as e:
        return {"error": f"QA failed: {str(e)}"}
