"""
Memory Search Tool — LLM-invocable tool for on-demand memory retrieval.

WHAT: Provides a `search_memory` tool that the LLM can invoke during generation
      to query the companion's memory backends. Instead of pre-fetching all 24
      context sources on every message, the LLM can request targeted memory
      searches when it actually needs them.

WHY:  Pre-fetching all sources on every message wastes ~3-5K tokens on simple
      greetings and casual replies. Agent-controlled retrieval produces more
      targeted queries because the LLM can decompose complex questions into
      focused sub-queries and skip retrieval entirely for casual messages.

HOW:  The tool routes queries to the appropriate backend:
      - conversations: pgvector semantic search of past messages
      - facts: fact_store with spreading activation
      - episodes: episodic memory similarity search
      - graph: Graphiti knowledge graph search
      - all: parallel search across all backends

Pipeline position: Lightweight context build -> LLM pass 1 (may call tool) ->
                   Tool results injected -> LLM pass 2 (final response)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# --- Tool definition (OpenAI function-calling schema) ---

SEARCH_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Search your memories about the user. Use this when you need to recall "
            "specific conversations, facts, or events. You do NOT need to call this "
            "for casual greetings or small talk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — be specific (e.g., 'conversation about mom's surgery' rather than just 'mom')"
                },
                "source": {
                    "type": "string",
                    "enum": ["all", "conversations", "facts", "episodes", "graph"],
                    "description": "Which memory backend to search. Use 'all' when unsure."
                },
                "time_range": {
                    "type": "string",
                    "enum": ["recent", "last_week", "last_month", "last_year", "all"],
                    "description": "Time window to search. 'recent' = last 3 days, 'all' = entire history."
                }
            },
            "required": ["query"]
        }
    }
}


@dataclass
class MemorySearchResult:
    """Result from a memory search invocation."""
    query: str
    source: str
    results: List[str] = field(default_factory=list)
    result_count: int = 0
    search_time_ms: int = 0
    error: Optional[str] = None


def search_memory(
    query: str,
    user_email: str,
    source: str = "all",
    time_range: str = "all",
    limit: int = 10
) -> MemorySearchResult:
    """
    Execute a memory search across the specified backend(s).

    Args:
        query: Natural language search query
        user_email: User's email for filtering
        source: Backend to search (all, conversations, facts, episodes, graph)
        time_range: Time window filter
        limit: Maximum results per backend

    Returns:
        MemorySearchResult with formatted results for prompt injection
    """
    start_time = time.time()
    all_results = []

    try:
        if source == "all":
            all_results = _search_all(query, user_email, time_range, limit)
        elif source == "conversations":
            all_results = _search_conversations(query, user_email, time_range, limit)
        elif source == "facts":
            all_results = _search_facts(query, user_email, limit)
        elif source == "episodes":
            all_results = _search_episodes(query, user_email, limit)
        elif source == "graph":
            all_results = _search_graph(query, limit)
        else:
            return MemorySearchResult(
                query=query, source=source,
                error=f"Unknown source: {source}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Memory search: query='{query[:50]}' source={source} "
            f"results={len(all_results)} time={elapsed_ms}ms"
        )

        return MemorySearchResult(
            query=query,
            source=source,
            results=all_results[:limit],
            result_count=len(all_results),
            search_time_ms=elapsed_ms
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Memory search failed: {e}")
        return MemorySearchResult(
            query=query, source=source,
            search_time_ms=elapsed_ms,
            error=str(e)
        )


def _search_all(query: str, user_email: str, time_range: str, limit: int) -> List[str]:
    """Search all backends in parallel and merge results."""
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_search_conversations, query, user_email, time_range, limit // 2): "conversations",
            executor.submit(_search_facts, query, user_email, limit // 2): "facts",
            executor.submit(_search_episodes, query, user_email, limit // 4): "episodes",
            executor.submit(_search_graph, query, limit // 4): "graph",
        }
        for future in as_completed(futures, timeout=10):
            source_name = futures[future]
            try:
                source_results = future.result()
                results.extend(source_results)
            except Exception as e:
                logger.warning(f"Memory search source '{source_name}' failed: {e}")

    return results[:limit]


def _search_conversations(query: str, user_email: str, time_range: str, limit: int) -> List[str]:
    """Search past conversations via pgvector semantic search."""
    try:
        from src.memory.semantic_search import search_memory as pgvector_search

        min_similarity = 0.3
        results = pgvector_search(
            query=query,
            email=user_email,
            limit=limit,
            min_similarity=min_similarity
        )

        formatted = []
        for r in results:
            sender = r.get('sender_name', 'Unknown')
            text = r.get('message_text', r.get('content', ''))[:200]
            timestamp = r.get('timestamp', r.get('created_at', ''))
            ts_str = f" ({timestamp})" if timestamp else ""
            formatted.append(f"[conversation{ts_str}] {sender}: {text}")

        return formatted

    except Exception as e:
        logger.warning(f"Conversation search failed: {e}")
        return []


def _search_facts(query: str, user_email: str, limit: int) -> List[str]:
    """Search the fact store with spreading activation."""
    try:
        from src.memory.fact_store import get_fact_store

        store = get_fact_store()
        facts = store.search_with_spreading_activation(
            query=query[:150],
            limit=limit,
            activation_depth=2,
            min_activation=0.25
        )

        formatted = []
        for fact in facts:
            text = fact.get('text', fact.get('fact_text', ''))
            category = fact.get('category', '')
            cat_str = f" ({category})" if category else ""
            formatted.append(f"[fact{cat_str}] {text}")

        return formatted

    except Exception as e:
        logger.warning(f"Fact search failed: {e}")
        return []


def _search_episodes(query: str, user_email: str, limit: int) -> List[str]:
    """Search episodic memory for similar past conversation episodes."""
    try:
        from src.memory.episodic_episodes import (
            get_episode_store,
            detect_topic,
            detect_emotional_state,
        )

        store = get_episode_store()
        topic = detect_topic(query, use_llm=False)
        emotion = detect_emotional_state(query, use_llm=False)

        episodes = store.find_similar(
            topic=topic,
            emotional_state=emotion,
            limit=limit
        )

        formatted = []
        for ep in episodes:
            summary = ep.get('summary', ep.get('description', ''))[:200]
            formatted.append(f"[episode] {summary}")

        return formatted

    except Exception as e:
        logger.warning(f"Episode search failed: {e}")
        return []


def _search_graph(query: str, limit: int) -> List[str]:
    """Search the Graphiti knowledge graph."""
    try:
        from src.memory.graphiti_search import get_graphiti_context_with_importance

        context = get_graphiti_context_with_importance(
            query=query,
            limit=limit
        )

        if context:
            return [f"[knowledge] {context}"]
        return []

    except Exception as e:
        logger.warning(f"Graph search failed: {e}")
        return []


def format_tool_results_for_prompt(result: MemorySearchResult) -> str:
    """Format a MemorySearchResult for injection into the LLM prompt."""
    if result.error:
        return f"\n[Memory search failed: {result.error}]\n"

    if not result.results:
        return (
            f"\n[Memory search for '{result.query}' returned no results. "
            f"You may not have memories about this topic.]\n"
        )

    lines = [
        f"\n[MEMORY SEARCH RESULTS for '{result.query}' "
        f"({result.result_count} results, {result.search_time_ms}ms)]"
    ]
    for r in result.results:
        lines.append(f"  - {r}")
    lines.append("[END MEMORY SEARCH RESULTS]\n")

    return "\n".join(lines)


# Singleton
_tool_instance = None


def get_memory_search_tool():
    """Get the memory search tool definition."""
    return SEARCH_MEMORY_TOOL
