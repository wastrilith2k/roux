"""
Semantic Memory Search - Vector search over conversation history.

WHAT: Provides semantic search over the messages table using pgvector for
fast cosine-similarity retrieval, with optional Fireworks Qwen3 reranking
for improved relevance ordering.

WHY: The companion needs to recall specific past messages that are
semantically relevant to the current conversation. Embedding-based search
finds matches that keyword search would miss (e.g., "beverage preferences"
matching a message about "peppermint tea").

HOW it fits:
  - context_builder.py calls get_context_for_message() to retrieve relevant
    past messages for inclusion in the LLM prompt.
  - episodic_memory.py handles the write path (embedding messages at
    insert time); this module handles the read path (searching at query time).

Two-stage retrieval:
  1. pgvector similarity search: fast, returns 3x candidates.
  2. Fireworks Qwen3 reranker: reorders by true relevance (optional,
     controlled by ENABLE_RERANKING env var).

Usage:
    from src.memory.semantic_search import search_memory

    results = search_memory("What does James like to drink?", email="user@example.com")
    for r in results:
        print(f"{r['similarity']:.0%}: {r['message_text'][:100]}")
"""
import os
import logging
from typing import List, Dict, Optional

from src.memory.embeddings import generate_embedding
from src.database.db import get_db

logger = logging.getLogger(__name__)

# Enable/disable reranking via environment variable
ENABLE_RERANKING = os.environ.get('ENABLE_RERANKING', 'true').lower() == 'true'
RERANK_MULTIPLIER = int(os.environ.get('RERANK_MULTIPLIER', '3'))  # Get 3x candidates for reranking


def search_memory(query: str, email: str = None, limit: int = 10,
                  min_similarity: float = 0.5, use_reranking: bool = None) -> List[Dict]:
    """
    Search conversation history for messages semantically similar to a query.

    Uses two-stage retrieval:
    1. pgvector similarity search (fast, gets 3x candidates)
    2. Fireworks reranking (accurate, reorders by true relevance)

    Args:
        query: Natural language query (e.g., "What does James like to drink?")
        email: Optional user email to filter results
        limit: Max number of results to return
        min_similarity: Minimum cosine similarity threshold (0-1)
        use_reranking: Override ENABLE_RERANKING setting

    Returns:
        List of messages with similarity scores, ordered by relevance
        Each result has: id, sender_name, message_text, timestamp, email, similarity
        If reranking enabled, also includes: rerank_score
    """
    try:
        # Determine if we should use reranking
        do_rerank = use_reranking if use_reranking is not None else ENABLE_RERANKING

        # Get more candidates if we're going to rerank
        fetch_limit = limit * RERANK_MULTIPLIER if do_rerank else limit

        # Generate embedding for the query
        query_embedding = generate_embedding(query)

        # Search the database (get more candidates for reranking)
        db = get_db()
        results = db.search_similar_messages(
            query_embedding=query_embedding,
            email=email,
            limit=fetch_limit,
            min_similarity=min_similarity
        )

        if not results:
            logger.info(f"Memory search for '{query[:50]}...' returned 0 results")
            return []

        # Apply reranking if enabled
        if do_rerank and len(results) > 1:
            results = _rerank_results(query, results, limit)
            logger.info(f"Memory search for '{query[:50]}...' returned {len(results)} reranked results")
        else:
            results = results[:limit]
            logger.info(f"Memory search for '{query[:50]}...' returned {len(results)} results")

        return results

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        return []


def _rerank_results(query: str, results: List[Dict], limit: int) -> List[Dict]:
    """
    Rerank results using Fireworks Qwen3 Reranker.

    Args:
        query: Original search query
        results: List of results from pgvector search
        limit: Final number of results to return

    Returns:
        Reranked and limited results
    """
    try:
        from src.memory.fireworks_reranker import rerank_results

        # Extract texts for reranking
        texts = [r.get('message_text', '') for r in results]

        # Rerank
        ranked = rerank_results(query, texts, top_n=limit)

        # Reorder results based on ranking
        reordered = []
        for r in ranked:
            idx = r["index"]
            if idx < len(results):
                result_copy = results[idx].copy()
                result_copy["rerank_score"] = r["relevance_score"]
                reordered.append(result_copy)

        logger.debug(f"Reranked {len(results)} results to top {len(reordered)}")
        return reordered

    except ImportError:
        logger.warning("Fireworks reranker not available, using original order")
        return results[:limit]
    except Exception as e:
        logger.warning(f"Reranking failed, using original order: {e}")
        return results[:limit]


def search_memory_formatted(query: str, email: str = None, limit: int = 5) -> str:
    """
    Search memory and return formatted text for LLM context injection.

    Args:
        query: Natural language query
        email: Optional user email
        limit: Max results

    Returns:
        Formatted string ready for prompt injection
    """
    results = search_memory(query, email=email, limit=limit)

    if not results:
        return ""

    lines = ["[RELEVANT MEMORIES]"]
    lines.append(
        "(These are past messages. Check timestamps — older memories "
        "may no longer reflect current reality.)"
    )
    for r in results:
        sender = r['sender_name']
        text = r['message_text'][:200]
        sim = r['similarity']
        ts = r.get('timestamp', '')
        # Format timestamp as relative time
        time_label = ''
        if ts:
            try:
                from datetime import datetime, timezone
                if isinstance(ts, str):
                    msg_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                else:
                    msg_time = ts
                if msg_time.tzinfo is None:
                    from zoneinfo import ZoneInfo
                    msg_time = msg_time.replace(tzinfo=ZoneInfo('America/Los_Angeles'))
                now = datetime.now(timezone.utc)
                diff = now - msg_time.astimezone(timezone.utc)
                if diff.days == 0:
                    time_label = "today"
                elif diff.days == 1:
                    time_label = "yesterday"
                elif diff.days < 7:
                    time_label = f"{diff.days}d ago"
                elif diff.days < 30:
                    time_label = f"{diff.days // 7}w ago"
                else:
                    time_label = f"{diff.days // 30}mo ago"
            except Exception:
                pass
        prefix = f"[{time_label}] " if time_label else ""
        lines.append(f"- {prefix}{sender}: {text}")

    return "\n".join(lines)


def get_context_for_message(user_message: str, email: str, limit: int = 5) -> List[Dict]:
    """
    Get relevant context from memory for responding to a user message.

    This is the function to call from the conversation pipeline.

    Args:
        user_message: The incoming user message
        email: User's email
        limit: Max number of relevant memories to retrieve

    Returns:
        List of relevant message dicts with similarity scores
    """
    # Skip very short messages
    if len(user_message.strip()) < 10:
        return []

    return search_memory(user_message, email=email, limit=limit, min_similarity=0.6)
