"""
Memory and Fact Search Tools -- PostgreSQL-backed memory access for the code executor.

WHAT: Functions to search conversation memories (full-text via tsvector),
      retrieve facts about a person, list recent conversation topics, and
      store new observations. Used by the companion when writing code to
      look up or record information.

WHY:  The companion needs to answer questions like "what restaurants have we
      discussed?" by searching its own memory. These functions expose the
      memory layer in a simple, error-safe API the LLM can call.

HOW:  Direct psycopg2 connections to the shared PostgreSQL instance. Full-text
      search uses PostgreSQL's tsvector/tsquery with ts_rank for relevance
      scoring. All functions return lists of dicts or error dicts (never raise).
"""
import os
import json
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


def _get_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', 5432),
        database=os.environ.get('POSTGRES_DB', 'companion'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', '')
    )


def search_memories(query: str, limit: int = 5) -> List[Dict]:
    """Search the companion's memories for relevant information.

    Performs a text search across stored memories and conversation history.

    Args:
        query: What to search for (e.g., "coffee shops", "his work project")
        limit: Maximum number of results to return (default 5)

    Returns:
        List of dictionaries with keys:
        - content: The memory content
        - timestamp: When it was stored
        - relevance: How relevant it is to the query

    Example:
        >>> from tools import memory
        >>> results = memory.search_memories("favorite restaurants", limit=3)
        >>> for r in results:
        ...     print(r['content'])
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Simple text search - can be enhanced with pgvector embeddings
            cur.execute("""
                SELECT
                    content,
                    created_at as timestamp,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as relevance
                FROM semantic_memories
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY relevance DESC
                LIMIT %s
            """, (query, query, limit))

            results = cur.fetchall()
            conn.close()

            return [dict(r) for r in results]

    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def get_facts_about(person_name: str) -> List[Dict]:
    """Get known facts about a specific person.

    Args:
        person_name: Name of the person (e.g., "James")

    Returns:
        List of facts with keys:
        - fact: The fact content
        - category: Category of the fact (preferences, work, personal, etc.)
        - confidence: How confident we are in this fact

    Example:
        >>> from tools import memory
        >>> facts = memory.get_facts_about("James")
        >>> for f in facts:
        ...     print(f"{f['category']}: {f['fact']}")
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search for person in entity_facts table
            # Use case-insensitive match on entity_name or entity_type
            cur.execute("""
                SELECT
                    fact_text as fact,
                    category,
                    confidence,
                    created_at
                FROM entity_facts
                WHERE LOWER(entity_name) LIKE LOWER(%s)
                   OR LOWER(entity_type) LIKE LOWER(%s)
                ORDER BY confidence DESC, created_at DESC
                LIMIT 50
            """, (f'%{person_name}%', f'%{person_name}%'))

            results = cur.fetchall()
            conn.close()

            return [dict(r) for r in results]

    except Exception as e:
        return [{"error": f"Failed to get facts: {str(e)}"}]


def get_recent_conversation_topics(days: int = 7) -> List[str]:
    """Get topics from recent conversations.

    Args:
        days: Number of days to look back (default 7)

    Returns:
        List of topic strings

    Example:
        >>> from tools import memory
        >>> topics = memory.get_recent_conversation_topics(days=3)
        >>> print(topics)
        ['work stress', 'weekend plans', 'new coffee shop']
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT topic
                FROM conversation_topics
                WHERE created_at > NOW() - INTERVAL '%s days'
                ORDER BY created_at DESC
                LIMIT 20
            """, (days,))

            results = cur.fetchall()
            conn.close()

            return [r['topic'] for r in results if r.get('topic')]

    except Exception as e:
        return [f"Error: {str(e)}"]


def store_observation(observation: str, category: str = "general") -> Dict:
    """Store a new observation about James.

    Use this when the companion notices something important to remember.

    Args:
        observation: What the companion observed (e.g., "James seems tired today")
        category: Category for the observation (general, mood, preference, etc.)

    Returns:
        {"status": "success"} or {"status": "error", "message": "..."}

    Example:
        >>> from tools import memory
        >>> memory.store_observation("James mentioned he prefers morning coffee over afternoon", "preference")
        {'status': 'success'}
    """
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO companion_observations (observation, category, created_at)
                VALUES (%s, %s, NOW())
            """, (observation, category))
            conn.commit()
            conn.close()

            return {"status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
