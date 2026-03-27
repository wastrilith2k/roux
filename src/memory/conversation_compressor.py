"""
Conversation Compressor — Compress old conversation history into session summaries.

WHAT: When conversation history exceeds a threshold, compresses older messages
      into a structured summary using an LLM. The summary captures topics,
      decisions, emotional arc, and open threads — preserving narrative context
      that would otherwise be lost when messages age out of the history window.

WHY:  The conversation history is limited to MESSAGE_HISTORY_LIMIT (25) messages.
      In long sessions (50+ messages), the first half vanishes entirely — the
      companion has no awareness of what was discussed 30 minutes ago. This
      creates an "amnesia cliff" where the companion forgets established context.
      Compression preserves the narrative arc in ~500 tokens instead of losing it.

HOW:  _get_conversation_history_structured() fetches more messages than the
      display limit. If the total exceeds COMPRESSION_THRESHOLD, messages
      beyond the most recent RECENT_MESSAGES_KEEP are compressed into a
      session summary via LLM. The summary is cached in Redis (keyed by
      user email + a hash of the compressed message IDs) so compression
      only runs once per batch. Rolling compression appends new batches
      to the existing summary.

See issue #23 for design rationale.
"""

import hashlib
import json
import logging
import os
from typing import Optional, List, Dict

import redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# When total messages exceed this, compress older ones into a summary.
COMPRESSION_THRESHOLD = int(os.getenv('COMPRESSION_THRESHOLD', '25'))

# Keep this many recent messages as raw turns (not compressed).
RECENT_MESSAGES_KEEP = int(os.getenv('RECENT_MESSAGES_KEEP', '15'))

# How many messages to fetch from DB (larger than history limit to have
# material to compress).
FETCH_LIMIT = int(os.getenv('COMPRESSION_FETCH_LIMIT', '50'))

# Redis TTL for cached summaries (24 hours — sessions rarely span longer).
SUMMARY_CACHE_TTL = int(os.getenv('SUMMARY_CACHE_TTL', str(24 * 3600)))

# ---------------------------------------------------------------------------
# Redis connection (reuse message_condenser's pattern)
# ---------------------------------------------------------------------------

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# Compression prompt
# ---------------------------------------------------------------------------

COMPRESSION_PROMPT = """Summarize the following conversation excerpt into a structured session summary.
This summary will be used as context for an ongoing conversation — the most recent messages
are provided separately, so focus on what came BEFORE them.

Format your summary EXACTLY like this (keep the bracketed headers):

[Session summary — {message_count} earlier messages]
- Topics discussed: <comma-separated list of topics>
- Key decisions/agreements: <what was decided or agreed upon, or "None" if nothing>
- Emotional arc: <how the emotional tone evolved>
- Open threads: <topics mentioned but not fully explored, or "None">
- Important context: <any facts, names, or details the companion should remember>

Be concise. Target 100-200 words. Preserve concrete details (names, times, plans, promises).
Do NOT include pleasantries or filler. Do NOT repeat the most recent messages.

Conversation to summarize:
"""


# ---------------------------------------------------------------------------
# Core compression logic
# ---------------------------------------------------------------------------

def _build_cache_key(user_email: str, message_ids: List[int]) -> str:
    """Build a Redis cache key from user email and the set of message IDs being compressed."""
    id_hash = hashlib.md5(
        ",".join(str(mid) for mid in sorted(message_ids)).encode()
    ).hexdigest()[:12]
    return f"session_summary:{user_email}:{id_hash}"


def compress_messages(
    messages: List[Dict],
    existing_summary: str = "",
) -> str:
    """
    Compress a list of messages into a session summary using an LLM.

    Args:
        messages: List of message dicts with 'sender_name' and 'message_text' keys.
        existing_summary: If rolling compression, the previous summary to extend.

    Returns:
        Structured session summary string.
    """
    if not messages:
        return existing_summary

    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()

    # Format messages for the LLM
    formatted_lines = []
    for msg in messages:
        sender = msg.get('sender_name', 'Unknown')
        text = msg.get('message_text', '')
        if text:
            formatted_lines.append(f"{sender}: {text}")

    conversation_text = "\n".join(formatted_lines)

    # If there's an existing summary, include it for rolling compression
    if existing_summary:
        prompt_text = (
            f"EXISTING SUMMARY (extend this with new information):\n{existing_summary}\n\n"
            f"NEW MESSAGES TO INCORPORATE:\n{conversation_text}"
        )
    else:
        prompt_text = conversation_text

    message_count = len(messages)
    system_prompt = COMPRESSION_PROMPT.replace("{message_count}", str(message_count))

    try:
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        chain = get_resilient_provider_chain(
            primary="openai:gpt-4o-mini",
            fallback="openai:gpt-4o-mini",
        )

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        summary = generate_sync(
            messages=llm_messages,
            temperature=0.2,
            max_tokens=500,
            chain=chain,
        )

        if summary and len(summary.strip()) > 20:
            logger.info(
                f"Compressed {message_count} messages into summary "
                f"({len(summary)} chars)"
            )
            return summary.strip()

        logger.warning("Compression returned empty/short summary, skipping")
        return existing_summary

    except Exception as e:
        logger.error(f"Conversation compression failed: {e}")
        return existing_summary


def get_or_create_session_summary(
    user_email: str,
    all_messages: List[Dict],
    recent_count: int = None,
) -> tuple:
    """
    Given a full list of messages (oldest-first), split into a compressed
    summary + recent raw messages.

    Args:
        user_email: User's email for cache keying.
        all_messages: All messages in chronological order (oldest first).
        recent_count: How many recent messages to keep raw. Defaults to RECENT_MESSAGES_KEEP.

    Returns:
        Tuple of (session_summary, recent_messages) where:
        - session_summary: Compressed summary string (empty if not enough messages)
        - recent_messages: List of message dicts to use as raw conversation turns
    """
    if recent_count is None:
        recent_count = RECENT_MESSAGES_KEEP

    total = len(all_messages)

    # Not enough messages to trigger compression — return all as-is
    if total <= COMPRESSION_THRESHOLD:
        return "", all_messages

    # Split: older messages get compressed, recent stay raw
    older_messages = all_messages[:-recent_count]
    recent_messages = all_messages[-recent_count:]

    # Build cache key from the IDs of messages being compressed
    older_ids = [msg.get('id', 0) for msg in older_messages if msg.get('id')]

    if not older_ids:
        return "", all_messages

    cache_key = _build_cache_key(user_email, older_ids)

    # Check Redis cache first
    try:
        r = _get_redis()
        cached = r.get(cache_key)
        if cached:
            logger.debug(f"Session summary cache hit ({len(older_ids)} messages)")
            return cached, recent_messages
    except Exception as e:
        logger.warning(f"Redis cache check failed: {e}")

    # Cache miss — compress via LLM
    summary = compress_messages(older_messages)

    # Cache the result
    if summary:
        try:
            r = _get_redis()
            r.set(cache_key, summary, ex=SUMMARY_CACHE_TTL)
            logger.info(f"Cached session summary for {user_email} ({len(older_ids)} messages)")
        except Exception as e:
            logger.warning(f"Failed to cache session summary: {e}")

    return summary, recent_messages
