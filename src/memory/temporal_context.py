"""
Temporal Context - Always-on recent significant events section.

WHAT: Provides a "Recent Significant Events" block that is unconditionally
included in every prompt, regardless of whether the current message
semantically matches those events. Combines high-importance facts from the
last 2 weeks with keyword-matched notable messages from the last 3 days.

WHY: Semantic search only surfaces memories that match the current query.
If James asks "how are you?" after Jesse ran away yesterday, semantic search
won't retrieve the runaway event because the query doesn't mention Jesse.
Temporal context ensures the companion always remembers what happened
recently, preventing the "Jesse ran away but the companion doesn't
remember" problem.

HOW it fits:
  - context_builder.py calls get_temporal_context() every turn and injects
    the result into the system prompt unconditionally.
  - Two complementary data sources:
    1. High-importance facts (from facts table, importance >= 6, last 14 days)
    2. Keyword-matched notable messages (from messages table, last 72 hours)

Key distinction from semantic search:
  - Semantic search: query-dependent -- finds what matches the current message.
  - Temporal context: time-dependent -- includes what happened recently.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


def get_recent_significant_events(
    user_email: str,
    days_back: int = 14,
    min_importance: int = 4,  # Lowered from 6 to catch more life changes
    max_events: int = 10
) -> List[Dict[str, Any]]:
    """
    Get recent significant facts (high importance, recent time).

    Args:
        user_email: User to get events for
        days_back: How many days to look back
        min_importance: Minimum importance score to include
        max_events: Maximum events to return

    Returns:
        List of fact dicts with: subject, object, importance, created_at
    """
    try:
        from src.database.db import get_db
        db = get_db()

        cutoff = datetime.now(PST) - timedelta(days=days_back)

        with db._get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT id, subject, predicate, object, importance, created_at
                FROM facts
                WHERE (user_email = %s OR user_email IS NULL)
                  AND created_at > %s
                  AND importance >= %s
                ORDER BY importance DESC, created_at DESC
                LIMIT %s
            """, (user_email, cutoff, min_importance, max_events))

            facts = cursor.fetchall()
            cursor.close()

        return [dict(f) for f in facts]

    except Exception as e:
        logger.error(f"Error getting recent significant events: {e}")
        return []


def get_recent_notable_messages(
    user_email: str,
    hours_back: int = 48,
    max_messages: int = 5
) -> List[Dict[str, Any]]:
    """
    Get recent user messages that likely contain significant life events.

    Uses keyword matching (ILIKE ANY) to find messages mentioning crisis
    events, career changes, health issues, milestones, or school events.
    This catches important events even if the fact extraction pipeline
    has not yet processed them.

    Args:
        user_email: User to get messages for
        hours_back: How many hours to look back
        max_messages: Maximum messages to return

    Returns:
        List of message dicts with id, sender_name, message_text, timestamp
    """
    try:
        from src.database.db import get_db
        db = get_db()

        cutoff = datetime.now(PST) - timedelta(hours=hours_back)

        # Keywords organized by category for maintainability.
        # These are used as ILIKE patterns against message_text.
        significant_keywords = [
            # Crisis events
            'police', 'hospital', 'emergency', 'accident', 'crisis',
            'ran away', 'took off', 'running away', 'missing', 'found',
            'arrested', 'court', 'evicted',
            # Career/job events
            'interview', 'job', 'offer', 'hired', 'fired', 'quit',
            'promotion', 'salary', 'laid off', 'contract',
            # Health events
            'sick', 'doctor', 'therapy', 'therapist', 'appointment',
            'diagnosis', 'medication', 'surgery',
            # Life milestones
            'pregnant', 'baby', 'born', 'died', 'passed away', 'funeral',
            'moved', 'moving', 'married', 'engaged',
            'divorce', 'separated', 'breakup',
            # School/kid events
            'school', 'grades', 'expelled', 'suspended', 'homework',
            'teacher', 'principal', 'detention',
            # Relationship events
            'girlfriend', 'boyfriend', 'dating', 'broke up',
            # Known company names
            'act-on', 'leantaas', 'snapsheet', 'cavallo',
        ]

        with db._get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build keyword search using ANY with LIKE patterns
            # This is safer than f-string interpolation
            patterns = [f'%{kw}%' for kw in significant_keywords]

            cursor.execute("""
                SELECT id, sender_name, message_text, timestamp
                FROM messages
                WHERE email = %s
                  AND timestamp > %s
                  AND sender_name = 'User'
                  AND LOWER(message_text) LIKE ANY(%s)
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_email, cutoff, patterns, max_messages))

            messages = cursor.fetchall()
            cursor.close()

        return [dict(m) for m in messages]

    except Exception as e:
        logger.error(f"Error getting recent notable messages: {e}")
        return []


def format_temporal_context(
    user_email: str,
    include_facts: bool = True,
    include_messages: bool = True
) -> Optional[str]:
    """
    Format recent significant events for inclusion in prompt.

    Returns formatted section or None if nothing significant found.
    """
    sections = []

    # Get recent significant facts
    if include_facts:
        facts = get_recent_significant_events(
            user_email,
            days_back=14,
            min_importance=6,
            max_events=15  # Increased from 8 to include more topic diversity
        )

        if facts:
            fact_lines = ["**Recent Important Facts** (last 2 weeks):"]
            for f in facts:
                # Calculate days ago
                created = f['created_at']
                if created.tzinfo is None:
                    created = created.replace(tzinfo=PST)
                days_ago = (datetime.now(PST) - created).days

                time_str = "today" if days_ago == 0 else f"{days_ago}d ago"
                fact_lines.append(f"- [{time_str}] {f['object']}")

            sections.append('\n'.join(fact_lines))

    # Get recent notable messages (crisis events, etc.)
    if include_messages:
        messages = get_recent_notable_messages(
            user_email,
            hours_back=72,  # 3 days for messages
            max_messages=3
        )

        if messages:
            msg_lines = ["**Recent Notable Events** (last 3 days):"]
            for m in messages:
                ts = m['timestamp']
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=PST)
                hours_ago = (datetime.now(PST) - ts).total_seconds() / 3600

                if hours_ago < 24:
                    time_str = f"{int(hours_ago)}h ago"
                else:
                    time_str = f"{int(hours_ago/24)}d ago"

                # Truncate long messages
                text = m['message_text'][:150]
                if len(m['message_text']) > 150:
                    text += "..."

                msg_lines.append(f"- [{time_str}] James said: {text}")

            sections.append('\n'.join(msg_lines))

    if not sections:
        return None

    return "[RECENT SIGNIFICANT EVENTS - DO NOT FORGET THESE]\n\n" + "\n\n".join(sections)


def get_temporal_context(user_email: str) -> Optional[str]:
    """
    Get temporal context for a user.

    This is the main entry point for the context builder.
    """
    return format_temporal_context(user_email)
