"""
Communication Learning — Outcome-driven communication preferences.

Analyzes interaction outcome data to learn what communication patterns
work well with the user and what patterns don't. These learned preferences
are injected into the companion's response prompt so it naturally adapts its
communication style.

This closes the feedback loop:
  Companion's message → User's response → Outcome tracking → Pattern analysis
  → Communication preferences → Future companion messages (better ones)

The preferences are NOT hardcoded rules. They're LLM-analyzed insights
that get passed as context to the response LLM, which then naturally
incorporates them. Example:
  "The user engages more when you ask about their projects than when you
   share articles. They tend to give brief responses to emotional
   check-ins before 10 AM."

Feature flag: COMPANION_COMMUNICATION_LEARNING_ENABLED (default: true)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.core.clock import now as clock_now
from src.database import tables as T

logger = logging.getLogger(__name__)

COMPANION_COMMUNICATION_LEARNING_ENABLED = os.environ.get(
    'COMPANION_COMMUNICATION_LEARNING_ENABLED', 'true'
).lower() == 'true'

# File to cache learned preferences (refreshed weekly)
PREFERENCES_CACHE_FILE = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'communication_preferences.json'
)

# Cache TTL in hours
CACHE_TTL_HOURS = 24 * 7  # 7 days (refreshed by weekly reflection)


def get_communication_preferences(user_email: str) -> Optional[str]:
    """
    Get learned communication preferences for injection into response prompt.

    Returns cached preferences if fresh, otherwise returns None.
    This is designed to be called on every response generation (fast path).
    """
    if not COMPANION_COMMUNICATION_LEARNING_ENABLED:
        return None

    try:
        if os.path.exists(PREFERENCES_CACHE_FILE):
            with open(PREFERENCES_CACHE_FILE, 'r') as f:
                cache = json.load(f)

            # Check freshness
            generated_at = cache.get('generated_at', '')
            if generated_at:
                gen_time = datetime.fromisoformat(generated_at)
                age_hours = (clock_now() - gen_time).total_seconds() / 3600
                if age_hours < CACHE_TTL_HOURS:
                    return cache.get('preferences_text', '')

    except Exception as e:
        logger.debug(f"Could not read communication preferences cache: {e}")

    return None


def generate_communication_preferences(user_email: str) -> Optional[str]:
    """
    Analyze interaction outcomes and generate communication preferences.

    Uses the outcome aggregator data + LLM analysis to produce natural
    language preferences that guide the companion's communication style.

    This should be called during reflection tasks (not per-message).
    """
    if not COMPANION_COMMUNICATION_LEARNING_ENABLED:
        return None

    try:
        from src.core.outcome_aggregator import (
            get_outcome_patterns,
            get_topics_that_land,
            get_topics_that_miss,
        )

        # Gather outcome data
        patterns_7d = get_outcome_patterns(user_email, days=7)
        patterns_30d = get_outcome_patterns(user_email, days=30)
        landing = get_topics_that_land(user_email, days=30)
        missing = get_topics_that_miss(user_email, days=30)

        if not patterns_30d and not landing and not missing:
            return None

        # Get time-of-day patterns
        time_patterns = _get_time_patterns(user_email)

        # Get proactive vs reactive engagement comparison
        proactive_engagement = _get_proactive_engagement(user_email)

        # Build context for LLM analysis
        context_parts = []

        if patterns_30d:
            context_parts.append("ENGAGEMENT BY APPROACH (30 days):")
            for action, data in patterns_30d.items():
                context_parts.append(
                    f"  {action}: {data['count']}x, "
                    f"engagement: {data['engagement_rate']}%, "
                    f"resonance: {data['avg_resonance']}, "
                    f"deflection: {data['deflection_rate']}%"
                )

        if patterns_7d:
            context_parts.append("\nRECENT TREND (7 days):")
            for action, data in patterns_7d.items():
                context_parts.append(
                    f"  {action}: engagement: {data['engagement_rate']}%"
                )

        if landing:
            context_parts.append(f"\nTOPICS THAT LAND WELL: {', '.join(landing)}")
        if missing:
            context_parts.append(f"TOPICS THAT MISS: {', '.join(missing)}")

        if time_patterns:
            context_parts.append(f"\nTIME PATTERNS: {time_patterns}")

        if proactive_engagement:
            context_parts.append(f"\nPROACTIVE ENGAGEMENT: {proactive_engagement}")

        if not context_parts:
            return None

        # LLM analysis
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        prompt = f"""Analyze these interaction outcome patterns and generate communication preferences.

{chr(10).join(context_parts)}

Based on this data, generate 3-5 brief communication insights.
These should be natural observations about what works, not rigid rules.

Format as a short paragraph (3-5 sentences), written in second person
("you tend to...", "the user responds well when...").

Focus on:
- What communication approaches get the best engagement
- What topics or styles to lean into vs back off from
- Any timing patterns worth noting
- Any recent trends that differ from long-term patterns

Keep it concise and actionable. No preamble or analysis explanation."""

        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[
                {"role": "system", "content": "Generate brief communication preference insights. Be concise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
            chain=chain,
        )

        preferences_text = response.strip() if response else None

        if preferences_text:
            _cache_preferences(preferences_text)
            logger.info(f"Generated communication preferences: {preferences_text[:80]}...")

        return preferences_text

    except Exception as e:
        logger.warning(f"Communication preference generation failed: {e}")
        return None


def _get_time_patterns(user_email: str) -> str:
    """Get engagement patterns by time of day."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        from src.database.schema_manager import set_search_path
        set_search_path(conn, user_email)

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT
                    CASE
                        WHEN EXTRACT(hour FROM created_at) BETWEEN 6 AND 11 THEN 'morning'
                        WHEN EXTRACT(hour FROM created_at) BETWEEN 12 AND 17 THEN 'afternoon'
                        ELSE 'evening'
                    END as period,
                    AVG(emotional_resonance) as avg_resonance,
                    COUNT(*) as cnt
                FROM {T.INTERACTION_OUTCOMES}
                WHERE user_email = %s
                AND created_at >= NOW() - INTERVAL '30 days'
                GROUP BY period
                ORDER BY avg_resonance DESC
            """, (user_email,))

            rows = cursor.fetchall()
            if rows:
                parts = [f"{r['period']}: resonance {float(r['avg_resonance'] or 0):.2f} ({r['cnt']}x)" for r in rows]
                conn.close()
                return ', '.join(parts)

        conn.close()
    except Exception as e:
        logger.debug(f"Could not get time patterns: {e}")
    return ""


def _get_proactive_engagement(user_email: str) -> str:
    """Compare engagement for proactive vs reactive messages."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        from src.database.schema_manager import set_search_path
        set_search_path(conn, user_email)

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT
                    CASE WHEN esme_action_type IN ('topic_introduction', 'curiosity_followup')
                         THEN 'proactive' ELSE 'reactive' END as msg_type,
                    AVG(emotional_resonance) as avg_resonance,
                    COUNT(*) FILTER (WHERE engagement_level IN ('enthusiastic', 'engaged')) as engaged,
                    COUNT(*) as total
                FROM {T.INTERACTION_OUTCOMES}
                WHERE user_email = %s
                AND created_at >= NOW() - INTERVAL '30 days'
                GROUP BY msg_type
            """, (user_email,))

            rows = cursor.fetchall()
            if rows:
                parts = []
                for r in rows:
                    total = r['total']
                    rate = round(r['engaged'] / total * 100, 1) if total > 0 else 0
                    parts.append(f"{r['msg_type']}: {rate}% engaged ({total}x)")
                conn.close()
                return ', '.join(parts)

        conn.close()
    except Exception as e:
        logger.debug(f"Could not get proactive engagement: {e}")
    return ""


def _cache_preferences(text: str) -> None:
    """Cache preferences to file."""
    try:
        os.makedirs(os.path.dirname(PREFERENCES_CACHE_FILE), exist_ok=True)
        cache = {
            'generated_at': clock_now().isoformat(),
            'preferences_text': text,
        }
        with open(PREFERENCES_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.debug(f"Could not cache preferences: {e}")


def format_preferences_for_prompt(user_email: str) -> str:
    """
    Get preferences formatted for injection into the response prompt.

    Returns empty string if no preferences available.
    """
    prefs = get_communication_preferences(user_email)
    if not prefs:
        return ""

    return (
        f"[COMMUNICATION INSIGHTS (from past interaction patterns)]:\n"
        f"{prefs}\n"
        f"(Use these naturally — don't announce them or force changes.)\n"
    )
