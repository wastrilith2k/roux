"""
Outcome Aggregator -- SQL analytics over interaction outcomes.

WHAT: Queries PostgreSQL for engagement rates by action type, topics that
      "land" (get positive follow-up) vs. topics that "miss" (get deflected
      or ignored), and overall engagement trends over time windows.

WHY:  The personality-evolution and proactive-curiosity systems need data on
      what resonates with the user. This module provides the raw statistics
      that drive those adaptation loops.

HOW:  Direct psycopg2 queries against the messages and interaction_outcomes
      tables. Not routed through the DB abstraction layer because the queries
      are analytics-specific aggregations.

Consumers: reflection Celery tasks, personality evolution, context builder.
"""

import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def _get_connection():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', '')
    )


def get_outcome_patterns(user_email: str, days: int = 30) -> Dict[str, Any]:
    """
    Get engagement rates by action type.

    Returns dict with action_type -> {count, avg_resonance, engagement_breakdown}
    """
    try:
        from psycopg2.extras import RealDictCursor
        conn = _get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT
                    companion_action_type,
                    COUNT(*) as count,
                    AVG(emotional_resonance) as avg_resonance,
                    COUNT(*) FILTER (WHERE engagement_level = 'enthusiastic') as enthusiastic,
                    COUNT(*) FILTER (WHERE engagement_level = 'engaged') as engaged,
                    COUNT(*) FILTER (WHERE engagement_level = 'brief') as brief,
                    COUNT(*) FILTER (WHERE engagement_level = 'deflected') as deflected,
                    COUNT(*) FILTER (WHERE engagement_level = 'ignored') as ignored
                FROM interaction_outcomes
                WHERE user_email = %s
                AND created_at >= NOW() - INTERVAL '%s days'
                GROUP BY companion_action_type
                ORDER BY count DESC
            """, (user_email, days))

            patterns = {}
            for row in cursor.fetchall():
                action = row['companion_action_type'] or 'general'
                total = row['count']
                patterns[action] = {
                    'count': total,
                    'avg_resonance': round(float(row['avg_resonance'] or 0), 2),
                    'engagement_rate': round(
                        (row['enthusiastic'] + row['engaged']) / total * 100, 1
                    ) if total > 0 else 0,
                    'deflection_rate': round(
                        (row['deflected'] + row['ignored']) / total * 100, 1
                    ) if total > 0 else 0,
                }

        conn.close()
        return patterns

    except Exception as e:
        logger.debug(f"Could not get outcome patterns: {e}")
        return {}


def get_topics_that_land(user_email: str, days: int = 30) -> List[str]:
    """Get topics with high engagement."""
    try:
        from psycopg2.extras import RealDictCursor
        conn = _get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT companion_topic, AVG(emotional_resonance) as avg_res,
                       COUNT(*) as cnt
                FROM interaction_outcomes
                WHERE user_email = %s
                AND created_at >= NOW() - INTERVAL '%s days'
                AND engagement_level IN ('enthusiastic', 'engaged')
                AND companion_topic IS NOT NULL AND companion_topic != ''
                GROUP BY companion_topic
                HAVING COUNT(*) >= 2
                ORDER BY avg_res DESC
                LIMIT 5
            """, (user_email, days))

            return [row['companion_topic'] for row in cursor.fetchall()]

        conn.close()

    except Exception as e:
        logger.debug(f"Could not get landing topics: {e}")
        return []


def get_topics_that_miss(user_email: str, days: int = 30) -> List[str]:
    """Get topics that tend to get deflected or ignored."""
    try:
        from psycopg2.extras import RealDictCursor
        conn = _get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT companion_topic, AVG(emotional_resonance) as avg_res,
                       COUNT(*) as cnt
                FROM interaction_outcomes
                WHERE user_email = %s
                AND created_at >= NOW() - INTERVAL '%s days'
                AND engagement_level IN ('deflected', 'ignored', 'brief')
                AND companion_topic IS NOT NULL AND companion_topic != ''
                GROUP BY companion_topic
                HAVING COUNT(*) >= 2
                ORDER BY avg_res ASC
                LIMIT 5
            """, (user_email, days))

            return [row['companion_topic'] for row in cursor.fetchall()]

        conn.close()

    except Exception as e:
        logger.debug(f"Could not get missing topics: {e}")
        return []


def format_outcome_patterns_for_reflection(user_email: str) -> str:
    """
    Format outcome patterns for injection into daily reflection prompt.

    Returns a concise summary of engagement patterns.
    """
    patterns = get_outcome_patterns(user_email, days=7)
    landing = get_topics_that_land(user_email, days=14)
    missing = get_topics_that_miss(user_email, days=14)

    if not patterns and not landing and not missing:
        return ""

    lines = ["[INTERACTION OUTCOME PATTERNS - How James responds to different approaches]"]

    if patterns:
        lines.append("\nEngagement by approach (last 7 days):")
        for action, data in patterns.items():
            lines.append(
                f"  {action}: {data['count']}x, "
                f"{data['engagement_rate']}% engaged, "
                f"avg resonance {data['avg_resonance']}"
            )

    if landing:
        lines.append(f"\nTopics that land well: {', '.join(landing)}")

    if missing:
        lines.append(f"Topics that miss: {', '.join(missing)}")

    return '\n'.join(lines)
