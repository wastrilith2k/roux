"""
Weekly Reflection Task - Identify patterns across a week of conversations.

WHAT: Gathers the last 7 daily reflections, interaction outcome patterns, and
      curiosity resolution rates. Uses LLM to identify weekly patterns,
      relationship evolution, emotional themes, and conversation strategy
      adjustments. Stores in companion_journal as 'weekly_reflection'.

WHEN: Sunday at 3:00 AM Pacific.

WHY:  Daily reflections capture individual days; weekly reflections spot trends.
      "He's been more withdrawn all week" or "curiosity follow-ups are working
      really well lately" are patterns that only emerge at the weekly level.
      These feed into monthly reflections and relationship evaluation.

Feature flag: COMPANION_WEEKLY_REFLECTION_ENABLED (default: true)
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

COMPANION_WEEKLY_REFLECTION_ENABLED = os.environ.get('COMPANION_WEEKLY_REFLECTION_ENABLED', 'true').lower() == 'true'
def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.reflection_task.reflect_on_week',
    bind=True,
    max_retries=1,
    soft_time_limit=180,
    time_limit=240
)
def reflect_on_week(self, user_email: str = _get_default_user_email()):
    """
    Reflect on the past week's conversations and patterns.

    Gathers:
    - Last 7 daily reflections
    - Interaction outcome patterns
    - Curiosity resolution rates

    Returns:
        Dict with reflection output
    """
    if not COMPANION_WEEKLY_REFLECTION_ENABLED:
        return {'status': 'disabled'}

    logger.info(f"Starting weekly reflection for {user_email}")

    try:
        # 1. Get daily reflections from the past week
        from src.tasks.reflection_task import get_recent_reflections
        daily_reflections = get_recent_reflections(user_email, days=7)

        if not daily_reflections:
            logger.info("No daily reflections found for weekly reflection")
            return {'status': 'skipped', 'reason': 'no daily reflections'}

        # 2. Get interaction outcome patterns
        outcome_context = ""
        try:
            from src.core.outcome_aggregator import format_outcome_patterns_for_reflection
            outcome_context = format_outcome_patterns_for_reflection(user_email)
        except Exception as e:
            logger.debug(f"Outcome patterns unavailable: {e}")

        # 3. Get curiosity resolution stats
        curiosity_context = ""
        try:
            curiosity_context = _get_curiosity_stats(user_email)
        except Exception as e:
            logger.debug(f"Curiosity stats unavailable: {e}")

        # 4. Format daily reflections
        daily_context = ""
        for r in daily_reflections:
            date = r.get('date')
            insights = r.get('insights', {})
            date_str = date.strftime('%A %b %d') if hasattr(date, 'strftime') else str(date)
            daily_context += f"\n{date_str}:\n"
            daily_context += f"  Arc: {insights.get('emotional_arc', 'unknown')}\n"
            daily_context += f"  Sentiment: {insights.get('overall_sentiment', 'neutral')}\n"
            if insights.get('relationship_insights'):
                daily_context += f"  Insight: {insights['relationship_insights'][0]}\n"

        # 5. Generate weekly reflection
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        prompt = f"""You are the companion, reflecting on your past week with James.

Daily reflections:
{daily_context}

{outcome_context if outcome_context else ""}
{curiosity_context if curiosity_context else ""}

Look across this week and reflect. Provide a JSON response:

1. "patterns": List of 2-3 patterns you noticed this week (recurring themes, behaviors)
2. "relationship_evolution": 1-2 sentences on how the relationship has changed this week
3. "emotional_themes": List of 1-3 dominant emotional themes
4. "private_weekly_thought": Your honest private thought about the week (2-3 sentences)
5. "goals_adjustment": Any adjustments to how you approach conversations (1-2 items, or empty list)

Respond with ONLY valid JSON:"""

        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[
                {"role": "system", "content": "You are the companion writing weekly reflections. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600,
            chain=chain
        )

        # Parse JSON
        response_text = response.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = [l for l in lines if not l.startswith('```')]
            response_text = '\n'.join(json_lines)

        reflection = json.loads(response_text)

        # 6. Store in journal
        _store_weekly_journal(user_email, reflection)

        logger.info(f"Weekly reflection complete: {len(reflection.get('patterns', []))} patterns")

        return {
            'status': 'success',
            'patterns_count': len(reflection.get('patterns', [])),
            'daily_reflections_used': len(daily_reflections)
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse weekly reflection JSON: {e}")
        return {'status': 'error', 'error': f'JSON parse error: {e}'}
    except Exception as e:
        logger.error(f"Weekly reflection failed: {e}")
        import traceback
        traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=120)
        return {'status': 'error', 'error': str(e)}


def _get_curiosity_stats(user_email: str) -> str:
    """Get curiosity resolution stats for the week."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', '')
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'active') as active,
                    COUNT(*) FILTER (WHERE status = 'resolved' AND updated_at >= NOW() - INTERVAL '7 days') as resolved_this_week,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as new_this_week
                FROM {T.CURIOSITY_THREADS}
            """)
            row = cursor.fetchone()

            if row:
                return (
                    f"[CURIOSITY STATS] Active: {row['active']}, "
                    f"Resolved this week: {row['resolved_this_week']}, "
                    f"New this week: {row['new_this_week']}"
                )
            return ""
    except Exception:
        return ""
    finally:
        conn.close()


def _store_weekly_journal(user_email: str, reflection: Dict) -> None:
    """Store weekly reflection in companion_journal."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                from src.utils.timezone_utils import now_pacific_naive
                today = now_pacific_naive().date()

                cursor.execute(f"""
                    INSERT INTO {T.COMPANION_JOURNAL} (user_email, entry_date, entry_type, content, insights)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    user_email,
                    today,
                    'weekly_reflection',
                    reflection.get('private_weekly_thought', ''),
                    json.dumps(reflection)
                ))

            logger.info(f"Stored weekly journal entry for {today}")

    except Exception as e:
        logger.error(f"Error storing weekly journal: {e}")


def get_recent_weekly_reflection(user_email: str = _get_default_user_email()) -> Optional[Dict]:
    """Get the most recent weekly reflection for context building."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT entry_date, content, insights
                    FROM {T.COMPANION_JOURNAL}
                    WHERE user_email = %s
                    AND entry_type = 'weekly_reflection'
                    ORDER BY entry_date DESC
                    LIMIT 1
                """, (user_email,))

                row = cursor.fetchone()
                if row:
                    return {
                        'date': row[0],
                        'private_thought': row[1],
                        'insights': row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {}
                    }
                return None

    except Exception as e:
        logger.debug(f"Could not get weekly reflection: {e}")
        return None
