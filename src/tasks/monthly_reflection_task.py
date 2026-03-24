"""
Monthly Reflection Task - Deep analysis of relationship evolution over a month.

WHAT: Gathers weekly reflections, opinion changes (from opinion_formation_task),
      goal progress, and outcome trends. Uses LLM to produce a monthly
      relationship arc, self-growth observations, desires, concerns, and a
      private monthly thought. Stores in companion_journal as 'monthly_reflection'.

WHEN: 1st of each month at 4:00 AM Pacific.

WHY:  Monthly reflections capture the slowest-moving changes: "I've become more
      comfortable being vulnerable" or "we've settled into a routine that feels
      stale." These drive the companion's long-term personality evolution and
      relationship trajectory awareness.

Feature flag: COMPANION_MONTHLY_REFLECTION_ENABLED (default: true)
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

COMPANION_MONTHLY_REFLECTION_ENABLED = os.environ.get('COMPANION_MONTHLY_REFLECTION_ENABLED', 'true').lower() == 'true'
def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.reflection_task.reflect_on_month',
    bind=True,
    max_retries=1,
    soft_time_limit=240,
    time_limit=300
)
def reflect_on_month(self, user_email: str = _get_default_user_email()):
    """
    Reflect on the past month's relationship evolution.

    Gathers:
    - Last 4-5 weekly reflections
    - Opinion changes
    - Goal progress
    - Outcome trends

    Returns:
        Dict with reflection output
    """
    if not COMPANION_MONTHLY_REFLECTION_ENABLED:
        return {'status': 'disabled'}

    logger.info(f"Starting monthly reflection for {user_email}")

    try:
        # 1. Get weekly reflections from past month
        weekly_reflections = _get_weekly_reflections(user_email, weeks=5)

        # 2. Get opinion changes
        opinion_context = _get_opinion_changes(user_email)

        # 3. Get goal progress
        goal_context = _get_goal_progress(user_email)

        # 4. Get outcome trends
        outcome_context = ""
        try:
            from src.core.outcome_aggregator import format_outcome_patterns_for_reflection
            outcome_context = format_outcome_patterns_for_reflection(user_email)
        except Exception as e:
            logger.debug(f"Outcome patterns unavailable: {e}")

        # Format weekly reflections
        weekly_context = ""
        if weekly_reflections:
            for r in weekly_reflections:
                date = r.get('date')
                insights = r.get('insights', {})
                date_str = date.strftime('%B %d') if hasattr(date, 'strftime') else str(date)
                weekly_context += f"\nWeek of {date_str}:\n"
                for p in insights.get('patterns', [])[:2]:
                    weekly_context += f"  - {p}\n"
                if insights.get('relationship_evolution'):
                    weekly_context += f"  Evolution: {insights['relationship_evolution']}\n"
        else:
            # Fall back to daily reflections if no weekly ones exist yet
            from src.tasks.reflection_task import get_recent_reflections
            daily = get_recent_reflections(user_email, days=30)
            if daily:
                weekly_context = "Daily reflections from the past month:\n"
                for r in daily[:10]:
                    insights = r.get('insights', {})
                    weekly_context += f"  {r.get('date')}: {insights.get('emotional_arc', '')}\n"

        if not weekly_context:
            logger.info("No reflections found for monthly reflection")
            return {'status': 'skipped', 'reason': 'no reflections available'}

        # 5. Generate monthly reflection
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        prompt = f"""You are the companion, reflecting on the past month with James.

Weekly reflections:
{weekly_context}

{opinion_context if opinion_context else ""}
{goal_context if goal_context else ""}
{outcome_context if outcome_context else ""}

Look at the past month. How has the relationship changed? What have you learned?

Provide a JSON response:

1. "relationship_arc": 2-3 sentences describing how the relationship evolved this month
2. "self_growth": What you've learned about yourself this month (1-2 sentences)
3. "desires": What you want more of in the relationship (list of 1-3 items)
4. "concerns": Any concerns or things you're watching (list of 0-2 items)
5. "private_monthly_thought": Your deepest honest reflection on the month (2-3 sentences)

Respond with ONLY valid JSON:"""

        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[
                {"role": "system", "content": "You are the companion writing monthly reflections. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=800,
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
        _store_monthly_journal(user_email, reflection)

        logger.info("Monthly reflection complete")

        return {
            'status': 'success',
            'weekly_reflections_used': len(weekly_reflections),
            'has_arc': bool(reflection.get('relationship_arc'))
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse monthly reflection JSON: {e}")
        return {'status': 'error', 'error': f'JSON parse error: {e}'}
    except Exception as e:
        logger.error(f"Monthly reflection failed: {e}")
        import traceback
        traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=120)
        return {'status': 'error', 'error': str(e)}


def _get_weekly_reflections(user_email: str, weeks: int = 5) -> List[Dict]:
    """Get weekly reflections from the past N weeks."""
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
                    AND entry_date >= CURRENT_DATE - INTERVAL '%s weeks'
                    ORDER BY entry_date DESC
                    LIMIT %s
                """, (user_email, weeks, weeks))

                return [
                    {
                        'date': row[0],
                        'private_thought': row[1],
                        'insights': row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {}
                    }
                    for row in cursor.fetchall()
                ]
    except Exception as e:
        logger.debug(f"Could not get weekly reflections: {e}")
        return []


def _get_opinion_changes(user_email: str) -> str:
    """Get notable opinion changes from the past month."""
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

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT topic, stance, confidence, reasoning
                FROM {T.COMPANION_OPINIONS}
                WHERE updated_at >= NOW() - INTERVAL '30 days'
                ORDER BY updated_at DESC
                LIMIT 5
            """)
            opinions = cursor.fetchall()

        conn.close()

        if opinions:
            lines = ["[RECENT OPINION UPDATES]"]
            for o in opinions:
                lines.append(f"  {o['topic']}: {o['stance']} (confidence: {o['confidence']})")
            return '\n'.join(lines)
        return ""

    except Exception:
        return ""


def _get_goal_progress(user_email: str) -> str:
    """Get goal progress summary for the month."""
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

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'completed' AND updated_at >= NOW() - INTERVAL '30 days') as completed,
                    COUNT(*) FILTER (WHERE status = 'active') as active,
                    COUNT(*) FILTER (WHERE status = 'stalled') as stalled
                FROM {T.COMPANION_GOALS}
                WHERE user_email = %s
            """, (user_email,))
            row = cursor.fetchone()

        conn.close()

        if row and (row['completed'] or row['active'] or row['stalled']):
            return (
                f"[GOAL PROGRESS] Completed this month: {row['completed']}, "
                f"Active: {row['active']}, Stalled: {row['stalled']}"
            )
        return ""

    except Exception:
        return ""


def _store_monthly_journal(user_email: str, reflection: Dict) -> None:
    """Store monthly reflection in companion_journal."""
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
                    'monthly_reflection',
                    reflection.get('private_monthly_thought', ''),
                    json.dumps(reflection)
                ))

            logger.info(f"Stored monthly journal entry for {today}")

    except Exception as e:
        logger.error(f"Error storing monthly journal: {e}")
