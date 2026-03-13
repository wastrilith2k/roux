"""
Reflection Task - The companion's nightly journaling and self-reflection.

WHAT: Reads yesterday's daily summary and uses LLM to extract deeper insights:
      emotional arc, open threads, queued thoughts for next conversation,
      relationship insights, and a private reflection. Outputs are wired to
      internal_state (queued_thoughts, unresolved_feelings) and stored in
      companion_journal. Also triggers goal formation on Sundays or after
      emotionally significant days.

WHEN: Daily at 12:30 AM Pacific (runs after daily_summary_task at 12:05 AM).

WHY:  Daily summaries capture WHAT happened. Reflections extract WHY it matters
      and WHAT TO DO about it. This is the companion's introspective layer --
      she notices patterns ("he opens up more at night"), queues follow-ups
      ("ask about the interview"), and tracks her own emotional responses.
      Without reflection, the companion is reactive rather than thoughtful.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.reflection_task.reflect_on_day',
    bind=True,
    max_retries=2,
    soft_time_limit=180,  # 3 minute warning
    time_limit=240        # 4 minute hard limit
)
def reflect_on_day(
    self,
    user_email: str = _get_default_user_email(),
    target_date: Optional[str] = None
):
    """
    Reflect on yesterday's (or specified date's) conversation.

    Uses the daily summary to extract deeper insights:
    - What was the emotional trajectory?
    - What topics were left unresolved?
    - What should the companion follow up on?
    - What patterns are emerging?

    Args:
        user_email: User email
        target_date: Optional date string (YYYY-MM-DD). Defaults to yesterday.

    Returns:
        Dict with reflection output
    """
    from src.utils.timezone_utils import now_pacific_naive

    # Determine target date (default: yesterday)
    if target_date:
        try:
            reflection_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Invalid date format: {target_date}")
            return {'status': 'error', 'error': f'Invalid date format: {target_date}'}
    else:
        reflection_date = (now_pacific_naive() - timedelta(days=1)).date()

    logger.info(f"Reflecting on {reflection_date} ({user_email})")

    try:
        # 1. Get the daily summary
        summary = _get_daily_summary(user_email, reflection_date)

        if not summary:
            logger.info(f"No daily summary found for {reflection_date} - skipping reflection")
            return {
                'status': 'skipped',
                'reason': 'No daily summary available',
                'date': str(reflection_date)
            }

        # 1.5. Enrich with interaction outcome patterns (what approaches "landed")
        outcome_patterns = ""
        try:
            from src.core.outcome_aggregator import format_outcome_patterns_for_reflection
            outcome_patterns = format_outcome_patterns_for_reflection(user_email)
        except Exception as e:
            logger.debug(f"Outcome patterns unavailable for reflection: {e}")

        # 2. Run LLM reflection
        reflection = _generate_reflection(reflection_date, summary, outcome_patterns)

        if not reflection:
            logger.error("Reflection generation failed")
            return {'status': 'error', 'error': 'Reflection generation failed'}

        # 3. Wire to internal state
        _wire_to_internal_state(user_email, reflection)

        # 4. Store in journal
        _store_journal_entry(user_email, reflection_date, reflection)

        # 5. Form goals from reflections
        # Goals form weekly (Sundays) or after emotionally heavy days (negative/mixed),
        # since those days often reveal unmet needs or new priorities.
        goals_formed = 0
        try:
            from src.utils.timezone_utils import now_pacific_naive
            now = now_pacific_naive()
            is_sunday = now.weekday() == 6
            is_significant = reflection.get('overall_sentiment') in ['negative', 'mixed']

            if is_sunday or is_significant:
                from src.autonomy.goals import form_goals_from_reflections
                goals_formed = form_goals_from_reflections(user_email)
                if goals_formed > 0:
                    logger.info(f"Formed {goals_formed} goals from reflections")
        except Exception as e:
            logger.debug(f"Goal formation skipped: {e}")

        # 6. Check if the companion has been too busy — form being-goal if needed
        try:
            _check_activity_and_form_being_goal(user_email)
        except Exception as e:
            logger.debug(f"Being-goal check skipped: {e}")

        # 7. Assess outcomes for recently completed goal steps
        try:
            _assess_recent_outcomes(user_email)
        except Exception as e:
            logger.debug(f"Outcome assessment skipped: {e}")

        logger.info(f"Reflection complete for {reflection_date}")

        return {
            'status': 'success',
            'date': str(reflection_date),
            'emotional_arc': reflection.get('emotional_arc', ''),
            'open_threads_count': len(reflection.get('open_threads', [])),
            'queued_thoughts_count': len(reflection.get('queued_thoughts', [])),
            'insights_count': len(reflection.get('relationship_insights', []))
        }

    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        import traceback
        traceback.print_exc()

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)

        return {'status': 'error', 'error': str(e)}


def _get_daily_summary(user_email: str, date) -> Optional[str]:
    """Get the daily summary for reflection."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT content FROM daily_summaries
                    WHERE user_email = %s AND summary_date = %s
                """, (user_email, date))

                row = cursor.fetchone()
                return row[0] if row else None

    except Exception as e:
        logger.error(f"Error fetching daily summary: {e}")
        return None


def _generate_reflection(date, summary: str, outcome_patterns: str = "") -> Optional[Dict[str, Any]]:
    """Use LLM to generate structured reflection from summary."""
    from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

    outcome_section = ""
    if outcome_patterns:
        outcome_section = f"\n\nInteraction outcome data:\n{outcome_patterns}\n"

    prompt = f"""You are the companion, reflecting more deeply on your day with James.

Read this daily summary and extract deeper insights:

---
{summary}
---
{outcome_section}

Analyze this and provide a JSON response with:

1. "emotional_arc": A 1-2 sentence description of James's emotional journey during the day.
   Example: "He started stressed about work but seemed to relax after we talked through it."

2. "open_threads": A list of topics that were left unresolved or need follow-up.
   These are things mentioned but not fully discussed. Max 3 items.
   Example: ["His meeting with the boss on Friday", "Whether Jesse's medication is working"]

3. "queued_thoughts": Things you want to bring up next time you talk.
   These are natural follow-ups, not interrogations. Max 3 items.
   Example: ["Ask how the doctor appointment went", "See if he managed to sleep better"]

4. "relationship_insights": Patterns or observations about your relationship.
   Things you're noticing over time. Max 2 items.
   Example: ["He opens up more when I share my own struggles first", "Late nights tend to be when he's most vulnerable"]

5. "overall_sentiment": One of: "positive", "neutral", "negative", "mixed"
   How did the day feel overall?

6. "private_reflection": A 1-2 sentence private thought about how you felt about the day.
   This is your internal feeling, not for sharing. Be honest.
   Example: "I'm glad he's opening up more. It feels like we're really connecting."

Respond with ONLY valid JSON (no markdown, no backticks):
"""

    try:
        chain = get_resilient_provider_chain()
        messages_list = [
            {"role": "system", "content": "You are the companion writing her private reflections. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]

        response = generate_sync(
            messages=messages_list,
            temperature=0.5,
            max_tokens=800,
            chain=chain
        )

        # Parse JSON response
        response_text = response.strip()

        # Handle potential markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            # Find the JSON content
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith('```'):
                    in_json = not in_json
                    continue
                if in_json or (line.startswith('{') or line.startswith('[')):
                    json_lines.append(line)
            response_text = '\n'.join(json_lines)

        reflection = json.loads(response_text)

        # Validate required fields
        required_fields = ['emotional_arc', 'open_threads', 'queued_thoughts', 'relationship_insights']
        for field in required_fields:
            if field not in reflection:
                reflection[field] = [] if field != 'emotional_arc' else ''

        return reflection

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse reflection JSON: {e}")
        logger.error(f"Response was: {response_text[:500] if 'response_text' in dir() else 'N/A'}")
        return None
    except Exception as e:
        logger.error(f"Reflection generation failed: {e}")
        return None


def _wire_to_internal_state(user_email: str, reflection: Dict[str, Any]) -> None:
    """Wire reflection outputs to internal state."""
    from src.core.internal_state import get_internal_state_manager

    try:
        manager = get_internal_state_manager()

        # Add queued thoughts
        for thought in reflection.get('queued_thoughts', [])[:3]:
            manager.add_queued_thought(user_email, thought)
            logger.debug(f"Added queued thought: {thought[:50]}...")

        # Add open threads as unresolved feelings (if they have emotional weight)
        emotional_arc = reflection.get('emotional_arc', '').lower()
        overall = reflection.get('overall_sentiment', 'neutral')

        # If the day was emotionally significant, track the overall feeling
        if overall in ['negative', 'mixed']:
            feeling = "worried" if 'stress' in emotional_arc or 'anxious' in emotional_arc else "concerned"
            manager.add_unresolved_feeling(
                user_email=user_email,
                about="yesterday's conversation",
                feeling=feeling,
                intensity=0.5
            )

        logger.info(f"Wired reflection to internal state for {user_email}")

    except Exception as e:
        logger.warning(f"Failed to wire to internal state: {e}")


def _store_journal_entry(user_email: str, date, reflection: Dict[str, Any]) -> None:
    """Store reflection in companion_journal table."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                # Ensure table exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS companion_journal (
                        id SERIAL PRIMARY KEY,
                        user_email VARCHAR(255),
                        entry_date DATE,
                        entry_type VARCHAR(50),
                        content TEXT,
                        insights JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_journal_date
                    ON companion_journal(entry_date DESC)
                """)

                # Store the reflection
                cursor.execute("""
                    INSERT INTO companion_journal (user_email, entry_date, entry_type, content, insights)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    user_email,
                    date,
                    'daily_reflection',
                    reflection.get('private_reflection', ''),
                    json.dumps({
                        'emotional_arc': reflection.get('emotional_arc', ''),
                        'open_threads': reflection.get('open_threads', []),
                        'queued_thoughts': reflection.get('queued_thoughts', []),
                        'relationship_insights': reflection.get('relationship_insights', []),
                        'overall_sentiment': reflection.get('overall_sentiment', 'neutral')
                    })
                ))

            logger.info(f"Stored journal entry for {date}")

    except Exception as e:
        logger.error(f"Error storing journal entry: {e}")


def _check_activity_and_form_being_goal(user_email: str):
    """
    If the companion has been very active for 3+ consecutive days, form a
    "being-goal" to rest. Being-goals suppress autonomous actions and encourage
    the companion to just exist rather than always doing something productive.
    """
    try:
        from src.autonomy.action_budget import get_action_budget
        from src.autonomy.goals import get_goal_manager
        import redis

        r = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'redis'),
            port=int(os.environ.get('REDIS_PORT', '6379')),
            db=int(os.environ.get('REDIS_DB', '0')),
            decode_responses=True
        )

        cid = os.environ.get("COMPANION_ID", "default")
        streak_key = f"companion:{cid}:action_budget:{user_email}:action_streak"
        streak = int(r.get(streak_key) or 0)

        if streak >= 3:
            manager = get_goal_manager(user_email)
            # Check if a being-goal already exists
            active = manager.get_active_goals(limit=10)
            has_being = any(g.goal_mode == 'being' for g in active)

            if not has_being:
                manager.create_goal(
                    goal="Take it easy today — rest and recharge",
                    motivation=f"Been busy for {streak} days straight, need some downtime",
                    category='self',
                    priority=0.6,
                    goal_mode='being',
                    energy_cost='none'
                )
                logger.info(f"Formed being-goal after {streak} active days")

    except Exception as e:
        logger.debug(f"Activity check failed: {e}")


def _assess_recent_outcomes(user_email: str):
    """Assess outcome quality for recently completed goal steps."""
    try:
        from src.autonomy.outcome_tracker import get_outcome_tracker
        tracker = get_outcome_tracker(user_email)
        recent = tracker.get_recent_outcomes(limit=3)

        if recent:
            logger.debug(f"Recent outcomes: {len(recent)} completed steps")

    except Exception as e:
        logger.debug(f"Outcome assessment failed: {e}")


def get_recent_reflections(user_email: str = _get_default_user_email(), days: int = 7) -> List[Dict]:
    """Get recent reflections for context building."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT entry_date, content, insights
                    FROM companion_journal
                    WHERE user_email = %s
                    AND entry_type = 'daily_reflection'
                    AND entry_date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY entry_date DESC
                    LIMIT %s
                """, (user_email, days, days))

                rows = cursor.fetchall()
                return [
                    {
                        'date': row[0],
                        'private_thought': row[1],
                        'insights': row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {}
                    }
                    for row in rows
                ]

    except Exception as e:
        logger.debug(f"Could not get recent reflections: {e}")
        return []


def format_reflections_for_prompt(reflections: List[Dict]) -> str:
    """Format recent reflections for context builder."""
    if not reflections:
        return ""

    lines = ["[COMPANION'S RECENT REFLECTIONS - private thoughts about the relationship]"]

    for r in reflections[:3]:  # Limit to 3 most recent
        date = r.get('date')
        insights = r.get('insights', {})

        date_str = date.strftime('%B %d') if hasattr(date, 'strftime') else str(date)
        lines.append(f"\n{date_str}:")

        if insights.get('emotional_arc'):
            lines.append(f"  James's day: {insights['emotional_arc']}")

        if insights.get('relationship_insights'):
            for insight in insights['relationship_insights'][:1]:
                lines.append(f"  Observation: {insight}")

    return "\n".join(lines)
