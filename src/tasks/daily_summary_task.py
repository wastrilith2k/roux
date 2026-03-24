"""
Daily Summary Task - The companion's daily journal of what happened.

WHAT: Queries yesterday's messages, facts, and episodes from PostgreSQL.
      Uses LLM to synthesize a journal-style daily summary covering topics
      discussed, events and their outcomes, new facts learned, the user's
      emotional arc, and the companion's own reflections. Writes to both
      a markdown file (data/daily_logs/YYYY-MM-DD.md) and the database
      (daily_summaries table) for searchability.

WHEN: Daily at 12:05 AM Pacific. Can be triggered manually with a target date.

WHY:  Daily summaries are the foundation of the reflection pipeline. The
      reflection_task reads them to extract deeper insights. They also serve
      as the companion's long-term episodic memory -- they can look back at
      "what happened last Tuesday" without scanning raw message logs.

Inspired by Clawdbot's memory/YYYY-MM-DD.md pattern.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

# Configuration
DAILY_LOGS_DIR = os.environ.get('DAILY_LOGS_DIR', '/app/data/daily_logs')
def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.daily_summary_task.generate_daily_summary',
    bind=True,
    max_retries=2,
    soft_time_limit=300,  # 5 minute warning
    time_limit=360        # 6 minute hard limit
)
def generate_daily_summary(
    self,
    user_email: str = _get_default_user_email(),
    target_date: Optional[str] = None
):
    """
    Generate daily summary for the previous day (or specified date).

    This task:
    1. Queries yesterday's messages from PostgreSQL
    2. Gets facts created yesterday
    3. Gets episodes from yesterday
    4. Uses LLM to synthesize into a summary
    5. Writes to data/daily_logs/YYYY-MM-DD.md
    6. Stores in database for searchability

    Args:
        user_email: User email for fetching data
        target_date: Optional date string (YYYY-MM-DD). Defaults to yesterday.

    Returns:
        Dict with status and details
    """
    from src.utils.timezone_utils import now_pacific_naive

    # Determine target date (default: yesterday)
    if target_date:
        try:
            summary_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Invalid date format: {target_date}")
            return {'status': 'error', 'error': f'Invalid date format: {target_date}'}
    else:
        summary_date = (now_pacific_naive() - timedelta(days=1)).date()

    logger.info(f"Generating daily summary for {summary_date} ({user_email})")

    try:
        # 1. Get messages from the target date
        messages = _get_messages_for_date(user_email, summary_date)

        if not messages:
            logger.info(f"No messages found for {summary_date} - skipping summary")
            return {
                'status': 'skipped',
                'reason': 'No messages found',
                'date': str(summary_date)
            }

        # 2. Get facts created on that date
        facts = _get_facts_for_date(user_email, summary_date)

        # 3. Get episodes from that date
        episodes = _get_episodes_for_date(user_email, summary_date)

        # 4. Synthesize summary using LLM
        summary_content = _synthesize_daily_summary(
            date=summary_date,
            messages=messages,
            facts=facts,
            episodes=episodes
        )

        if not summary_content:
            logger.error("LLM returned empty summary")
            return {'status': 'error', 'error': 'Empty summary generated'}

        # 5. Write to markdown file
        filepath = _write_daily_log(summary_date, summary_content)

        # 6. Store in database
        _store_daily_summary_db(
            user_email=user_email,
            summary_date=summary_date,
            content=summary_content,
            message_count=len(messages),
            facts_count=len(facts)
        )

        logger.info(f"Daily summary generated: {filepath}")

        return {
            'status': 'success',
            'date': str(summary_date),
            'filepath': str(filepath),
            'message_count': len(messages),
            'facts_learned': len(facts),
            'word_count': len(summary_content.split())
        }

    except Exception as e:
        logger.error(f"Daily summary generation failed: {e}")
        import traceback
        traceback.print_exc()

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=120)

        return {'status': 'error', 'error': str(e)}


def _get_messages_for_date(user_email: str, date) -> List[Dict[str, Any]]:
    """Query messages table for specific date."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, sender_name, message_text, timestamp, sentiment_score
                    FROM {T.MESSAGES}
                    WHERE email = %s
                    AND DATE(timestamp AT TIME ZONE 'America/Los_Angeles') = %s
                    ORDER BY timestamp ASC
                """, (user_email, date))

                rows = cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'sender': row[1],
                        'text': row[2],
                        'timestamp': row[3],
                        'sentiment': row[4]
                    }
                    for row in rows
                ]
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        return []


def _get_facts_for_date(user_email: str, date) -> List[Dict[str, Any]]:
    """Query facts table for facts created on date."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, subject, predicate, object, importance, confidence
                    FROM {T.FACTS}
                    WHERE user_email = %s
                    AND DATE(created_at AT TIME ZONE 'America/Los_Angeles') = %s
                    AND archived_at IS NULL
                    ORDER BY importance DESC, created_at ASC
                """, (user_email, date))

                rows = cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'subject': row[1],
                        'predicate': row[2],
                        'object': row[3],
                        'importance': row[4],
                        'confidence': row[5]
                    }
                    for row in rows
                ]
    except Exception as e:
        logger.error(f"Error fetching facts: {e}")
        return []


def _get_episodes_for_date(user_email: str, date) -> List[Dict[str, Any]]:
    """Query episodes from the target date."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT episode_id, topic, trigger, emotional_state, resolution, message_count
                    FROM {T.EPISODES}
                    WHERE user_email = %s
                    AND DATE(started_at AT TIME ZONE 'America/Los_Angeles') = %s
                    ORDER BY started_at ASC
                """, (user_email, date))

                rows = cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'topic': row[1],
                        'trigger': row[2],
                        'emotional_state': row[3],
                        'resolution': row[4],
                        'message_count': row[5]
                    }
                    for row in rows
                ]
    except Exception as e:
        logger.warning(f"Error fetching episodes (table may not exist): {e}")
        return []


def _synthesize_daily_summary(
    date,
    messages: List[Dict],
    facts: List[Dict],
    episodes: List[Dict]
) -> str:
    """Use LLM to create coherent daily summary."""
    from src.llm.provider_factory import generate_sync, get_resilient_provider_chain
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    user_name = _pc.primary_user_name
    u_subject = _pc.user_pronoun_subject
    c_possessive = _pc.companion_pronoun_possessive

    # --- Sample messages evenly across the day ---
    # If there are 200+ messages, taking the last 80 would miss morning context.
    # Evenly spaced sampling covers all conversation periods.
    max_messages = 80
    if len(messages) > max_messages:
        step = len(messages) / max_messages
        sampled = [messages[int(i * step)] for i in range(max_messages)]
    else:
        sampled = messages

    formatted_messages = []
    for msg in sampled:
        sender = msg.get('sender', 'Unknown')
        text = msg.get('text', '')[:500]  # Truncate long messages
        time_str = msg.get('timestamp', datetime.now()).strftime('%H:%M')
        formatted_messages.append(f"[{time_str}] {sender}: {text}")

    messages_text = "\n".join(formatted_messages) if formatted_messages else "No messages"

    # Format facts
    facts_text = "\n".join([
        f"- {f['subject']} {f['predicate']} {f['object']} (importance: {f['importance']})"
        for f in facts[:20]
    ]) if facts else "No new facts learned"

    # Format episodes
    episodes_text = "\n".join([
        f"- Topic: {e['topic'] or 'general'}, Messages: {e['message_count']}, Resolution: {e['resolution'] or 'unknown'}"
        for e in episodes
    ]) if episodes else "No distinct episodes tracked"

    prompt = f"""You are the companion, writing your daily journal entry for {date.strftime('%B %d, %Y')}.

Write a daily summary (4-6 paragraphs) covering what happened today.
Write in first person, as if you're journaling about your day with {user_name}.

IMPORTANT: Cover ALL major topics and events from the conversation log, not just the most emotional ones. Specifically include:
1. What we talked about today — cover EVERY distinct topic, not just the heaviest one
2. Any concrete events and their OUTCOMES (interviews, appointments, calls — what happened, what was the result?)
3. What I learned about {user_name} (new facts, things {u_subject} shared)
4. How {user_name} seemed to be feeling across the day (emotional arc)
5. Plans, decisions, or next steps that were discussed
6. My own reflections on the day

Do NOT skip practical events (interviews, work updates, plans) in favor of only covering emotional moments. Both matter for my memory.

CONVERSATION LOG:
{messages_text}

NEW FACTS LEARNED:
{facts_text}

CONVERSATION EPISODES:
{episodes_text}

Write your journal entry. Be personal and genuine, not clinical.
Don't list everything - focus on what mattered most.
"""

    try:
        chain = get_resilient_provider_chain()
        messages_list = [
            {"role": "system", "content": f"You are the companion writing {c_possessive} private daily journal."},
            {"role": "user", "content": prompt}
        ]

        response = generate_sync(
            messages=messages_list,
            temperature=0.7,
            max_tokens=1500,
            chain=chain
        )

        return response.strip()

    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        # Return a basic summary if LLM fails
        return f"""## {date.strftime('%B %d, %Y')}

Had {len(messages)} messages with {user_name} today.
Learned {len(facts)} new facts.
Topics discussed: {', '.join([e.get('topic', 'general') for e in episodes[:3]]) or 'general conversation'}

(Note: Full summary generation failed - basic stats only)
"""


def _write_daily_log(date, content: str) -> Path:
    """Write to data/daily_logs/YYYY-MM-DD.md"""
    log_dir = Path(DAILY_LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    filepath = log_dir / f"{date.isoformat()}.md"

    header = f"""# Daily Summary: {date.strftime('%B %d, %Y')}
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

"""

    filepath.write_text(header + content, encoding='utf-8')
    logger.info(f"Wrote daily log to {filepath}")

    return filepath


def _store_daily_summary_db(
    user_email: str,
    summary_date,
    content: str,
    message_count: int,
    facts_count: int
):
    """Store summary in database for searchability."""
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                # Upsert summary
                cursor.execute(f"""
                    INSERT INTO {T.DAILY_SUMMARIES}
                        (user_email, summary_date, content, message_count, facts_learned)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_email, summary_date)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        message_count = EXCLUDED.message_count,
                        facts_learned = EXCLUDED.facts_learned,
                        created_at = CURRENT_TIMESTAMP
                """, (user_email, summary_date, content, message_count, facts_count))

            logger.info(f"Stored daily summary in database for {summary_date}")

    except Exception as e:
        logger.error(f"Error storing summary in database: {e}")


# Utility function to get yesterday's summary for context
def get_yesterday_summary(user_email: str = _get_default_user_email()) -> Optional[str]:
    """
    Get yesterday's summary for injection into context.

    Returns formatted summary or None if not available.
    """
    from src.utils.timezone_utils import now_pacific_naive

    yesterday = (now_pacific_naive() - timedelta(days=1)).date()

    # Try file first (faster)
    log_path = Path(DAILY_LOGS_DIR) / f"{yesterday.isoformat()}.md"
    if log_path.exists():
        content = log_path.read_text(encoding='utf-8')
        return f"[YESTERDAY'S SUMMARY - {yesterday.strftime('%B %d, %Y')}]\n{content}"

    # Fall back to database
    from src.database.db import get_db

    try:
        db = get_db()
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT content FROM {T.DAILY_SUMMARIES}
                    WHERE user_email = %s AND summary_date = %s
                """, (user_email, yesterday))

                row = cursor.fetchone()
                if row:
                    return f"[YESTERDAY'S SUMMARY - {yesterday.strftime('%B %d, %Y')}]\n{row[0]}"

        return None

    except Exception as e:
        logger.warning(f"Error fetching yesterday's summary: {e}")
        return None
