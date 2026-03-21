"""
Reminder Check Task - Surface due reminders to the companion's awareness.

WHAT: Queries companion_reminders for incomplete items due within 1 hour.
      Adds each to queued_thoughts (surfaced in interjections/reach-outs).
      Very overdue reminders (>6h past due) bump reach-out pressure to
      increase urgency of the companion proactively messaging James.

WHEN: Every 2 hours during waking hours (8, 10, 12, 14, 16, 18, 20 PST).

WHY:  Without this, reminders the companion set would never fire. This is
      the polling mechanism that checks "do I have anything I need to remind
      James about?" and escalates urgency for forgotten items.

Feature flag: COMPANION_REMINDER_CHECK_ENABLED (default: true)
"""

import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


def _is_enabled() -> bool:
    return os.environ.get('COMPANION_REMINDER_CHECK_ENABLED', 'true').lower() in ('true', '1')


@celery_app.task(
    name='tasks.reminder_check.check_due_reminders',
    soft_time_limit=60,
    time_limit=90,
)
def check_due_reminders(user_email: str = None):
    """Check for due reminders and surface them to the companion's awareness."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    if not _is_enabled():
        return {'status': 'disabled'}

    try:
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', ''),
        )

        now = datetime.now(PST)
        due_reminders = []

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Reminders due within 1 hour (upcoming + overdue)
            cur.execute(f"""
                SELECT id, title, due_date, notes
                FROM {T.COMPANION_REMINDERS}
                WHERE completed = FALSE
                  AND due_date IS NOT NULL
                  AND due_date <= NOW() + INTERVAL '1 hour'
                ORDER BY due_date ASC
            """)
            due_reminders = cur.fetchall()

        conn.close()

        if not due_reminders:
            logger.debug("No due reminders found")
            return {'status': 'success', 'due_count': 0}

        # Add to queued thoughts
        from src.core.internal_state import get_internal_state_manager
        state_manager = get_internal_state_manager()

        very_overdue = []
        # Compare with naive local time because the Celery worker runs with
        # TZ=America/Los_Angeles and user-entered dates are PST-intended.
        now_local = datetime.now()
        for reminder in due_reminders:
            due_date = reminder['due_date']
            title = reminder['title']

            due_naive = due_date.replace(tzinfo=None) if due_date.tzinfo else due_date
            hours_overdue = (now_local - due_naive).total_seconds() / 3600

            if hours_overdue > 0:
                thought = f"reminder (overdue): {title}"
            else:
                thought = f"reminder (coming up): {title}"

            state_manager.add_queued_thought(user_email, thought)

            if hours_overdue > 6:
                very_overdue.append(title)

        # Bump reach-out pressure for very overdue reminders
        if very_overdue:
            try:
                from src.autonomy.reach_out_pressure import get_reach_out_pressure, save_pressure
                pressure = get_reach_out_pressure()
                pressure.accumulate(f"overdue reminders: {', '.join(very_overdue[:2])}")
                save_pressure(pressure)
                logger.info(f"Bumped reach-out pressure for {len(very_overdue)} very overdue reminders")
            except Exception as e:
                logger.debug(f"Could not bump pressure: {e}")

        logger.info(f"Surfaced {len(due_reminders)} due reminders to queued_thoughts")
        return {'status': 'success', 'due_count': len(due_reminders)}

    except Exception as e:
        logger.error(f"Reminder check failed: {e}")
        return {'status': 'error', 'error': str(e)}
