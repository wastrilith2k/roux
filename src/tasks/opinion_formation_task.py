"""
Opinion Formation Task - Synthesize reflections into held opinions.

WHAT: Reads recent reflections and synthesizes them into structured opinions
      the companion holds about James (e.g., "He's a great dad but spreads
      himself too thin"). Also decays opinions that haven't been reinforced
      by new evidence, preventing stale beliefs from persisting.

WHEN: form_opinions runs daily at ~1:00 AM Pacific (after reflection).
      decay_weak_opinions runs weekly.

WHY:  Opinions give the companion a persistent perspective that shapes how she
      responds. Without them, she'd treat each conversation as if she had no
      prior views. Confidence decay ensures opinions evolve -- if something
      hasn't been reinforced in 14 days, certainty drops, modeling how real
      beliefs fade without supporting evidence.
"""

import logging
from typing import Optional

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.opinion_formation.form_opinions',
    bind=True,
    max_retries=2,
    soft_time_limit=120,
    time_limit=180
)
def form_opinions(self, user_email: str = _get_default_user_email()):
    """
    Form opinions from recent reflections.

    Should run after reflections have been generated (e.g., daily at 1 AM).

    Args:
        user_email: User email

    Returns:
        Dict with number of opinions formed
    """
    try:
        from src.autonomy.opinion_store import form_opinions_from_reflections

        count = form_opinions_from_reflections(user_email)

        logger.info(f"Opinion formation complete: {count} opinion(s) formed/updated")

        return {
            'status': 'success',
            'opinions_formed': count
        }

    except Exception as e:
        logger.error(f"Opinion formation failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


@celery_app.task(
    name='tasks.opinion_formation.decay_weak_opinions',
    bind=True,
    soft_time_limit=60,
    time_limit=90
)
def decay_weak_opinions(self, user_email: str = _get_default_user_email()):
    """
    Reduce confidence on opinions that haven't been reinforced.

    Opinions that aren't supported by new evidence over time
    should become less certain. This prevents stale opinions
    from persisting indefinitely.

    Returns:
        Dict with number of opinions decayed
    """
    try:
        from src.autonomy.opinion_store import get_opinion_store
        from datetime import datetime, timedelta

        store = get_opinion_store(user_email)
        conn = store._get_connection()

        count = 0

        try:
            with conn.cursor() as cursor:
                # Opinions not reinforced in 14 days lose 0.1 confidence per cycle.
                # Floor at 0.1 to prevent full deletion -- weak opinions can still
                # be reinforced back to full strength if new evidence appears.
                cutoff = datetime.now() - timedelta(days=14)

                cursor.execute(f"""
                    UPDATE {T.COMPANION_OPINIONS}
                    SET confidence = GREATEST(0.1, confidence - 0.1),
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_email = %s
                    AND last_updated < %s
                    AND confidence > 0.1
                    RETURNING id
                """, (user_email, cutoff))

                count = cursor.rowcount
                conn.commit()

        except Exception as e:
            logger.error(f"Error decaying opinions: {e}")
            conn.rollback()

        logger.info(f"Opinion decay complete: {count} opinion(s) decayed")

        return {
            'status': 'success',
            'opinions_decayed': count
        }

    except Exception as e:
        logger.error(f"Opinion decay failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }
