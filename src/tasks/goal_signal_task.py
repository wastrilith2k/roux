"""
Goal Signal Task — form goals from multiple signal sources.

Runs every 12h (9 AM and 9 PM). Looks at high-urgency curiosities,
strong opinions, and unshared findings to propose new goals.
Supplements (doesn't replace) reflection-based goal formation.

Schedule: 9:00 AM and 9:00 PM daily
"""

import logging

from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name='tasks.goal_signals.form_from_signals',
    soft_time_limit=120,
    time_limit=180,
)
def form_goals_from_signals_task(user_email: str = None):
    """Form goals from curiosity, opinion, and finding signals."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    try:
        from src.autonomy.goal_formation import form_goals_from_signals
        count = form_goals_from_signals(user_email)
        logger.info(f"Signal-based goal formation: {count} goals created")
        return {'status': 'success', 'goals_created': count}
    except Exception as e:
        logger.error(f"Signal-based goal formation failed: {e}")
        return {'status': 'error', 'error': str(e)}
