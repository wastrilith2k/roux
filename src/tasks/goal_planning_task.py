"""
Goal Planning Task - Decomposes unplanned goals into actionable steps.

Runs every 6 hours to check for goals that don't have steps yet
and decomposes them via LLM.

Schedule: Every 6 hours (6 AM, 12 PM, 6 PM, 12 AM)
"""

import logging
from src.celery_app import celery_app

logger = logging.getLogger(__name__)

def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.goal_planning.decompose_unplanned_goals',
    bind=True,
    max_retries=1,
    soft_time_limit=120,
    time_limit=180
)
def decompose_unplanned_goals(self, user_email: str = _get_default_user_email()):
    """
    Check for active goals without steps and decompose them.

    This runs periodically to ensure all active goals have actionable steps.
    New goals created by reflections or other systems get picked up here.
    """
    try:
        from src.autonomy.goal_planner import get_goal_planner

        planner = get_goal_planner(user_email)
        unplanned = planner.get_unplanned_goals(user_email)

        if not unplanned:
            logger.debug("No unplanned goals to decompose")
            return {'status': 'skipped', 'reason': 'no unplanned goals'}

        total_steps = 0
        for goal in unplanned:
            logger.info(f"Decomposing goal: {goal.goal[:50]}...")
            steps = planner.decompose_goal(goal)
            total_steps += len(steps)
            logger.info(f"  → {len(steps)} steps created")

        logger.info(f"Decomposed {len(unplanned)} goals into {total_steps} total steps")
        return {
            'status': 'success',
            'goals_decomposed': len(unplanned),
            'steps_created': total_steps
        }

    except Exception as e:
        logger.error(f"Goal decomposition failed: {e}")
        import traceback
        traceback.print_exc()

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)

        return {'status': 'error', 'error': str(e)}
