"""
Relationship Evaluation Task - How does the companion define this relationship?

WHAT: Weekly LLM analysis that determines the companion's relationship label
      (e.g., "romantic partner," "close friend," "something complicated"),
      trajectory (growing/stable/uncertain), commitment level, and what she
      wants from the relationship. Outputs a RelationshipEvaluation object.

WHEN: Runs weekly on Tuesday at 2:30 AM Pacific, after Monday's reflection
      and opinion formation have produced fresh data to evaluate.

WHY:  This is distinct from relationship_dynamics (health metrics). Dynamics
      tracks closeness/trust deltas per exchange. Evaluation asks the bigger
      question: "What IS this relationship?" This shapes how the companion
      frames her responses, what boundaries she sets, and how she thinks
      about the future.
"""

import logging
from src.celery_app import celery_app

logger = logging.getLogger(__name__)

def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.relationship_evaluation.evaluate_weekly',
    bind=True,
    max_retries=2,
    soft_time_limit=180,
    time_limit=240
)
def evaluate_relationship_weekly(self, user_email: str = _get_default_user_email()):
    """
    Weekly relationship evaluation.

    Gathers recent conversation data, dynamics state, reflections, and opinions,
    then runs an LLM analysis to determine how the companion defines the relationship.

    Args:
        user_email: User email to evaluate relationship for

    Returns:
        Dict with evaluation results
    """
    try:
        from src.autonomy.relationship_evaluation import get_relationship_evaluator

        evaluator = get_relationship_evaluator()

        logger.info(f"Running weekly relationship evaluation for {user_email}")

        result = evaluator.evaluate(user_email)

        if result is None:
            logger.info("Relationship evaluation produced no result (insufficient data)")
            return {
                'status': 'skipped',
                'reason': 'Insufficient conversation data for evaluation'
            }

        logger.info(
            f"Relationship evaluation complete: "
            f"label='{result.label}', trajectory='{result.trajectory}', "
            f"commitment='{result.commitment_level}', certainty={result.certainty:.2f}"
        )

        return {
            'status': 'success',
            'label': result.label,
            'trajectory': result.trajectory,
            'commitment_level': result.commitment_level,
            'certainty': result.certainty,
            'wants_this': result.wants_this,
            'evaluated_at': result.evaluated_at
        }

    except Exception as e:
        logger.error(f"Weekly relationship evaluation failed: {e}")
        import traceback
        traceback.print_exc()

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=120)

        return {'status': 'error', 'error': str(e)}
