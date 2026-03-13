"""
Core Memory Refresh Task - Regenerate the companion's curated memory document.

WHAT: Refreshes COMPANION_MEMORY.md with an LLM-synthesized narrative about James
      and the companion's world. Combines entity profiles (YAML ground truth),
      high-importance facts from the database, and synthesized biographies
      into a coherent document that serves as the companion's "who I know
      and what I know about them" reference.

WHEN: Weekly on Sundays at 4:00 AM Pacific. Can also be triggered manually.

WHY:  The core memory document is injected into every conversation's system
      prompt. It gives the companion persistent knowledge without requiring
      runtime database queries for basic facts. Refreshing weekly ensures it
      stays current as new facts are learned and old ones evolve.
"""

import logging
from src.celery_app import celery_app

logger = logging.getLogger(__name__)

# Default user email (can be overridden)
def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.core_memory_task.refresh_core_memory',
    bind=True,
    max_retries=2,
    soft_time_limit=180,  # 3 minute warning
    time_limit=240        # 4 minute hard limit
)
def refresh_core_memory(self, user_email: str = _get_default_user_email()):
    """
    Refresh the companion's core memory file.

    This task:
    1. Loads entity profiles (YAML ground truth)
    2. Gets high-importance facts from the database
    3. Gets synthesized biographies
    4. Uses LLM to synthesize into a narrative
    5. Writes to data/COMPANION_MEMORY.md

    Args:
        user_email: User email for fetching user-specific data

    Returns:
        Dict with status and details
    """
    logger.info(f"Starting core memory refresh for {user_email}")

    try:
        from src.memory.core_memory import get_core_memory_manager

        manager = get_core_memory_manager()
        success = manager.refresh_core_memory(user_email)

        if success:
            # Get the new content for logging
            content = manager.get_core_memory()
            word_count = len(content.split()) if content else 0

            logger.info(f"Core memory refresh completed: {word_count} words")

            return {
                'status': 'success',
                'user_email': user_email,
                'word_count': word_count,
                'char_count': len(content) if content else 0
            }
        else:
            logger.error("Core memory refresh failed")
            return {
                'status': 'error',
                'error': 'Refresh returned False'
            }

    except Exception as e:
        logger.error(f"Core memory refresh task failed: {e}")
        import traceback
        traceback.print_exc()

        # Retry on failure
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)

        return {
            'status': 'error',
            'error': str(e)
        }


@celery_app.task(
    name='tasks.core_memory_task.refresh_core_memory_sync',
    bind=True
)
def refresh_core_memory_sync(self, user_email: str = _get_default_user_email()):
    """
    Synchronous version for direct calling (not via queue).

    Useful for testing or manual triggers from Python code.
    """
    from src.memory.core_memory import get_core_memory_manager

    manager = get_core_memory_manager()
    success = manager.refresh_core_memory(user_email)

    return {
        'status': 'success' if success else 'error',
        'user_email': user_email
    }
