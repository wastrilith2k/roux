"""
Biography Refresh Task - Periodically refreshes synthesized biographies

This task:
1. Gets all unique subjects from the facts table
2. For each subject, groups facts by theme
3. Synthesizes a coherent paragraph for each theme
4. Stores with temporal decay-aware importance scoring

Should be run:
- Daily (via scheduler)
- After bulk fact extraction
- After significant events are learned

The synthesized biographies provide better context than scattered facts
because related information is grouped into coherent paragraphs.
"""

import logging
from typing import Dict, Any

from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name='tasks.refresh_biographies')
def refresh_biographies_task(user_email: str = None) -> Dict[str, Any]:
    """
    Celery task to refresh all synthesized biographies.

    Args:
        user_email: Optional user context (default: None for all users)

    Returns:
        Dict with results per subject
    """
    logger.info("Starting biography refresh task")

    try:
        from src.memory.synthesized_biographies import refresh_all_biographies

        results = refresh_all_biographies(user_email)

        # Log summary
        success_count = sum(1 for r in results.values() if r.get('success'))
        fail_count = len(results) - success_count

        logger.info(
            f"Biography refresh complete: {success_count} succeeded, "
            f"{fail_count} failed, {len(results)} total subjects"
        )

        return {
            'success': True,
            'subjects_processed': len(results),
            'succeeded': success_count,
            'failed': fail_count,
            'details': results
        }

    except Exception as e:
        logger.error(f"Biography refresh task failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def refresh_biographies_sync(user_email: str = None) -> Dict[str, Any]:
    """
    Synchronous version for direct calls (not via Celery).

    Use this for testing or when you need immediate results.
    """
    logger.info("Starting synchronous biography refresh")

    try:
        from src.memory.synthesized_biographies import refresh_all_biographies

        results = refresh_all_biographies(user_email)

        success_count = sum(1 for r in results.values() if r.get('success'))

        logger.info(f"Biography refresh complete: {success_count}/{len(results)} succeeded")

        return {
            'success': True,
            'subjects_processed': len(results),
            'details': results
        }

    except Exception as e:
        logger.error(f"Biography refresh failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    # Allow running directly for testing
    import sys

    logging.basicConfig(level=logging.INFO)

    user_email = sys.argv[1] if len(sys.argv) > 1 else None
    result = refresh_biographies_sync(user_email)

    print(f"Results: {result}")
