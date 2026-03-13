"""
Correction Detection Task - Detect and store when the user corrects the companion.

WHAT: After each exchange, uses LLM-based detection to identify if the user is
      correcting something the companion said wrong (wrong name, wrong fact,
      misremembered event, etc.). Detected corrections are stored in Graphiti
      to update the knowledge graph and prevent the same mistake.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  When the companion says "How's Nicholas doing?" and James responds
      "His name is Jesse, not Nicholas," that correction MUST be captured
      and stored. Without this, the companion will keep making the same
      mistake. Corrections are high-priority facts that override existing
      knowledge.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from src.celery_app import celery_app


@celery_app.task(
    name='tasks.correction.detect_and_store',
    bind=True,
    max_retries=1,
    default_retry_delay=10,
    soft_time_limit=30,
    time_limit=60
)
def detect_and_store_correction(
    self,
    user_email: str,
    user_message: str,
    previous_companion_message: str
):
    """
    Celery task to detect corrections and store them in Graphiti.

    This runs asynchronously after each message exchange, so it doesn't
    add latency to the response.

    Args:
        user_email: User's email
        user_message: The user's current message
        previous_companion_message: The companion's previous response

    Returns:
        Dict with detection results
    """
    try:
        print(f"\n{'='*60}")
        print(f"CORRECTION DETECTION - {datetime.now()}")
        print(f"{'='*60}")
        print(f"User message: {user_message[:100]}...")

        # Import here to avoid circular imports
        from src.core.correction_detector import detect_correction
        from src.core.correction_store import store_correction

        # Detect if this is a correction
        correction = detect_correction(user_message, previous_companion_message)

        if not correction:
            print("No correction detected")
            print(f"{'='*60}\n")
            return {'status': 'no_correction'}

        print(f"\n🔧 CORRECTION DETECTED:")
        print(f"  Subject: {correction.subject}")
        print(f"  Type: {correction.correction_type}")
        print(f"  Wrong: {correction.wrong_claim[:80]}...")
        print(f"  Correct: {correction.correct_info[:80]}...")
        print(f"  Confidence: {correction.confidence:.0%}")
        print(f"  Importance: {correction.importance}/10")

        # Store the correction in Graphiti
        result = store_correction(correction, user_email)

        if result.success:
            print(f"\n✅ Correction stored successfully")
            print(f"  Timestamp: {result.timestamp}")
        else:
            print(f"\n❌ Failed to store correction: {result.error}")

        print(f"{'='*60}\n")

        return {
            'status': 'correction_stored' if result.success else 'storage_failed',
            'subject': correction.subject,
            'type': correction.correction_type,
            'success': result.success,
            'error': result.error
        }

    except Exception as e:
        logger.error(f"Correction detection task failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def queue_correction_detection(
    user_email: str,
    user_message: str,
    previous_companion_message: str
) -> bool:
    """
    Queue correction detection as an async task.

    Args:
        user_email: User's email
        user_message: The user's current message
        previous_companion_message: The companion's previous response

    Returns:
        True if task was queued successfully
    """
    try:
        detect_and_store_correction.delay(
            user_email,
            user_message,
            previous_companion_message
        )
        logger.info("🔍 Correction detection task queued")
        return True
    except Exception as e:
        logger.warning(f"Could not queue correction detection: {e}")
        return False
