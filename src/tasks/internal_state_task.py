"""
Internal State Update Task - Manage the companion's activity/energy lifecycle.

WHAT: Detects goodbye/goodnight signals (regex patterns) to transition the
      companion to idle mode. Also detects when the companion expressed
      tiredness, triggering energy restoration. Does NOT handle physical
      actions (eating, drinking) -- those are in scene_extraction_task.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  The companion needs to know when a conversation has ended (to stop
      generating reach-outs) and when she's resting (to restore energy).
      This is the simple keyword-based layer; the LLM-based mood/scene
      tracking happens elsewhere.
"""

import logging
import re
from src.celery_app import celery_app

logger = logging.getLogger(__name__)

# --- Regex patterns for conversation-ending signals ---
GOODBYE_PATTERNS = [
    r'\bgoodnight\b',
    r'\bgood night\b',
    r'\bnight\b.*\blove\b',
    r'\bsweet dreams\b',
    r'\bsleep well\b',
    r'\bgoodbye\b',
    r'\bgotta go\b',
    r'\bhave to go\b',
    r'\bneed to go\b',
    r'\btalk.*(later|tomorrow|soon)\b',
    r'\bsee you\b',
    r'\bbye\b',
    r'\bheading to (bed|sleep)\b',
    r'\bgoing to (bed|sleep)\b',
    r'\btime for (bed|sleep)\b',
]

# --- Regex patterns for companion tiredness/rest signals ---
REST_PATTERNS = [
    r'need(s)? to (rest|sleep|nap)',
    r'should (rest|sleep|nap)',
    r"i('m| am) (going to |gonna )?(rest|sleep|nap)",
    r'time to (rest|sleep|nap)',
    r'getting sleepy',
    r'so tired',
    r'exhausted',
]


@celery_app.task(
    name='internal_state_update',
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    autoretry_for=(Exception,),
    acks_late=True
)
def update_internal_state(self, email: str, user_message: str, companion_response: str):
    """
    Update the companion's internal state based on conversation.

    Args:
        email: User email
        user_message: User's message
        companion_response: The companion's response
    """
    try:
        from src.core.internal_state import get_internal_state_manager

        manager = get_internal_state_manager()

        user_lower = user_message.lower()
        companion_lower = companion_response.lower()

        # Check if this is a goodbye/goodnight
        is_goodbye = any(
            re.search(pattern, user_lower, re.IGNORECASE)
            for pattern in GOODBYE_PATTERNS
        )

        # Also check if the companion said goodnight
        companion_goodbye = any(
            re.search(pattern, companion_lower, re.IGNORECASE)
            for pattern in GOODBYE_PATTERNS
        )

        if is_goodbye or companion_goodbye:
            # Transition to idle
            manager.transition_to_idle(email)
            logger.info(f"💤 Goodbye detected, transitioning to idle for {email}")
            return

        # Check if the companion expressed need to rest
        companion_tired = any(
            re.search(pattern, companion_lower, re.IGNORECASE)
            for pattern in REST_PATTERNS
        )

        if companion_tired:
            # The companion rests - restore some energy
            state = manager.rest(email, energy_restored=0.2)
            logger.info(f"😴 Companion resting, energy now {state.energy:.2f}")
            return

        # Could add mood detection here in the future
        # For now, mood is tracked via energy depletion and
        # could be enhanced with LLM-based mood extraction

        logger.debug("Internal state update complete")

    except Exception as e:
        logger.error(f"Internal state update failed: {e}")
        raise
