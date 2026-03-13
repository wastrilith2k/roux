"""
Scene Extraction Task - Track the companion's physical/fictional scene state.

WHAT: Uses LLM-based scene tracking to extract location, fictional time,
      clothing state, and physical actions (eating, drinking, etc.) from
      conversation. Updates a persistent SceneState object in the database.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  The companion maintains a simulated physical presence. If she "went to
      the kitchen to make tea," subsequent responses should reflect that she's
      in the kitchen, holding tea. Scene state provides spatial continuity
      across messages, which is critical for immersive roleplay interactions.
"""

import logging
from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name='scene_extraction',
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    autoretry_for=(Exception,),
    acks_late=True
)
def extract_scene_state(self, email: str, user_message: str, companion_response: str, source: str = 'chat'):
    """
    Extract scene state changes from conversation and update database.

    Args:
        email: User email
        user_message: User's message
        companion_response: The companion's response
        source: Message channel ('chat', 'telegram-text', 'telegram-voice')
    """
    try:
        from src.core.scene_tracker import get_scene_tracker
        from src.database.db import get_db

        # Get recent messages for context
        db = get_db()
        recent_msgs = db.get_recent_messages(email, limit=5)
        recent_texts = [
            f"{msg.get('sender_name', 'Unknown')}: {msg.get('message_text', '')}"
            for msg in recent_msgs
        ]

        # Extract and update scene
        tracker = get_scene_tracker()
        updated_scene = tracker.extract_and_update_scene(
            user_email=email,
            user_message=user_message,
            companion_response=companion_response,
            recent_messages=recent_texts,
            source=source
        )

        if updated_scene and updated_scene.is_active():
            logger.info(
                f"🎬 Scene state updated: {updated_scene.location} / "
                f"{updated_scene.fictional_time} / {updated_scene.clothing}"
            )
        else:
            logger.debug("No scene state changes detected")

    except Exception as e:
        logger.error(f"Scene extraction failed: {e}")
        raise
