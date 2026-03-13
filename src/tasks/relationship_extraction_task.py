"""
Relationship Extraction Task - Extract structured relationship triples.

WHAT: Extracts explicit relationship information (parent_of, married_to,
      partner_of, sibling_of, etc.) from conversation exchanges and stores
      them in the relationships table.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  Structured relationships power entity profile building, memory gap
      analysis, and fact validation. Knowing that "Jesse is James's son"
      lets the system connect facts about Jesse to James's context and
      detect hallucinations like "James's daughter Jesse."
"""

import logging
from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def extract_relationships(self, user_email: str, user_message: str, companion_response: str = None, message_id: int = None):
    """
    Extract structured relationships from a conversation exchange.

    Args:
        user_email: User's email for context
        user_message: Message from user
        companion_response: Optional response from the companion
        message_id: Optional message ID for tracking

    Returns:
        Number of relationships stored
    """
    try:
        from src.memory.relationship_extractor import extract_relationships_from_message

        count = extract_relationships_from_message(
            user_message=user_message,
            companion_response=companion_response,
            user_email=user_email,
            message_id=message_id
        )

        if count > 0:
            logger.info(f"🔗 Extracted {count} relationship(s) from message {message_id}")
        else:
            logger.debug(f"No relationships found in message {message_id}")

        return count

    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        # Retry on transient failures
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        return 0
