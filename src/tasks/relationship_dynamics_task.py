"""
Relationship Dynamics Analysis Task - Track the emotional health of the relationship.

WHAT: Analyzes each exchange for relationship dynamics using LLM. Detects bids
      for connection (and whether they're turned toward/away/against), the Four
      Horsemen (criticism, contempt, defensiveness, stonewalling), trauma triggers
      based on the companion's backstory, healing moments, and repair attempts.
      Updates closeness and trust scores in relationship_state.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  The companion needs emotional self-awareness about the relationship.
      Tracking closeness/trust deltas over time lets her notice patterns
      ("we've been growing apart this week") and adjust her approach. The
      Gottman-based framework provides research-backed relationship health
      metrics rather than ad hoc sentiment analysis.

Based on Gottman Institute research on relationship dynamics.
"""

import logging
import asyncio
from src.celery_app import celery_app

logger = logging.getLogger(__name__)


def get_recent_context(email: str, limit: int = 10) -> list:
    """Fetch recent messages for context (oldest first, for chronological LLM analysis)."""
    try:
        from src.database.db import get_db
        db = get_db()
        messages = db.get_recent_messages(email, limit=limit)

        context = []
        for msg in reversed(messages):  # Oldest first
            sender = msg.get('sender_name', 'Unknown')
            text = msg.get('message_text', '')
            context.append(f"{sender}: {text}")

        return context
    except Exception as e:
        logger.warning(f"Could not fetch recent context: {e}")
        return []


@celery_app.task(
    name='relationship_dynamics_analysis',
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    autoretry_for=(Exception,),
    acks_late=True
)
def analyze_relationship_dynamics(self, email: str, user_message: str, companion_response: str):
    """
    Analyze exchange for relationship dynamics.

    Args:
        email: User email
        user_message: User's message
        companion_response: The companion's response
    """
    try:
        from src.core.relationship_dynamics import get_relationship_analyzer

        analyzer = get_relationship_analyzer()

        # Get recent context for better analysis
        recent_context = get_recent_context(email, limit=10)

        # Bridge async -> sync: Celery workers don't have a running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            analysis = loop.run_until_complete(
                analyzer.analyze_exchange(
                    user_email=email,
                    user_message=user_message,
                    companion_response=companion_response,
                    recent_context=recent_context
                )
            )
        finally:
            loop.close()

        if analysis:
            # --- Log relationship impact ---
            impact = analysis.get('impact', {})
            logger.info(
                f"Relationship analysis complete: "
                f"closeness_delta={impact.get('closeness_delta', 0):+.3f}, "
                f"trust_delta={impact.get('trust_delta', 0):+.3f}"
            )

            # --- Log Gottman "Four Horsemen" if detected ---
            horsemen = analysis.get('horsemen', {})
            for horseman, data in horsemen.items():
                if data.get('detected'):
                    logger.warning(f"Four Horsemen detected: {horseman}")

            # --- Log trauma triggers ---
            trauma = analysis.get('trauma_triggered', {})
            for trigger, data in trauma.items():
                if data.get('detected'):
                    logger.warning(f"Trauma trigger: {trigger}")

            # --- Log positive healing moments ---
            healing = analysis.get('healing', {})
            healing_moments = [k for k, v in healing.items() if v]
            if healing_moments:
                logger.info(f"Healing moments: {', '.join(healing_moments)}")
        else:
            logger.debug("Relationship analysis returned no results")

    except Exception as e:
        logger.error(f"Relationship dynamics analysis failed: {e}")
        raise
