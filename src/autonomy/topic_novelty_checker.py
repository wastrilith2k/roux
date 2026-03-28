"""
Topic Novelty Checker -- Prevents proactive messages from repeating recently
discussed topics.

WHAT: An LLM-based gate that compares a proposed proactive message against the
      last ~25 conversation messages and returns NOVEL or DUPLICATE.

WHY:  Without this, the companion can re-ask questions the user already answered
      or re-share information already discussed.  The interjection and reach-out
      engines both call this before sending.

HOW IT FITS:
  - Called from AlwaysOnService._check_interjection() and
    _check_reach_out_with_pressure() right before a message would be sent.
  - Uses a fast, low-token LLM call (timeout=3s) via the resilient provider
    chain.  On failure it "fails open" -- the message goes through.

Feature flag: COMPANION_TOPIC_NOVELTY_CHECK_ENABLED (default: true)
"""

import os
import logging
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Runtime feature flag -- can be toggled without restart via env var
COMPANION_TOPIC_NOVELTY_CHECK_ENABLED = os.environ.get(
    'COMPANION_TOPIC_NOVELTY_CHECK_ENABLED', 'true'
).lower() == 'true'


# =============================================================================
# Checker
# =============================================================================

class TopicNoveltyChecker:
    """Checks whether a proposed proactive message covers an already-discussed topic."""

    def is_novel(
        self,
        proposed_message: str,
        recent_messages: List[str],
        context_label: str = "proactive message"
    ) -> bool:
        """
        Check if a proposed message is novel relative to recent conversation.

        Args:
            proposed_message: The message the companion wants to send.
            recent_messages: Last ~25 messages from the conversation (both parties).
            context_label: Label for logging ("interjection", "reach-out", etc.).

        Returns:
            True if the topic is novel (safe to send), False if duplicate.
        """
        # Bypass entirely when feature is disabled
        if not COMPANION_TOPIC_NOVELTY_CHECK_ENABLED:
            return True

        # Nothing to compare against -- assume novel
        if not proposed_message or not recent_messages:
            return True

        start = time.time()

        try:
            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

            # Format recent messages compactly (last 25, 200 chars each)
            recent_text = "\n".join(
                f"- {msg[:200]}" for msg in recent_messages[-25:]
            )

            prompt = f"""Compare this proposed message against recent conversation history.

PROPOSED MESSAGE:
{proposed_message[:500]}

RECENT CONVERSATION (last 25 messages):
{recent_text}

Is the proposed message about a topic that was ALREADY discussed in the recent conversation?

Consider it a DUPLICATE if:
- It asks about something already answered
- It brings up a topic already covered (even if worded differently)
- It shares information already shared
- It revisits a question already asked

Consider it NOVEL if:
- It's a genuinely new topic or angle
- It follows up with new information on a previous topic
- It asks a deeper question about something briefly mentioned

Respond with ONLY:
NOVEL or DUPLICATE
Then a brief reason (one sentence).

Example: DUPLICATE Already asked about his weekend plans 10 messages ago."""

            chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[
                    {"role": "system", "content": "Classify if a message topic is novel or duplicate. Reply with ONLY NOVEL or DUPLICATE followed by a reason."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
                chain=chain,
                timeout=3  # tight timeout -- fail open if slow
            )
            from src.services.cost_tracker import track_llm_call
            track_llm_call(chain, call_purpose='topic_novelty')

            processing_time = int((time.time() - start) * 1000)

            # Parse the first line of the response
            response_clean = response.strip().upper()
            is_duplicate = response_clean.startswith("DUPLICATE")
            reason = response.strip().split('\n')[0] if response.strip() else ""

            if is_duplicate:
                logger.info(f"Topic novelty: {context_label} is DUPLICATE ({processing_time}ms) - {reason}")
                return False
            else:
                logger.debug(f"Topic novelty: {context_label} is NOVEL ({processing_time}ms)")
                return True

        except Exception as e:
            processing_time = int((time.time() - start) * 1000)
            logger.warning(f"Topic novelty check failed ({processing_time}ms): {e}")
            return True  # fail open -- allow the message if the check itself fails


# =============================================================================
# Singleton accessor
# =============================================================================

_checker: Optional[TopicNoveltyChecker] = None


def get_topic_novelty_checker() -> TopicNoveltyChecker:
    """Get singleton TopicNoveltyChecker instance."""
    global _checker
    if _checker is None:
        _checker = TopicNoveltyChecker()
    return _checker
