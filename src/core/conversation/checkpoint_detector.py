"""
Checkpoint Detector — Companion Framework

WHAT: Detects when a long conversation should trigger a "memory checkpoint" —
      a prompt injection that nudges the companion to consolidate important
      information in its response.
WHY:  The context window only holds ~25 messages. In long conversations, important
      facts shared early can fall off the end. The checkpoint prompt reminds the
      companion to naturally reference key details so the memory extraction pipeline
      captures them before they age out.
HOW:  Three independent thresholds (any one triggers a checkpoint):
      1. Turn count: every 15 turns after the initial threshold
      2. Token estimate: when conversation exceeds ~5000 tokens (rough 4-char estimate)
      3. Duration: when conversation has lasted 20+ minutes
      Checkpoints respect a cooldown (10 turns) to avoid repeated injection.
      The checkpoint prompt is invisible to the user — it's a system-level hint
      that the companion processes internally.

Inspired by Clawdbot's "write durable memories" pattern.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (configurable via environment)
# ---------------------------------------------------------------------------

TURN_THRESHOLD = int(os.environ.get('CHECKPOINT_TURN_THRESHOLD', '15'))
TOKEN_THRESHOLD = int(os.environ.get('CHECKPOINT_TOKEN_THRESHOLD', '5000'))
DURATION_THRESHOLD_MINUTES = int(os.environ.get('CHECKPOINT_DURATION_MINUTES', '20'))
CHECKPOINT_ENABLED = os.environ.get('MEMORY_CHECKPOINT_ENABLED', 'true').lower() == 'true'


class CheckpointDetector:
    """
    Detects when a memory checkpoint should be injected into the conversation.

    A checkpoint prompts the companion to consolidate important information before
    the conversation gets too long, ensuring key facts are captured by
    the memory extraction system.
    """

    def __init__(self):
        self.turn_threshold = TURN_THRESHOLD
        self.token_threshold = TOKEN_THRESHOLD
        self.duration_threshold_minutes = DURATION_THRESHOLD_MINUTES
        self._last_checkpoint_turn = 0  # Track to avoid repeated checkpoints

    def should_checkpoint(
        self,
        conversation_turns: List[Dict[str, Any]],
        continuity_context: Optional[str] = None
    ) -> bool:
        """
        Check if checkpoint is needed based on thresholds.

        Args:
            conversation_turns: List of {"role": "user"|"assistant", "content": "..."}
            continuity_context: String describing conversation state (e.g., "ACTIVE CONVERSATION")

        Returns:
            True if checkpoint should be triggered
        """
        if not CHECKPOINT_ENABLED:
            return False

        turn_count = len(conversation_turns)

        # Don't checkpoint too early
        if turn_count < self.turn_threshold:
            return False

        # Don't checkpoint repeatedly - wait at least 10 more turns after last checkpoint
        if turn_count < self._last_checkpoint_turn + 10:
            return False

        # Check turn count threshold
        if turn_count >= self.turn_threshold:
            # Only checkpoint at specific intervals (every 15 turns after threshold)
            if (turn_count - self.turn_threshold) % 15 == 0:
                logger.info(f"Checkpoint triggered: {turn_count} turns (threshold: {self.turn_threshold})")
                self._last_checkpoint_turn = turn_count
                return True

        # Token-based threshold: rough estimate at 4 chars per token
        total_chars = sum(len(turn.get('content', '')) for turn in conversation_turns)
        token_estimate = total_chars // 4

        if token_estimate >= self.token_threshold:
            # Trigger at ~2000 token intervals past threshold (within 500-token window)
            if (token_estimate - self.token_threshold) % 2000 < 500:
                logger.info(f"Checkpoint triggered: ~{token_estimate} tokens (threshold: {self.token_threshold})")
                self._last_checkpoint_turn = turn_count
                return True

        # Check duration from continuity context
        # Continuity context contains phrases like "ACTIVE CONVERSATION (X minutes)"
        if continuity_context and self._check_duration_threshold(continuity_context):
            logger.info(f"Checkpoint triggered: duration threshold met")
            self._last_checkpoint_turn = turn_count
            return True

        return False

    def _check_duration_threshold(self, continuity_context: str) -> bool:
        """
        Check if conversation duration exceeds threshold.

        Parses duration from continuity context string.
        """
        import re

        # Look for patterns like "(X minutes)" or "(X hours)"
        minute_match = re.search(r'\((\d+)\s*minutes?\)', continuity_context, re.IGNORECASE)
        hour_match = re.search(r'\((\d+)\s*hours?\)', continuity_context, re.IGNORECASE)

        total_minutes = 0

        if hour_match:
            total_minutes += int(hour_match.group(1)) * 60
        if minute_match:
            total_minutes += int(minute_match.group(1))

        return total_minutes >= self.duration_threshold_minutes

    def get_checkpoint_prompt(self) -> str:
        """
        Return the checkpoint prompt to inject into the system prompt.

        This prompt guides the companion to consolidate important information
        without being obvious to the user.
        """
        return """
[MEMORY CHECKPOINT - Internal reminder]

This conversation has been substantial. Before responding to the user's message,
take a moment to note (internally, not out loud) any important information:

1. NEW FACTS: Did the user share anything new about themselves, their family, work, or plans?
2. CORRECTIONS: Did they correct anything you said or clarify something you misunderstood?
3. EMOTIONAL STATE: How are they feeling? Is something bothering them or exciting them?
4. CONTEXT: What's the current situation or topic that should be remembered?

You don't need to mention this checkpoint to the user. Just ensure your response
naturally incorporates any important details so they get captured by memory systems.

If nothing significant was shared recently, simply continue the conversation naturally.
"""

    def get_checkpoint_prompt_brief(self) -> str:
        """
        Return a briefer checkpoint prompt for less intrusive injection.
        """
        return """
[MEMORY NOTE: Long conversation - ensure important details from recent messages are reflected in your response for memory capture]
"""


# Singleton instance
_detector: Optional[CheckpointDetector] = None


def get_checkpoint_detector() -> CheckpointDetector:
    """Get or create CheckpointDetector singleton."""
    global _detector
    if _detector is None:
        _detector = CheckpointDetector()
    return _detector
