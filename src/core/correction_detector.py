"""
Correction Detector -- detects when the user is correcting the companion.

WHAT: Classifies user messages as corrections (or not) using a fast Fireworks
      LLM call. Returns a Correction dataclass with wrong_claim, correct_info,
      subject, type (fact/memory/relationship/timing), confidence, and
      importance (1-10 scale).

WHY:  When the companion says something wrong ("you work at LeanTaaS") and the
      user corrects it ("no, I work at Acme"), we need to detect that correction
      and store it so the claim verifier can prevent the same mistake in future.

HOW:  A keyword pre-filter (correction words like "no", "actually", "that's not")
      plus contradiction patterns ("I drink", "I am") gate the LLM call. If any
      match, the user message + previous companion message are sent to Fireworks
      for structured JSON classification. Detected corrections flow to
      CorrectionStore for persistence.

Singleton: `get_correction_detector()` at module bottom.
"""

import os
import json
import logging
from openai import OpenAI
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model config - use centralized definitions with fallbacks
from src.llm.fireworks_models import FAST_MODELS, call_fireworks


@dataclass
class Correction:
    """Represents a detected correction."""
    wrong_claim: str  # What the companion said incorrectly
    correct_info: str  # What the user says is correct
    subject: str  # Who/what this is about
    correction_type: str  # fact, memory, relationship, timing
    confidence: float  # How confident we are this is a correction
    importance: int = 5  # 1-10 scale: how important is this to remember?


class CorrectionDetector:
    """
    Detects when user is correcting the companion.

    Uses Fireworks for fast classification.
    """

    def __init__(self):
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            logger.warning("FIREWORKS_API_KEY not set - correction detection disabled")
            self.client = None
        else:
            self.client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=self.api_key
            )

    def detect(
        self,
        user_message: str,
        previous_companion_message: str
    ) -> Optional[Correction]:
        """
        Detect if user is correcting something the companion said.

        Args:
            user_message: The user's current message
            previous_companion_message: The companion's previous response

        Returns:
            Correction object if detected, None otherwise
        """
        if not self.client:
            return None

        if not user_message or not previous_companion_message:
            return None

        # Skip very short messages
        if len(user_message) < 10:
            return None

        # Quick keyword pre-filter to avoid unnecessary API calls
        correction_keywords = [
            "no,", "no ", "actually", "that's not", "that's wrong",
            "incorrect", "you got", "i never", "i didn't", "i don't",
            "it's not", "it was", "it's", "not ", "wrong", "isn't",
            "wasn't", "correction", "correct ", "remember,"
        ]

        message_lower = user_message.lower()
        if not any(kw in message_lower for kw in correction_keywords):
            # Also check for statements that contradict (e.g., "I drink tea")
            # These might not have explicit correction keywords
            contradiction_patterns = [
                "i drink", "i am", "i have", "i was", "i did",
                "it was", "it is", "she is", "he is", "they are",
                "we were", "we are", "my ", "the interview"
            ]
            if not any(p in message_lower for p in contradiction_patterns):
                return None

        try:
            prompt = f"""Analyze if this user message is correcting something the AI said.

Previous AI message (companion):
"{previous_companion_message[:1000]}"

User message:
"{user_message}"

Correction patterns to look for:
- "No, that's not right..." / "That's incorrect..."
- "Actually, it's..." / "It's X, not Y"
- "You got X wrong, it's Y"
- "I never said/did that..."
- "I don't drink coffee" (contradicting an assumption)
- Stating a fact that contradicts what the companion said
- Correcting timing ("it was 2 months ago, not last week")
- Correcting who/what ("it's X, not Y")

If this IS a correction, respond with ONLY this JSON:
{{
  "is_correction": true,
  "wrong_claim": "what the companion said that was wrong (be specific)",
  "correct_info": "what the user says is correct",
  "subject": "person/thing/event this is about (e.g., 'interview', 'Lena', 'piggyback ride')",
  "correction_type": "fact|memory|relationship|timing",
  "confidence": 0.0-1.0,
  "importance": 1-10
}}

Importance scale:
- 9-10: Core identity facts (coffee preference, family names, job history, major life events)
- 7-8: Relationship dynamics, recurring preferences, significant memories
- 5-6: General facts, occasional preferences
- 3-4: Situational/scenario corrections (items on this trip, today's plans)
- 1-2: Trivial roleplay details that won't matter later

If NOT a correction (just continuing conversation, agreeing, asking questions), respond with ONLY:
{{
  "is_correction": false
}}

Respond with ONLY valid JSON, no other text."""

            result_text = call_fireworks(
                self.client,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )

            if not result_text:
                return None

            # Parse JSON response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.warning(f"Failed to parse correction detection response: {result_text[:100]}")
                    return None

            if not result.get("is_correction"):
                return None

            return Correction(
                wrong_claim=result.get("wrong_claim", ""),
                correct_info=result.get("correct_info", ""),
                subject=result.get("subject", ""),
                correction_type=result.get("correction_type", "fact"),
                confidence=float(result.get("confidence", 0.8)),
                importance=int(result.get("importance", 5))
            )

        except Exception as e:
            logger.error(f"Correction detection error: {e}")
            return None


# Singleton instance
_detector: Optional[CorrectionDetector] = None


def get_correction_detector() -> CorrectionDetector:
    """Get singleton CorrectionDetector instance."""
    global _detector
    if _detector is None:
        _detector = CorrectionDetector()
    return _detector


def detect_correction(
    user_message: str,
    previous_companion_message: str
) -> Optional[Correction]:
    """
    Convenience function to detect corrections.

    Args:
        user_message: The user's current message
        previous_companion_message: The companion's previous response

    Returns:
        Correction object if detected, None otherwise
    """
    detector = get_correction_detector()
    return detector.detect(user_message, previous_companion_message)
