"""
Two-Level Persona Adaptation — AutoPal-inspired micro+macro personality evolution.

Inspired by the AutoPal paper (arxiv 2406.13960). The companion's personality evolves
at two levels:

MICRO (per-message):
  After each message, detect if the user revealed new information about
  interests, preferences, or traits. If so, check compatibility with
  the companion's existing persona and generate a lightweight micro-adaptation.
  Example: User mentions they started learning guitar → companion might
  develop curiosity about music or relate it to their own interests.

MACRO (weekly, existing):
  Value inference already handles this — weekly analysis of conversation
  history to update values, boundaries, preferences, etc.

The micro-level is fast (< 1s, rule-based pre-filter + optional LLM)
and runs as a background task after each message. It stores detected
traits in a lightweight format that the macro-level can use.

Feature flag: COMPANION_PERSONA_ADAPTATION_ENABLED (default: true)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.core.clock import now as clock_now
from src.database import tables as T

logger = logging.getLogger(__name__)

COMPANION_PERSONA_ADAPTATION_ENABLED = os.environ.get(
    'COMPANION_PERSONA_ADAPTATION_ENABLED', 'true'
).lower() == 'true'

# Keywords that suggest new user traits/interests
TRAIT_INDICATORS = [
    'started', 'began', 'learning', 'got into', 'picked up',
    'new hobby', 'trying out', 'discovered', 'obsessed with',
    'addicted to', 'really into', 'falling in love with',
    'changed my mind', 'used to', 'not anymore', 'quit',
    'stopped', 'switched to', 'converted to', 'moved to',
    'promoted', 'hired', 'fired', 'enrolled', 'graduated',
]

# Maximum micro-adaptations stored before macro-level processes them
MAX_PENDING_ADAPTATIONS = 50

# File for pending micro-adaptations
ADAPTATIONS_FILE = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'pending_persona_adaptations.json'
)


@dataclass
class MicroAdaptation:
    """A lightweight per-message persona adaptation."""
    detected_trait: str      # What was detected (e.g., "started learning guitar")
    trait_type: str          # Type: interest, preference, lifestyle, career, relationship
    user_or_companion: str   # "user" (about user) or "companion" (companion should adapt)
    suggested_adaptation: str  # How companion might adapt (e.g., "develop curiosity about music")
    timestamp: str           # ISO timestamp
    source_message: str      # Truncated source message

    def to_dict(self) -> Dict:
        return {
            'detected_trait': self.detected_trait,
            'trait_type': self.trait_type,
            'user_or_companion': self.user_or_companion,
            'suggested_adaptation': self.suggested_adaptation,
            'timestamp': self.timestamp,
            'source_message': self.source_message,
        }


def detect_trait_signals(message: str) -> List[str]:
    """
    Fast rule-based pre-filter to detect if a message contains trait signals.

    Returns a list of matching indicator phrases found.
    This is the cheap gate before any LLM call.
    """
    if not message or len(message) < 15:
        return []

    message_lower = message.lower()
    matches = []

    for indicator in TRAIT_INDICATORS:
        if indicator in message_lower:
            matches.append(indicator)

    return matches


def extract_adaptation(
    user_message: str,
    indicators: List[str],
) -> Optional[MicroAdaptation]:
    """
    Extract a micro-adaptation from a message with detected trait signals.

    Uses LLM for nuanced extraction. Only called when rule-based
    pre-filter detected indicators.
    """
    try:
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        prompt = f"""A user message contains signals of a new trait, interest, or life change.

Message: "{user_message[:300]}"
Detected indicators: {', '.join(indicators)}

Extract:
1. TRAIT: What new trait/interest/change was revealed? (1 sentence)
2. TYPE: interest, preference, lifestyle, career, or relationship
3. ADAPTATION: How should the companion naturally adapt? (1 sentence)
   Think: What would a real partner do? Show curiosity? Connect it to shared interests? Ask questions?

/no_think
Respond as JSON:
{{"trait": "...", "type": "...", "adaptation": "..."}}"""

        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[
                {"role": "system", "content": "Extract trait information. Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150,
            chain=chain,
            timeout=3  # Keep it fast
        )

        if not response:
            return None

        content = response.strip()
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        return MicroAdaptation(
            detected_trait=result.get('trait', ''),
            trait_type=result.get('type', 'interest'),
            user_or_companion='companion',
            suggested_adaptation=result.get('adaptation', ''),
            timestamp=clock_now().isoformat(),
            source_message=user_message[:100],
        )

    except Exception as e:
        logger.debug(f"Micro-adaptation extraction failed: {e}")
        return None


def store_pending_adaptation(adaptation: MicroAdaptation) -> None:
    """Store a micro-adaptation for later processing by macro-level."""
    try:
        existing = []
        if os.path.exists(ADAPTATIONS_FILE):
            with open(ADAPTATIONS_FILE, 'r') as f:
                existing = json.load(f)

        existing.append(adaptation.to_dict())

        # Cap at MAX_PENDING_ADAPTATIONS
        if len(existing) > MAX_PENDING_ADAPTATIONS:
            existing = existing[-MAX_PENDING_ADAPTATIONS:]

        os.makedirs(os.path.dirname(ADAPTATIONS_FILE), exist_ok=True)
        with open(ADAPTATIONS_FILE, 'w') as f:
            json.dump(existing, f, indent=2)

    except Exception as e:
        logger.debug(f"Could not store micro-adaptation: {e}")


def get_pending_adaptations() -> List[Dict]:
    """Get pending micro-adaptations (for macro-level processing)."""
    try:
        if os.path.exists(ADAPTATIONS_FILE):
            with open(ADAPTATIONS_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def clear_pending_adaptations() -> None:
    """Clear pending adaptations after macro-level has processed them."""
    try:
        if os.path.exists(ADAPTATIONS_FILE):
            with open(ADAPTATIONS_FILE, 'w') as f:
                json.dump([], f)
    except Exception:
        pass


def process_message_for_adaptation(user_message: str) -> Optional[MicroAdaptation]:
    """
    Main entry point: check a user message for persona adaptation signals.

    Called as a background task after each message. Fast path:
    1. Rule-based pre-filter (< 1ms)
    2. If indicators found, LLM extraction (< 3s)
    3. Store pending adaptation

    Returns the adaptation if one was detected, None otherwise.
    """
    if not COMPANION_PERSONA_ADAPTATION_ENABLED:
        return None

    # Fast pre-filter
    indicators = detect_trait_signals(user_message)
    if not indicators:
        return None

    logger.info(f"Trait signals detected: {indicators}")

    # LLM extraction
    adaptation = extract_adaptation(user_message, indicators)

    if adaptation and adaptation.detected_trait:
        store_pending_adaptation(adaptation)
        logger.info(
            f"Micro-adaptation: {adaptation.trait_type} - "
            f"{adaptation.detected_trait[:60]}"
        )
        return adaptation

    return None


def format_adaptations_for_value_inference() -> str:
    """
    Format pending micro-adaptations as context for weekly value inference.

    This is the bridge between micro and macro levels: pending adaptations
    are fed into the macro-level value inference so it can incorporate
    recent per-message observations into the broader personality update.
    """
    pending = get_pending_adaptations()
    if not pending:
        return ""

    lines = ["RECENT PERSONA SIGNALS (from per-message detection):"]
    for a in pending[-20:]:  # Last 20
        lines.append(
            f"- [{a.get('trait_type', 'unknown')}] {a.get('detected_trait', '')}"
            f" → {a.get('suggested_adaptation', '')}"
        )

    return '\n'.join(lines)
