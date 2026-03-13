"""
Memory Confidence Decay - Temporal confidence modeling for facts.

WHAT: Computes an "effective confidence" for each fact by combining the
stored base confidence with recency decay and mention-frequency boost.
Also provides natural-language hedging qualifiers ("I think...",
"if I remember right...") calibrated to the effective confidence level.

WHY: Human memory fades for things not recently discussed. Without decay,
the companion would state a two-month-old offhand remark with the same
certainty as something confirmed yesterday. This module makes the companion
appropriately uncertain about stale, rarely-mentioned facts while remaining
confident about frequently-reinforced ones.

HOW it fits:
  - fact_store.get_facts_for_subject() calls enrich_facts_with_effective_confidence()
    to annotate each fact before it reaches the prompt.
  - context_builder uses format_facts_with_confidence() to add [UNCERTAIN]
    prefixes to low-confidence facts so the LLM knows to hedge.

Formula: effective = base_confidence * recency_factor * mention_boost
  - recency_factor: linear decay from 1.0 to RECENCY_FLOOR over FULL_DECAY_DAYS
  - mention_boost: +0.1 per additional mention, capped at MENTION_BOOST_CAP
  - Result clamped to [CONFIDENCE_FLOOR, CONFIDENCE_CEILING]

Feature flag: COMPANION_CONFIDENCE_DECAY_ENABLED (default: true)
"""

import os
import logging
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

COMPANION_CONFIDENCE_DECAY_ENABLED = os.environ.get('COMPANION_CONFIDENCE_DECAY_ENABLED', 'true').lower() == 'true'

FULL_DECAY_DAYS = 90     # Days for recency factor to reach its floor
RECENCY_FLOOR = 0.3      # Minimum recency factor (facts never fully vanish)
MENTION_BOOST_CAP = 1.5  # Maximum multiplier from repeated mentions
CONFIDENCE_FLOOR = 0.1   # Absolute minimum effective confidence
CONFIDENCE_CEILING = 0.99  # Absolute maximum effective confidence


def calculate_effective_confidence(
    base_confidence: float,
    mention_count: int,
    last_mentioned: Optional[datetime],
    created_at: Optional[datetime]
) -> float:
    """
    Calculate time-adjusted confidence for a fact.

    Formula: effective = base * recency_factor * mention_boost

    Args:
        base_confidence: Stored confidence value (0.0-1.0)
        mention_count: Number of times this fact has been mentioned
        last_mentioned: When the fact was last referenced
        created_at: When the fact was first stored

    Returns:
        Effective confidence (0.1 - 0.99)
    """
    if not COMPANION_CONFIDENCE_DECAY_ENABLED:
        return base_confidence

    now = clock_now()

    # --- Recency factor: how recently was this fact last referenced? ---
    # Prefer last_mentioned; fall back to created_at; assume old if neither.
    if last_mentioned:
        days_since = (now - last_mentioned).days
    elif created_at:
        days_since = (now - created_at).days
    else:
        days_since = FULL_DECAY_DAYS  # Worst case: treat as maximally decayed

    recency_factor = max(RECENCY_FLOOR, 1.0 - days_since / FULL_DECAY_DAYS)

    # --- Mention boost: frequently referenced facts stay vivid ---
    mention_count = mention_count or 1
    mention_boost = min(MENTION_BOOST_CAP, 1.0 + (mention_count - 1) * 0.1)

    # --- Combine and clamp ---
    effective = (base_confidence or 0.7) * recency_factor * mention_boost
    return max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, effective))


def get_confidence_qualifier(effective_confidence: float) -> str:
    """
    Get a natural hedging qualifier based on confidence level.

    Returns empty string for high confidence (no hedging needed).
    """
    if effective_confidence >= 0.8:
        return ""
    elif effective_confidence >= 0.6:
        return random.choice(["I think", "if I remember right"])
    elif effective_confidence >= 0.4:
        return random.choice(["I'm not sure but", "I vaguely remember"])
    else:
        return random.choice(["I might be wrong but", "I'm fuzzy on this but"])


def format_facts_with_confidence(facts: List[Dict[str, Any]]) -> str:
    """
    Format facts with confidence qualifiers for prompt injection.

    High-confidence facts are presented as-is.
    Low-confidence facts get hedging prefixes.

    Args:
        facts: List of fact dicts with 'effective_confidence' key

    Returns:
        Formatted string for prompt injection
    """
    if not facts:
        return ""

    lines = []
    for fact in facts:
        eff_conf = fact.get('effective_confidence', fact.get('confidence', 0.7))
        subject = fact.get('subject', '')
        predicate = fact.get('predicate', '')
        obj = fact.get('object', '')

        qualifier = get_confidence_qualifier(eff_conf)

        if qualifier:
            lines.append(f"[UNCERTAIN] {qualifier}, {subject} {predicate} {obj}")
        else:
            lines.append(f"{subject} {predicate} {obj}")

    return '\n'.join(lines)


def enrich_facts_with_effective_confidence(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add 'effective_confidence' field to each fact dict.

    Modifies facts in-place and returns them.
    """
    for fact in facts:
        fact['effective_confidence'] = calculate_effective_confidence(
            base_confidence=fact.get('confidence', 0.7),
            mention_count=fact.get('mention_count', 1),
            last_mentioned=fact.get('last_mentioned'),
            created_at=fact.get('created_at')
        )
    return facts
