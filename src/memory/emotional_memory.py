"""
Emotional Memory Weighting — Bayesian emotional confidence for facts.

Inspired by "Dynamic Affective Memory Management for Personalized LLM
Agents" (arxiv 2510.27418). Each fact carries an emotional confidence
distribution across three dimensions: positive, negative, neutral.

Key insights:
- Emotionally charged memories (high positive OR high negative) naturally
  resist decay — just like human memory.
- Ambiguous/neutral memories fade faster.
- Bayesian updates allow emotional associations to evolve as new evidence
  arrives.

The emotional valence modulates the confidence decay formula:
  emotional_persistence = |positive - negative| + 0.5 * max(positive, negative)
  decay_modifier = 1.0 + emotional_persistence * 0.5

High-valence facts decay slower (multiplied into the access-based decay).

Feature flag: COMPANION_EMOTIONAL_MEMORY_ENABLED (default: true)
"""

import os
import json
import math
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

COMPANION_EMOTIONAL_MEMORY_ENABLED = os.environ.get(
    'COMPANION_EMOTIONAL_MEMORY_ENABLED', 'true'
).lower() == 'true'


@dataclass
class EmotionalProfile:
    """
    Bayesian emotional confidence distribution for a memory.

    Each dimension is 0.0-1.0 and they sum to ~1.0 (normalized).
    """
    positive: float = 0.33
    negative: float = 0.33
    neutral: float = 0.34

    def normalize(self) -> 'EmotionalProfile':
        """Ensure dimensions sum to 1.0."""
        total = self.positive + self.negative + self.neutral
        if total == 0:
            return EmotionalProfile()
        return EmotionalProfile(
            positive=self.positive / total,
            negative=self.negative / total,
            neutral=self.neutral / total,
        )

    @property
    def valence(self) -> float:
        """
        Emotional valence: -1.0 (very negative) to +1.0 (very positive).
        0.0 is neutral.
        """
        return self.positive - self.negative

    @property
    def intensity(self) -> float:
        """
        Emotional intensity: 0.0 (neutral) to 1.0 (very emotional).
        High intensity = strong positive OR strong negative.
        """
        return max(self.positive, self.negative)

    @property
    def entropy(self) -> float:
        """
        Belief entropy H(m) = -sum(p_k * log2(p_k)).
        Low entropy = clear emotional signal.
        High entropy = ambiguous/contradictory.

        Thresholds:
        - < 0.8: Clear emotional signal (healthy)
        - 0.8-1.4: Mixed signal
        - > 1.4: High confusion (candidate for review)
        """
        h = 0.0
        for p in [self.positive, self.negative, self.neutral]:
            if p > 0:
                h -= p * math.log2(p)
        return h

    @property
    def persistence_modifier(self) -> float:
        """
        How much this emotional profile should slow decay.

        High-valence memories (strongly positive or negative) get a
        persistence bonus that slows their decay rate.

        Returns:
            1.0 (no modifier) to 1.5 (50% slower decay)
        """
        if not COMPANION_EMOTIONAL_MEMORY_ENABLED:
            return 1.0

        # Stronger emotions = more persistent
        emotional_strength = abs(self.valence) + 0.5 * self.intensity
        # Map to 1.0-1.5 range
        return 1.0 + min(0.5, emotional_strength * 0.3)

    def to_dict(self) -> Dict[str, float]:
        return {
            'positive': round(self.positive, 3),
            'negative': round(self.negative, 3),
            'neutral': round(self.neutral, 3),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'EmotionalProfile':
        return cls(
            positive=d.get('positive', 0.33),
            negative=d.get('negative', 0.33),
            neutral=d.get('neutral', 0.34),
        )

    @classmethod
    def from_json(cls, s: str) -> 'EmotionalProfile':
        try:
            return cls.from_dict(json.loads(s))
        except (json.JSONDecodeError, TypeError):
            return cls()


def bayesian_update(
    current: EmotionalProfile,
    new_sentiment: Dict[str, float],
    evidence_strength: float = 0.5,
) -> EmotionalProfile:
    """
    Update emotional profile using Bayesian formula.

    C_new = (C * W + S * P) / (W + S)

    Where:
        C = current emotional confidence
        W = prior belief strength (based on how established the profile is)
        S = evidence strength (correlated with emotional intensity)
        P = new sentiment values

    Args:
        current: Current emotional profile
        new_sentiment: New emotional evidence {'positive': 0-1, 'negative': 0-1, 'neutral': 0-1}
        evidence_strength: How strong the new evidence is (0.0-1.0)

    Returns:
        Updated emotional profile
    """
    # Prior strength: more extreme current beliefs = stronger prior
    prior_strength = 1.0 + current.intensity

    s = max(0.1, evidence_strength)

    new_positive = (current.positive * prior_strength + new_sentiment.get('positive', 0.33) * s) / (prior_strength + s)
    new_negative = (current.negative * prior_strength + new_sentiment.get('negative', 0.33) * s) / (prior_strength + s)
    new_neutral = (current.neutral * prior_strength + new_sentiment.get('neutral', 0.34) * s) / (prior_strength + s)

    return EmotionalProfile(
        positive=new_positive,
        negative=new_negative,
        neutral=new_neutral,
    ).normalize()


def classify_sentiment(text: str) -> Dict[str, float]:
    """
    Quick rule-based sentiment classification for a text.

    For high-stakes classification, use an LLM instead. This is a
    lightweight heuristic for common patterns.

    Returns:
        Dict with positive/negative/neutral scores summing to ~1.0
    """
    text_lower = text.lower()

    # Positive indicators
    positive_words = [
        'happy', 'love', 'great', 'wonderful', 'excited', 'amazing',
        'joy', 'celebrate', 'proud', 'grateful', 'beautiful', 'fun',
        'perfect', 'awesome', 'glad', 'enjoy', 'nice', 'good',
    ]
    # Negative indicators
    negative_words = [
        'sad', 'angry', 'frustrated', 'worried', 'anxious', 'stress',
        'hurt', 'scared', 'afraid', 'upset', 'depressed', 'crisis',
        'terrible', 'awful', 'horrible', 'hate', 'sick', 'pain',
        'died', 'death', 'lost', 'broke', 'fight', 'argument',
    ]

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    total = pos_count + neg_count

    if total == 0:
        return {'positive': 0.2, 'negative': 0.2, 'neutral': 0.6}

    pos_ratio = pos_count / total
    neg_ratio = neg_count / total

    # Scale so emotional content has higher confidence
    intensity = min(1.0, total / 3.0)  # 3+ emotional words = max intensity

    return {
        'positive': pos_ratio * intensity * 0.7 + 0.1,
        'negative': neg_ratio * intensity * 0.7 + 0.1,
        'neutral': max(0.1, 1.0 - intensity * 0.7),
    }


def get_emotional_profile_for_fact(fact: Dict[str, Any]) -> EmotionalProfile:
    """
    Get or compute emotional profile for a fact.

    Checks for stored emotional_profile JSONB first, falls back to
    rule-based classification of the fact text.
    """
    # Check for stored profile
    stored = fact.get('emotional_profile')
    if stored:
        if isinstance(stored, str):
            return EmotionalProfile.from_json(stored)
        elif isinstance(stored, dict):
            return EmotionalProfile.from_dict(stored)

    # Classify from fact text
    text_parts = [
        fact.get('subject', ''),
        fact.get('predicate', ''),
        fact.get('object', ''),
        fact.get('context', ''),
    ]
    text = ' '.join(p for p in text_parts if p)
    sentiment = classify_sentiment(text)

    return EmotionalProfile.from_dict(sentiment)


def apply_emotional_persistence(
    effective_confidence: float,
    emotional_profile: EmotionalProfile,
) -> float:
    """
    Apply emotional persistence modifier to confidence.

    High-valence memories decay slower (modifier > 1.0 amplifies confidence).

    Args:
        effective_confidence: Base confidence after access-based decay
        emotional_profile: The fact's emotional profile

    Returns:
        Modified confidence (still clamped to 0.1-0.99)
    """
    if not COMPANION_EMOTIONAL_MEMORY_ENABLED:
        return effective_confidence

    modifier = emotional_profile.persistence_modifier
    result = effective_confidence * modifier

    return max(0.1, min(0.99, result))
