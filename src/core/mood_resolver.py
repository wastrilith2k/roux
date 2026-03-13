"""
Unified Mood Resolver -- single authoritative mood for the companion.

WHAT: Merges three independent mood systems into one canonical mood + metadata
      dict. Returns (primary_mood, mood_info) where mood_info contains intensity,
      contributing sources, and category.

WHY:  Three systems independently track mood (EmotionalState per-message,
      MoodPersistence for long-lasting event moods, AvatarPersonality for
      avatar selection). Without a resolver, the avatar might show "happy"
      while the response text reflects "hurt". This module is the single
      source of truth.

HOW:  Priority chain:
      1. Strong persistent mood from a significant event (intensity > 0.7)
      2. Current contextual mood from EmotionalState
      3. Energy/engagement modifiers (tired, engaged)
      4. Neutral fallback
      The resolved mood is consumed by avatar selection, response generation,
      and time-passage narration.

Entry point: `get_resolved_mood()` at module bottom.
"""

import logging
from typing import List, Dict, Optional, Tuple
from src.core.emotional_state import get_emotional_state
from src.core.mood_persistence import get_mood_persistence

logger = logging.getLogger(__name__)


# Unified mood set - combines all three systems
UNIFIED_MOODS = {
    # Positive moods
    'elated': 'positive',
    'excited': 'positive',
    'happy': 'positive',
    'playful': 'positive',
    'proud': 'positive',
    'grateful': 'positive',
    'touched': 'positive',
    'creative': 'positive',

    # Neutral moods
    'neutral': 'neutral',
    'thoughtful': 'neutral',
    'peaceful': 'neutral',

    # Tired moods
    'tired': 'tired',

    # Negative moods
    'sad': 'negative',
    'melancholy': 'negative',
    'worried': 'negative',
    'anxious': 'negative',
    'frustrated': 'negative',
    'annoyed': 'negative',
    'hurt': 'negative',
}


def get_resolved_mood() -> Tuple[str, Dict[str, any]]:
    """
    Get the current resolved mood by intelligently merging all mood systems.

    Returns:
        Tuple of (primary_mood: str, mood_info: dict with details)

    Example return:
        ('excited', {
            'primary_mood': 'excited',
            'sources': ['persistent', 'contextual'],
            'intensity': 0.8,
            'category': 'positive',
            'from_persistent': {'mood': 'excited', 'intensity': 0.7, 'cause': '...'},
            'from_emotional_state': {'mood': 'excited', 'energy': 0.85, 'engagement': 0.9}
        })
    """
    try:
        # Get both mood systems
        emotional_state = get_emotional_state()
        mood_persistence = get_mood_persistence()

        # Get active moods from persistence (events/long-lasting moods)
        persistent_moods = mood_persistence.get_active_moods()

        # Get current contextual mood from emotional state
        contextual_mood = emotional_state.mood
        contextual_energy = emotional_state.energy_level
        contextual_engagement = emotional_state.topic_engagement

        # Determine primary mood based on priority
        primary_mood = None
        mood_sources = []
        persistent_mood_info = None

        # Priority 1: Strong persistent moods (intensity > 0.5)
        strong_persistent = [m for m in persistent_moods if m['intensity'] > 0.5]
        if strong_persistent:
            # Take the strongest mood
            persistent_mood_info = max(strong_persistent, key=lambda m: m['intensity'])
            primary_mood = persistent_mood_info['mood']
            mood_sources.append('persistent')
            logger.info(f"🎭 Primary mood from persistence: {primary_mood} ({persistent_mood_info['intensity']:.0%})")

        # Priority 2: Contextual mood + energy/engagement modifiers
        if not primary_mood:
            primary_mood = contextual_mood
            mood_sources.append('contextual')

            # Add special moods based on context if high engagement/energy
            if contextual_engagement > 0.8 and contextual_energy > 0.7:
                primary_mood = 'creative'  # High engagement + high energy = creative flow
                mood_sources.append('creative_flow')
                logger.info(f"🎭 Creative flow detected: high engagement ({contextual_engagement:.0%}) + high energy ({contextual_energy:.0%})")
            elif contextual_energy < 0.3:
                primary_mood = 'tired'
                mood_sources.append('exhaustion')

            logger.info(f"🎭 Primary mood from context: {primary_mood}")

        # Calculate overall intensity
        intensity = 0.5  # Default neutral intensity
        if persistent_mood_info:
            intensity = persistent_mood_info['intensity']
        elif contextual_energy < 0.3:
            intensity = 0.6  # Tired moods are moderately intense
        elif contextual_energy > 0.8:
            intensity = 0.8  # High energy moods are intense
        else:
            intensity = 0.5

        # Apply modulation from weak persistent moods
        if persistent_moods and not persistent_mood_info:
            # There are persistent moods but all weak - slightly boost intensity
            avg_persistent_intensity = sum(m['intensity'] for m in persistent_moods) / len(persistent_moods)
            intensity = min(0.9, intensity + avg_persistent_intensity * 0.2)

        # Get category for this mood
        mood_category = UNIFIED_MOODS.get(primary_mood, 'neutral')

        # Build return info
        mood_info = {
            'primary_mood': primary_mood,
            'sources': mood_sources,
            'intensity': intensity,
            'category': mood_category,
            'from_persistent': persistent_mood_info,
            'from_emotional_state': {
                'mood': contextual_mood,
                'energy': contextual_energy,
                'engagement': contextual_engagement
            }
        }

        logger.debug(f"✨ Resolved mood: {primary_mood} ({intensity:.0%}, category: {mood_category}, sources: {', '.join(mood_sources)})")

        return (primary_mood, mood_info)

    except Exception as e:
        logger.error(f"❌ Error resolving mood: {e}")
        logger.debug(f"   Falling back to neutral mood")
        return ('neutral', {
            'primary_mood': 'neutral',
            'sources': ['fallback'],
            'intensity': 0.5,
            'category': 'neutral',
            'from_persistent': None,
            'from_emotional_state': None,
            'error': str(e)
        })


def get_mood_info() -> Dict[str, any]:
    """Get detailed mood information for debugging/logging."""
    mood, info = get_resolved_mood()
    return info


def is_positive_mood(mood: Optional[str] = None) -> bool:
    """Check if a mood is positive. Uses resolved mood if not specified."""
    if mood is None:
        mood, _ = get_resolved_mood()
    return UNIFIED_MOODS.get(mood, 'neutral') == 'positive'


def is_negative_mood(mood: Optional[str] = None) -> bool:
    """Check if a mood is negative. Uses resolved mood if not specified."""
    if mood is None:
        mood, _ = get_resolved_mood()
    return UNIFIED_MOODS.get(mood, 'neutral') == 'negative'


def is_tired_mood(mood: Optional[str] = None) -> bool:
    """Check if a mood is tired. Uses resolved mood if not specified."""
    if mood is None:
        mood, _ = get_resolved_mood()
    return mood == 'tired'


def format_mood_for_prompt() -> str:
    """Format resolved mood information for inclusion in LLM prompts."""
    mood, info = get_resolved_mood()

    lines = [
        "\n## Current Mood (Unified System)",
        f"- Primary Mood: {mood.title()}",
        f"- Category: {info['category'].title()}",
        f"- Intensity: {info['intensity']:.0%}",
        f"- Sources: {', '.join(info['sources'])}"
    ]

    # Add persistent mood details if applicable
    if info['from_persistent']:
        pm = info['from_persistent']
        lines.append(f"\n**Persistent Mood Event:**")
        lines.append(f"- Type: {pm['mood'].title()}")
        lines.append(f"- Intensity: {pm['intensity']:.0%}")
        lines.append(f"- Cause: {pm['cause']}")

    # Add emotional state details
    if info['from_emotional_state']:
        es = info['from_emotional_state']
        lines.append(f"\n**Contextual State:**")
        lines.append(f"- Energy: {es['energy']:.0%}")
        lines.append(f"- Engagement: {es['engagement']:.0%}")
        lines.append(f"- Base Mood: {es['mood'].title()}")

    return '\n'.join(lines)


# Test function
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    print("Testing Unified Mood Resolver\n")
    print("=" * 60)

    mood, info = get_resolved_mood()
    print(f"\n🎭 Resolved Mood: {mood.upper()}")
    print(f"Category: {info['category']}")
    print(f"Intensity: {info['intensity']:.0%}")
    print(f"Sources: {', '.join(info['sources'])}")

    print(f"\n{format_mood_for_prompt()}")
    print("\n" + "=" * 60)
    print("✅ Mood resolver test complete")
