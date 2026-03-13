"""
Entity Profile Context Filter -- embedding-based profile section selection.

WHAT: Uses cosine similarity between conversation context and reference
      embeddings to decide which entity-profile sections to include in the
      prompt. Always includes core identity; conditionally includes intimacy/
      boundaries, appearance, family, and memory-gap sections.

WHY:  Full entity profiles can be thousands of tokens. Most messages don't need
      the companion's trauma history or family tree. This filter keeps prompt
      size down while guaranteeing safety-critical sections (trauma, boundaries)
      appear whenever intimacy is detected -- even at a low threshold.

HOW:  Reference embeddings for each context type (intimate, appearance, family,
      memory) are cached on first use. For each message, a combined embedding
      of recent messages is compared against each reference. If similarity
      exceeds the per-type threshold, that category's sections are included.
      Crisis/health sections (`ALWAYS_INCLUDE_IF_EXISTS`) bypass filtering
      entirely.

Singleton: `get_entity_profile_context_filter()` at module bottom.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Section categories by profile type — keyed by entity name from entity_profiles/
# Override per-entity with sections that should always be included regardless of context.
# Any entity not listed here falls back to 'default'.
ALWAYS_INCLUDE_SECTIONS = {
    'default': {'name', 'birthday', 'relationship_status'},
}

# CRITICAL: These sections are ALWAYS included if they exist (crisis info, health emergencies)
ALWAYS_INCLUDE_IF_EXISTS = {'current_crisis', 'health'}

# Context triggers mapped to sections
CONTEXT_SECTION_MAPPING = {
    'intimate': {'trauma_history', 'boundaries'},
    'appearance': {'physical', 'visual'},
    'family': {'family'},
    'memory': {'memory_gaps'},
}

# Reference text for context detection embeddings
CONTEXT_REFERENCE_TEXT = {
    'intimate': (
        'intimate romantic sexual physical touch vulnerability trust '
        'boundaries consent belt restraint undress kiss hold bed bedroom '
        'cuddle close together naked body'
    ),
    'appearance': (
        'looks appearance physical body face hair eyes skin clothes '
        'wearing beautiful pretty handsome tall short build dress outfit '
        'how you look what you look like'
    ),
    'family': (
        'mother father parent sibling sister brother family childhood '
        'grew up home background parents mom dad kids children'
    ),
    'memory': (
        'remember forgot memory confused what happened earlier yesterday '
        'did we when was forget forgotten lost track blank foggy'
    ),
}

# Similarity thresholds for context detection
CONTEXT_THRESHOLDS = {
    'intimate': 0.50,  # Lower threshold for intimacy safety
    'appearance': 0.55,
    'family': 0.55,
    'memory': 0.55,
}


@dataclass
class ContextSignals:
    """Detected context signals from message analysis."""
    intimacy_detected: bool = False
    appearance_mentioned: bool = False
    family_mentioned: bool = False
    memory_confusion: bool = False


class EntityProfileContextFilter:
    """
    Filters entity profile sections based on conversation context.

    Uses embeddings for semantic context detection instead
    of keyword lists, following LLM-first design philosophy.
    """

    def __init__(self):
        """Initialize filter with embedding cache."""
        self._reference_cache: Dict[str, List[float]] = {}

    def get_filtered_profile(
        self,
        entity_name: str,
        user_message: str,
        recent_messages: List[str] = None,
        full_profile: Dict = None
    ) -> Dict:
        """
        Get context-filtered entity profile.

        Args:
            entity_name: Entity name (e.g., 'companion', 'james')
            user_message: Current user message
            recent_messages: Recent conversation messages
            full_profile: Full profile dict (loaded if not provided)

        Returns:
            Filtered profile dict with only relevant sections.
        """
        if full_profile is None:
            from src.core.entity_profile_loader import get_entity_profile_loader
            loader = get_entity_profile_loader()
            full_profile = loader.get_profile(entity_name)

        if not full_profile:
            return {}

        # Detect context signals
        context = self.detect_context(user_message, recent_messages or [])

        # Get sections to include
        sections_to_include = self._get_sections_for_context(entity_name, context)

        # Filter profile
        filtered = {}
        for key, value in full_profile.items():
            # Include if: in sections_to_include OR 'notes' OR in always-include-if-exists
            if key in sections_to_include or key == 'notes' or key in ALWAYS_INCLUDE_IF_EXISTS:
                filtered[key] = value

        logger.debug(
            f"Filtered {entity_name} profile: {len(filtered)}/{len(full_profile)} sections "
            f"(context: intimate={context.intimacy_detected}, appearance={context.appearance_mentioned})"
        )

        return filtered

    def detect_context(
        self,
        user_message: str,
        recent_messages: List[str]
    ) -> ContextSignals:
        """
        Detect conversation context using embedding similarity.

        Args:
            user_message: Current user message
            recent_messages: Recent conversation messages

        Returns:
            ContextSignals with detected context flags.
        """
        # Combine recent messages with current for context
        combined_text = user_message
        if recent_messages:
            combined_text = " ".join(recent_messages[-3:]) + " " + user_message

        # Generate embedding for combined context
        context_embedding = self._get_embedding(combined_text)
        if not context_embedding:
            # Fallback: no context detected, only always-include sections
            return ContextSignals()

        signals = ContextSignals()

        # Check each context type using semantic similarity
        for context_type, reference_text in CONTEXT_REFERENCE_TEXT.items():
            reference_embedding = self._get_reference_embedding(context_type, reference_text)
            if reference_embedding:
                similarity = self._cosine_similarity(context_embedding, reference_embedding)
                threshold = CONTEXT_THRESHOLDS.get(context_type, 0.55)

                if similarity >= threshold:
                    if context_type == 'intimate':
                        signals.intimacy_detected = True
                        logger.debug(f"Intimacy context detected (similarity: {similarity:.2f})")
                    elif context_type == 'appearance':
                        signals.appearance_mentioned = True
                        logger.debug(f"Appearance context detected (similarity: {similarity:.2f})")
                    elif context_type == 'family':
                        signals.family_mentioned = True
                    elif context_type == 'memory':
                        signals.memory_confusion = True

        return signals

    def _get_sections_for_context(
        self,
        entity_name: str,
        context: ContextSignals
    ) -> Set[str]:
        """
        Get set of sections to include based on context.

        Args:
            entity_name: Entity name
            context: Detected context signals

        Returns:
            Set of section names to include.
        """
        entity_lower = entity_name.lower()

        # Start with always-include sections
        sections = set(ALWAYS_INCLUDE_SECTIONS.get(
            entity_lower,
            ALWAYS_INCLUDE_SECTIONS['default']
        ))

        # Add context-triggered sections
        if context.intimacy_detected:
            sections.update(CONTEXT_SECTION_MAPPING['intimate'])
        if context.appearance_mentioned:
            sections.update(CONTEXT_SECTION_MAPPING['appearance'])
        if context.family_mentioned:
            sections.update(CONTEXT_SECTION_MAPPING['family'])
        if context.memory_confusion:
            sections.update(CONTEXT_SECTION_MAPPING['memory'])

        return sections

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        try:
            from src.memory.embeddings import generate_embedding
            return generate_embedding(text)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def _get_reference_embedding(
        self,
        name: str,
        description: str
    ) -> Optional[List[float]]:
        """Get or create cached reference embedding."""
        if name not in self._reference_cache:
            embedding = self._get_embedding(description)
            if embedding:
                self._reference_cache[name] = embedding

        return self._reference_cache.get(name)

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """Compute cosine similarity between vectors."""
        try:
            from src.memory.embeddings import cosine_similarity
            return cosine_similarity(a, b)
        except Exception:
            return 0.0


# Singleton accessor
_filter: Optional[EntityProfileContextFilter] = None


def get_entity_profile_context_filter() -> EntityProfileContextFilter:
    """Get or create EntityProfileContextFilter singleton."""
    global _filter
    if _filter is None:
        _filter = EntityProfileContextFilter()
    return _filter


def clear_filter_cache():
    """Clear singleton for testing."""
    global _filter
    _filter = None
