"""Regression tests for issue #25: Retrieval-based context injection.

The bug: Three large context sections (observations ~3,400 tokens, core memory
~1,000 tokens, recent significant events ~600 tokens) were injected on every
message regardless of relevance, consuming ~5,000 tokens (38%) of the prompt.

The fix:
1. Observations omitted from lightweight (fast) and medium paths — only
   injected on the full/complex path.
2. Core memory omitted from lightweight (fast) path — only injected on
   medium and full paths.
3. Recent significant events deduplicated by word-level Jaccard similarity
   and capped at 10 entries.
"""

import pytest
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePersonaConfig:
    companion_short_name: str = "Kai"
    companion_name: str = "Kai Tanaka"
    primary_user_name: str = "James"
    c_possessive: str = "her"
    c_pronoun_subject: str = "she"
    c_pronoun_object: str = "her"
    u_possessive: str = "his"
    u_pronoun_subject: str = "he"
    u_pronoun_object: str = "him"
    primary_user_email: str = "test@test.com"


def _get_fake_persona_config():
    return FakePersonaConfig()


def _make_builder():
    """Create a ContextBuilder with all source fetchers mocked."""
    from src.core.conversation.context_builder import ContextBuilder

    builder = ContextBuilder.__new__(ContextBuilder)
    builder.db = MagicMock()
    builder._executor = ThreadPoolExecutor(max_workers=4)
    builder._source_timings = {}

    # Mock all _get_* methods to return empty strings by default
    for method_name in dir(builder):
        if method_name.startswith('_get_') and callable(getattr(builder, method_name)):
            setattr(builder, method_name, MagicMock(return_value=""))

    # Special return types for methods that return tuples
    builder._get_conversation_history_structured = MagicMock(return_value=([], "", ""))
    builder._get_memories = MagicMock(return_value=("", ""))

    return builder


# ---------------------------------------------------------------------------
# 1. Observations: omitted from lightweight and medium, present in full
# ---------------------------------------------------------------------------

class TestObservationsRetrievalBased:
    """Observations context should only be fetched on the full/complex path."""

    def test_lightweight_build_omits_observations(self):
        """build_lightweight() should not populate observations_context."""
        builder = _make_builder()
        builder._get_observations_context = MagicMock(return_value="OBSERVATIONS_MARKER")

        ctx = builder.build_lightweight("test@test.com", "hey", 50)

        builder._get_observations_context.assert_not_called()
        assert ctx.observations_context == ""

    def test_medium_build_omits_observations(self):
        """build_medium() should not populate observations_context."""
        builder = _make_builder()
        builder._get_observations_context = MagicMock(return_value="OBSERVATIONS_MARKER")

        ctx = builder.build_medium("test@test.com", "how's your day going?", 50)

        builder._get_observations_context.assert_not_called()
        assert ctx.observations_context == ""

    def test_full_build_includes_observations(self):
        """The full build path should still fetch observations_context."""
        builder = _make_builder()
        builder._get_observations_context = MagicMock(return_value="OBSERVATIONS_MARKER")

        ctx = builder._build_parallel("test@test.com", "I need to talk about something", 50)

        builder._get_observations_context.assert_called_once()
        assert ctx.observations_context == "OBSERVATIONS_MARKER"


# ---------------------------------------------------------------------------
# 2. Core Memory: omitted from lightweight, present in medium and full
# ---------------------------------------------------------------------------

class TestCoreMemoryRetrievalBased:
    """Core memory should be omitted from the fast/lightweight path."""

    def test_lightweight_build_omits_core_memory(self):
        """build_lightweight() should not populate core_memory."""
        builder = _make_builder()
        builder._get_core_memory = MagicMock(return_value="CORE_MEMORY_MARKER")

        ctx = builder.build_lightweight("test@test.com", "hey!", 50)

        builder._get_core_memory.assert_not_called()
        assert ctx.core_memory == ""

    def test_medium_build_includes_core_memory(self):
        """build_medium() should still populate core_memory."""
        builder = _make_builder()
        builder._get_core_memory = MagicMock(return_value="CORE_MEMORY_MARKER")

        ctx = builder.build_medium("test@test.com", "what have you been up to?", 50)

        builder._get_core_memory.assert_called_once()
        assert ctx.core_memory == "CORE_MEMORY_MARKER"

    def test_full_build_includes_core_memory(self):
        """The full build should still populate core_memory."""
        builder = _make_builder()
        builder._get_core_memory = MagicMock(return_value="CORE_MEMORY_MARKER")

        ctx = builder._build_parallel("test@test.com", "I need to talk about something", 50)

        builder._get_core_memory.assert_called_once()
        assert ctx.core_memory == "CORE_MEMORY_MARKER"


# ---------------------------------------------------------------------------
# 3. Recent Significant Events deduplication
# ---------------------------------------------------------------------------

class TestDeduplicateFacts:
    """Test the _deduplicate_facts function for near-duplicate removal."""

    def test_identical_facts_deduplicated(self):
        """Exact duplicate facts should be reduced to one."""
        from src.memory.temporal_context import _deduplicate_facts

        facts = [
            {'object': 'James deflects emotional topics', 'importance': 8},
            {'object': 'James deflects emotional topics', 'importance': 7},
        ]
        result = _deduplicate_facts(facts)
        assert len(result) == 1
        assert result[0]['importance'] == 8

    def test_near_duplicate_facts_deduplicated(self):
        """Facts with high word overlap should be deduplicated."""
        from src.memory.temporal_context import _deduplicate_facts

        # "James deflects emotional topics" is fully contained in the longer version
        facts = [
            {'object': 'James deflects emotional topics', 'importance': 8},
            {'object': 'James deflects emotional topics when asked directly', 'importance': 6},
        ]
        result = _deduplicate_facts(facts)
        assert len(result) == 1
        assert result[0]['importance'] == 8

    def test_distinct_facts_preserved(self):
        """Facts about different topics should all be kept."""
        from src.memory.temporal_context import _deduplicate_facts

        facts = [
            {'object': 'James got a new job at Google', 'importance': 9},
            {'object': 'Jesse ran away from home yesterday', 'importance': 8},
            {'object': 'Kyler has a math test on Friday', 'importance': 5},
        ]
        result = _deduplicate_facts(facts)
        assert len(result) == 3

    def test_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        from src.memory.temporal_context import _deduplicate_facts

        assert _deduplicate_facts([]) == []

    def test_single_fact_returned_unchanged(self):
        """A single fact passes through unchanged."""
        from src.memory.temporal_context import _deduplicate_facts

        facts = [{'object': 'James is interviewing at Act-On', 'importance': 7}]
        result = _deduplicate_facts(facts)
        assert len(result) == 1

    def test_multiple_near_duplicates_reduced(self):
        """Multiple phrasings of the same fact should collapse significantly."""
        from src.memory.temporal_context import _deduplicate_facts

        # Simulates the production pattern from the issue
        facts = [
            {'object': 'James deflects emotional topics', 'importance': 8},
            {'object': 'James deflects emotional topics when asked', 'importance': 7},
            {'object': 'James often deflects emotional topics in conversation', 'importance': 6},
        ]
        result = _deduplicate_facts(facts)
        # The shorter phrase is a high-containment subset of the longer ones
        assert len(result) <= 2, \
            f"Expected at most 2 after dedup, got {len(result)}: {[f['object'] for f in result]}"

    def test_custom_threshold(self):
        """Custom similarity threshold is respected."""
        from src.memory.temporal_context import _deduplicate_facts

        # These share very few words — only deduped at low threshold
        facts = [
            {'object': 'The weather was nice today for a walk', 'importance': 4},
            {'object': 'James mentioned the nice sunny weather', 'importance': 3},
        ]
        # High threshold — these should NOT be deduped
        result = _deduplicate_facts(facts, similarity_threshold=0.95)
        assert len(result) == 2

        # Very low threshold — these SHOULD be deduped
        result = _deduplicate_facts(facts, similarity_threshold=0.20)
        assert len(result) == 1

    def test_fact_with_empty_object_preserved(self):
        """Facts with empty object text are kept (not crashed on)."""
        from src.memory.temporal_context import _deduplicate_facts

        facts = [
            {'object': '', 'importance': 5},
            {'object': 'James got a promotion', 'importance': 8},
        ]
        result = _deduplicate_facts(facts)
        assert len(result) == 2


class TestFormatTemporalContextCap:
    """Test that format_temporal_context caps facts at 10 after dedup."""

    @patch('src.memory.temporal_context.get_recent_significant_events')
    @patch('src.memory.temporal_context.get_recent_notable_messages')
    def test_facts_capped_at_10(self, mock_messages, mock_events):
        """Even with many distinct facts, output is capped at 10."""
        from src.memory.temporal_context import format_temporal_context
        from datetime import datetime
        from zoneinfo import ZoneInfo

        PST = ZoneInfo('America/Los_Angeles')
        now = datetime.now(PST)

        # 15 distinct facts
        mock_events.return_value = [
            {'object': f'Unique fact number {i}', 'importance': 9 - (i % 3),
             'created_at': now}
            for i in range(15)
        ]
        mock_messages.return_value = []

        result = format_temporal_context("test@test.com")

        assert result is not None
        fact_lines = [line for line in result.split('\n') if line.startswith('- [')]
        assert len(fact_lines) <= 10, \
            f"Facts should be capped at 10 after dedup, got {len(fact_lines)}"

    @patch('src.memory.temporal_context.get_recent_significant_events')
    @patch('src.memory.temporal_context.get_recent_notable_messages')
    def test_dedup_reduces_redundant_facts(self, mock_messages, mock_events):
        """Redundant facts about the same topic should be deduplicated."""
        from src.memory.temporal_context import format_temporal_context
        from datetime import datetime
        from zoneinfo import ZoneInfo

        PST = ZoneInfo('America/Los_Angeles')
        now = datetime.now(PST)

        mock_events.return_value = [
            {'object': 'James deflects emotional topics', 'importance': 8, 'created_at': now},
            {'object': 'James deflects emotional topics when asked', 'importance': 7, 'created_at': now},
            {'object': 'James got promoted at work', 'importance': 9, 'created_at': now},
            {'object': 'Jesse has a basketball game Friday', 'importance': 6, 'created_at': now},
        ]
        mock_messages.return_value = []

        result = format_temporal_context("test@test.com")

        assert result is not None
        fact_lines = [line for line in result.split('\n') if line.startswith('- [')]
        # 2 near-duplicates should collapse to 1, plus 2 unique = 3 total
        assert len(fact_lines) == 3, \
            f"Expected 3 facts after dedup (1 deflects + 1 promoted + 1 basketball), got {len(fact_lines)}: {fact_lines}"
