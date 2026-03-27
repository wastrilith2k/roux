"""
Tests for schedule context: learned user routine from conversations (issue #28).

Covers:
- ScheduleContextProvider retrieves schedule-related facts
- Fact classification: schedule predicates and keyword detection
- Time relevance scoring: facts are ranked by current time of day and day type
- Prompt formatting: learned routine is formatted for LLM injection
- TimeAwareness integration: learned schedule section is included in time context
- Graceful degradation: empty/error cases return empty strings
"""

import os
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

from src.core.schedule_context import (
    ScheduleContextProvider,
    get_schedule_context_provider,
    SCHEDULE_PREDICATES,
    SCHEDULE_KEYWORDS,
)


# =========================================================================
# Helper: create fake facts
# =========================================================================

def _make_fact(subject, predicate, obj, importance=5, confidence=0.8):
    """Create a fake fact dict matching the fact store format."""
    return {
        'id': hash(f"{subject}{predicate}{obj}") % 10000,
        'subject': subject,
        'predicate': predicate,
        'object': obj,
        'importance': importance,
        'confidence': confidence,
        'effective_confidence': confidence,
        'temporal': 'current',
        'archived_at': None,
    }


# =========================================================================
# Fact Classification Tests
# =========================================================================

class TestFactClassification:
    """Test _is_schedule_fact correctly identifies schedule-related facts."""

    def setup_method(self):
        self.provider = ScheduleContextProvider()

    def test_schedule_predicate_detected(self):
        """Facts with schedule predicates are identified."""
        fact = _make_fact('James', 'has_routine', 'works 9-5 on weekdays')
        assert self.provider._is_schedule_fact(fact) is True

    def test_works_at_predicate_detected(self):
        fact = _make_fact('James', 'works_at', 'a tech company from 9am to 5pm')
        assert self.provider._is_schedule_fact(fact) is True

    def test_picks_up_predicate_detected(self):
        fact = _make_fact('James', 'picks_up', 'the kids at 3:30 PM on weekdays')
        assert self.provider._is_schedule_fact(fact) is True

    def test_schedule_keyword_in_object(self):
        """Facts with schedule keywords in the object are identified."""
        fact = _make_fact('James', 'mentioned', 'usually free on weekends')
        assert self.provider._is_schedule_fact(fact) is True

    def test_time_keyword_in_object(self):
        fact = _make_fact('James', 'said', 'goes to the gym every morning')
        assert self.provider._is_schedule_fact(fact) is True

    def test_non_schedule_fact_rejected(self):
        """Facts without schedule indicators are not classified as schedule facts."""
        fact = _make_fact('James', 'likes', 'peppermint tea')
        assert self.provider._is_schedule_fact(fact) is False

    def test_non_schedule_fact_about_preferences(self):
        fact = _make_fact('James', 'favorite_color', 'blue')
        assert self.provider._is_schedule_fact(fact) is False

    def test_predicate_matching_is_case_insensitive(self):
        fact = _make_fact('James', 'HAS_ROUTINE', 'morning jog at 6am')
        assert self.provider._is_schedule_fact(fact) is True

    def test_keyword_matching_is_case_insensitive(self):
        fact = _make_fact('James', 'does', 'Goes to the GYM every Morning')
        assert self.provider._is_schedule_fact(fact) is True

    def test_empty_fact_fields_handled(self):
        """Facts with None/empty fields don't crash."""
        fact = _make_fact('James', None, None)
        fact['predicate'] = None
        fact['object'] = None
        assert self.provider._is_schedule_fact(fact) is False


# =========================================================================
# Time Relevance Scoring Tests
# =========================================================================

class TestTimeRelevanceScoring:
    """Test compute_time_relevance scores facts by current time context."""

    def setup_method(self):
        self.provider = ScheduleContextProvider()

    def test_morning_fact_scores_higher_in_morning(self):
        """A fact about morning routine should score higher when it's morning."""
        fact = _make_fact('James', 'has_routine', 'goes for a jog every morning')
        morning_score = self.provider.compute_time_relevance(fact, 'morning', 'weekday')
        evening_score = self.provider.compute_time_relevance(fact, 'evening', 'weekday')
        assert morning_score > evening_score

    def test_weekend_fact_scores_higher_on_weekend(self):
        """A fact about weekends should score higher on weekends."""
        fact = _make_fact('James', 'free_on', 'usually free on weekends')
        weekend_score = self.provider.compute_time_relevance(fact, 'morning', 'weekend')
        weekday_score = self.provider.compute_time_relevance(fact, 'morning', 'weekday')
        assert weekend_score > weekday_score

    def test_work_fact_scores_higher_on_weekday(self):
        """A fact about work should score higher on weekdays."""
        fact = _make_fact('James', 'works', 'at the office 9-5')
        weekday_score = self.provider.compute_time_relevance(fact, 'afternoon', 'weekday')
        weekend_score = self.provider.compute_time_relevance(fact, 'afternoon', 'weekend')
        assert weekday_score > weekend_score

    def test_high_importance_boosts_score(self):
        """Higher importance facts should score higher."""
        high = _make_fact('James', 'has_routine', 'picks up kids at 3:30', importance=9)
        low = _make_fact('James', 'has_routine', 'picks up kids at 3:30', importance=2)
        # Same time context, different importance
        high_score = self.provider.compute_time_relevance(high, 'afternoon', 'weekday')
        low_score = self.provider.compute_time_relevance(low, 'afternoon', 'weekday')
        assert high_score > low_score

    def test_low_confidence_reduces_score(self):
        """Facts with lower confidence should score lower."""
        high_conf = _make_fact('James', 'has_routine', 'gym at 6am', confidence=0.9)
        low_conf = _make_fact('James', 'has_routine', 'gym at 6am', confidence=0.3)
        high_score = self.provider.compute_time_relevance(high_conf, 'morning', 'weekday')
        low_score = self.provider.compute_time_relevance(low_conf, 'morning', 'weekday')
        assert high_score > low_score

    def test_score_never_exceeds_one(self):
        """Relevance score should be capped at 1.0."""
        fact = _make_fact('James', 'has_routine', 'morning work commute weekday office', importance=10, confidence=1.0)
        score = self.provider.compute_time_relevance(fact, 'morning', 'weekday')
        assert score <= 1.0

    def test_score_is_non_negative(self):
        """Relevance score should never be negative."""
        fact = _make_fact('James', 'has_routine', 'something', importance=0, confidence=0.1)
        score = self.provider.compute_time_relevance(fact, 'night', 'weekend')
        assert score >= 0.0


# =========================================================================
# Schedule Retrieval Tests (with mocked fact store)
# =========================================================================

class TestScheduleRetrieval:
    """Test get_learned_schedule_facts and get_relevant_schedule_context."""

    def _make_provider_with_facts(self, facts):
        """Create a provider with a mocked fact store returning given facts."""
        provider = ScheduleContextProvider()
        mock_store = MagicMock()
        mock_store.get_facts_for_subject.return_value = facts
        provider._fact_store = mock_store
        return provider

    def test_filters_schedule_facts_only(self):
        """Only schedule-related facts are returned."""
        facts = [
            _make_fact('James', 'has_routine', 'works 9-5'),
            _make_fact('James', 'likes', 'peppermint tea'),
            _make_fact('James', 'picks_up', 'kids at 3:30'),
            _make_fact('James', 'favorite_food', 'pizza'),
        ]
        provider = self._make_provider_with_facts(facts)
        result = provider.get_learned_schedule_facts('James')
        assert len(result) == 2
        predicates = [f['predicate'] for f in result]
        assert 'has_routine' in predicates
        assert 'picks_up' in predicates

    def test_returns_empty_when_no_facts(self):
        provider = self._make_provider_with_facts([])
        result = provider.get_learned_schedule_facts('James')
        assert result == []

    def test_ranked_by_relevance(self):
        """Facts are ranked by time relevance."""
        facts = [
            _make_fact('James', 'has_routine', 'gym every morning', importance=7),
            _make_fact('James', 'has_routine', 'works at office weekday afternoon', importance=7),
        ]
        provider = self._make_provider_with_facts(facts)

        # Wednesday afternoon at 2pm
        afternoon = datetime(2026, 3, 25, 14, 0, tzinfo=ZoneInfo('America/Los_Angeles'))
        ranked = provider.get_relevant_schedule_context('James', current_time=afternoon)

        # The work/office/afternoon fact should rank higher than morning gym
        assert len(ranked) == 2
        top_fact = ranked[0][0]
        assert 'office' in top_fact['object'] or 'afternoon' in top_fact['object']

    def test_max_facts_limit(self):
        """Respects the max_facts parameter."""
        facts = [_make_fact('James', 'has_routine', f'activity {i}') for i in range(20)]
        provider = self._make_provider_with_facts(facts)
        ranked = provider.get_relevant_schedule_context('James', max_facts=3)
        assert len(ranked) <= 3

    def test_graceful_on_fact_store_error(self):
        """Returns empty list when fact store raises an exception."""
        provider = ScheduleContextProvider()
        mock_store = MagicMock()
        mock_store.get_facts_for_subject.side_effect = Exception("DB connection failed")
        provider._fact_store = mock_store
        result = provider.get_learned_schedule_facts('James')
        assert result == []


# =========================================================================
# Prompt Formatting Tests
# =========================================================================

class TestPromptFormatting:
    """Test format_for_prompt produces correct output."""

    def _make_provider_with_facts(self, facts):
        provider = ScheduleContextProvider()
        mock_store = MagicMock()
        mock_store.get_facts_for_subject.return_value = facts
        provider._fact_store = mock_store
        return provider

    def test_format_includes_header(self):
        facts = [_make_fact('James', 'has_routine', 'works 9-5')]
        provider = self._make_provider_with_facts(facts)
        now = datetime(2026, 3, 25, 10, 0, tzinfo=ZoneInfo('America/Los_Angeles'))
        result = provider.format_for_prompt('James', current_time=now)
        assert "James's LEARNED ROUTINE" in result
        assert "past conversations" in result

    def test_format_includes_facts(self):
        facts = [
            _make_fact('James', 'has_routine', 'works 9-5'),
            _make_fact('James', 'picks_up', 'kids at 3:30 PM'),
        ]
        provider = self._make_provider_with_facts(facts)
        now = datetime(2026, 3, 25, 10, 0, tzinfo=ZoneInfo('America/Los_Angeles'))
        result = provider.format_for_prompt('James', current_time=now)
        assert 'works 9-5' in result
        assert 'kids at 3:30 PM' in result

    def test_format_returns_empty_when_no_facts(self):
        provider = self._make_provider_with_facts([])
        now = datetime(2026, 3, 25, 10, 0, tzinfo=ZoneInfo('America/Los_Angeles'))
        result = provider.format_for_prompt('James', current_time=now)
        assert result == ""

    def test_format_includes_usage_guidance(self):
        """Output should tell the LLM how to use the information."""
        facts = [_make_fact('James', 'has_routine', 'works 9-5')]
        provider = self._make_provider_with_facts(facts)
        now = datetime(2026, 3, 25, 10, 0, tzinfo=ZoneInfo('America/Los_Angeles'))
        result = provider.format_for_prompt('James', current_time=now)
        assert 'naturally' in result


# =========================================================================
# TimeAwareness Integration Tests
# =========================================================================

class TestTimeAwarenessIntegration:
    """Test that learned schedule is integrated into the time awareness system."""

    @patch('src.core.time_awareness.TimeAwareness._get_calendar_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_routine_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_companion_schedule_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_work_projects_section', return_value='')
    def test_learned_schedule_included_in_time_context(self, _work, _comp, _routine, _cal):
        """When learned schedule facts exist, they appear in get_time_context()."""
        from src.core.time_awareness import TimeAwareness

        mock_provider = MagicMock()
        mock_provider.format_for_prompt.return_value = (
            "[James's LEARNED ROUTINE — from past conversations]\n"
            "- has_routine: works 9-5 on weekdays"
        )

        with patch('src.core.schedule_context.get_schedule_context_provider', return_value=mock_provider):
            ta = TimeAwareness()
            context = ta.get_time_context()

        assert "LEARNED ROUTINE" in context
        assert "works 9-5" in context

    @patch('src.core.time_awareness.TimeAwareness._get_calendar_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_routine_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_companion_schedule_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_work_projects_section', return_value='')
    def test_no_learned_schedule_when_empty(self, _work, _comp, _routine, _cal):
        """When no schedule facts exist, no learned section appears."""
        from src.core.time_awareness import TimeAwareness

        mock_provider = MagicMock()
        mock_provider.format_for_prompt.return_value = ""

        with patch('src.core.schedule_context.get_schedule_context_provider', return_value=mock_provider):
            ta = TimeAwareness()
            context = ta.get_time_context()

        assert "LEARNED ROUTINE" not in context

    @patch('src.core.time_awareness.TimeAwareness._get_calendar_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_routine_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_companion_schedule_section', return_value='')
    @patch('src.core.time_awareness.TimeAwareness._get_work_projects_section', return_value='')
    def test_graceful_degradation_on_error(self, _work, _comp, _routine, _cal):
        """If schedule context provider throws, time context still works."""
        from src.core.time_awareness import TimeAwareness

        with patch('src.core.time_awareness.TimeAwareness._get_learned_schedule_section', side_effect=Exception("boom")):
            ta = TimeAwareness()
            # Should not raise — the method handles errors internally
            # But since we patched the method itself to throw, let's test
            # the internal error handling path instead
            pass

        # Test the internal error handling of _get_learned_schedule_section
        ta = TimeAwareness()
        with patch('src.core.schedule_context.get_schedule_context_provider', side_effect=Exception("import fail")):
            result = ta._get_learned_schedule_section(datetime.now())
        assert result == ""


# =========================================================================
# Singleton Tests
# =========================================================================

class TestSingleton:
    """Test get_schedule_context_provider singleton behavior."""

    def test_returns_same_instance(self):
        """Singleton should return the same instance."""
        import src.core.schedule_context as mod
        old = mod._provider
        mod._provider = None  # Reset
        try:
            a = get_schedule_context_provider()
            b = get_schedule_context_provider()
            assert a is b
        finally:
            mod._provider = old

    def test_returns_schedule_context_provider_instance(self):
        import src.core.schedule_context as mod
        old = mod._provider
        mod._provider = None
        try:
            p = get_schedule_context_provider()
            assert isinstance(p, ScheduleContextProvider)
        finally:
            mod._provider = old
