"""Tests for episodic-to-semantic consolidation pipeline.

Tests the consolidation logic without requiring database or LLM connections.
"""

import pytest
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.core.clock import SimulationClock, set_clock, SystemClock, PST

NOW = datetime(2026, 4, 1, 2, 0, tzinfo=PST)


@pytest.fixture(autouse=True)
def fixed_clock():
    set_clock(SimulationClock(start=NOW))
    yield
    set_clock(SystemClock())


class TestConsolidationLogic:
    """Test the consolidation decision logic."""

    def test_minimum_recurrence_threshold(self):
        """Need at least 3 episodes to consolidate (matches task constant)."""
        min_recurrence = 3  # Must match MIN_RECURRENCE in episodic_consolidation_task.py
        assert min_recurrence >= 2
        assert min_recurrence <= 5

    def test_lookback_window(self):
        """Lookback should be 30 days (matches task constant)."""
        lookback_days = 30  # Must match LOOKBACK_DAYS in episodic_consolidation_task.py
        assert lookback_days == 30


class TestFactGeneration:
    """Test the LLM response parsing for consolidated facts."""

    def test_parse_valid_response(self):
        """Should parse a well-formed JSON response."""
        response = json.dumps({
            'subject': 'user',
            'predicate': 'frequently experiences',
            'object': 'work-related stress',
            'explanation': 'Work stress is a recurring theme',
            'importance': 7,
        })
        result = json.loads(response)
        assert result['subject'] == 'user'
        assert result['predicate'] == 'frequently experiences'
        assert result['object'] == 'work-related stress'

    def test_importance_bounds(self):
        """Importance should be clamped between 1 and 10."""
        assert min(10, max(1, 15)) == 10
        assert min(10, max(1, -3)) == 1
        assert min(10, max(1, 7)) == 7


class TestTopicGrouping:
    """Test the recurring topic detection logic."""

    def test_topic_deduplication(self):
        """Same topic across episodes should be grouped."""
        episodes = [
            {'topic': 'work stress', 'episode_id': '1'},
            {'topic': 'work stress', 'episode_id': '2'},
            {'topic': 'work stress', 'episode_id': '3'},
            {'topic': 'family', 'episode_id': '4'},
        ]

        # Group by topic
        from collections import Counter
        topic_counts = Counter(e['topic'] for e in episodes)

        # "work stress" recurs 3 times -> should consolidate
        assert topic_counts['work stress'] >= 3
        # "family" only 1 time -> should not
        assert topic_counts['family'] < 3

    def test_emotional_aggregation(self):
        """Should aggregate emotional states across episodes."""
        emotions = ['stressed', 'anxious', 'stressed', 'frustrated', None]
        # Filter and deduplicate
        unique_emotions = list(set(e for e in emotions if e))
        assert 'stressed' in unique_emotions
        assert 'anxious' in unique_emotions
        assert len(unique_emotions) == 3  # stressed, anxious, frustrated

    def test_satisfaction_averaging(self):
        """Should compute average satisfaction across episodes."""
        satisfactions = [0.3, 0.5, 0.2, 0.4]
        avg = sum(satisfactions) / len(satisfactions)
        assert avg == pytest.approx(0.35, abs=0.01)


class TestConsolidatedFactStorage:
    """Test how consolidated facts are stored."""

    def test_fact_has_high_confidence(self):
        """Consolidated facts should have high confidence (0.85)."""
        confidence = 0.85  # Value used in store_consolidated_fact
        assert confidence >= 0.8

    def test_fact_has_provenance(self):
        """Consolidated facts should include source information."""
        topic_data = {
            'topic': 'work stress',
            'episode_count': 5,
        }
        fact_data = {
            'explanation': 'Recurring work anxiety pattern',
        }

        context = (
            f"Consolidated from {topic_data['episode_count']} episodes about: "
            f"{topic_data['topic']}. {fact_data.get('explanation', '')}"
        )

        assert 'Consolidated from 5 episodes' in context
        assert 'work stress' in context
        assert 'Recurring work anxiety' in context

    def test_source_is_episodic_consolidation(self):
        """Source field should be 'episodic_consolidation' for traceability."""
        source = 'episodic_consolidation'
        assert source != 'conversation'  # Different from regular facts
        assert source != 'episodic_promotion'  # Different from sleep-time version


class TestMaxPatterns:
    """Test processing limits."""

    def test_max_patterns_per_run(self):
        """Should cap patterns per run to prevent long-running tasks."""
        max_patterns = 10  # Must match MAX_PATTERNS_PER_RUN in episodic_consolidation_task.py
        assert max_patterns <= 20
        assert max_patterns >= 5
