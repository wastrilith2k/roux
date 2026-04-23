"""
Retrieval Feedback Tests (RF01-RF04)

Tests retrieval value tracking storage and aggregation.
Unit tests — no live DB required.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestRetrievalFeedbackModel:
    """Test RetrievalFeedback dataclass."""

    def test_rf01_create_feedback_record(self):
        """RF01: RetrievalFeedback can be created with required fields."""
        from src.memory.retrieval_feedback import RetrievalFeedback

        fb = RetrievalFeedback(
            message_id=42,
            retrieval_plan={'entities': ['jesse'], 'sources': ['events']},
            sources_used=['events', 'graphiti'],
            queries_run=['Jesse health', 'Jesse current status'],
            memories_retrieved=[{'source': 'events', 'snippet': 'Jesse crisis...', 'score': 0.8}],
            active_tip_ids=['tip-001'],
        )

        assert fb.message_id == 42
        assert fb.outcome_engagement is None
        assert fb.outcome_resonance is None
        assert len(fb.active_tip_ids) == 1

    def test_rf02_format_source_performance(self):
        """RF02: Source performance formats correctly for retrieval agent."""
        from src.memory.retrieval_feedback import format_source_performance

        stats = {
            'events': {'uses': 23, 'engagement_rate': 0.78, 'avg_resonance': 0.45},
            'graphiti': {'uses': 41, 'engagement_rate': 0.65, 'avg_resonance': 0.32},
        }

        formatted = format_source_performance(stats)

        assert 'events' in formatted
        assert '78%' in formatted or '78.0%' in formatted
        assert '23' in formatted

    def test_rf03_format_empty_stats(self):
        """RF03: Empty stats returns empty string."""
        from src.memory.retrieval_feedback import format_source_performance

        assert format_source_performance({}) == ""

    def test_rf04_format_query_patterns(self):
        """RF04: Query pattern insights format correctly."""
        from src.memory.retrieval_feedback import format_retrieval_insights

        patterns = [
            {'pattern': 'emotional state queries', 'engagement_rate': 0.82, 'count': 15},
            {'pattern': 'broad recall queries', 'engagement_rate': 0.45, 'count': 12},
        ]

        formatted = format_retrieval_insights(patterns)

        assert 'emotional state' in formatted
        assert '82%' in formatted or '82.0%' in formatted
