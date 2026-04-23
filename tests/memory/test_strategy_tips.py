"""
Strategy Tips Tests (ST01-ST06)

Tests tip storage, retrieval, and scoring.
Unit tests use mocked database — no live DB required.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


class TestStrategyTipModel:
    """Test StrategyTip dataclass and scoring."""

    def test_st01_create_strategy_tip(self):
        """ST01: StrategyTip can be created with required fields."""
        from src.memory.strategy_tips import StrategyTip

        tip = StrategyTip(
            tip_type='strategy',
            content='When user mentions stress about parenting, lead with empathy.',
            trigger_condition='emotional_support + family topic',
            context_category='family_children',
            emotional_context='stressed',
            source_type='episode',
            source_ids=['ep-001', 'ep-002'],
        )

        assert tip.tip_type == 'strategy'
        assert tip.times_used == 0
        assert tip.is_principled is False

    def test_st02_value_score_no_usage(self):
        """ST02: Value score is 0 when tip has never been used."""
        from src.memory.strategy_tips import StrategyTip

        tip = StrategyTip(
            tip_type='insight',
            content='test',
            trigger_condition='test',
            source_type='episode',
        )

        assert tip.value_score == 0.0

    def test_st03_value_score_with_usage(self):
        """ST03: Value score reflects helpful/total ratio."""
        from src.memory.strategy_tips import StrategyTip

        tip = StrategyTip(
            tip_type='strategy',
            content='test',
            trigger_condition='test',
            source_type='episode',
            times_used=10,
            times_helpful=8,
            times_unhelpful=1,
        )

        # value_score = times_helpful / (times_used + 1) = 8/11 ≈ 0.727
        assert 0.7 < tip.value_score < 0.8

    def test_st04_principled_tip(self):
        """ST04: Principled tips have low engagement but high authenticity."""
        from src.memory.strategy_tips import StrategyTip

        tip = StrategyTip(
            tip_type='insight',
            content='user deflects when asked about work directly.',
            trigger_condition='work topic + direct question',
            source_type='interaction_outcome',
            is_principled=True,
            authenticity_avg=0.9,
            engagement_avg=0.3,
        )

        assert tip.is_principled is True
        assert tip.authenticity_avg > tip.engagement_avg

    def test_st05_format_for_context(self):
        """ST05: Tips format correctly for context injection."""
        from src.memory.strategy_tips import StrategyTip

        tip = StrategyTip(
            tip_type='recovery',
            content='When a tease lands wrong, acknowledge it directly.',
            trigger_condition='playful misfire',
            source_type='episode',
            source_ids=['ep-003'],
        )

        formatted = tip.format_for_context()

        assert 'Recovery note' in formatted
        assert 'acknowledge it directly' in formatted
        assert '1 similar situation' in formatted

    def test_st06_format_strategy_and_insight(self):
        """ST06: Different tip types have appropriate labels."""
        from src.memory.strategy_tips import StrategyTip

        strategy = StrategyTip(
            tip_type='strategy',
            content='Lead with empathy.',
            trigger_condition='stress',
            source_type='episode',
            source_ids=['a', 'b', 'c'],
        )
        insight = StrategyTip(
            tip_type='insight',
            content='user opens up after companion shares first.',
            trigger_condition='feelings topic',
            source_type='interaction_outcome',
            source_ids=['x'],
        )

        assert 'Observation' in strategy.format_for_context()
        assert '3 similar situations' in strategy.format_for_context()
        assert 'Pattern' in insight.format_for_context()
