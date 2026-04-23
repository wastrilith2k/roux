"""Tests for 8-heuristic proactive scoring.

Tests the individual heuristic functions and composite scoring logic.
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.core.clock import SimulationClock, set_clock, SystemClock, PST
from src.autonomy.proactive_scoring import (
    ProactiveScore,
    score_relevance,
    score_information_gap,
    score_expected_impact,
    score_urgency,
    score_coherence,
    score_originality,
    score_balance,
    score_timing,
    evaluate_proactive_scores,
    format_scores_for_llm,
    MIN_COMPOSITE_SCORE,
    HEURISTIC_WEIGHTS,
)


NOW = datetime(2026, 4, 1, 10, 0, tzinfo=PST)  # 10 AM on a weekday


@pytest.fixture(autouse=True)
def fixed_clock():
    set_clock(SimulationClock(start=NOW))
    yield
    set_clock(SystemClock())


class TestProactiveScore:
    """Test the ProactiveScore dataclass."""

    def test_composite_calculation(self):
        score = ProactiveScore(
            relevance=1.0, information_gap=1.0, expected_impact=1.0,
            urgency=1.0, coherence=1.0, originality=1.0,
            balance=1.0, timing=1.0,
        )
        assert score.composite == pytest.approx(1.0, abs=0.01)

    def test_zero_composite(self):
        score = ProactiveScore()
        assert score.composite == 0.0

    def test_should_proceed_above_threshold(self):
        score = ProactiveScore(
            relevance=0.5, information_gap=0.5, expected_impact=0.5,
            urgency=0.5, coherence=0.5, originality=0.5,
            balance=0.5, timing=0.5,
        )
        assert score.should_proceed is True

    def test_should_not_proceed_below_threshold(self):
        score = ProactiveScore(
            relevance=0.1, information_gap=0.1, expected_impact=0.1,
            urgency=0.1, coherence=0.1, originality=0.1,
            balance=0.1, timing=0.1,
        )
        assert score.should_proceed is False

    def test_weights_sum_to_one(self):
        total = sum(HEURISTIC_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_to_dict(self):
        score = ProactiveScore(relevance=0.7, timing=0.3)
        d = score.to_dict()
        assert 'relevance' in d
        assert 'composite' in d
        assert d['relevance'] == 0.7

    def test_summary_format(self):
        score = ProactiveScore(relevance=0.9, timing=0.1)
        summary = score.summary()
        assert 'Composite' in summary
        assert 'Strongest' in summary


class TestRelevanceScoring:
    """Test relevance heuristic."""

    def test_high_relevance_with_content(self):
        context = {
            'queued_thoughts': ['something to say'],
            'curiosity_threads': 'topic1, topic2',
            'active_trigger': 'morning_greeting',
        }
        assert score_relevance(context) >= 0.7

    def test_low_relevance_empty_context(self):
        context = {}
        assert score_relevance(context) == 0.0

    def test_capped_at_one(self):
        context = {
            'queued_thoughts': ['thought'],
            'curiosity_threads': 'topic',
            'goal_relate_hints': ['hint'],
            'active_trigger': 'morning',
            'has_research_to_share': True,
            'private_thoughts': 'something',
        }
        assert score_relevance(context) <= 1.0


class TestExpectedImpact:
    """Test expected impact heuristic."""

    def test_penalty_for_double_text(self):
        context = {
            'waiting_for_response': True,
            'hours_since_last_message': 0.5,
        }
        assert score_expected_impact(context) < 0.3

    def test_bonus_for_content(self):
        context = {
            'waiting_for_response': False,
            'queued_thoughts': ['interesting thought'],
        }
        assert score_expected_impact(context) >= 0.5

    def test_long_wait_reduces_penalty(self):
        recent = score_expected_impact({
            'waiting_for_response': True,
            'hours_since_last_message': 1.0,
        })
        longer = score_expected_impact({
            'waiting_for_response': True,
            'hours_since_last_message': 8.0,
        })
        assert longer > recent


class TestUrgencyScoring:
    """Test urgency heuristic."""

    def test_high_urgency_curiosity(self):
        context = {'high_urgency_topic': 'important question'}
        assert score_urgency(context) >= 0.5

    def test_long_silence_creates_urgency(self):
        context = {'hours_since_last_message': 30}
        assert score_urgency(context) >= 0.4

    def test_no_urgency_default(self):
        context = {'hours_since_last_message': 1}
        assert score_urgency(context) < 0.3


class TestOriginalityScoring:
    """Test originality heuristic."""

    def test_no_history_is_original(self):
        assert score_originality({}, None) == 0.7

    def test_similar_message_reduces_score(self):
        context = {'opener_hint': 'good morning how did you sleep'}
        recent = ['good morning! how did you sleep?']
        score = score_originality(context, recent)
        assert score < 0.6

    def test_different_message_keeps_score(self):
        context = {'opener_hint': 'I found something interesting about cooking'}
        recent = ['good morning! how are you?']
        score = score_originality(context, recent)
        assert score >= 0.6


class TestBalanceScoring:
    """Test balance heuristic."""

    def test_waiting_for_response_lowers_balance(self):
        context = {'waiting_for_response': True, 'hours_since_last_message': 1}
        assert score_balance(context) < 0.5

    def test_long_silence_raises_balance(self):
        context = {'waiting_for_response': False, 'hours_since_last_message': 30}
        assert score_balance(context) >= 0.7

    def test_user_messaged_last_raises_balance(self):
        context = {'waiting_for_response': False, 'hours_since_last_message': 2}
        assert score_balance(context) >= 0.5


class TestTimingScoring:
    """Test timing heuristic."""

    def test_morning_is_good(self):
        context = {'hour': 9, 'is_weekend': False}
        assert score_timing(context) >= 0.5

    def test_late_night_is_bad(self):
        context = {'hour': 23, 'is_weekend': False}
        assert score_timing(context) < 0.4

    def test_user_busy_lowers_timing(self):
        context = {'hour': 10, 'james_calendar_busy': True}
        busy_score = score_timing(context)
        context['james_calendar_busy'] = False
        free_score = score_timing(context)
        assert free_score > busy_score


class TestEvaluateProactiveScores:
    """Test the full evaluation pipeline."""

    def test_rich_context_scores_high(self):
        context = {
            'queued_thoughts': ['a thought'],
            'curiosity_threads': 'topics',
            'high_urgency_topic': 'important',
            'active_trigger': 'morning_greeting',
            'hours_since_last_message': 10,
            'waiting_for_response': False,
            'hour': 9,
            'is_weekend': False,
        }
        score = evaluate_proactive_scores(context)
        assert score.should_proceed is True
        assert score.composite > 0.5

    def test_empty_context_scores_low(self):
        context = {
            'hours_since_last_message': 1,
            'waiting_for_response': True,
            'hour': 23,
        }
        score = evaluate_proactive_scores(context)
        assert score.composite < 0.4


class TestFormatScoresForLLM:
    """Test LLM-friendly formatting."""

    def test_format_includes_key_info(self):
        score = ProactiveScore(
            relevance=0.8, information_gap=0.7, expected_impact=0.6,
            urgency=0.3, coherence=0.5, originality=0.9,
            balance=0.6, timing=0.5,
        )
        text = format_scores_for_llm(score)
        assert 'PROACTIVE MOTIVATION ANALYSIS' in text
        assert 'motivation' in text.lower()
