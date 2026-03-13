"""Tests for memory confidence decay calculations."""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.core.clock import SimulationClock, set_clock, SystemClock, PST
from src.memory.confidence_decay import (
    calculate_effective_confidence,
    get_confidence_qualifier,
    format_facts_with_confidence,
    enrich_facts_with_effective_confidence,
    FULL_DECAY_DAYS,
    RECENCY_FLOOR,
    MENTION_BOOST_CAP,
    CONFIDENCE_FLOOR,
    CONFIDENCE_CEILING,
)


NOW = datetime(2026, 3, 1, 12, 0, tzinfo=PST)


@pytest.fixture(autouse=True)
def fixed_clock():
    """Pin clock for all decay tests."""
    set_clock(SimulationClock(start=NOW))
    yield
    set_clock(SystemClock())


class TestCalculateEffectiveConfidence:
    """Test the core decay formula."""

    def test_fresh_fact_no_decay(self):
        """A fact mentioned just now should have minimal decay."""
        result = calculate_effective_confidence(
            base_confidence=0.8,
            mention_count=1,
            last_mentioned=NOW,
            created_at=NOW,
        )
        assert result == pytest.approx(0.8, abs=0.01)

    def test_fully_decayed_fact(self):
        """A fact not mentioned for FULL_DECAY_DAYS should hit recency floor."""
        old = NOW - timedelta(days=FULL_DECAY_DAYS)
        result = calculate_effective_confidence(
            base_confidence=0.8,
            mention_count=1,
            last_mentioned=old,
            created_at=old,
        )
        expected = 0.8 * RECENCY_FLOOR  # 0.8 * 0.3 = 0.24
        assert result == pytest.approx(expected, abs=0.01)

    def test_half_decayed(self):
        """A fact at half the decay period."""
        half = NOW - timedelta(days=FULL_DECAY_DAYS / 2)
        result = calculate_effective_confidence(
            base_confidence=0.8,
            mention_count=1,
            last_mentioned=half,
            created_at=half,
        )
        # recency_factor = 1.0 - 45/90 = 0.5
        expected = 0.8 * 0.5
        assert result == pytest.approx(expected, abs=0.05)

    def test_mention_boost(self):
        """Multiple mentions should boost confidence."""
        result = calculate_effective_confidence(
            base_confidence=0.7,
            mention_count=5,
            last_mentioned=NOW,
            created_at=NOW,
        )
        # mention_boost = 1.0 + (5-1)*0.1 = 1.4
        expected = 0.7 * 1.0 * 1.4  # 0.98
        assert result == pytest.approx(expected, abs=0.01)

    def test_mention_boost_capped(self):
        """Mention boost should not exceed MENTION_BOOST_CAP."""
        result = calculate_effective_confidence(
            base_confidence=0.7,
            mention_count=100,
            last_mentioned=NOW,
            created_at=NOW,
        )
        # boost capped at 1.5, so 0.7 * 1.0 * 1.5 = 1.05 -> clamped to 0.99
        assert result <= CONFIDENCE_CEILING

    def test_floor_clamp(self):
        """Effective confidence should never go below CONFIDENCE_FLOOR."""
        very_old = NOW - timedelta(days=FULL_DECAY_DAYS * 3)
        result = calculate_effective_confidence(
            base_confidence=0.1,
            mention_count=1,
            last_mentioned=very_old,
            created_at=very_old,
        )
        assert result >= CONFIDENCE_FLOOR

    def test_ceiling_clamp(self):
        """Effective confidence should never exceed CONFIDENCE_CEILING."""
        result = calculate_effective_confidence(
            base_confidence=1.0,
            mention_count=20,
            last_mentioned=NOW,
            created_at=NOW,
        )
        assert result <= CONFIDENCE_CEILING

    def test_none_last_mentioned_uses_created_at(self):
        """If last_mentioned is None, falls back to created_at."""
        week_ago = NOW - timedelta(days=7)
        result = calculate_effective_confidence(
            base_confidence=0.8,
            mention_count=1,
            last_mentioned=None,
            created_at=week_ago,
        )
        # recency_factor = 1.0 - 7/90 ≈ 0.922
        expected = 0.8 * (1.0 - 7 / FULL_DECAY_DAYS)
        assert result == pytest.approx(expected, abs=0.02)

    def test_none_both_dates_worst_case(self):
        """If both dates are None, assume maximally decayed."""
        result = calculate_effective_confidence(
            base_confidence=0.8,
            mention_count=1,
            last_mentioned=None,
            created_at=None,
        )
        expected = 0.8 * RECENCY_FLOOR
        assert result == pytest.approx(expected, abs=0.01)

    def test_none_base_confidence_defaults(self):
        """If base_confidence is None/falsy, defaults to 0.7."""
        result = calculate_effective_confidence(
            base_confidence=None,
            mention_count=1,
            last_mentioned=NOW,
            created_at=NOW,
        )
        assert result == pytest.approx(0.7, abs=0.01)

    def test_zero_mention_count_handled(self):
        """mention_count=0 should be treated as 1 (no crash, no zero-multiply)."""
        result = calculate_effective_confidence(
            base_confidence=0.8,
            mention_count=0,
            last_mentioned=NOW,
            created_at=NOW,
        )
        assert result == pytest.approx(0.8, abs=0.01)

    def test_negative_days_since(self):
        """Future last_mentioned should give recency_factor=1.0 (no decay)."""
        future = NOW + timedelta(days=10)
        result = calculate_effective_confidence(
            base_confidence=0.8,
            mention_count=1,
            last_mentioned=future,
            created_at=NOW,
        )
        # recency_factor = max(0.3, 1.0 - (-10)/90) = max(0.3, 1.11) = 1.11
        # But that's > 1 which means boost. Result clamped to ceiling.
        assert result <= CONFIDENCE_CEILING
        assert result >= 0.8  # Should be at least base


class TestGetConfidenceQualifier:
    """Test hedging qualifiers."""

    def test_high_confidence_no_qualifier(self):
        assert get_confidence_qualifier(0.9) == ""
        assert get_confidence_qualifier(0.8) == ""

    def test_medium_confidence_hedging(self):
        q = get_confidence_qualifier(0.65)
        assert q in ("I think", "if I remember right")

    def test_low_confidence_hedging(self):
        q = get_confidence_qualifier(0.45)
        assert q in ("I'm not sure but", "I vaguely remember")

    def test_very_low_confidence_hedging(self):
        q = get_confidence_qualifier(0.2)
        assert q in ("I might be wrong but", "I'm fuzzy on this but")


class TestFormatFactsWithConfidence:
    """Test prompt formatting."""

    def test_empty_list(self):
        assert format_facts_with_confidence([]) == ""

    def test_high_confidence_no_prefix(self):
        facts = [{'subject': 'James', 'predicate': 'likes', 'object': 'tea', 'effective_confidence': 0.9}]
        result = format_facts_with_confidence(facts)
        assert result == "James likes tea"
        assert "[UNCERTAIN]" not in result

    def test_low_confidence_has_prefix(self):
        facts = [{'subject': 'James', 'predicate': 'likes', 'object': 'hiking', 'effective_confidence': 0.3}]
        result = format_facts_with_confidence(facts)
        assert "[UNCERTAIN]" in result
        assert "James" in result
        assert "hiking" in result


class TestEnrichFacts:
    """Test batch enrichment."""

    def test_adds_effective_confidence_field(self):
        facts = [
            {'confidence': 0.8, 'mention_count': 1, 'last_mentioned': NOW, 'created_at': NOW},
            {'confidence': 0.5, 'mention_count': 3, 'last_mentioned': NOW - timedelta(days=30), 'created_at': NOW - timedelta(days=60)},
        ]
        enriched = enrich_facts_with_effective_confidence(facts)
        assert len(enriched) == 2
        for fact in enriched:
            assert 'effective_confidence' in fact
            assert CONFIDENCE_FLOOR <= fact['effective_confidence'] <= CONFIDENCE_CEILING

    def test_empty_list(self):
        assert enrich_facts_with_effective_confidence([]) == []
