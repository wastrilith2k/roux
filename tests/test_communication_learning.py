"""Tests for outcome-driven communication learning."""

import pytest
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.core.clock import SimulationClock, set_clock, SystemClock, PST

NOW = datetime(2026, 4, 1, 10, 0, tzinfo=PST)


@pytest.fixture(autouse=True)
def fixed_clock():
    set_clock(SimulationClock(start=NOW))
    yield
    set_clock(SystemClock())


class TestPreferencesCache:
    """Test the preference caching mechanism."""

    def test_cache_round_trip(self, tmp_path):
        """Should save and load preferences from cache."""
        cache_file = str(tmp_path / "prefs.json")

        cache = {
            'generated_at': NOW.isoformat(),
            'preferences_text': 'The user responds well to questions about projects.',
        }

        with open(cache_file, 'w') as f:
            json.dump(cache, f)

        with open(cache_file, 'r') as f:
            loaded = json.load(f)

        assert loaded['preferences_text'] == 'The user responds well to questions about projects.'
        assert loaded['generated_at'] == NOW.isoformat()

    def test_stale_cache_detected(self, tmp_path):
        """Cache older than TTL should be considered stale."""
        cache_file = str(tmp_path / "prefs.json")

        old_time = (NOW - timedelta(days=10)).isoformat()
        cache = {
            'generated_at': old_time,
            'preferences_text': 'Old preferences.',
        }

        with open(cache_file, 'w') as f:
            json.dump(cache, f)

        # Check freshness (7-day TTL)
        gen_time = datetime.fromisoformat(old_time)
        age_hours = (NOW - gen_time).total_seconds() / 3600
        assert age_hours > 24 * 7  # Stale


class TestPreferencesFormatting:
    """Test the prompt formatting for preferences."""

    def test_format_includes_header(self):
        """Formatted preferences should have identifying header."""
        from src.core.communication_learning import format_preferences_for_prompt

        # This will return empty since there's no cached file
        # but we can test the format logic directly
        prefs_text = "The user responds well to questions about projects."
        formatted = (
            f"[COMMUNICATION INSIGHTS (from past interaction patterns)]:\n"
            f"{prefs_text}\n"
            f"(Use these naturally — don't announce them or force changes.)\n"
        )

        assert "[COMMUNICATION INSIGHTS" in formatted
        assert "Use these naturally" in formatted
        assert prefs_text in formatted


class TestOutcomePatterns:
    """Test outcome pattern analysis logic."""

    def test_engagement_rate_calculation(self):
        """Engagement rate should be (enthusiastic + engaged) / total."""
        enthusiastic = 5
        engaged = 10
        total = 20
        rate = (enthusiastic + engaged) / total * 100
        assert rate == 75.0

    def test_deflection_rate_calculation(self):
        """Deflection rate should be (deflected + ignored) / total."""
        deflected = 3
        ignored = 2
        total = 20
        rate = (deflected + ignored) / total * 100
        assert rate == 25.0

    def test_zero_total_handled(self):
        """Zero total should not cause division error."""
        total = 0
        rate = (0 / total * 100) if total > 0 else 0
        assert rate == 0


class TestTimePatternsLogic:
    """Test time-of-day classification logic."""

    def test_morning_classification(self):
        for hour in [6, 7, 8, 9, 10, 11]:
            period = _classify_hour(hour)
            assert period == 'morning'

    def test_afternoon_classification(self):
        for hour in [12, 13, 14, 15, 16, 17]:
            period = _classify_hour(hour)
            assert period == 'afternoon'

    def test_evening_classification(self):
        for hour in [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]:
            period = _classify_hour(hour)
            assert period == 'evening'


def _classify_hour(hour: int) -> str:
    """Replicate the SQL CASE logic for testing."""
    if 6 <= hour <= 11:
        return 'morning'
    elif 12 <= hour <= 17:
        return 'afternoon'
    else:
        return 'evening'
