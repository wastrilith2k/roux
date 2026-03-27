"""Tests for temporal awareness: device clock + timezone as companion context (issue #27)."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.time_awareness import derive_temporal_context
from src.core.entity_profile_loader import get_current_time_context


class TestDeriveTemporalContext:
    """Tests for derive_temporal_context() — converts client clock data to structured context."""

    def test_basic_derivation(self):
        """Client timestamp + timezone produces all expected fields."""
        result = derive_temporal_context(
            '2026-03-24T20:30:00-04:00',
            'America/New_York'
        )
        assert result['time_of_day'] == 'evening'
        assert result['day_of_week'] == 'Tuesday'
        assert result['is_weekend'] is False
        assert result['date_str'] == 'March 24, 2026'
        assert result['timezone'] == 'America/New_York'
        assert 'formatted' in result
        assert 'Current time:' in result['formatted']

    def test_morning(self):
        result = derive_temporal_context('2026-03-25T08:15:00-04:00', 'America/New_York')
        assert result['time_of_day'] == 'morning'

    def test_afternoon(self):
        result = derive_temporal_context('2026-03-25T14:00:00-04:00', 'America/New_York')
        assert result['time_of_day'] == 'afternoon'

    def test_evening(self):
        result = derive_temporal_context('2026-03-25T19:00:00-04:00', 'America/New_York')
        assert result['time_of_day'] == 'evening'

    def test_night(self):
        result = derive_temporal_context('2026-03-25T23:30:00-04:00', 'America/New_York')
        assert result['time_of_day'] == 'night'

    def test_early_morning_is_night(self):
        """Before 5 AM should be 'night', not 'morning'."""
        result = derive_temporal_context('2026-03-25T03:00:00-04:00', 'America/New_York')
        assert result['time_of_day'] == 'night'

    def test_weekend_saturday(self):
        # March 28, 2026 is a Saturday
        result = derive_temporal_context('2026-03-28T10:00:00-04:00', 'America/New_York')
        assert result['is_weekend'] is True
        assert result['day_of_week'] == 'Saturday'

    def test_weekend_sunday(self):
        # March 29, 2026 is a Sunday
        result = derive_temporal_context('2026-03-29T10:00:00-04:00', 'America/New_York')
        assert result['is_weekend'] is True
        assert result['day_of_week'] == 'Sunday'

    def test_weekday(self):
        # March 25, 2026 is a Wednesday
        result = derive_temporal_context('2026-03-25T10:00:00-04:00', 'America/New_York')
        assert result['is_weekend'] is False

    def test_utc_timestamp_converted_to_local(self):
        """A UTC timestamp should be converted to the user's local timezone."""
        # 2026-03-25 00:00 UTC = 2026-03-24 20:00 EDT (still Tuesday evening)
        result = derive_temporal_context('2026-03-25T00:00:00Z', 'America/New_York')
        assert result['day_of_week'] == 'Tuesday'
        assert result['time_of_day'] == 'evening'

    def test_different_timezone_same_utc_instant(self):
        """Same UTC instant should produce different local times for different timezones."""
        ts = '2026-03-25T12:00:00Z'
        ny = derive_temporal_context(ts, 'America/New_York')
        tokyo = derive_temporal_context(ts, 'Asia/Tokyo')
        # NYC: 8 AM (morning), Tokyo: 9 PM (night)
        assert ny['time_of_day'] == 'morning'
        assert tokyo['time_of_day'] == 'night'

    def test_invalid_timezone_returns_empty(self):
        """Invalid timezone should gracefully return empty dict (server fallback)."""
        result = derive_temporal_context('2026-03-25T12:00:00Z', 'Not/A/Timezone')
        assert result == {}

    def test_invalid_timestamp_returns_empty(self):
        """Invalid timestamp should gracefully return empty dict."""
        result = derive_temporal_context('not-a-timestamp', 'America/New_York')
        assert result == {}

    def test_empty_inputs_return_empty(self):
        result = derive_temporal_context('', 'America/New_York')
        assert result == {}


class TestGetCurrentTimeContextWithClientTemporal:
    """Tests that get_current_time_context() uses client temporal data when provided."""

    def test_uses_client_temporal_when_provided(self):
        """When client_temporal has 'formatted', it should be used instead of server clock."""
        temporal = {
            'formatted': 'Current time: Tuesday, March 24, 2026 at 8:30 PM EDT',
            'time_of_day': 'evening',
            'day_of_week': 'Tuesday',
            'is_weekend': False,
        }
        result = get_current_time_context(temporal)
        assert 'EDT' in result
        assert 'Tuesday' in result
        assert 'evening' in result
        assert 'weekday' in result

    def test_weekend_label(self):
        temporal = {
            'formatted': 'Current time: Saturday, March 28, 2026 at 10:00 AM EDT',
            'time_of_day': 'morning',
            'day_of_week': 'Saturday',
            'is_weekend': True,
        }
        result = get_current_time_context(temporal)
        assert 'weekend' in result
        assert 'Saturday' in result

    def test_falls_back_to_server_clock_when_none(self):
        """When no client temporal data, falls back to server PST clock."""
        result = get_current_time_context(None)
        assert 'Current time:' in result
        assert 'PST' in result or 'PDT' in result

    def test_falls_back_when_empty_dict(self):
        """Empty dict (failed derivation) should fall back to server clock."""
        result = get_current_time_context({})
        assert 'Current time:' in result
        assert 'PST' in result or 'PDT' in result

    def test_no_hardcoded_timezone_in_client_mode(self):
        """When using client time, PST should not appear (user may be in a different zone)."""
        temporal = {
            'formatted': 'Current time: Wednesday, March 25, 2026 at 9:00 AM JST',
            'time_of_day': 'morning',
            'day_of_week': 'Wednesday',
            'is_weekend': False,
        }
        result = get_current_time_context(temporal)
        assert 'JST' in result
        assert 'PST' not in result


class TestEndToEndTemporalFlow:
    """Integration test: client data flows through derive → get_current_time_context."""

    def test_full_flow(self):
        """Derive from client clock, then format for prompt — no hardcoded assumptions."""
        temporal = derive_temporal_context(
            '2026-03-28T22:15:00-04:00',  # Saturday night in NYC
            'America/New_York'
        )
        prompt_line = get_current_time_context(temporal)

        assert 'Saturday' in prompt_line
        assert 'night' in prompt_line
        assert 'weekend' in prompt_line
        assert 'March 28, 2026' in prompt_line
        # Should NOT contain hardcoded PST
        assert 'PST' not in prompt_line
