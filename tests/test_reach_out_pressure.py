"""Tests for the reach-out pressure model."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.clock import SimulationClock, set_clock, SystemClock, PST
from src.autonomy.reach_out_pressure import ReachOutPressure


NOW = datetime(2026, 3, 1, 12, 0, tzinfo=PST)


@pytest.fixture(autouse=True)
def fixed_clock():
    set_clock(SimulationClock(start=NOW))
    yield
    set_clock(SystemClock())


class TestAccumulation:
    def test_first_accumulation(self):
        p = ReachOutPressure()
        assert p.level == 0.0
        p.accumulate("test")
        # increment = 0.15 / (1 + 1*0.2) = 0.125
        assert p.level == pytest.approx(0.125, abs=0.01)
        assert p.suppressed_count == 1

    def test_diminishing_returns(self):
        p = ReachOutPressure()
        p.accumulate("test1")
        level1 = p.level
        p.accumulate("test2")
        level2 = p.level - level1
        p.accumulate("test3")
        level3 = p.level - level1 - level2
        # Each increment should be smaller
        assert level2 < level1 or level2 == pytest.approx(level1, abs=0.02)
        assert level3 < level2 or level3 == pytest.approx(level2, abs=0.02)

    def test_capped_at_one(self):
        p = ReachOutPressure()
        for _ in range(50):
            p.accumulate("test")
        assert p.level <= 1.0

    def test_reason_tracked(self):
        p = ReachOutPressure()
        p.accumulate("quiet_hours")
        assert p.last_suppressed_reason == "quiet_hours"


class TestBump:
    def test_bump_increases_level(self):
        p = ReachOutPressure()
        p.bump(0.1, "activity_done")
        assert p.level == pytest.approx(0.1)

    def test_bump_does_not_increment_count(self):
        p = ReachOutPressure()
        p.bump(0.1)
        assert p.suppressed_count == 0

    def test_bump_capped(self):
        p = ReachOutPressure(level=0.95)
        p.bump(0.2)
        assert p.level <= 1.0


class TestRelease:
    def test_release_resets_everything(self):
        p = ReachOutPressure(level=0.8, suppressed_count=5, last_suppressed_reason="test")
        p.release()
        assert p.level == 0.0
        assert p.suppressed_count == 0
        assert p.last_suppressed_reason == ""
        assert p.last_successful is not None


class TestDecay:
    def test_decay_reduces_level(self):
        p = ReachOutPressure(level=0.5)
        p.decay(minutes_elapsed=10)
        assert p.level < 0.5

    def test_decay_formula(self):
        p = ReachOutPressure(level=0.5)
        # decay_amount = 0.5 * 0.015 * 10 = 0.075
        p.decay(minutes_elapsed=10)
        assert p.level == pytest.approx(0.5 - 0.075, abs=0.01)

    def test_decay_never_negative(self):
        p = ReachOutPressure(level=0.01)
        p.decay(minutes_elapsed=1000)
        assert p.level >= 0.0

    def test_no_decay_at_zero(self):
        p = ReachOutPressure(level=0.0)
        p.decay(minutes_elapsed=60)
        assert p.level == 0.0

    def test_suppressed_count_resets_at_low_level(self):
        p = ReachOutPressure(level=0.15, suppressed_count=3)
        p.decay(minutes_elapsed=60)  # should drop below 0.1
        assert p.suppressed_count == 0


class TestIntervalCalculation:
    def test_zero_pressure_gives_base_interval(self):
        p = ReachOutPressure(level=0.0)
        interval = p.get_next_interval_minutes()
        # Should be around BASE_INTERVAL (12) +/- 15% jitter
        assert 9.0 <= interval <= 15.0

    def test_max_pressure_gives_min_interval(self):
        p = ReachOutPressure(level=1.0)
        interval = p.get_next_interval_minutes()
        # Should be around MIN_INTERVAL (3) +/- jitter
        assert 2.5 <= interval <= 5.0

    def test_higher_pressure_shorter_interval(self):
        # Test with fixed seed for reproducibility isn't needed;
        # just check the trend over multiple samples
        low_intervals = []
        high_intervals = []
        for _ in range(20):
            p_low = ReachOutPressure(level=0.2)
            low_intervals.append(p_low.get_next_interval_minutes())
            p_high = ReachOutPressure(level=0.8)
            high_intervals.append(p_high.get_next_interval_minutes())
        # Average should show the trend
        assert sum(high_intervals) / len(high_intervals) < sum(low_intervals) / len(low_intervals)


class TestSerialization:
    def test_roundtrip(self):
        p = ReachOutPressure(level=0.42, suppressed_count=3, last_suppressed_reason="test")
        d = p.to_dict()
        p2 = ReachOutPressure.from_dict(d)
        assert p2.level == pytest.approx(0.42)
        assert p2.suppressed_count == 3
        assert p2.last_suppressed_reason == "test"

    def test_from_empty_dict(self):
        p = ReachOutPressure.from_dict({})
        assert p.level == 0.0

    def test_from_none(self):
        p = ReachOutPressure.from_dict(None)
        assert p.level == 0.0
