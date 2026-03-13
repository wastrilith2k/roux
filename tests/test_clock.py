"""Tests for the clock abstraction."""
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.core.clock import SystemClock, SimulationClock, set_clock, get_clock, now, PST


class TestSystemClock:
    def test_returns_datetime_with_pst(self):
        clock = SystemClock()
        t = clock.now()
        assert t.tzinfo is not None
        assert isinstance(t, datetime)

    def test_advances_with_real_time(self):
        clock = SystemClock()
        t1 = clock.now()
        t2 = clock.now()
        assert t2 >= t1


class TestSimulationClock:
    def test_returns_set_time(self):
        start = datetime(2026, 3, 1, 10, 0, tzinfo=PST)
        clock = SimulationClock(start=start)
        assert clock.now() == start

    def test_advance_hours(self):
        start = datetime(2026, 3, 1, 10, 0, tzinfo=PST)
        clock = SimulationClock(start=start)
        clock.advance(hours=3)
        assert clock.now() == start + timedelta(hours=3)

    def test_advance_days(self):
        start = datetime(2026, 3, 1, 10, 0, tzinfo=PST)
        clock = SimulationClock(start=start)
        clock.advance(days=7)
        assert clock.now().day == 8

    def test_set_time(self):
        start = datetime(2026, 3, 1, 10, 0, tzinfo=PST)
        clock = SimulationClock(start=start)
        new_time = datetime(2026, 6, 15, 14, 30, tzinfo=PST)
        clock.set(new_time)
        assert clock.now() == new_time

    def test_multiple_advances(self):
        start = datetime(2026, 1, 1, 7, 0, tzinfo=PST)
        clock = SimulationClock(start=start)
        clock.advance(hours=3)
        clock.advance(minutes=30)
        clock.advance(days=1)
        expected = start + timedelta(hours=3, minutes=30, days=1)
        assert clock.now() == expected


class TestGlobalClock:
    def test_default_is_system_clock(self):
        # Reset to default
        set_clock(SystemClock())
        clock = get_clock()
        assert isinstance(clock, SystemClock)

    def test_set_simulation_clock(self):
        sim = SimulationClock(start=datetime(2026, 1, 1, tzinfo=PST))
        set_clock(sim)
        assert get_clock() is sim
        assert now() == datetime(2026, 1, 1, tzinfo=PST)

    def test_now_convenience(self):
        start = datetime(2026, 5, 15, 12, 0, tzinfo=PST)
        set_clock(SimulationClock(start=start))
        assert now() == start

    def teardown_method(self):
        """Reset to system clock after each test."""
        set_clock(SystemClock())
