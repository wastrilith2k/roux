"""Shared test fixtures for the companion framework."""

import os
import pytest

# Set test environment before any src imports
os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test_password')
os.environ.setdefault('POSTGRES_DB', 'companion_test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'companion')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/15')

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.core.clock import SimulationClock, SystemClock, set_clock, PST


@pytest.fixture
def sim_clock():
    """Provide a SimulationClock starting at a known time, reset after test."""
    start = datetime(2026, 3, 1, 12, 0, tzinfo=PST)
    clock = SimulationClock(start=start)
    set_clock(clock)
    yield clock
    set_clock(SystemClock())


@pytest.fixture
def mock_db(tmp_path, monkeypatch):
    """Provide a mock database connection params pointing nowhere.

    Use this for tests that need to verify DB calls are made correctly
    without actually connecting.
    """
    monkeypatch.setenv('POSTGRES_HOST', 'localhost')
    monkeypatch.setenv('POSTGRES_PORT', '59999')
    monkeypatch.setenv('POSTGRES_DB', 'companion_test')
    monkeypatch.setenv('POSTGRES_USER', 'test')
    monkeypatch.setenv('POSTGRES_PASSWORD', 'test_password')
