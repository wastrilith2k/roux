"""
Clock Abstraction — Companion Framework

WHAT: Provides an injectable time source for the entire framework.
WHY:  The simulation system (scripts/simulate_relationship.py) needs to compress weeks
      of relationship development into minutes. Every module that cares about "now" must
      go through this abstraction so simulation can control time globally.
HOW:  A module-level singleton clock (default: SystemClock) can be swapped to a
      SimulationClock via set_clock(). All code calls clock.now() instead of
      datetime.now() directly. The convenience function now() avoids importing
      the singleton everywhere.

Production: SystemClock -> real wall clock in Pacific time.
Simulation: SimulationClock -> manually advanced via advance()/set().
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# All timestamps in the framework default to Pacific time
PST = ZoneInfo('America/Los_Angeles')


# ---------------------------------------------------------------------------
# Clock interface and implementations
# ---------------------------------------------------------------------------

class Clock:
    """Abstract clock interface. Subclass and implement now()."""

    def now(self) -> datetime:
        raise NotImplementedError


class SystemClock(Clock):
    """Production clock — returns real wall-clock time in Pacific."""

    def now(self) -> datetime:
        return datetime.now(PST)


class SimulationClock(Clock):
    """Manually-controlled clock for time-compressed simulation and testing."""

    def __init__(self, start: datetime):
        self._time = start

    def now(self) -> datetime:
        return self._time

    def advance(self, **kwargs):
        """Advance the clock by a timedelta. Usage: advance(hours=2) or advance(days=1)."""
        self._time += timedelta(**kwargs)

    def set(self, time: datetime):
        """Jump the clock to an exact point in time."""
        self._time = time


# ---------------------------------------------------------------------------
# Module-level singleton — the single source of "what time is it?"
# ---------------------------------------------------------------------------

_clock: Clock = SystemClock()


def get_clock() -> Clock:
    """Get the current clock instance."""
    return _clock


def set_clock(clock: Clock):
    """Override the global clock (for simulation/testing)."""
    global _clock
    _clock = clock


def now() -> datetime:
    """Convenience: get current time from the active clock.

    Preferred import style throughout the codebase:
        from src.core.clock import now as clock_now
    """
    return _clock.now()
