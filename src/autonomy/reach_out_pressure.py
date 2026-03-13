"""
Reach-Out Pressure -- Tracks the companion's accumulated desire to communicate.

WHAT: A numeric "pressure" model (0.0-1.0) that represents how much the
      companion wants to reach out right now.  Higher pressure shortens the
      interval between reach-out checks; releasing it resets the cycle.

WHY:  Real people feel a growing urge to talk when they can't.  Rather than
      reaching out on a fixed timer, pressure makes the companion's behavior
      adaptive -- she checks more often when she has something to say and less
      when she doesn't.

HOW IT FITS:
  - The AlwaysOnService (_run_loop) reads pressure to decide its sleep interval.
  - The ReachOutEngine accumulates pressure on "external" suppressions (she wanted
    to talk but couldn't) and releases it on a successful send.
  - Pressure decays naturally over time (~45-min half-life) so stale desire fades.
  - Persisted to disk as JSON so the state survives restarts.

Lifecycle:
  accumulate()  -> called when reach-out is suppressed externally
  bump()        -> small nudge from contextual events (activity transitions)
  release()     -> called after a successful reach-out
  decay()       -> called every loop iteration to model natural fade
"""

import json
import logging
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)


# =============================================================================
# Pressure dataclass
# =============================================================================

@dataclass
class ReachOutPressure:
    """Tracks accumulated desire to reach out."""

    # ---- Core pressure state ----
    level: float = 0.0              # 0.0 (no desire) to 1.0 (maximal desire)
    suppressed_count: int = 0       # consecutive external suppressions
    last_suppressed_reason: str = ""

    # ---- Timestamps ----
    last_successful: Optional[str] = None   # ISO -- when she last reached out
    last_updated: Optional[str] = None      # ISO -- last mutation

    # ---- Interval bounds (minutes) ----
    MIN_INTERVAL: float = 3.0       # fastest possible re-check
    MAX_INTERVAL: float = 25.0      # slowest possible re-check
    BASE_INTERVAL: float = 12.0     # starting point at zero pressure

    # -----------------------------------------------------------------
    # Mutation helpers
    # -----------------------------------------------------------------

    def accumulate(self, reason: str):
        """
        Increase pressure when she wanted to reach out but was blocked.

        Each consecutive suppression increases pressure with diminishing
        returns so it doesn't spike to 1.0 immediately.
        Formula: increment = 0.15 / (1 + count * 0.2)
          1st: +0.15, 2nd: +0.12, 3rd: +0.10, ...
        """
        self.suppressed_count += 1
        self.last_suppressed_reason = reason

        increment = 0.15 / (1 + self.suppressed_count * 0.2)
        self.level = min(1.0, self.level + increment)

        self.last_updated = clock_now().isoformat()

        logger.info(
            f"Pressure accumulated: {self.level:.2f} "
            f"(+{increment:.2f}, count={self.suppressed_count}, reason={reason})"
        )

    def bump(self, amount: float, reason: str = ""):
        """
        Small pressure bump for contextual events (activity transitions, etc.).

        Lighter than accumulate() -- does NOT increment suppressed_count.
        """
        old = self.level
        self.level = min(1.0, self.level + amount)
        self.last_updated = clock_now().isoformat()

        logger.debug(f"Pressure bumped: {old:.2f} -> {self.level:.2f} ({reason})")

    def release(self):
        """Reset pressure after a successful reach-out."""
        self.level = 0.0
        self.suppressed_count = 0
        self.last_suppressed_reason = ""
        self.last_successful = clock_now().isoformat()
        self.last_updated = clock_now().isoformat()

        logger.info("Pressure released (successful reach-out)")

    def decay(self, minutes_elapsed: float):
        """
        Natural decay over time.

        Pressure doesn't persist forever -- if nothing triggers for a while
        the desire fades.  Modeled as linear-approximation of exponential
        decay with a ~45-minute half-life (decay_rate=0.015/min).
        """
        if self.level <= 0:
            return

        decay_rate = 0.015  # per minute
        decay_amount = self.level * decay_rate * minutes_elapsed
        old = self.level
        self.level = max(0.0, self.level - decay_amount)

        # Once pressure drops below threshold, reset suppressed count
        if self.level < 0.1:
            self.suppressed_count = 0

        self.last_updated = clock_now().isoformat()

        if old - self.level > 0.01:
            logger.debug(f"Pressure decayed: {old:.2f} -> {self.level:.2f} ({minutes_elapsed:.0f}min)")

    # -----------------------------------------------------------------
    # Interval calculation
    # -----------------------------------------------------------------

    def get_next_interval_minutes(self) -> float:
        """
        Calculate next check interval based on current pressure.

        High pressure  -> shorter interval (closer to MIN_INTERVAL)
        Low  pressure  -> longer  interval (closer to MAX_INTERVAL)
        +-15 % random jitter to avoid robotic regularity.
        """
        # Linear interpolation: high pressure = short interval
        base = self.BASE_INTERVAL - (self.BASE_INTERVAL - self.MIN_INTERVAL) * self.level
        interval = max(self.MIN_INTERVAL, min(self.MAX_INTERVAL, base))

        # Jitter (+/-15%) to feel more natural
        jitter = interval * random.uniform(-0.15, 0.15)
        interval = max(self.MIN_INTERVAL, interval + jitter)

        return round(interval, 1)

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level,
            'suppressed_count': self.suppressed_count,
            'last_suppressed_reason': self.last_suppressed_reason,
            'last_successful': self.last_successful,
            'last_updated': self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReachOutPressure':
        if not data:
            return cls()
        return cls(
            level=data.get('level', 0.0),
            suppressed_count=data.get('suppressed_count', 0),
            last_suppressed_reason=data.get('last_suppressed_reason', ''),
            last_successful=data.get('last_successful'),
            last_updated=data.get('last_updated'),
        )


# =============================================================================
# File-based persistence
# =============================================================================

def _get_pressure_file() -> str:
    """Get path to pressure state JSON file."""
    data_dir = os.environ.get('DATA_DIR', '/app/data')
    return os.path.join(data_dir, 'reach_out_pressure.json')


def load_pressure() -> ReachOutPressure:
    """Load pressure state from disk (returns fresh state on any error)."""
    try:
        filepath = _get_pressure_file()
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                pressure = ReachOutPressure.from_dict(data)
                logger.debug(f"Loaded pressure state: level={pressure.level:.2f}")
                return pressure
    except Exception as e:
        logger.debug(f"Could not load pressure state: {e}")

    return ReachOutPressure()


def save_pressure(pressure: ReachOutPressure):
    """Persist pressure state to disk."""
    try:
        filepath = _get_pressure_file()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(pressure.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save pressure state: {e}")


# =============================================================================
# Singleton accessor
# =============================================================================

_pressure: Optional[ReachOutPressure] = None


def get_reach_out_pressure() -> ReachOutPressure:
    """Get singleton pressure instance (loads from file on first call)."""
    global _pressure
    if _pressure is None:
        _pressure = load_pressure()
    return _pressure
