"""
Background Life -- tracks the companion's activities between conversations.

WHAT: Records what the companion was doing while the user was away, based on
      the companion's schedule and the time of day. Produces an activity log and a narrative
      summary that bridges the gap when conversation resumes.

WHY:  Without this, the companion either says nothing about the gap ("hi again")
      or invents something random. By grounding background activities in the
      actual schedule, the narrative feels consistent: "I finished my morning
      focus block and then read for a bit" instead of a generic filler.

HOW:  When the companion goes idle, `start_background_life()` snapshots her
      mood/energy and schedule context. Periodically (or on resume),
      `infer_activities()` walks through elapsed time blocks and assigns
      activities from the schedule. `generate_narrative()` calls the LLM to
      produce a natural sentence about what she did. State is persisted as a
      JSON file in data/ (no DB migration needed).

Singleton: `get_background_life_tracker()` at module bottom.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

from src.utils.path_utils import get_data_dir

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')
ACTIVITIES_FILE = Path(os.path.join(get_data_dir(), "companion_activities.json"))


@dataclass
class Activity:
    """A recorded activity from the companion's background life."""
    timestamp: str  # ISO format
    activity_type: str  # work, personal, rest, social, etc.
    description: str  # What she was doing
    duration_hours: float  # How long (estimated)
    mood: str = "neutral"  # Her mood at the time
    energy: float = 0.5  # Energy level (0-1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Activity':
        return cls(**data)


class BackgroundLife:
    """
    Tracks what the companion was doing between conversations.

    Provides narrative continuity - the companion remembers what they were doing
    and can reference it naturally in conversation.
    """

    def __init__(self):
        self._activities: List[Activity] = []
        self._load_activities()

    def _load_activities(self):
        """Load activities from file."""
        try:
            if ACTIVITIES_FILE.exists():
                with open(ACTIVITIES_FILE, 'r') as f:
                    data = json.load(f)
                    self._activities = [Activity.from_dict(a) for a in data]
                    # Prune old activities (keep last 7 days)
                    self._prune_old_activities()
        except Exception as e:
            logger.warning(f"Could not load activities: {e}")
            self._activities = []

    def _save_activities(self):
        """Save activities to file."""
        try:
            ACTIVITIES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(ACTIVITIES_FILE, 'w') as f:
                json.dump([a.to_dict() for a in self._activities], f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save activities: {e}")

    def _prune_old_activities(self):
        """Remove activities older than 7 days."""
        cutoff = datetime.now(PST) - timedelta(days=7)
        cutoff_str = cutoff.isoformat()
        self._activities = [a for a in self._activities if a.timestamp > cutoff_str]

    def record_activity(
        self,
        activity_type: str,
        description: str,
        duration_hours: float,
        mood: str = "neutral",
        energy: float = 0.5
    ):
        """
        Record what the companion was doing.

        Called when transitioning to idle or at regular intervals.
        """
        activity = Activity(
            timestamp=datetime.now(PST).isoformat(),
            activity_type=activity_type,
            description=description,
            duration_hours=duration_hours,
            mood=mood,
            energy=energy
        )
        self._activities.append(activity)
        self._save_activities()
        logger.info(f"Recorded activity: {activity_type} - {description[:50]}...")

    def get_recent_activities(self, hours: int = 24) -> List[Activity]:
        """Get activities from the last N hours."""
        cutoff = datetime.now(PST) - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()
        return [a for a in self._activities if a.timestamp > cutoff_str]

    def get_last_activity(self) -> Optional[Activity]:
        """Get the most recent activity."""
        if self._activities:
            return self._activities[-1]
        return None

    def generate_idle_activity(
        self,
        gap_hours: float,
        current_mood: str = "neutral",
        current_energy: float = 0.5
    ) -> str:
        """
        Generate what the companion was doing during idle time.

        Uses the companion's schedule + time of day to infer realistic activities.
        Returns a natural description for the prompt.
        """
        from src.scheduling.companion_schedule import get_companion_schedule
        from src.utils.timezone_utils import now_pacific_naive

        now = now_pacific_naive()
        schedule = get_companion_schedule()

        # Get what time block(s) the companion was in during the gap
        gap_start = now - timedelta(hours=gap_hours)

        # Determine activity based on schedule and time
        activity_type = "personal"
        description = "relaxing"

        try:
            # Check if the companion was working during the gap
            if schedule.is_working_now(gap_start):
                activity_type = "work"
                # Get current work projects for context
                today_schedule = schedule.get_today_schedule()
                if today_schedule:
                    workload = today_schedule.get('workload', 'normal')
                    if workload == 'heavy':
                        description = "deep in documentation work - it's been a heavy day"
                    else:
                        description = "working on documentation"

            # Check if she was sleeping
            elif schedule.is_asleep(gap_start) or gap_hours >= 6:
                activity_type = "rest"
                description = "sleeping"

            # Evening/personal time
            elif gap_start.hour >= 18:
                activity_type = "personal"
                if current_energy < 0.4:
                    description = "unwinding - been a long day"
                else:
                    description = "relaxing, doing some reading"

            # Afternoon
            elif gap_start.hour >= 12:
                if gap_start.weekday() < 5:  # Weekday
                    activity_type = "work"
                    description = "finishing up some docs"
                else:  # Weekend
                    activity_type = "personal"
                    description = "enjoying the weekend"

            # Morning
            else:
                activity_type = "personal"
                description = "getting started with the day"

            # Record this activity
            self.record_activity(
                activity_type=activity_type,
                description=description,
                duration_hours=gap_hours,
                mood=current_mood,
                energy=current_energy
            )

        except Exception as e:
            logger.warning(f"Could not generate schedule-based activity: {e}")
            description = "doing some stuff"

        return description

    def generate_narrative(self, gap_hours: float) -> str:
        """
        Generate a natural narrative of what she was doing.

        Uses LLM to synthesize recent activities into a coherent description.
        Falls back to simple description if LLM unavailable.
        """
        recent = self.get_recent_activities(hours=int(gap_hours) + 1)

        if not recent:
            return self.generate_idle_activity(gap_hours)

        # Simple narrative from recent activities
        if len(recent) == 1:
            return recent[0].description

        # Multiple activities - summarize
        activity_types = set(a.activity_type for a in recent)
        if 'work' in activity_types and 'rest' in activity_types:
            return "working and taking breaks"
        elif 'work' in activity_types:
            return "focused on work"
        elif 'rest' in activity_types:
            return "resting and recharging"
        else:
            return recent[-1].description

    def format_for_prompt(self, hours: int = 24) -> str:
        """
        Format recent activities for inclusion in prompt.

        Gives the companion context about what they've been doing recently
        so she can reference it naturally in conversation.
        """
        recent = self.get_recent_activities(hours)

        if not recent:
            return ""

        lines = ["[COMPANION'S RECENT ACTIVITIES - what they've been doing]"]

        for activity in recent[-5:]:  # Last 5 activities
            time = datetime.fromisoformat(activity.timestamp)
            time_str = time.strftime("%I:%M %p")
            lines.append(f"- {time_str}: {activity.description} ({activity.activity_type})")

        lines.append("\nUse these naturally if relevant - you remember what you've been doing.")

        return "\n".join(lines)


# Singleton instance
_background_life: Optional[BackgroundLife] = None


def get_background_life() -> BackgroundLife:
    """Get singleton BackgroundLife instance."""
    global _background_life
    if _background_life is None:
        _background_life = BackgroundLife()
    return _background_life
