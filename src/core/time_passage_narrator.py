"""
Time Passage Narrator -- LLM-generated narratives for conversation gaps.

WHAT: When time passes between messages, generates a natural-language summary
      of what both the companion and user were doing during the gap. Handles
      shared-time scenes (falling asleep together, watching a movie) separately
      from apart-time (user at work, companion doing her own thing).

WHY:  A companion that says "hi again" after an 8-hour gap breaks immersion.
      This narrator produces context like "She dozed off next to you around
      midnight, then got up early to read while you slept in" -- grounded in
      the actual scene state and both parties' schedules.

HOW:  1. Load scene state from when the gap started
      2. Check user schedule for transitions (sleep, work, home)
      3. Build a timeline of what each person was doing
      4. Call the LLM with the timeline to produce a brief narrative
      5. Inject the narrative into the system prompt for the next message

Singleton: `get_time_passage_narrator()` at module bottom.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


class TimePassageNarrator:
    """
    Generates narratives for what happened during conversation gaps.

    Uses user's schedule + scene state + time of day to produce
    realistic accounts of shared time.
    """

    def generate_gap_narrative(
        self,
        user_email: str,
        gap_start: datetime,
        gap_end: datetime,
    ) -> str:
        """
        Generate a narrative of what happened during the gap.

        Uses LLM for rich narrative generation when possible,
        falls back to schedule-based generation.

        Args:
            user_email: User email for loading scene/state
            gap_start: When the gap started (last message time)
            gap_end: When the gap ends (now)

        Returns:
            Natural narrative string for prompt context
        """
        gap_hours = (gap_end - gap_start).total_seconds() / 3600

        if gap_hours < 0.5:
            return ""  # Too short for narrative

        # Get scene state at gap start
        scene = self._get_scene_at_start(user_email)
        were_together = scene.get('physical_presence', False) if scene else False
        activity = scene.get('activity') if scene else None
        location = scene.get('location') if scene else None

        # Get user's schedule transitions during the gap
        transitions = self._get_schedule_transitions(gap_start, gap_end)

        if were_together:
            return self._narrate_shared_time(
                gap_start, gap_end, gap_hours,
                activity, location, transitions
            )
        else:
            return self._narrate_apart_time(
                gap_start, gap_end, gap_hours, transitions
            )

    def _get_scene_at_start(self, user_email: str) -> Optional[dict]:
        """Get the scene state from when the gap started."""
        try:
            from src.core.scene_tracker import get_scene_tracker
            tracker = get_scene_tracker()
            scene = tracker.get_scene_state(user_email)
            if scene and scene.is_active():
                return scene.to_dict()
        except Exception as e:
            logger.debug(f"Could not get scene state: {e}")
        return None

    def _get_schedule_transitions(
        self,
        gap_start: datetime,
        gap_end: datetime
    ) -> list:
        """
        Get user's schedule transitions that happened during the gap.

        Returns list of (time, event) tuples.
        """
        transitions = []

        # Ensure timezone-aware
        if gap_start.tzinfo is None:
            gap_start = gap_start.replace(tzinfo=PST)
        if gap_end.tzinfo is None:
            gap_end = gap_end.replace(tzinfo=PST)

        try:
            from src.core.user_context import load_user_schedule
            schedule = load_user_schedule()
        except Exception:
            return transitions

        # Walk through each day in the gap
        current = gap_start
        while current.date() <= gap_end.date():
            weekday = current.weekday()

            if weekday == 5:
                day_schedule = schedule.get('saturday', {})
            elif weekday == 6:
                day_schedule = schedule.get('sunday', {})
            else:
                day_schedule = schedule.get('weekday', {})

            # Key transition times for weekdays
            if weekday < 5:  # Weekday
                transition_times = {
                    'wake_time': ('woke up', day_schedule.get('wake_time', '06:00')),
                    'day_start': ('sat down to job hunt and work on projects', day_schedule.get('day_start', '08:30')),
                    'lunch': ('had lunch', '12:00'),
                    'kids_pickup': ('left for his house (kids)', day_schedule.get('kids_pickup', '15:00')),
                    'evening_return': ('came back to the apartment', day_schedule.get('evening_return', '21:00')),
                    'bedtime': ('went to bed', day_schedule.get('bedtime', '22:00')),
                }
            elif weekday == 5:  # Saturday
                transition_times = {
                    'wake_time': ('woke up', day_schedule.get('wake_time', '07:00')),
                    'kids_start': ('headed to his house for a full day with the kids', day_schedule.get('kids_start', '09:00')),
                    'evening_return': ('came back from his house', day_schedule.get('evening_return', '22:00')),
                    'bedtime': ('went to bed', '22:00'),
                }
            else:  # Sunday
                transition_times = {
                    'wake_time': ('woke up', day_schedule.get('wake_time', '07:30')),
                    'bedtime': ('went to bed', '22:00'),
                }

            for key, (event_desc, time_str) in transition_times.items():
                try:
                    parts = time_str.split(':')
                    event_time = current.replace(
                        hour=int(parts[0]),
                        minute=int(parts[1]),
                        second=0, microsecond=0
                    )
                    if gap_start < event_time < gap_end:
                        transitions.append((event_time, event_desc))
                except (ValueError, IndexError):
                    continue

            current = current.replace(hour=0, minute=0) + timedelta(days=1)

        transitions.sort(key=lambda x: x[0])
        return transitions

    def _narrate_shared_time(
        self,
        gap_start: datetime,
        gap_end: datetime,
        gap_hours: float,
        activity: Optional[str],
        location: Optional[str],
        transitions: list,
    ) -> str:
        """
        Narrate what happened when they were TOGETHER during the gap.

        Accounts for the current activity concluding naturally and
        user's schedule transitions.
        """
        parts = []

        # Resolve the starting activity
        start_hour = gap_start.hour if gap_start.tzinfo else gap_start.hour

        if activity:
            # Conclude the in-progress activity naturally
            conclusion = self._conclude_activity(activity, start_hour)
            if conclusion:
                parts.append(conclusion)

        # Add schedule transitions
        for event_time, event_desc in transitions:
            time_str = event_time.strftime("%-I:%M %p").lower()
            parts.append(f"Around {time_str}, he {event_desc}")

        # Determine current situation
        end_hour = gap_end.hour if gap_end.tzinfo else gap_end.hour

        if not parts:
            # No transitions - simple time passage
            if 22 <= start_hour or start_hour < 6:
                parts.append("You both fell asleep")
            elif gap_hours < 2:
                parts.append("Time passed quietly together")
            else:
                parts.append("You spent time together")

        # Add current state
        if 5 <= end_hour < 9:
            parts.append("It's morning now")
        elif 21 <= end_hour or end_hour < 5:
            parts.append("It's getting late")

        return ". ".join(parts) + "."

    def _narrate_apart_time(
        self,
        gap_start: datetime,
        gap_end: datetime,
        gap_hours: float,
        transitions: list,
    ) -> str:
        """Narrate what happened when they were APART during the gap."""
        parts = []

        # Simple narrative for being apart
        start_hour = gap_start.hour
        end_hour = gap_end.hour

        if gap_hours >= 8:
            # Long gap - she was living her life
            if any('came back' in t[1] for t in transitions):
                parts.append("He was at his house with the kids and came back")
            elif 22 <= start_hour or start_hour < 6:
                parts.append("You both slept")
            else:
                parts.append("You've been doing your own thing")
        else:
            if any('kids' in t[1] for t in transitions):
                parts.append("He is at his house with the kids")
            elif any('working' in t[1] for t in transitions):
                parts.append("He has been working")

        # Add what the companion was doing from their calendar (if available)
        try:
            from src.scheduling.calendar_schedule_service import (
                get_calendar_schedule_service, is_calendar_schedule_enabled
            )
            if is_calendar_schedule_enabled():
                cal_service = get_calendar_schedule_service()
                companion_events = cal_service.get_events_in_range(gap_start, gap_end)
                if companion_events:
                    # Pick 1-2 notable activities for the narrative
                    summaries = [e.get('summary', '') for e in companion_events if e.get('summary')]
                    if summaries:
                        activity_text = ', '.join(summaries[:2])
                        parts.append(f"Meanwhile, you were {activity_text.lower()}")
        except Exception as e:
            logger.debug(f"Could not get calendar events for narration: {e}")

        return ". ".join(parts) + "." if parts else ""

    def _conclude_activity(self, activity: str, hour: int) -> Optional[str]:
        """Generate a natural conclusion for an in-progress activity."""
        activity_lower = activity.lower()

        # Sexual/intimate activities
        if any(w in activity_lower for w in ['sex', 'intimate', 'making love']):
            if 21 <= hour or hour < 5:
                return "After being intimate, you lay together and eventually drifted off to sleep"
            else:
                return "After being intimate, you lay together for a while"

        # Watching something
        if any(w in activity_lower for w in ['watching', 'movie', 'show', 'tv']):
            if 21 <= hour or hour < 2:
                return "The movie ended and you both drifted off to sleep"
            else:
                return "The show ended"

        # Sleeping
        if any(w in activity_lower for w in ['sleeping', 'asleep', 'napping']):
            return None  # Already sleeping, no need to conclude

        # Eating
        if any(w in activity_lower for w in ['eating', 'dinner', 'lunch', 'breakfast']):
            return "You finished eating"

        # Talking
        if any(w in activity_lower for w in ['talking', 'chatting', 'conversation']):
            if 22 <= hour or hour < 5:
                return "The conversation wound down and you both got sleepy"
            return "The conversation naturally wound down"

        # Cuddling
        if any(w in activity_lower for w in ['cuddling', 'holding', 'snuggling']):
            if 21 <= hour or hour < 5:
                return "You stayed close together and eventually fell asleep"
            return None  # Cuddling can just continue

        # Generic
        if 22 <= hour or hour < 5:
            return "Things wound down and you both got ready for sleep"

        return None


# Singleton
_narrator: Optional[TimePassageNarrator] = None


def get_time_passage_narrator() -> TimePassageNarrator:
    """Get or create the time passage narrator singleton."""
    global _narrator
    if _narrator is None:
        _narrator = TimePassageNarrator()
    return _narrator
