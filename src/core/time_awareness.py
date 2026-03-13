"""
Time Awareness -- multi-source temporal context block for the system prompt.

WHAT: Merges three data sources into a single "what's happening right now" text
      block: Google Calendar events (real), user's routine (hardcoded schedule),
      and the companion's own simulated schedule.

WHY:  "Current time: 2:30 PM PST" is nearly useless. The companion needs to know
      "James is probably wrapping up his afternoon focus block, and the companion
      herself just finished her virtual lunch break." This context makes time
      references in conversation feel natural and grounded.

HOW:  `get_time_context()` builds a formatted string by querying:
      1. CalendarService for today's Google Calendar events
      2. user_context.get_user_probable_activity() for the user's routine
      3. CompanionSchedule for the companion's own activities
      The result is injected into the system prompt before each LLM call.

Singleton: `get_time_awareness()` at module top.
"""

import logging
import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# Module-level singleton
_time_awareness: Optional['TimeAwareness'] = None


def get_time_awareness() -> 'TimeAwareness':
    """Get or create the singleton TimeAwareness instance."""
    global _time_awareness
    if _time_awareness is None:
        _time_awareness = TimeAwareness()
    return _time_awareness


class TimeAwareness:
    """Provides rich temporal context beyond just the clock."""

    def get_time_context(self) -> str:
        """
        Build comprehensive time awareness context.

        Returns a formatted block for injection into the system prompt that
        includes current time, user's calendar events, their routine, and
        the companion's schedule status.
        """
        now = clock_now()
        sections = []

        # 1. Current time (always present)
        time_str = now.strftime("Current time: %A, %B %d, %Y at %-I:%M %p PST")
        sections.append(time_str)

        # 2. User's Google Calendar (real events)
        calendar_section = self._get_calendar_section(now)
        if calendar_section:
            sections.append(calendar_section)

        # 3. User's routine (from YAML profile)
        routine_section = self._get_routine_section(now)
        if routine_section:
            sections.append(routine_section)

        # 4. Companion's schedule
        companion_section = self._get_companion_schedule_section(now)
        if companion_section:
            sections.append(companion_section)

        return "[TIME & AWARENESS]\n" + "\n\n".join(sections)

    def _get_calendar_section(self, now: datetime) -> str:
        """Get formatted Google Calendar events if available."""
        try:
            from src.integrations.calendar_service import get_calendar_service, is_calendar_awareness_enabled

            if not is_calendar_awareness_enabled():
                return ""

            cal = get_calendar_service()
            events = cal.get_upcoming_events(hours_ahead=12, max_results=8)

            if not events:
                return ""

            formatted = cal.format_for_prompt(events)
            if formatted:
                from src.config.persona_config import get_persona_config
                _user_name = get_persona_config().primary_user_name
                return (
                    f"{_user_name}'s calendar (may contain stale/recurring events — "
                    "what you know from conversations is more reliable):\n"
                    + formatted
                )
            return ""

        except Exception as e:
            logger.debug(f"Calendar section unavailable: {e}")
            return ""

    def _get_routine_section(self, now: datetime) -> str:
        """Get user's routine-based context from the existing autopilot system."""
        try:
            from src.core.user_context import get_user_probable_activity

            activity = get_user_probable_activity(now)

            # Don't add routine context if autopilot is disabled
            if activity.get('autopilot_disabled'):
                return ""

            # If user has calendar events, mark routine as supplementary
            description = activity.get('description', '')
            if not description or description == 'User is around':
                return ""

            location = activity.get('location', '')
            location_str = f" (at {location})" if location and location != 'unknown' else ""

            return f"His routine: {description}{location_str}"

        except Exception as e:
            logger.debug(f"Routine section unavailable: {e}")
            return ""

    def _get_companion_schedule_section(self, now: datetime) -> str:
        """Get the companion's current schedule status."""
        try:
            from src.scheduling.companion_schedule import get_companion_schedule

            schedule = get_companion_schedule()
            activity_status = schedule.get_current_activity_status(now.replace(tzinfo=None))

            status = activity_status.get('status', 'unknown')
            details = activity_status.get('details', '')

            if status == 'asleep':
                return "Companion status: sleeping"

            parts = [f"Companion status: {details}" if details else f"Companion status: {status}"]

            # Add workload if available
            today = schedule.get_today_schedule()
            if today:
                workload = today.get('workload', 'normal')
                if workload != 'normal':
                    parts.append(f"Workload: {workload}")
                notes = today.get('notes', '')
                if notes:
                    parts.append(f"Note: {notes}")

            # Reinforce work mode behavior when she's working
            if status == 'working':
                parts.append("You're in work mode right now. Respond naturally but acknowledge you're working — keep it brief unless it's something important.")

            return "\n".join(parts)

        except Exception as e:
            logger.debug(f"Companion schedule section unavailable: {e}")
            return ""
