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


def derive_temporal_context(client_timestamp: str, client_timezone: str) -> dict:
    """
    Derive structured temporal context from client-provided clock data.

    Args:
        client_timestamp: ISO 8601 timestamp from the client device
        client_timezone: IANA timezone string (e.g., 'America/New_York')

    Returns:
        Dict with keys: local_time, time_of_day, day_of_week, is_weekend,
        date_str, timezone, formatted (human-readable summary)
    """
    try:
        tz = ZoneInfo(client_timezone)
    except (KeyError, ValueError):
        logger.warning(f"Invalid client timezone '{client_timezone}', falling back to server clock")
        return {}

    try:
        from datetime import timezone as dt_timezone
        parsed = datetime.fromisoformat(client_timestamp.replace('Z', '+00:00'))
        local_time = parsed.astimezone(tz)
    except (ValueError, TypeError):
        logger.warning(f"Invalid client timestamp '{client_timestamp}', falling back to server clock")
        return {}

    hour = local_time.hour
    if 5 <= hour < 12:
        time_of_day = 'morning'
    elif 12 <= hour < 17:
        time_of_day = 'afternoon'
    elif 17 <= hour < 21:
        time_of_day = 'evening'
    else:
        time_of_day = 'night'

    day_of_week = local_time.strftime('%A')
    is_weekend = local_time.weekday() >= 5
    date_str = local_time.strftime('%B %d, %Y')

    # Abbreviate timezone for display (e.g., 'America/New_York' -> 'EST'/'EDT')
    tz_abbr = local_time.strftime('%Z') or client_timezone.split('/')[-1]

    formatted = local_time.strftime(f"Current time: %A, %B %d, %Y at %-I:%M %p {tz_abbr}")

    return {
        'local_time': local_time,
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'date_str': date_str,
        'timezone': client_timezone,
        'tz_abbr': tz_abbr,
        'formatted': formatted,
    }


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

        # 5. Companion's work projects
        work_section = self._get_work_projects_section()
        if work_section:
            sections.append(work_section)

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
        """
        Get the companion's schedule as first-person behavioral context.

        Tries the rich calendar schedule service first (DB-backed, with
        completed/in-progress/upcoming awareness), then falls back to
        the structural skeleton.
        """
        # Try the rich calendar schedule service first
        try:
            from src.scheduling.calendar_schedule_service import (
                get_calendar_schedule_service, is_calendar_schedule_enabled
            )
            if is_calendar_schedule_enabled():
                cal_service = get_calendar_schedule_service()
                behavior_context = cal_service.format_schedule_behavior_context()
                if behavior_context:
                    return f"[YOUR SCHEDULE TODAY]\n{behavior_context}"
        except Exception as e:
            logger.debug(f"Calendar schedule service unavailable: {e}")

        # Fall back to structural skeleton
        try:
            from src.scheduling.companion_schedule import get_companion_schedule

            schedule = get_companion_schedule()
            activity_status = schedule.get_current_activity_status(
                now.replace(tzinfo=None)
            )

            status = activity_status.get('status', 'unknown')
            details = activity_status.get('details', '')

            if status == 'asleep':
                return "You're sleeping right now."

            parts = []
            if details:
                parts.append(f"You're currently: {details}")
            else:
                parts.append(f"Status: {status}")

            today = schedule.get_today_schedule()
            if today:
                workload = today.get('workload', 'normal')
                if workload != 'normal':
                    parts.append(f"Workload today: {workload}")

            if status == 'working':
                parts.append(
                    "You're in work mode right now. Respond naturally but "
                    "acknowledge you're working — keep it brief unless "
                    "it's something important."
                )

            return "\n".join(parts)

        except Exception as e:
            logger.debug(f"Companion schedule section unavailable: {e}")
            return ""

    def _get_work_projects_section(self) -> str:
        """Get the companion's current work projects for prompt context."""
        try:
            from src.autonomy.work_projects import get_work_project_manager
            wpm = get_work_project_manager()
            return wpm.format_for_prompt()
        except Exception as e:
            logger.debug(f"Work projects unavailable: {e}")
            return ""
