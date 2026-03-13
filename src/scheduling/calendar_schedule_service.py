"""
Calendar Schedule Service -- read-side facade for the companion's calendar events.

WHAT: Fetches today's (or any day's) events from Google Calendar and formats
      them for system-prompt injection, reach-out eligibility checks, and
      time-passage narration. Includes in-memory TTL caching and graceful
      fallback to local JSON plan files when Google API is unavailable.

WHY:  Multiple subsystems need the companion's schedule in different formats.
      This service centralises the read path and shields callers from Google
      API failure modes with a two-tier cache (memory + local JSON).

HOW:  Primary: Google Calendar API via `google_service.py`.
      Fallback: JSON files in `data/companion_daily_plans/`.
      Cache TTLs: 10 min for today's events, 1 hr for recent events.

Singleton: `get_calendar_schedule_service()` at module bottom.

Usage:
    service = get_calendar_schedule_service()
    events = service.get_today_events()
    prompt_text = service.format_for_prompt(events)
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cache TTLs
TODAY_EVENTS_CACHE_TTL = 600  # 10 minutes
RECENT_EVENTS_CACHE_TTL = 3600  # 1 hour

# Daily plans directory (local fallback)
DAILY_PLANS_DIR = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'companion_daily_plans'
)


class CalendarScheduleService:
    """Read-side service for the companion's Google Calendar schedule."""

    def __init__(self):
        self._today_cache = None  # (timestamp, events_list)
        self._recent_cache = None  # (timestamp, events_list)

    def _get_calendar_id(self) -> Optional[str]:
        """Get the stored calendar ID, or None if not created yet."""
        cal_id_file = os.path.join(
            os.environ.get('DATA_DIR', '/app/data'),
            'companion_calendar_id.txt'
        )
        if os.path.exists(cal_id_file):
            with open(cal_id_file, 'r') as f:
                cal_id = f.read().strip()
                return cal_id if cal_id else None
        return None

    def get_today_events(self) -> List[Dict]:
        """
        Get today's events from Google Calendar.

        Cached for 10 minutes. Falls back to local JSON cache.

        Returns:
            List of event dicts with summary, start, end, description, category
        """
        # Check cache
        if self._today_cache:
            cache_time, cached_events = self._today_cache
            if time.time() - cache_time < TODAY_EVENTS_CACHE_TTL:
                return cached_events

        from src.utils.timezone_utils import now_pacific_naive
        today = now_pacific_naive()
        date_str = today.strftime('%Y-%m-%d')

        events = self._fetch_events_for_date(date_str)

        # Cache the result
        self._today_cache = (time.time(), events)
        return events

    def get_current_activity(self) -> Optional[Dict]:
        """
        Get the event that overlaps with the current time.

        Returns:
            Event dict if currently in an event, None otherwise
        """
        from src.utils.timezone_utils import now_pacific_naive
        now = now_pacific_naive()
        current_time = now.strftime('%H:%M')

        events = self.get_today_events()
        for event in events:
            start = event.get('start_time', event.get('start', ''))
            end = event.get('end_time', event.get('end', ''))

            # Normalize to HH:MM
            if 'T' in start:
                start = start.split('T')[1][:5]
            if 'T' in end:
                end = end.split('T')[1][:5]

            if start <= current_time < end:
                return event

        return None

    def get_recent_events(self, days: int = 3) -> List[Dict]:
        """
        Get events from the past N days for narrative context.

        Cached for 1 hour.

        Args:
            days: Number of past days to fetch (default: 3)

        Returns:
            List of event dicts grouped by date
        """
        # Check cache
        if self._recent_cache:
            cache_time, cached_events = self._recent_cache
            if time.time() - cache_time < RECENT_EVENTS_CACHE_TTL:
                return cached_events

        from src.utils.timezone_utils import now_pacific_naive
        today = now_pacific_naive()

        all_events = []
        for i in range(1, days + 1):
            past_date = today - timedelta(days=i)
            date_str = past_date.strftime('%Y-%m-%d')
            day_events = self._fetch_events_for_date(date_str)
            if day_events:
                all_events.append({
                    'date': date_str,
                    'day_name': past_date.strftime('%A'),
                    'events': day_events,
                })

        self._recent_cache = (time.time(), all_events)
        return all_events

    def get_upcoming_events(self, hours: int = 4) -> List[Dict]:
        """
        Get the next few events coming up.

        Args:
            hours: Look-ahead window in hours (default: 4)

        Returns:
            List of upcoming event dicts
        """
        from src.utils.timezone_utils import now_pacific_naive
        now = now_pacific_naive()
        current_time = now.strftime('%H:%M')
        cutoff_time = (now + timedelta(hours=hours)).strftime('%H:%M')

        events = self.get_today_events()
        upcoming = []
        for event in events:
            start = event.get('start_time', event.get('start', ''))
            if 'T' in start:
                start = start.split('T')[1][:5]

            if current_time < start <= cutoff_time:
                upcoming.append(event)

        return upcoming

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Dict]:
        """
        Get events within an arbitrary time range (for time passage narration).

        Args:
            start: Start of range
            end: End of range

        Returns:
            List of event dicts
        """
        all_events = []
        current = start
        while current.date() <= end.date():
            date_str = current.strftime('%Y-%m-%d')
            day_events = self._fetch_events_for_date(date_str)
            for event in day_events:
                event_start = event.get('start_time', event.get('start', ''))
                if 'T' in event_start:
                    event_start = event_start.split('T')[1][:5]

                # Build full datetime for comparison
                try:
                    event_dt = datetime.strptime(
                        f"{date_str} {event_start}",
                        '%Y-%m-%d %H:%M'
                    )
                    if start <= event_dt <= end:
                        event_copy = dict(event)
                        event_copy['date'] = date_str
                        all_events.append(event_copy)
                except ValueError:
                    continue

            current = current.replace(hour=0, minute=0) + timedelta(days=1)

        return all_events

    def format_for_prompt(self, events: List[Dict]) -> str:
        """
        Format events into an LLM-consumable text block.

        Args:
            events: List of event dicts

        Returns:
            Formatted string for system prompt injection
        """
        if not events:
            return ""

        lines = []
        for event in events:
            start = event.get('start_time', event.get('start', ''))
            end = event.get('end_time', event.get('end', ''))

            # Normalize to HH:MM
            if 'T' in start:
                start = start.split('T')[1][:5]
            if 'T' in end:
                end = end.split('T')[1][:5]

            summary = event.get('summary', 'Activity')
            description = event.get('description', '')

            line = f"- {start}-{end}: {summary}"
            if description:
                line += f" ({description})"
            lines.append(line)

        return '\n'.join(lines)

    def format_current_context(self) -> str:
        """
        Format today's schedule into a concise context block for the system prompt.

        Returns:
            Formatted string like:
            "Your schedule today: [events]. You just finished [last]. Coming up: [next]."
        """
        from src.utils.timezone_utils import now_pacific_naive
        now = now_pacific_naive()
        current_time = now.strftime('%H:%M')

        events = self.get_today_events()
        if not events:
            return ""

        parts = []

        # Current activity
        current = self.get_current_activity()
        if current:
            parts.append(f"Right now you're: {current.get('summary', 'busy')}")
            desc = current.get('description', '')
            if desc:
                parts.append(f"({desc})")

        # What just finished
        last_event = None
        for event in events:
            end = event.get('end_time', event.get('end', ''))
            if 'T' in end:
                end = end.split('T')[1][:5]
            if end <= current_time:
                last_event = event

        if last_event and not current:
            parts.append(f"You just finished: {last_event.get('summary', 'something')}")

        # Coming up next
        upcoming = self.get_upcoming_events(hours=3)
        if upcoming:
            next_events = [e.get('summary', '?') for e in upcoming[:2]]
            parts.append(f"Coming up: {', '.join(next_events)}")

        return '. '.join(parts) if parts else ''

    def _fetch_events_for_date(self, date_str: str) -> List[Dict]:
        """
        Fetch events for a specific date, trying Google Calendar first,
        then falling back to local JSON cache.
        """
        # Try Google Calendar API
        try:
            cal_id = self._get_calendar_id()
            if cal_id:
                from src.integrations.google_service import get_google_service
                service = get_google_service()
                time_min = f"{date_str}T00:00:00-08:00"
                time_max = f"{date_str}T23:59:59-08:00"
                events = service.list_calendar_events(cal_id, time_min, time_max)
                if events:
                    # Normalize the events to our format
                    normalized = []
                    for e in events:
                        start = e.get('start', '')
                        end = e.get('end', '')
                        normalized.append({
                            'summary': e.get('summary', ''),
                            'start_time': start.split('T')[1][:5] if 'T' in start else start,
                            'end_time': end.split('T')[1][:5] if 'T' in end else end,
                            'description': e.get('description', ''),
                        })
                    return normalized
        except Exception as e:
            logger.debug(f"Could not fetch from Google Calendar: {e}")

        # Fall back to local JSON cache
        return self._fetch_events_from_cache(date_str)

    def _fetch_events_from_cache(self, date_str: str) -> List[Dict]:
        """Fall back to local JSON file cache."""
        plan_file = os.path.join(DAILY_PLANS_DIR, f'{date_str}.json')
        if os.path.exists(plan_file):
            try:
                with open(plan_file, 'r') as f:
                    plan = json.load(f)
                return plan.get('events', [])
            except Exception as e:
                logger.debug(f"Could not read cached plan for {date_str}: {e}")
        return []


# Singleton
_service: Optional[CalendarScheduleService] = None


def get_calendar_schedule_service() -> CalendarScheduleService:
    """Get or create the CalendarScheduleService singleton."""
    global _service
    if _service is None:
        _service = CalendarScheduleService()
    return _service


def is_calendar_schedule_enabled() -> bool:
    """Check if the calendar schedule feature is enabled."""
    return os.environ.get('COMPANION_CALENDAR_SCHEDULE_ENABLED', os.environ.get('COMPANION_CALENDAR_SCHEDULE_ENABLED', 'false')).lower() in ('true', '1', 'on')
