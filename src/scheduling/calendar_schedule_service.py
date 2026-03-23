"""
Calendar Schedule Service -- read-side facade for the companion's schedule events.

WHAT: Fetches today's (or any day's) events and formats them for system-prompt
      injection, reach-out eligibility checks, and time-passage narration.

WHY:  Multiple subsystems need the companion's schedule in different formats.
      This service centralises the read path with a three-tier fallback:
      PostgreSQL -> JSON cache -> Google Calendar.

HOW:  Primary:  PostgreSQL (companion_schedule_events table)
      Fallback: JSON files in data/companion_daily_plans/
      Last:     Google Calendar API (if sync is enabled)
      Cache:    In-memory TTL cache (10 min today, 1 hr recent)

Singleton: get_calendar_schedule_service() at module bottom.

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

# Daily plans directory (JSON cache)
DAILY_PLANS_DIR = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'companion_daily_plans'
)


class CalendarScheduleService:
    """Read-side service for the companion's schedule."""

    def __init__(self):
        self._today_cache = None  # (timestamp, events_list)
        self._recent_cache = None  # (timestamp, events_list)

    def get_today_events(self) -> List[Dict]:
        """
        Get today's events. Cached for 10 minutes.

        Tries: DB -> JSON cache -> Google Calendar.
        Also advances event statuses (planned -> in_progress -> completed).
        """
        if self._today_cache:
            cache_time, cached_events = self._today_cache
            if time.time() - cache_time < TODAY_EVENTS_CACHE_TTL:
                return cached_events

        from src.utils.timezone_utils import now_pacific_naive
        today = now_pacific_naive()
        date_str = today.strftime('%Y-%m-%d')

        # Advance statuses before reading
        try:
            from src.scheduling.calendar_schedule_generator import (
                advance_event_statuses
            )
            advance_event_statuses()
        except Exception as e:
            logger.debug(f"Could not advance event statuses: {e}")

        events = self._fetch_events_for_date(date_str)
        self._today_cache = (time.time(), events)
        return events

    def get_current_activity(self) -> Optional[Dict]:
        """Get the event that overlaps with the current time."""
        from src.utils.timezone_utils import now_pacific_naive
        now = now_pacific_naive()
        current_time = now.strftime('%H:%M')

        events = self.get_today_events()
        for event in events:
            start = event.get('start_time', event.get('start', ''))
            end = event.get('end_time', event.get('end', ''))
            if 'T' in start:
                start = start.split('T')[1][:5]
            if 'T' in end:
                end = end.split('T')[1][:5]
            if start <= current_time < end:
                return event
        return None

    def get_completed_events(self) -> List[Dict]:
        """Get today's events that have been completed (past end_time)."""
        from src.utils.timezone_utils import now_pacific_naive
        now = now_pacific_naive()
        current_time = now.strftime('%H:%M')

        events = self.get_today_events()
        completed = []
        for event in events:
            end = event.get('end_time', event.get('end', ''))
            if 'T' in end:
                end = end.split('T')[1][:5]
            if end <= current_time:
                completed.append(event)
        return completed

    def get_recent_events(self, days: int = 3) -> List[Dict]:
        """Get events from the past N days for narrative context."""
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
        """Get the next few events coming up."""
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

    def get_events_in_range(self, start: datetime, end: datetime) -> List[Dict]:
        """Get events within an arbitrary time range."""
        all_events = []
        current = start
        while current.date() <= end.date():
            date_str = current.strftime('%Y-%m-%d')
            day_events = self._fetch_events_for_date(date_str)
            for event in day_events:
                event_start = event.get('start_time', event.get('start', ''))
                if 'T' in event_start:
                    event_start = event_start.split('T')[1][:5]
                try:
                    event_dt = datetime.strptime(
                        f"{date_str} {event_start}", '%Y-%m-%d %H:%M'
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
        """Format events into an LLM-consumable text block."""
        if not events:
            return ""
        lines = []
        for event in events:
            start = event.get('start_time', event.get('start', ''))
            end = event.get('end_time', event.get('end', ''))
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
        """Format today's schedule into a concise context block."""
        from src.utils.timezone_utils import now_pacific_naive
        now = now_pacific_naive()
        current_time = now.strftime('%H:%M')

        events = self.get_today_events()
        if not events:
            return ""

        parts = []
        current = self.get_current_activity()
        if current:
            parts.append(f"Right now you're: {current.get('summary', 'busy')}")
            desc = current.get('description', '')
            if desc:
                parts.append(f"({desc})")

        last_event = None
        for event in events:
            end = event.get('end_time', event.get('end', ''))
            if 'T' in end:
                end = end.split('T')[1][:5]
            if end <= current_time:
                last_event = event

        if last_event and not current:
            parts.append(
                f"You just finished: {last_event.get('summary', 'something')}"
            )

        upcoming = self.get_upcoming_events(hours=3)
        if upcoming:
            next_events = [e.get('summary', '?') for e in upcoming[:2]]
            parts.append(f"Coming up: {', '.join(next_events)}")

        return '. '.join(parts) if parts else ''

    def format_schedule_behavior_context(self) -> str:
        """
        Format schedule into first-person behavioral guidance for the LLM.

        Instead of "Right now you're: Deep work", gives the LLM actual
        behavioral cues and first-person awareness of the day.
        """
        from src.utils.timezone_utils import now_pacific_naive

        now = now_pacific_naive()
        current_time = now.strftime('%H:%M')
        events = self.get_today_events()

        if not events:
            return ""

        current = self.get_current_activity()
        upcoming = self.get_upcoming_events(hours=2)
        completed = self.get_completed_events()

        parts = []

        # What you've done so far today (first-person awareness)
        if completed:
            done_summaries = [e.get('summary', '?') for e in completed[-3:]]
            parts.append(
                f"Earlier today you: {', '.join(done_summaries)}."
            )

        # Check for paused events (interrupted by conversation)
        paused = [e for e in events if e.get('status') == 'paused']
        if paused:
            last_paused = paused[-1]
            parts.append(
                f"You were doing {last_paused.get('summary', 'something')} "
                f"but paused to chat. You can go back to it when the conversation wraps up."
            )

        if current:
            summary = current.get('summary', 'something')
            end_time = current.get('end_time', current.get('end', ''))
            if 'T' in end_time:
                end_time = end_time.split('T')[1][:5]

            try:
                end_h, end_m = int(end_time[:2]), int(end_time[3:5])
                mins_left = (end_h * 60 + end_m) - (now.hour * 60 + now.minute)
            except (ValueError, IndexError):
                mins_left = 30

            desc = current.get('description', '')
            is_work = any(
                kw in summary.lower()
                for kw in ('work', 'deep', 'focus', 'code', 'design',
                           'review', 'write', 'draft', 'edit')
            )
            is_meeting = any(
                kw in summary.lower()
                for kw in ('meeting', 'call', 'sync', 'standup', '1:1')
            )

            if is_work:
                parts.append(
                    f"You're in the middle of work right now ({summary}). "
                    f"You have about {mins_left} minutes left on this block. "
                    f"If they chat casually, you can engage briefly but you "
                    f"genuinely have things to do — it's okay to say you're "
                    f"busy or need to get back to work."
                )
            elif is_meeting:
                parts.append(
                    f"You're in a meeting right now ({summary}). "
                    f"You might only be able to send quick texts."
                )
            else:
                parts.append(f"You're currently doing: {summary}")
                if desc:
                    parts.append(f"({desc})")
                if mins_left < 15:
                    parts.append(
                        f"Almost done — wrapping up in about "
                        f"{mins_left} minutes."
                    )
        else:
            # Not in any event — what just ended?
            last_event = None
            for event in events:
                end = event.get('end_time', event.get('end', ''))
                if 'T' in end:
                    end = end.split('T')[1][:5]
                if end <= current_time:
                    last_event = event

            if last_event:
                parts.append(
                    f"You just finished {last_event.get('summary', 'something')}. "
                    f"You're between activities — taking a breather."
                )
            elif events:
                first = events[0]
                start = first.get('start_time', first.get('start', ''))
                parts.append(
                    f"Your day hasn't officially started yet. "
                    f"First up: {first.get('summary', 'something')} at {start}."
                )

        # Upcoming events
        if upcoming:
            next_event = upcoming[0]
            next_name = next_event.get('summary', 'something')
            next_start = next_event.get('start_time', next_event.get('start', ''))
            if 'T' in next_start:
                next_start = next_start.split('T')[1][:5]

            try:
                s_h, s_m = int(next_start[:2]), int(next_start[3:5])
                mins_until = (s_h * 60 + s_m) - (now.hour * 60 + now.minute)
            except (ValueError, IndexError):
                mins_until = 60

            if mins_until <= 15:
                parts.append(
                    f"Heads up — you have {next_name} starting in about "
                    f"{mins_until} minutes."
                )
            elif mins_until <= 45:
                parts.append(f"Coming up soon: {next_name} at {next_start}.")

        return " ".join(parts)

    def _fetch_events_for_date(self, date_str: str) -> List[Dict]:
        """
        Fetch events for a date. Priority: DB -> JSON cache -> Google Calendar.
        """
        # Try PostgreSQL first
        try:
            from src.scheduling.calendar_schedule_generator import (
                get_events_from_db
            )
            db_events = get_events_from_db(date_str)
            if db_events:
                return db_events
        except Exception as e:
            logger.debug(f"Could not fetch from DB: {e}")

        # Try JSON cache
        events = self._fetch_events_from_cache(date_str)
        if events:
            return events

        # Try Google Calendar API (last resort)
        try:
            from src.scheduling.calendar_schedule_generator import (
                is_google_calendar_sync_enabled
            )
            if is_google_calendar_sync_enabled():
                cal_id = self._get_calendar_id()
                if cal_id:
                    from src.integrations.google_service import get_google_service
                    service = get_google_service()
                    time_min = f"{date_str}T00:00:00-08:00"
                    time_max = f"{date_str}T23:59:59-08:00"
                    raw = service.list_calendar_events(
                        cal_id, time_min, time_max
                    )
                    if raw:
                        return [
                            {
                                'summary': e.get('summary', ''),
                                'start_time': (
                                    e.get('start', '').split('T')[1][:5]
                                    if 'T' in e.get('start', '')
                                    else e.get('start', '')
                                ),
                                'end_time': (
                                    e.get('end', '').split('T')[1][:5]
                                    if 'T' in e.get('end', '')
                                    else e.get('end', '')
                                ),
                                'description': e.get('description', ''),
                            }
                            for e in raw
                        ]
        except Exception as e:
            logger.debug(f"Could not fetch from Google Calendar: {e}")

        return []

    def _get_calendar_id(self) -> Optional[str]:
        """Get the stored Google Calendar ID."""
        cal_id_file = os.path.join(
            os.environ.get('DATA_DIR', '/app/data'),
            'companion_calendar_id.txt'
        )
        if os.path.exists(cal_id_file):
            with open(cal_id_file, 'r') as f:
                cal_id = f.read().strip()
                return cal_id if cal_id else None
        return None

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
    return os.environ.get(
        'COMPANION_CALENDAR_SCHEDULE_ENABLED', 'false'
    ).lower() in ('true', '1', 'on')
