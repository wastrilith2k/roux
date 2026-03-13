"""
Calendar Service -- cached read-side facade for the user's Google Calendar.

WHAT: Fetches events from ALL of the user's Google Calendars, normalises them
      into a common format, caches results for 5 minutes, and provides helper
      methods: time-until-next-event, is-user-busy, upcoming events list.

WHY:  Multiple subsystems need calendar data (context builder, reach-out engine,
      always-on service, time awareness). This service centralises the read
      path so we make one Google API call per 5-minute window instead of one
      per consumer.

HOW:  On `get_upcoming_events()`, checks the in-memory cache. If stale, calls
      GoogleService to list events from all calendars, merges and sorts them,
      caches the result. Events are normalised to dicts with start, end, summary,
      location, and calendar_name. Helper methods compute derived info (busy
      detection, minutes until next event).

Singleton: `get_calendar_service()` at module top.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# Module-level singleton
_calendar_service: Optional['CalendarService'] = None


def get_calendar_service() -> 'CalendarService':
    """Get or create the singleton CalendarService instance."""
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = CalendarService()
    return _calendar_service


def is_calendar_awareness_enabled() -> bool:
    """Check if calendar awareness feature is enabled."""
    return os.environ.get('COMPANION_CALENDAR_AWARENESS_ENABLED', os.environ.get('COMPANION_CALENDAR_AWARENESS_ENABLED', 'true')).lower() in ('true', '1', 'on')


class CalendarService:
    """Fetches and caches Google Calendar events via direct Google API."""

    CACHE_TTL_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._google_service = None
        self._cache: Dict[str, Tuple[float, List[Dict]]] = {}

    def _get_google_service(self):
        """Lazily initialize GoogleService for calendar reads.

        Uses the user's calendar token if available (credentials/user_calendar_token.json),
        otherwise falls back to the companion's token (credentials/google_token.json).
        """
        if self._google_service is None:
            try:
                from src.integrations.google_service import GoogleService
                from pathlib import Path

                # Check for user-specific calendar token
                user_token = os.environ.get('USER_CALENDAR_TOKEN_PATH')
                if not user_token:
                    # Check common locations
                    candidates = [
                        Path('/app/credentials/user_calendar_token.json'),
                        Path('/root/companion/credentials/user_calendar_token.json'),
                        Path(__file__).parent.parent.parent / 'credentials' / 'user_calendar_token.json',
                    ]
                    for p in candidates:
                        if p.exists():
                            user_token = str(p)
                            break

                if user_token and Path(user_token).exists():
                    self._google_service = GoogleService(token_path=user_token)
                    logger.info(f"Calendar service: using user's Google account (all calendars)")
                else:
                    self._google_service = GoogleService()
                    logger.info("Calendar service: using companion's Google account (no user token found)")
            except Exception as e:
                logger.error(f"Failed to initialize Google API for calendar: {e}")
                raise
        return self._google_service

    def _get_cache_key(self, hours_ahead: int, max_results: int) -> str:
        """Generate cache key from request params."""
        return f"{hours_ahead}:{max_results}"

    def _get_cached(self, cache_key: str) -> Optional[List[Dict]]:
        """Return cached events if still fresh, else None."""
        if cache_key in self._cache:
            cached_time, events = self._cache[cache_key]
            if time.time() - cached_time < self.CACHE_TTL_SECONDS:
                return events
            del self._cache[cache_key]
        return None

    def get_upcoming_events(self, hours_ahead: int = 24, max_results: int = 10) -> List[Dict]:
        """
        Fetch upcoming calendar events from ALL calendars, with caching.

        Returns normalized list of event dicts with keys:
        - summary: Event title
        - start: ISO datetime string
        - end: ISO datetime string
        - description: Event description (if any)
        - location: Event location (if any)
        - calendar_name: Which calendar this event is from
        - start_display: Human-readable start time (e.g., "2:30 PM")
        - end_display: Human-readable end time
        - is_now: Whether event is currently happening
        - time_until: Human-readable time until event starts (if in future)
        """
        if not is_calendar_awareness_enabled():
            return []

        cache_key = self._get_cache_key(hours_ahead, max_results)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Calendar cache hit for {cache_key}")
            return cached

        now = datetime.now(PST)
        time_min = now.isoformat()
        time_max = (now + timedelta(hours=hours_ahead)).isoformat()

        try:
            svc = self._get_google_service()
            raw_events = svc.list_all_calendar_events(
                time_min=time_min,
                time_max=time_max,
                max_results_per_calendar=max_results,
            )
        except Exception as e:
            logger.warning(f"Calendar fetch failed: {e}")
            return []

        events = self._normalize_events(raw_events, now)

        self._cache[cache_key] = (time.time(), events)
        logger.info(f"Calendar fetched {len(events)} events across all calendars (next {hours_ahead}h)")
        return events

    def _normalize_events(self, raw_events: List, now: datetime) -> List[Dict]:
        """Normalize Google Calendar event format into clean dicts."""
        events = []
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue

            # Google Calendar API returns start/end as objects with dateTime or date
            start_raw = raw.get('start', {})
            end_raw = raw.get('end', {})

            if isinstance(start_raw, dict):
                start_str = start_raw.get('dateTime', start_raw.get('date', ''))
            else:
                start_str = str(start_raw) if start_raw else ''

            if isinstance(end_raw, dict):
                end_str = end_raw.get('dateTime', end_raw.get('date', ''))
            else:
                end_str = str(end_raw) if end_raw else ''

            # Parse times for display
            start_dt = self._parse_datetime(start_str)
            end_dt = self._parse_datetime(end_str)

            is_now = False
            time_until = ""
            start_display = ""
            end_display = ""

            if start_dt:
                start_display = start_dt.strftime('%-I:%M %p')
                if end_dt and start_dt <= now <= end_dt:
                    is_now = True
                elif start_dt > now:
                    delta = start_dt - now
                    minutes = int(delta.total_seconds() / 60)
                    if minutes < 60:
                        time_until = f"in {minutes} min"
                    else:
                        hours = minutes // 60
                        remaining_min = minutes % 60
                        if remaining_min > 0:
                            time_until = f"in {hours}h {remaining_min}min"
                        else:
                            time_until = f"in {hours}h"

            if end_dt:
                end_display = end_dt.strftime('%-I:%M %p')

            events.append({
                'summary': raw.get('summary', 'Untitled Event'),
                'start': start_str,
                'end': end_str,
                'description': raw.get('description', ''),
                'location': raw.get('location', ''),
                'calendar_name': raw.get('calendar_name', ''),
                'start_display': start_display,
                'end_display': end_display,
                'is_now': is_now,
                'time_until': time_until,
            })

        return events

    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse ISO datetime string into timezone-aware datetime."""
        if not dt_str:
            return None
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.astimezone(PST)
        except (ValueError, TypeError):
            return None

    def peek_next_event(self) -> Optional[Dict]:
        """
        Lightweight: get the next upcoming event only.
        Uses cached data when available, or fetches minimal set.
        """
        events = self.get_upcoming_events(hours_ahead=8, max_results=5)
        # Find first event that hasn't ended yet
        for event in events:
            if not event.get('is_now') and event.get('time_until'):
                return event
        # If all current events are happening now, return the first one
        for event in events:
            if event.get('is_now'):
                return event
        return None

    def is_user_busy_now(self) -> Tuple[bool, Optional[str]]:
        """
        Quick check: is the user currently in a calendar event?

        Returns:
            (is_busy, event_name) - e.g., (True, "Dentist appointment")
        """
        events = self.get_upcoming_events(hours_ahead=2, max_results=5)
        for event in events:
            if event.get('is_now'):
                return True, event.get('summary', 'an event')
        return False, None

    # Backward compat alias
    # Backward compat alias removed

    def format_for_prompt(self, events: List[Dict]) -> str:
        """
        Format calendar events for LLM consumption.

        Returns a clean, readable block or empty string if no events.
        """
        if not events:
            return ""

        lines = []
        current_events = [e for e in events if e.get('is_now')]
        upcoming_events = [e for e in events if not e.get('is_now')]

        if current_events:
            for e in current_events:
                location_part = f" at {e['location']}" if e.get('location') else ""
                cal_part = f" [{e['calendar_name']}]" if e.get('calendar_name') else ""
                lines.append(f"- NOW: {e['summary']}{location_part} (until {e['end_display']}){cal_part}")

        for e in upcoming_events:
            time_part = f" ({e['time_until']})" if e.get('time_until') else ""
            location_part = f" at {e['location']}" if e.get('location') else ""
            cal_part = f" [{e['calendar_name']}]" if e.get('calendar_name') else ""
            lines.append(f"- {e['start_display']}: {e['summary']}{location_part}{time_part}{cal_part}")

        return "\n".join(lines)
