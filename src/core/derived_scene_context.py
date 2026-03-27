"""
Derived Scene Context — replaces decorative scene labels with real data.

WHAT: Composes a unified context block from real signals: time of day,
      schedule/calendar, presence mode, and conversation history. This gives
      the companion grounded awareness of the current situation instead of
      relying on decorative labels like "LIVING ROOM - EVENING".

WHY:  The companion shouldn't need to be told what's happening — it should know.
      By deriving scene context from real data (device clock, learned schedule,
      presence mode), the companion can reason about the user's situation
      accurately and adjust tone, pacing, and expectations.

HOW:  `DerivedSceneContextBuilder.build()` pulls from:
      1. Time of day — from derive_temporal_context() or server clock
      2. Schedule/calendar — what the user is likely doing (ScheduleContextProvider)
      3. Presence mode — In Person vs SMS (PresenceModeManager)
      4. Conversation recency — last message timestamps

      It outputs a structured [Context] block and a list of behavioral
      constraints the companion should follow.

Singleton: get_derived_scene_context_builder() at module bottom.
"""

import logging
import threading
from datetime import datetime
from typing import Optional, List

from src.core.clock import now as clock_now
from src.core.time_awareness import time_of_day_label

logger = logging.getLogger(__name__)

# Module-level singleton
_builder: Optional['DerivedSceneContextBuilder'] = None
_builder_lock = threading.Lock()


def get_derived_scene_context_builder() -> 'DerivedSceneContextBuilder':
    """Get or create the singleton DerivedSceneContextBuilder."""
    global _builder
    if _builder is None:
        with _builder_lock:
            if _builder is None:
                _builder = DerivedSceneContextBuilder()
    return _builder


def _infer_location(time_of_day: str, is_weekend: bool,
                    schedule_hints: List[str], presence_mode: str) -> str:
    """
    Infer probable location from available signals.

    Returns a human-readable location inference with reasoning.
    """
    schedule_lower = ' '.join(schedule_hints).lower()

    # Schedule mentions override time-based guesses
    if any(word in schedule_lower for word in ['office', 'work', 'commute', 'commuting']):
        return 'Likely at work (based on schedule)'
    if any(word in schedule_lower for word in ['gym', 'exercise', 'workout']):
        return 'Likely at gym (based on schedule)'
    if any(word in schedule_lower for word in ['school', 'class', 'university']):
        return 'Likely at school (based on schedule)'
    if any(word in schedule_lower for word in ['store', 'shopping', 'errand']):
        return 'Likely out (based on schedule)'

    # Time-based defaults
    if is_weekend:
        return 'Home (weekend, no events)'
    if time_of_day == 'night':
        return 'Home (based on time)'
    if time_of_day == 'morning' and not schedule_hints:
        return 'Home (based on time, no events)'
    if time_of_day in ('afternoon', 'evening') and not schedule_hints:
        return 'Home (based on time, no events)'

    return 'Unknown'


def _infer_user_status(time_of_day: str, is_weekend: bool,
                       schedule_hints: List[str]) -> str:
    """
    Infer user availability/status from signals.

    Returns a short status string.
    """
    schedule_lower = ' '.join(schedule_hints).lower()

    # Check for explicit busy signals
    if any(word in schedule_lower for word in ['meeting', 'busy', 'unavailable']):
        return 'Busy (based on schedule)'
    if any(word in schedule_lower for word in ['work', 'office', 'commute']):
        return 'At work (may be slow to respond)'

    # Time-based status
    if time_of_day == 'night':
        return 'Winding down (late)'
    if is_weekend:
        return 'Available (weekend)'

    return 'Available'


def _derive_behavioral_constraints(
    time_of_day: str,
    presence_mode: str,
    user_status: str,
    schedule_hints: List[str],
) -> List[str]:
    """
    Derive behavioral constraints from the current scene context.

    Returns a list of constraint strings that guide companion behavior.
    """
    constraints = []

    # Presence mode constraints
    if presence_mode == 'texting':
        constraints.append(
            'SMS mode: no physical actions, no shared environment descriptions'
        )

    # Time-based constraints
    if time_of_day == 'night':
        constraints.append(
            'Late night: user may be tired, be mindful of energy levels'
        )

    # Status-based constraints
    if 'work' in user_status.lower() or 'busy' in user_status.lower():
        constraints.append(
            'User may be busy: keep responses concise, '
            'don\'t expect immediate replies'
        )

    return constraints


class DerivedSceneContextBuilder:
    """
    Builds a derived scene context block from real signals.

    Replaces decorative scene labels with data-driven context that
    the companion can actually reason about.
    """

    def build(
        self,
        user_email: str,
        client_temporal: Optional[dict] = None,
        presence_mode_str: Optional[str] = None,
        schedule_context: Optional[str] = None,
    ) -> str:
        """
        Assemble the derived scene context block.

        Args:
            user_email: User's email for state lookups
            client_temporal: Output of derive_temporal_context() if available
            presence_mode_str: Raw presence mode context string (from PresenceModeManager)
            schedule_context: Raw schedule/time awareness context string

        Returns:
            Formatted context block string, or empty string if no signals available.
        """
        # 1. Resolve time context
        time_data = self._resolve_time(client_temporal)
        if not time_data:
            return ""

        time_of_day = time_data['time_of_day']
        is_weekend = time_data['is_weekend']
        formatted_time = time_data['formatted_time']
        day_of_week = time_data['day_of_week']

        # 2. Resolve presence mode
        presence = self._resolve_presence(presence_mode_str)

        # 3. Extract schedule hints
        schedule_hints = self._extract_schedule_hints(schedule_context)

        # 4. Derive location and status
        location = _infer_location(time_of_day, is_weekend, schedule_hints, presence)
        user_status = _infer_user_status(time_of_day, is_weekend, schedule_hints)

        # 5. Derive behavioral constraints
        constraints = _derive_behavioral_constraints(
            time_of_day, presence, user_status, schedule_hints
        )

        # 6. Format the context block
        return self._format_context_block(
            formatted_time=formatted_time,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
            presence=presence,
            location=location,
            user_status=user_status,
            constraints=constraints,
        )

    def _resolve_time(self, client_temporal: Optional[dict] = None) -> Optional[dict]:
        """
        Resolve time context from client temporal data or server clock.

        Returns dict with time_of_day, is_weekend, formatted_time, day_of_week
        or None if resolution fails.
        """
        if client_temporal and client_temporal.get('time_of_day'):
            return {
                'time_of_day': client_temporal['time_of_day'],
                'is_weekend': client_temporal.get('is_weekend', False),
                'formatted_time': client_temporal.get('formatted', ''),
                'day_of_week': client_temporal.get('day_of_week', ''),
            }

        # Fall back to server clock
        try:
            now = clock_now()
            hour = now.hour
            time_of_day = time_of_day_label(hour)
            is_weekend = now.weekday() >= 5
            day_of_week = now.strftime('%A')
            formatted_time = now.strftime('%A, %-I:%M %p')

            return {
                'time_of_day': time_of_day,
                'is_weekend': is_weekend,
                'formatted_time': formatted_time,
                'day_of_week': day_of_week,
            }
        except Exception as e:
            logger.warning(f"Failed to resolve time context: {e}")
            return None

    def _resolve_presence(self, presence_mode_str: Optional[str] = None) -> str:
        """
        Resolve presence mode from the presence mode context string.

        Returns 'in_person' or 'texting'.
        """
        if presence_mode_str and 'TEXTING' in presence_mode_str:
            return 'texting'
        return 'in_person'

    def _extract_schedule_hints(self, schedule_context: Optional[str] = None) -> List[str]:
        """
        Extract actionable hints from the schedule/time awareness context string.

        Returns a list of schedule-related strings (calendar events, routine items).
        """
        if not schedule_context:
            return []

        hints = []
        for line in schedule_context.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Skip header lines and formatting
            if line.startswith('[') or line.startswith('Current time:'):
                continue
            # Keep lines that contain actual schedule information
            if line.startswith('-') or line.startswith('His routine:') or ':' in line:
                hints.append(line.lstrip('- '))

        return hints

    def _format_context_block(
        self,
        formatted_time: str,
        day_of_week: str,
        time_of_day: str,
        presence: str,
        location: str,
        user_status: str,
        constraints: List[str],
    ) -> str:
        """Format the derived context into a structured block for the prompt."""
        presence_label = 'In Person' if presence == 'in_person' else 'Texting (SMS)'

        lines = [
            '[Derived Context]',
            f'Time: {formatted_time} ({time_of_day})',
            f'Presence: {presence_label}',
            f'Location inference: {location}',
            f'User status: {user_status}',
        ]

        if constraints:
            lines.append('')
            lines.append('Behavioral constraints:')
            for c in constraints:
                lines.append(f'- {c}')

        return '\n'.join(lines)
