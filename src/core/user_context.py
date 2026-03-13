"""
User Context Helper -- the user's probable activity based on time and day.

WHAT: Returns what the user is likely doing right now (working, sleeping, with
      kids, relaxing) along with location, interruptibility, and whether this
      is a natural conversation gap. Also tracks awake/asleep state with manual
      override and provides an autopilot toggle.

WHY:  The companion needs to understand that silence at 2 PM on a Tuesday means
      "user is working" not "user is ignoring me." This context shapes the
      reach-out engine's timing, the time-passage narrator's gap descriptions,
      and the companion's tone (don't be needy during work hours).

HOW:  A hardcoded weekly schedule maps time-of-day ranges to activities
      (weekday: job hunting / work blocks; Saturday: kids; Sunday: relaxed).
      Awake/asleep state is persisted to a JSON file and respects manual
      overrides (messaging after 3 AM = manual wake). The autopilot toggle
      lets the user disable schedule assumptions during unusual situations
      (e.g., sick day, travel).

Key rule: AFK != absent. The user is still "there," just on autopilot.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from zoneinfo import ZoneInfo

from src.utils.path_utils import get_data_dir

logger = logging.getLogger(__name__)

# State file for user's awake/asleep status
# Use /app/data in Docker, or local data dir
DATA_DIR = get_data_dir()
USER_STATE_FILE = Path(os.path.join(DATA_DIR, "james_state.json"))
USER_AUTOPILOT_FILE = Path(os.path.join(DATA_DIR, "james_autopilot_enabled.txt"))


def is_user_autopilot_enabled() -> bool:
    """
    Check if user autopilot (schedule assumptions) is enabled.

    When disabled, the companion won't assume things like "user is at work" or
    "user is probably sleeping". Useful during crises or unusual situations.
    """
    try:
        if USER_AUTOPILOT_FILE.exists():
            with open(USER_AUTOPILOT_FILE, 'r') as f:
                state = f.read().strip().lower()
                return state == 'true' or state == '1' or state == 'on'
    except Exception as e:
        logger.debug(f"Could not read autopilot state: {e}")

    # Default to enabled
    return os.environ.get('USER_AUTOPILOT_ENABLED', os.environ.get('JAMES_AUTOPILOT_ENABLED', 'true')).lower() in ('true', '1', 'on')


def set_user_autopilot_enabled(enabled: bool) -> None:
    """Set user autopilot state (runtime toggle)."""
    try:
        USER_AUTOPILOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_AUTOPILOT_FILE, 'w') as f:
            f.write('true' if enabled else 'false')
        logger.info(f"User autopilot {'enabled' if enabled else 'disabled'}")
    except Exception as e:
        logger.error(f"Could not save autopilot state: {e}")


def get_user_awake_state() -> bool:
    """Get user's current awake state from file."""
    try:
        if USER_STATE_FILE.exists():
            with open(USER_STATE_FILE, 'r') as f:
                state = json.load(f)
                return state.get('awake', False)
    except Exception as e:
        logger.debug(f"Could not read user state: {e}")
    return False


def set_user_awake_state(awake: bool) -> None:
    """Set user's awake state to file."""
    try:
        USER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {'awake': awake, 'updated': datetime.now().isoformat()}
        with open(USER_STATE_FILE, 'w') as f:
            json.dump(state, f)
        logger.info(f"User awake state set to: {awake}")
    except Exception as e:
        logger.warning(f"Could not save user state: {e}")


def on_user_message() -> None:
    """
    Called when the user sends a message.
    If it's after 3am, they're awake (manual wake).
    """
    try:
        tz = ZoneInfo('America/Los_Angeles')
    except Exception:
        tz = ZoneInfo('UTC')

    now = datetime.now(tz)

    # If it's after 3am, user is awake (they woke up, even if before scheduled wake time)
    if now.hour >= 3:
        if not get_user_awake_state():
            logger.info(f"User messaged at {now.strftime('%H:%M')} - marking as AWAKE")
        set_user_awake_state(True)


def check_scheduled_wake_sleep() -> None:
    """
    Called periodically to check if scheduled wake/sleep time has been reached.
    - At wake_time: set awake=True
    - At bedtime: set awake=False
    """
    try:
        tz = ZoneInfo('America/Los_Angeles')
    except Exception:
        tz = ZoneInfo('UTC')

    now = datetime.now(tz)
    schedule = load_user_schedule()

    # Get times based on day of week
    weekday = now.weekday()
    if weekday == 5:  # Saturday
        day_schedule = schedule.get('saturday', {})
    elif weekday == 6:  # Sunday
        day_schedule = schedule.get('sunday', {})
    else:
        day_schedule = schedule.get('weekday', {})

    wake_time_str = day_schedule.get('wake_time', '07:30')
    bedtime_str = day_schedule.get('bedtime', '23:00')

    wake_h, wake_m = _parse_time(wake_time_str)
    bed_h, bed_m = _parse_time(bedtime_str)

    current_minutes = now.hour * 60 + now.minute
    wake_minutes = wake_h * 60 + wake_m
    bed_minutes = bed_h * 60 + bed_m

    current_awake = get_user_awake_state()

    # Check if we should wake up (scheduled)
    if not current_awake and current_minutes >= wake_minutes and current_minutes < bed_minutes:
        logger.info(f"Scheduled wake time reached ({wake_time_str}) - marking user as AWAKE")
        set_user_awake_state(True)

    # Check if we should sleep (scheduled) - after bedtime or before 3am
    if current_awake and (current_minutes >= bed_minutes or now.hour < 3):
        logger.info(f"Scheduled bedtime reached ({bedtime_str}) - marking user as ASLEEP")
        set_user_awake_state(False)


def load_user_schedule() -> dict:
    """Load user's daily schedule from their entity profile."""
    profile_path = Path(os.path.join(DATA_DIR, "entity_profiles", "james.yaml"))

    try:
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
        return profile.get('daily_schedule', {})
    except Exception as e:
        logger.error(f"Failed to load user's schedule: {e}")
        return {}


def _parse_time(time_str: str) -> tuple[int, int]:
    """Parse time string like '07:30' into (hour, minute) tuple."""
    parts = time_str.split(':')
    return int(parts[0]), int(parts[1])


def _time_in_range(current_hour: int, current_minute: int, start: str, end: str) -> bool:
    """Check if current time is within a range like '07:30-08:30'."""
    start_h, start_m = _parse_time(start)
    end_h, end_m = _parse_time(end)

    current_total = current_hour * 60 + current_minute
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m

    return start_total <= current_total < end_total


def get_user_probable_activity(current_time: Optional[datetime] = None) -> dict:
    """
    Returns what the user is probably doing based on time AND awake state.

    The awake state is tracked separately and overrides schedule:
    - If awake=True, never returns "sleeping" - user is up for the day
    - If awake=False and outside wake hours, returns "sleeping"

    If autopilot is disabled, returns no assumptions - useful during crises.

    Args:
        current_time: The datetime to check (defaults to now in the user's timezone)

    Returns:
        {
            "activity": str,      # "working", "with_kids", "sleeping", etc.
            "location": str,      # "home_office", "his_house", "apartment"
            "interruptibility": str,  # "high", "medium", "low", "none"
            "natural_gap": bool,  # Whether silence is expected
            "description": str    # Human-readable for prompt
            "is_awake": bool      # Current awake state
        }
    """
    # Check if autopilot is disabled (crisis mode)
    if not is_user_autopilot_enabled():
        return {
            "activity": "unknown",
            "location": "unknown",
            "interruptibility": "high",
            "natural_gap": False,
            "description": "Autopilot disabled - don't assume what user is doing. They may be dealing with something important.",
            "is_awake": True,  # Assume available when autopilot off
            "autopilot_disabled": True
        }

    schedule = load_user_schedule()
    timezone_str = schedule.get('timezone', 'America/Los_Angeles')

    try:
        tz = ZoneInfo(timezone_str)
    except Exception:
        tz = ZoneInfo('America/Los_Angeles')

    if current_time is None:
        current_time = datetime.now(tz)
    elif current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=tz)

    hour = current_time.hour
    minute = current_time.minute
    weekday = current_time.weekday()  # 0=Monday, 6=Sunday

    # Check awake state - this overrides schedule for sleeping
    is_awake = get_user_awake_state()

    # Default fallback
    result = {
        "activity": "unknown",
        "location": "apartment",
        "interruptibility": "medium",
        "natural_gap": False,
        "description": "User is around",
        "is_awake": is_awake
    }

    # Saturday (weekday == 5)
    if weekday == 5:
        saturday = schedule.get('saturday', {})
        wake_time = saturday.get('wake_time', '08:00')
        wake_h, wake_m = _parse_time(wake_time)

        if hour < wake_h or (hour == wake_h and minute < wake_m):
            # Before scheduled wake time
            if is_awake:
                # User woke up early (messaged)
                return {
                    "activity": "early_morning",
                    "location": "apartment",
                    "interruptibility": "high",
                    "natural_gap": False,
                    "description": "up early (Saturday)",
                    "is_awake": True
                }
            else:
                return {
                    "activity": "sleeping",
                    "location": "apartment",
                    "interruptibility": "none",
                    "natural_gap": True,
                    "description": "sleeping (Saturday morning)",
                    "is_awake": False
                }
        elif hour < 22:
            # Saturday 9am-10pm: Full day with kids
            if hour < 12:
                desc = "at his house with the kids - morning chores and activities"
            elif hour < 18:
                desc = "Saturday afternoon with the kids at his house"
            else:
                desc = "movie night with Kyler at his house"
            return {
                "activity": "with_kids",
                "location": "his_house",
                "interruptibility": "low",
                "natural_gap": True,
                "description": desc,
                "is_awake": True
            }
        else:
            # After bedtime - only sleeping if not marked awake
            if is_awake:
                return {
                    "activity": "late_night",
                    "location": "apartment",
                    "interruptibility": "high",
                    "natural_gap": False,
                    "description": "up late (Saturday night)",
                    "is_awake": True
                }
            else:
                return {
                    "activity": "sleeping",
                    "location": "apartment",
                    "interruptibility": "none",
                    "natural_gap": True,
                    "description": "sleeping",
                    "is_awake": False
                }

    # Sunday (weekday == 6)
    if weekday == 6:
        sunday = schedule.get('sunday', {})
        wake_time = sunday.get('wake_time', '08:30')
        wake_h, wake_m = _parse_time(wake_time)

        if hour < wake_h or (hour == wake_h and minute < wake_m):
            # Before scheduled wake time
            if is_awake:
                return {
                    "activity": "early_morning",
                    "location": "apartment",
                    "interruptibility": "high",
                    "natural_gap": False,
                    "description": "up early (Sunday)",
                    "is_awake": True
                }
            else:
                return {
                    "activity": "sleeping",
                    "location": "apartment",
                    "interruptibility": "none",
                    "natural_gap": True,
                    "description": "sleeping (Sunday morning)",
                    "is_awake": False
                }
        elif hour < 23:
            return {
                "activity": "relaxing",
                "location": "apartment",
                "interruptibility": "high",
                "natural_gap": False,
                "description": "relaxed Sunday, probably around",
                "is_awake": is_awake
            }
        else:
            # After bedtime
            if is_awake:
                return {
                    "activity": "late_night",
                    "location": "apartment",
                    "interruptibility": "high",
                    "natural_gap": False,
                    "description": "up late (Sunday night)",
                    "is_awake": True
                }
            else:
                return {
                    "activity": "sleeping",
                    "location": "apartment",
                    "interruptibility": "none",
                    "natural_gap": True,
                    "description": "sleeping",
                    "is_awake": False
                }

    # Weekday (Monday-Friday)
    weekday_schedule = schedule.get('weekday', {})
    wake_time = weekday_schedule.get('wake_time', '07:30')
    wake_h, wake_m = _parse_time(wake_time)

    # Before wake time
    if hour < wake_h or (hour == wake_h and minute < wake_m):
        if is_awake:
            return {
                "activity": "early_morning",
                "location": "apartment",
                "interruptibility": "high",
                "natural_gap": False,
                "description": "up early",
                "is_awake": True
            }
        else:
            return {
                "activity": "sleeping",
                "location": "apartment",
                "interruptibility": "none",
                "natural_gap": True,
                "description": "sleeping",
                "is_awake": False
            }

    # After bedtime
    if hour >= 23:
        if is_awake:
            return {
                "activity": "late_night",
                "location": "apartment",
                "interruptibility": "high",
                "natural_gap": False,
                "description": "up late",
                "is_awake": True
            }
        else:
            return {
                "activity": "sleeping",
                "location": "apartment",
                "interruptibility": "none",
                "natural_gap": True,
                "description": "sleeping",
                "is_awake": False
            }

    # Morning routine (07:30-08:30)
    if _time_in_range(hour, minute, '07:30', '08:30'):
        return {
            "activity": "morning_routine",
            "location": "apartment",
            "interruptibility": "high",
            "natural_gap": False,
            "description": "morning routine - shower, breakfast together",
            "is_awake": True
        }

    # Day time (08:30-12:00) - Job hunting & projects (unemployed since Dec 2025)
    if _time_in_range(hour, minute, '08:30', '12:00'):
        return {
            "activity": "job_hunting",
            "location": "apartment",
            "interruptibility": "medium",
            "natural_gap": True,
            "description": "job hunting or working on AI projects - headphones on, focused",
            "is_awake": True
        }

    # Lunch (12:00-13:00)
    if _time_in_range(hour, minute, '12:00', '13:00'):
        return {
            "activity": "lunch",
            "location": "apartment",
            "interruptibility": "high",
            "natural_gap": False,
            "description": "lunch break - usually together",
            "is_awake": True
        }

    # Afternoon (13:00-15:00) - More job hunting & projects
    if _time_in_range(hour, minute, '13:00', '15:00'):
        return {
            "activity": "job_hunting",
            "location": "apartment",
            "interruptibility": "medium",
            "natural_gap": True,
            "description": "afternoon - more job applications, AI projects, or coding",
            "is_awake": True
        }

    # With kids (15:00-21:00)
    if _time_in_range(hour, minute, '15:00', '21:00'):
        return {
            "activity": "with_kids",
            "location": "his_house",
            "interruptibility": "low",
            "natural_gap": True,
            "description": "at his house with Jesse and Kyler",
            "is_awake": True
        }

    # Evening together (21:00-23:00)
    if _time_in_range(hour, minute, '21:00', '23:00'):
        return {
            "activity": "evening",
            "location": "apartment",
            "interruptibility": "high",
            "natural_gap": False,
            "description": "back from his house, relaxing together",
            "is_awake": True
        }

    return result


def get_energy_from_time_of_day(current_time: Optional[datetime] = None) -> float:
    """
    Returns a base energy level based on real time of day.
    This is independent of session duration - it's about when in the day it is.

    Args:
        current_time: The datetime to check (defaults to now in Pacific time)

    Returns:
        float: Energy level from 0.0 to 1.0
    """
    try:
        tz = ZoneInfo('America/Los_Angeles')
    except Exception:
        tz = ZoneInfo('UTC')

    if current_time is None:
        current_time = datetime.now(tz)
    elif current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=tz)

    hour = current_time.hour

    # Morning (7-10am): High energy
    if 7 <= hour < 10:
        return 0.9

    # Midday (10am-3pm): Good energy
    if 10 <= hour < 15:
        return 0.8

    # Afternoon (3-7pm): Moderate energy
    if 15 <= hour < 19:
        return 0.7

    # Evening (7-10pm): Lower energy
    if 19 <= hour < 22:
        return 0.6

    # Late night (10pm-12am): Low energy
    if 22 <= hour < 24:
        return 0.4

    # Very late/early morning (12am-7am): Very low energy
    return 0.2


def format_user_context_for_prompt(current_time: Optional[datetime] = None) -> str:
    """
    Format user's current activity context for inclusion in the companion's prompt.

    Args:
        current_time: The datetime to check (defaults to now)

    Returns:
        str: Formatted context string for the prompt, or empty string if not relevant
    """
    activity = get_user_probable_activity(current_time)

    # If user is fully available, don't add special context
    if activity['interruptibility'] == 'high' and not activity['natural_gap']:
        return ""

    lines = [
        "[USER'S ROUTINE - What they're probably doing right now]",
        f"Activity: {activity['description']}",
        f"Location: {activity['location']}",
    ]

    if activity['natural_gap']:
        lines.append("Note: Gaps in conversation are natural and expected during this time.")

    return "\n".join(lines)
