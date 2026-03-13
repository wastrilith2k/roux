"""
Calendar Schedule Generator -- produces rich daily events and writes them to Google Calendar.

WHAT: 5-step pipeline that turns the structural skeleton from CompanionSchedule
      (work hours, workload, meetings) into detailed, natural-language Google
      Calendar events and caches them locally as JSON for offline access.

WHY:  The companion's schedule needs to look realistic on a real Google Calendar
      (shared with the user) and be available to other subsystems (reach-out
      engine, time-passage narrator) even when the Google API is down.

HOW:  1. Load skeleton from CompanionSchedule
      2. Gather context (entity profiles, recent conversation topics, weather)
      3. Call LLM to generate specific event titles + descriptions
      4. Write events to the companion's dedicated Google Calendar via API
      5. Cache the plan locally in data/companion_daily_plans/<date>.json

Usage:
    from src.scheduling.calendar_schedule_generator import generate_daily_schedule

    events = generate_daily_schedule()  # today
    events = generate_daily_schedule(target_date=datetime(2026, 2, 24))
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to store the calendar ID
CALENDAR_ID_FILE = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'companion_calendar_id.txt'
)

# Path to store daily plans
DAILY_PLANS_DIR = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'companion_daily_plans'
)

# LLM model for generating schedule
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2-instruct-0905"


def get_or_create_companion_calendar() -> str:
    """
    Get or create the companion's dedicated Google Calendar.

    Idempotent - reads from file if already created, otherwise creates
    a new secondary calendar and stores the ID.

    Returns:
        Calendar ID string
    """
    # Check if we already have a calendar ID
    if os.path.exists(CALENDAR_ID_FILE):
        with open(CALENDAR_ID_FILE, 'r') as f:
            cal_id = f.read().strip()
            if cal_id:
                logger.debug(f"Using existing companion calendar: {cal_id}")
                return cal_id

    # Create a new secondary calendar
    from src.integrations.google_service import get_google_service
    service = get_google_service()
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    cal_id = service.create_secondary_calendar(f"{_pc.companion_short_name}'s Schedule")

    # Store the calendar ID
    os.makedirs(os.path.dirname(CALENDAR_ID_FILE), exist_ok=True)
    with open(CALENDAR_ID_FILE, 'w') as f:
        f.write(cal_id)

    logger.info(f"Created companion's calendar: {cal_id}")
    return cal_id


def _get_narrative_context() -> str:
    """
    Gather narrative context for LLM schedule generation.

    Fetches yesterday's summary, recent conversation snippets,
    and any work projects for richer event descriptions.
    """
    parts = []

    # Yesterday's daily summary (if available)
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Last few messages for conversational context
            cursor.execute("""
                SELECT sender_name, message_text, timestamp
                FROM messages
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            messages = cursor.fetchall()
            if messages:
                parts.append("Recent conversation:")
                for msg in reversed(messages):
                    text = (msg['message_text'] or '')[:100]
                    parts.append(f"  {msg['sender_name']}: {text}")
        conn.close()
    except Exception as e:
        logger.debug(f"Could not fetch conversation context: {e}")

    # Yesterday's calendar events (if we have them)
    from src.utils.timezone_utils import now_pacific_naive
    yesterday = now_pacific_naive() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    yesterday_plan_file = os.path.join(DAILY_PLANS_DIR, f'{yesterday_str}.json')
    if os.path.exists(yesterday_plan_file):
        try:
            with open(yesterday_plan_file, 'r') as f:
                yesterday_plan = json.load(f)
            event_summaries = [e.get('summary', '') for e in yesterday_plan.get('events', [])]
            if event_summaries:
                parts.append(f"\nYesterday's schedule included: {', '.join(event_summaries)}")
        except Exception:
            pass

    # User's calendar for today (so the companion can plan around their availability)
    try:
        from src.integrations.calendar_service import get_calendar_service, is_calendar_awareness_enabled
        if is_calendar_awareness_enabled():
            cal = get_calendar_service()
            james_events = cal.get_upcoming_events(hours_ahead=18, max_results=10)
            if james_events:
                parts.append("\nJames's calendar today:")
                for e in james_events:
                    start = e.get('start_display', '')
                    end = e.get('end_display', '')
                    summary = e.get('summary', 'event')
                    parts.append(f"  - {start}-{end}: {summary}")
                parts.append("(Consider James's availability when planning shared meals or evening time)")
    except Exception as e:
        logger.debug(f"Could not fetch James's calendar: {e}")

    return '\n'.join(parts) if parts else ''


def _generate_events_via_llm(
    skeleton: Dict,
    narrative_context: str,
    target_date: datetime,
) -> List[Dict]:
    """
    Use LLM to generate rich, specific calendar events from the structural skeleton.

    Args:
        skeleton: Output from CompanionSchedule.generate_daily_schedule()
        narrative_context: Context about recent conversation and yesterday
        target_date: The date being planned

    Returns:
        List of event dicts with summary, start, end, description, category
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv('FIREWORKS_API_KEY')
    )

    date_str = target_date.strftime('%Y-%m-%d')
    day_name = target_date.strftime('%A')
    is_workday = skeleton.get('is_workday', False)
    is_vacation = skeleton.get('is_vacation', False)
    workload = skeleton.get('workload', 'normal')

    # Build schedule skeleton description
    skeleton_desc = f"Day: {day_name}, {date_str}\n"
    if is_vacation:
        skeleton_desc += "This is a vacation day (day off from work).\n"
    elif not is_workday:
        skeleton_desc += f"This is a weekend day.\nWake time: {skeleton.get('wake_time', '07:00')}\n"
    else:
        skeleton_desc += (
            f"Workday with {workload} workload.\n"
            f"Wake time: {skeleton.get('wake_time', '06:00')}\n"
            f"Work: {skeleton.get('work_start', '09:00')} - {skeleton.get('work_end', '18:00')}\n"
            f"Lunch: {skeleton.get('lunch_start', '12:00')} - {skeleton.get('lunch_end', '13:00')}\n"
        )

    # Meetings
    meetings = skeleton.get('meetings', [])
    if meetings:
        skeleton_desc += "Meetings:\n"
        for m in meetings:
            skeleton_desc += f"  - {m.get('time')}: {m.get('topic')} ({m.get('duration')} min)\n"

    # Recurring activities
    recurring = skeleton.get('recurring_activities', [])
    if recurring:
        skeleton_desc += f"Recurring activities: {', '.join(recurring)}\n"

    # Work projects (if available)
    work_projects = skeleton.get('work_projects', [])
    if work_projects:
        skeleton_desc += "Current work projects:\n"
        for p in work_projects:
            skeleton_desc += f"  - {p.get('name', 'task')}: {p.get('description', '')}\n"

    prompt = f"""Generate the companion's daily schedule as calendar events.

The companion is a person who works in tech.
Generate a realistic daily schedule based on the skeleton below.

SCHEDULE SKELETON:
{skeleton_desc}

NARRATIVE CONTEXT (what's been happening recently):
{narrative_context or 'No recent context available.'}

Generate 8-12 calendar events for the day. Each event should be SPECIFIC and feel like a real person's schedule:

Rules:
- Events should cover wake-up to bedtime (~6 AM to ~10 PM)
- Include personal time (morning routine, meals, exercise, hobbies, wind-down)
- If workday: include work blocks with specific tasks (not just "working")
- Include meals (breakfast, lunch, dinner)
- Add 1-2 personal/hobby activities
- Make descriptions feel natural and specific, not generic
- Events should NOT overlap
- Use the meeting times from the skeleton if provided
- Respect the work hours from the skeleton

/no_think
Return as JSON array only, no other text:
[
  {{
    "summary": "short event title",
    "start_time": "HH:MM",
    "end_time": "HH:MM",
    "description": "1-2 sentence detail about what she's doing",
    "category": "personal|work|meal|exercise|social|meeting"
  }},
  ...
]"""

    response = client.chat.completions.create(
        model=FIREWORKS_MODEL,
        max_tokens=1500,
        temperature=0.85,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content.strip()

    # Clean up thinking tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
    content = content.strip()

    # Extract JSON from code blocks if present
    if '```' in content:
        content = content.split('```')[1]
        if content.startswith('json'):
            content = content[4:]
        content = content.strip()

    events = json.loads(content)

    if not isinstance(events, list) or len(events) == 0:
        raise ValueError("LLM returned invalid events format")

    # Validate and clean events
    cleaned = []
    for e in events:
        if not isinstance(e, dict) or 'summary' not in e:
            continue
        cleaned.append({
            'summary': e.get('summary', 'Activity'),
            'start_time': e.get('start_time', '09:00'),
            'end_time': e.get('end_time', '10:00'),
            'description': e.get('description', ''),
            'category': e.get('category', 'personal'),
        })

    logger.info(f"LLM generated {len(cleaned)} events for {date_str}")
    return cleaned


def clear_day_events(target_date: Optional[datetime] = None) -> int:
    """
    Delete existing events for a date on the companion's calendar.

    Idempotent - safe to call before regeneration.

    Args:
        target_date: Date to clear (default: today)

    Returns:
        Number of events deleted
    """
    from src.utils.timezone_utils import now_pacific_naive
    from src.integrations.google_service import get_google_service

    if target_date is None:
        target_date = now_pacific_naive()

    date_str = target_date.strftime('%Y-%m-%d')
    time_min = f"{date_str}T00:00:00-08:00"
    time_max = f"{date_str}T23:59:59-08:00"

    try:
        cal_id = get_or_create_companion_calendar()
        service = get_google_service()
        count = service.delete_calendar_events(cal_id, time_min, time_max)
        logger.info(f"Cleared {count} events for {date_str}")
        return count
    except Exception as e:
        logger.error(f"Failed to clear events for {date_str}: {e}")
        return 0


def generate_daily_schedule(target_date: Optional[datetime] = None, force: bool = False) -> Dict:
    """
    Generate the companion's daily schedule and write to Google Calendar.

    Main method:
    1. Get structural skeleton from CompanionSchedule
    2. Get narrative context
    3. LLM generates rich event list
    4. Write each event to Google Calendar
    5. Save to local cache

    Args:
        target_date: Date to generate for (default: today)
        force: If True, clear existing events and regenerate

    Returns:
        Dict with date, events list, and metadata
    """
    from src.utils.timezone_utils import now_pacific_naive
    from src.scheduling.companion_schedule import get_companion_schedule
    from src.integrations.google_service import get_google_service

    if target_date is None:
        target_date = now_pacific_naive()

    date_str = target_date.strftime('%Y-%m-%d')

    # Check if we already have a plan for today (unless forcing)
    os.makedirs(DAILY_PLANS_DIR, exist_ok=True)
    plan_file = os.path.join(DAILY_PLANS_DIR, f'{date_str}.json')
    if not force and os.path.exists(plan_file):
        logger.info(f"Daily plan already exists for {date_str}, skipping generation")
        with open(plan_file, 'r') as f:
            return json.load(f)

    logger.info(f"Generating daily schedule for {date_str} (force={force})")

    # Step 1: Get structural skeleton
    schedule = get_companion_schedule()
    skeleton = schedule.generate_daily_schedule(target_date, force=force)

    # Step 2: Get narrative context
    narrative_context = _get_narrative_context()

    # Step 3: LLM generates events
    events = _generate_events_via_llm(skeleton, narrative_context, target_date)

    # Step 4: Clear existing events if regenerating
    if force:
        clear_day_events(target_date)

    # Step 5: Write to Google Calendar
    cal_id = get_or_create_companion_calendar()
    service = get_google_service()
    calendar_event_ids = []

    for event in events:
        start_time = f"{date_str}T{event['start_time']}:00-08:00"
        end_time = f"{date_str}T{event['end_time']}:00-08:00"

        try:
            result = service.create_calendar_event(
                summary=event['summary'],
                start=start_time,
                end=end_time,
                description=event.get('description', ''),
                calendar_id=cal_id,
            )
            calendar_event_ids.append(result.get('id'))
        except Exception as e:
            logger.warning(f"Failed to create calendar event '{event['summary']}': {e}")

    # Step 6: Save to local cache
    plan_data = {
        'date': date_str,
        'day_name': target_date.strftime('%A'),
        'generated_at': now_pacific_naive().isoformat(),
        'calendar_id': cal_id,
        'calendar_event_ids': calendar_event_ids,
        'skeleton': {
            'is_workday': skeleton.get('is_workday', False),
            'is_vacation': skeleton.get('is_vacation', False),
            'workload': skeleton.get('workload'),
            'wake_time': skeleton.get('wake_time'),
            'work_start': skeleton.get('work_start'),
            'work_end': skeleton.get('work_end'),
        },
        'events': events,
    }

    with open(plan_file, 'w') as f:
        json.dump(plan_data, f, indent=2)

    logger.info(f"Daily schedule generated: {len(events)} events, {len(calendar_event_ids)} written to calendar")
    return plan_data
