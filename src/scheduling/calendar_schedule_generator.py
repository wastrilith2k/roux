"""
Calendar Schedule Generator -- produces rich daily events, stored internally first.

WHAT: 5-step pipeline that turns the structural skeleton from CompanionSchedule
      (work hours, workload, meetings) into detailed, natural-language calendar
      events stored in PostgreSQL. Optionally syncs to Google Calendar.

WHY:  The schedule is the companion's own sense of their day. It must work
      without any external service. Google Calendar is a nice-to-have mirror.

HOW:  1. Load skeleton from CompanionSchedule
      2. Gather context (recent conversation, yesterday's events)
      3. Call LLM to generate specific event titles + descriptions
      4. Store events in PostgreSQL (companion_schedule_events)
      5. Optionally sync to Google Calendar if configured
      6. Cache locally as JSON for fast reads

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
from typing import Dict, List, Optional

from src.database import tables as T

logger = logging.getLogger(__name__)

# Path to store the Google Calendar ID (only used if sync is enabled)
CALENDAR_ID_FILE = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'companion_calendar_id.txt'
)

# Path to store daily plans (JSON cache for fast reads)
DAILY_PLANS_DIR = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'companion_daily_plans'
)

# LLM model for generating schedule
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_db_connection():
    """Get a psycopg2 connection using standard env vars."""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', '')
    )


def _ensure_schedule_table(conn) -> None:
    """Create the schedule events table if it doesn't exist (idempotent)."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS companion_schedule_events (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                summary TEXT NOT NULL,
                start_time TIME NOT NULL,
                end_time TIME NOT NULL,
                description TEXT DEFAULT '',
                category TEXT DEFAULT 'personal',
                status TEXT DEFAULT 'planned',
                google_event_id TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_schedule_events_date
                ON companion_schedule_events (date)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_schedule_events_date_status
                ON companion_schedule_events (date, status)
        """)
    conn.commit()


def store_events_to_db(date_str: str, events: List[Dict]) -> List[int]:
    """
    Store generated events to PostgreSQL.

    Replaces any existing events for the date. Returns list of row IDs.
    """
    conn = _get_db_connection()
    try:
        _ensure_schedule_table(conn)
        with conn.cursor() as cur:
            # Clear existing events for this date
            cur.execute(
                "DELETE FROM companion_schedule_events WHERE date = %s",
                (date_str,)
            )
            # Insert new events
            ids = []
            for event in events:
                cur.execute("""
                    INSERT INTO companion_schedule_events
                        (date, summary, start_time, end_time, description, category)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    date_str,
                    event.get('summary', 'Activity'),
                    event.get('start_time', '09:00'),
                    event.get('end_time', '10:00'),
                    event.get('description', ''),
                    event.get('category', 'personal'),
                ))
                ids.append(cur.fetchone()[0])
            conn.commit()
            logger.info(f"Stored {len(ids)} events to DB for {date_str}")
            return ids
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to store events to DB: {e}")
        return []
    finally:
        conn.close()


def get_events_from_db(date_str: str) -> List[Dict]:
    """
    Read events for a date from PostgreSQL.

    Returns list of event dicts, or empty list if none found.
    """
    conn = _get_db_connection()
    try:
        _ensure_schedule_table(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, summary, start_time, end_time, description,
                       category, status, google_event_id
                FROM companion_schedule_events
                WHERE date = %s
                ORDER BY start_time
            """, (date_str,))
            rows = cur.fetchall()
            return [
                {
                    'id': row[0],
                    'summary': row[1],
                    'start_time': row[2].strftime('%H:%M') if row[2] else '',
                    'end_time': row[3].strftime('%H:%M') if row[3] else '',
                    'description': row[4] or '',
                    'category': row[5] or 'personal',
                    'status': row[6] or 'planned',
                    'google_event_id': row[7],
                }
                for row in rows
            ]
    except Exception as e:
        logger.warning(f"Failed to read events from DB for {date_str}: {e}")
        return []
    finally:
        conn.close()


def update_event_status(event_id: int, status: str) -> bool:
    """
    Update the status of a schedule event.

    Valid statuses: planned, in_progress, completed, skipped, paused
    """
    if status not in ('planned', 'in_progress', 'completed', 'skipped', 'paused'):
        logger.warning(f"Invalid event status: {status}")
        return False

    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE companion_schedule_events
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (status, event_id))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update event status: {e}")
        return False
    finally:
        conn.close()


def advance_event_statuses() -> int:
    """
    Automatically advance event statuses based on current time.

    - Events whose end_time has passed: planned -> completed
    - Events whose start_time has passed but end_time hasn't: planned -> in_progress

    Returns number of events updated.
    """
    from src.utils.timezone_utils import now_pacific_naive
    now = now_pacific_naive()
    date_str = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M')

    conn = _get_db_connection()
    try:
        updated = 0
        with conn.cursor() as cur:
            # Mark completed: end_time has passed and still planned/in_progress
            cur.execute("""
                UPDATE companion_schedule_events
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE date = %s
                  AND end_time <= %s::time
                  AND status IN ('planned', 'in_progress')
            """, (date_str, current_time))
            updated += cur.rowcount

            # Mark in_progress: start_time passed, end_time hasn't, still planned
            cur.execute("""
                UPDATE companion_schedule_events
                SET status = 'in_progress', updated_at = CURRENT_TIMESTAMP
                WHERE date = %s
                  AND start_time <= %s::time
                  AND end_time > %s::time
                  AND status = 'planned'
            """, (date_str, current_time, current_time))
            updated += cur.rowcount

        conn.commit()
        if updated:
            logger.debug(f"Advanced {updated} event statuses for {date_str}")
        return updated
    except Exception as e:
        conn.rollback()
        logger.warning(f"Failed to advance event statuses: {e}")
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Google Calendar sync (optional)
# ---------------------------------------------------------------------------

def is_google_calendar_sync_enabled() -> bool:
    """Check if Google Calendar sync is enabled."""
    return os.environ.get(
        'COMPANION_CALENDAR_GOOGLE_SYNC', 'false'
    ).lower() in ('true', '1', 'on')


def get_or_create_companion_calendar() -> Optional[str]:
    """
    Get or create the companion's dedicated Google Calendar.

    Returns None if Google Calendar sync is disabled or fails.
    """
    if not is_google_calendar_sync_enabled():
        return None

    if os.path.exists(CALENDAR_ID_FILE):
        with open(CALENDAR_ID_FILE, 'r') as f:
            cal_id = f.read().strip()
            if cal_id:
                return cal_id

    try:
        from src.integrations.google_service import get_google_service
        from src.config.persona_config import get_persona_config
        service = get_google_service()
        _pc = get_persona_config()
        cal_id = service.create_secondary_calendar(
            f"{_pc.companion_short_name}'s Schedule"
        )
        os.makedirs(os.path.dirname(CALENDAR_ID_FILE), exist_ok=True)
        with open(CALENDAR_ID_FILE, 'w') as f:
            f.write(cal_id)
        logger.info(f"Created companion's Google Calendar: {cal_id}")
        return cal_id
    except Exception as e:
        logger.warning(f"Could not create Google Calendar: {e}")
        return None


def sync_events_to_google_calendar(
    date_str: str,
    events: List[Dict],
    force: bool = False
) -> List[str]:
    """
    Sync events to Google Calendar. Returns list of Google event IDs.

    This is a secondary sync — the DB is the source of truth.
    Fails gracefully and returns empty list if Google API is unavailable.
    """
    if not is_google_calendar_sync_enabled():
        return []

    cal_id = get_or_create_companion_calendar()
    if not cal_id:
        return []

    try:
        from src.integrations.google_service import get_google_service
        service = get_google_service()

        # Clear existing events if forcing
        if force:
            time_min = f"{date_str}T00:00:00-08:00"
            time_max = f"{date_str}T23:59:59-08:00"
            try:
                service.delete_calendar_events(cal_id, time_min, time_max)
            except Exception as e:
                logger.warning(f"Failed to clear Google Calendar events: {e}")

        google_ids = []
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
                google_ids.append(result.get('id', ''))
            except Exception as e:
                logger.warning(
                    f"Failed to sync event '{event['summary']}' to Google: {e}"
                )
                google_ids.append('')

        # Store Google event IDs back to DB
        if any(google_ids):
            _store_google_ids(date_str, events, google_ids)

        logger.info(
            f"Synced {sum(1 for g in google_ids if g)} events to Google Calendar"
        )
        return google_ids

    except Exception as e:
        logger.warning(f"Google Calendar sync failed: {e}")
        return []


def _store_google_ids(
    date_str: str, events: List[Dict], google_ids: List[str]
) -> None:
    """Store Google event IDs back to the DB rows."""
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            for event, gid in zip(events, google_ids):
                if gid and event.get('id'):
                    cur.execute("""
                        UPDATE companion_schedule_events
                        SET google_event_id = %s
                        WHERE id = %s
                    """, (gid, event['id']))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Failed to store Google event IDs: {e}")
    finally:
        conn.close()


def clear_day_events(target_date: Optional[datetime] = None) -> int:
    """
    Delete existing events for a date from both DB and Google Calendar.
    """
    from src.utils.timezone_utils import now_pacific_naive

    if target_date is None:
        target_date = now_pacific_naive()

    date_str = target_date.strftime('%Y-%m-%d')
    count = 0

    # Clear from DB
    conn = _get_db_connection()
    try:
        _ensure_schedule_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM companion_schedule_events WHERE date = %s",
                (date_str,)
            )
            count = cur.rowcount
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to clear DB events for {date_str}: {e}")
    finally:
        conn.close()

    # Clear from Google Calendar (best effort)
    if is_google_calendar_sync_enabled():
        try:
            cal_id = get_or_create_companion_calendar()
            if cal_id:
                from src.integrations.google_service import get_google_service
                service = get_google_service()
                time_min = f"{date_str}T00:00:00-08:00"
                time_max = f"{date_str}T23:59:59-08:00"
                service.delete_calendar_events(cal_id, time_min, time_max)
        except Exception as e:
            logger.warning(f"Failed to clear Google Calendar events: {e}")

    # Clear JSON cache
    plan_file = os.path.join(DAILY_PLANS_DIR, f'{date_str}.json')
    if os.path.exists(plan_file):
        try:
            os.remove(plan_file)
        except OSError:
            pass

    logger.info(f"Cleared {count} events for {date_str}")
    return count


# ---------------------------------------------------------------------------
# Narrative context (for LLM prompt)
# ---------------------------------------------------------------------------

def _get_narrative_context() -> str:
    """
    Gather narrative context for LLM schedule generation.

    Fetches yesterday's summary, recent conversation snippets,
    and any work projects for richer event descriptions.
    """
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()

    parts = []

    # Recent conversation
    try:
        conn = _get_db_connection()
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Last few messages for conversational context
            cursor.execute(f"""
                SELECT sender_name, message_text, timestamp
                FROM {T.MESSAGES}
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

    # Yesterday's events (from DB first, then JSON cache)
    from src.utils.timezone_utils import now_pacific_naive
    yesterday = now_pacific_naive() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')

    yesterday_events = get_events_from_db(yesterday_str)
    if not yesterday_events:
        # Fall back to JSON cache
        plan_file = os.path.join(DAILY_PLANS_DIR, f'{yesterday_str}.json')
        if os.path.exists(plan_file):
            try:
                with open(plan_file, 'r') as f:
                    yesterday_events = json.load(f).get('events', [])
            except Exception:
                pass

    if yesterday_events:
        summaries = [e.get('summary', '') for e in yesterday_events]
        parts.append(f"\nYesterday's schedule included: {', '.join(summaries)}")

    # User's calendar (so companion can plan around their availability)
    try:
        from src.integrations.calendar_service import (
            get_calendar_service, is_calendar_awareness_enabled
        )
        if is_calendar_awareness_enabled():
            cal = get_calendar_service()
            user_events = cal.get_upcoming_events(hours_ahead=18, max_results=10)
            if user_events:
                parts.append(f"\n{_pc.primary_user_name}'s calendar today:")
                for e in user_events:
                    start = e.get('start_display', '')
                    end = e.get('end_display', '')
                    summary = e.get('summary', 'event')
                    parts.append(f"  - {start}-{end}: {summary}")
    except Exception as e:
        logger.debug(f"Could not fetch user's calendar: {e}")

    return '\n'.join(parts) if parts else ''


# ---------------------------------------------------------------------------
# LLM event generation
# ---------------------------------------------------------------------------

def _generate_events_via_llm(
    skeleton: Dict,
    narrative_context: str,
    target_date: datetime,
) -> List[Dict]:
    """
    Use LLM to generate rich, specific calendar events from the skeleton.
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

    skeleton_desc = f"Day: {day_name}, {date_str}\n"
    if is_vacation:
        skeleton_desc += "This is a vacation day (day off from work).\n"
    elif not is_workday:
        skeleton_desc += (
            f"This is a weekend day.\n"
            f"Wake time: {skeleton.get('wake_time', '07:00')}\n"
        )
    else:
        skeleton_desc += (
            f"Workday with {workload} workload.\n"
            f"Wake time: {skeleton.get('wake_time', '06:00')}\n"
            f"Work: {skeleton.get('work_start', '09:00')} - "
            f"{skeleton.get('work_end', '18:00')}\n"
            f"Lunch: {skeleton.get('lunch_start', '12:00')} - "
            f"{skeleton.get('lunch_end', '13:00')}\n"
        )

    meetings = skeleton.get('meetings', [])
    if meetings:
        skeleton_desc += "Meetings:\n"
        for m in meetings:
            skeleton_desc += (
                f"  - {m.get('time')}: {m.get('topic')} "
                f"({m.get('duration')} min)\n"
            )

    recurring = skeleton.get('recurring_activities', [])
    if recurring:
        skeleton_desc += f"Recurring activities: {', '.join(recurring)}\n"

    work_projects = skeleton.get('work_projects', [])
    if work_projects:
        skeleton_desc += "Current work projects:\n"
        for p in work_projects:
            skeleton_desc += (
                f"  - {p.get('name', 'task')}: {p.get('description', '')}\n"
            )

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
    "description": "1-2 sentence detail about what they're doing",
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_daily_schedule(
    target_date: Optional[datetime] = None,
    force: bool = False
) -> Dict:
    """
    Generate the companion's daily schedule.

    Flow:
    1. Check DB for existing events (unless force=True)
    2. Get structural skeleton from CompanionSchedule
    3. Get narrative context
    4. LLM generates rich event list
    5. Store events to PostgreSQL (primary)
    6. Sync to Google Calendar (optional, secondary)
    7. Cache as JSON (fast-read layer)

    Args:
        target_date: Date to generate for (default: today)
        force: If True, clear existing events and regenerate

    Returns:
        Dict with date, events list, and metadata
    """
    from src.utils.timezone_utils import now_pacific_naive
    from src.scheduling.companion_schedule import get_companion_schedule

    if target_date is None:
        target_date = now_pacific_naive()

    date_str = target_date.strftime('%Y-%m-%d')

    # Check DB for existing events (unless forcing)
    if not force:
        existing = get_events_from_db(date_str)
        if existing:
            logger.info(
                f"Daily plan already exists in DB for {date_str} "
                f"({len(existing)} events), skipping generation"
            )
            return {
                'date': date_str,
                'day_name': target_date.strftime('%A'),
                'events': existing,
                'source': 'database',
            }

        # Also check JSON cache
        os.makedirs(DAILY_PLANS_DIR, exist_ok=True)
        plan_file = os.path.join(DAILY_PLANS_DIR, f'{date_str}.json')
        if os.path.exists(plan_file):
            logger.info(f"Daily plan exists in JSON cache for {date_str}")
            with open(plan_file, 'r') as f:
                return json.load(f)

    logger.info(f"Generating daily schedule for {date_str} (force={force})")

    # Clear existing if forcing
    if force:
        clear_day_events(target_date)

    # Step 1: Get structural skeleton
    schedule = get_companion_schedule()
    skeleton = schedule.generate_daily_schedule(target_date, force=force)

    # Step 1.5: Inject work projects into skeleton
    try:
        from src.autonomy.work_projects import get_work_project_manager
        wpm = get_work_project_manager()
        projects = wpm.format_for_schedule()
        if projects:
            skeleton['work_projects'] = projects
    except Exception as e:
        logger.debug(f"Could not load work projects for schedule: {e}")

    # Step 2: Get narrative context
    narrative_context = _get_narrative_context()

    # Step 3: LLM generates events
    events = _generate_events_via_llm(skeleton, narrative_context, target_date)

    # Step 4: Store to PostgreSQL (primary)
    db_ids = store_events_to_db(date_str, events)

    # Attach DB IDs to events for reference
    for event, db_id in zip(events, db_ids):
        event['id'] = db_id

    # Step 5: Sync to Google Calendar (optional, secondary)
    google_ids = sync_events_to_google_calendar(date_str, events, force=force)

    # Step 6: Save JSON cache
    os.makedirs(DAILY_PLANS_DIR, exist_ok=True)
    plan_data = {
        'date': date_str,
        'day_name': target_date.strftime('%A'),
        'generated_at': now_pacific_naive().isoformat(),
        'skeleton': {
            'is_workday': skeleton.get('is_workday', False),
            'is_vacation': skeleton.get('is_vacation', False),
            'workload': skeleton.get('workload'),
            'wake_time': skeleton.get('wake_time'),
            'work_start': skeleton.get('work_start'),
            'work_end': skeleton.get('work_end'),
        },
        'events': events,
        'source': 'generated',
    }

    plan_file = os.path.join(DAILY_PLANS_DIR, f'{date_str}.json')
    with open(plan_file, 'w') as f:
        json.dump(plan_data, f, indent=2)

    logger.info(
        f"Daily schedule generated: {len(events)} events, "
        f"{len(db_ids)} in DB, {sum(1 for g in google_ids if g)} in Google"
    )
    return plan_data
