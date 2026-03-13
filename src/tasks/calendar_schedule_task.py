"""
Calendar Schedule Task - Generate the companion's daily schedule.

WHAT: Uses the calendar_schedule_generator to produce the companion's daily
      plan (work blocks, breaks, personal time, etc.) and writes events to
      Google Calendar. The generated schedule drives the companion's
      availability and activity status throughout the day.

WHEN: Daily at 5:30 AM Pacific (before the companion's 6 AM "wake up").

WHY:  The companion needs a simulated daily structure. Without a schedule,
      she has no concept of "I'm at work right now" or "I have a meeting."
      The schedule feeds into autonomous_action_task (is it a good time to
      act?) and scene_extraction_task (current activity context).
"""

import logging
from datetime import datetime
from typing import Optional

from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name='tasks.calendar_schedule.generate_daily_plan',
    bind=True,
    max_retries=2,
    default_retry_delay=300,  # 5 minutes between retries
)
def generate_daily_plan(self, target_date: Optional[str] = None):
    """
    Generate the companion's daily plan and write to Google Calendar.

    Args:
        target_date: Date string in YYYY-MM-DD format (default: today)
    """
    from src.scheduling.calendar_schedule_service import is_calendar_schedule_enabled

    if not is_calendar_schedule_enabled():
        logger.info("Calendar schedule generation is disabled (COMPANION_CALENDAR_SCHEDULE_ENABLED != true)")
        return {'status': 'skipped', 'reason': 'disabled'}

    try:
        from src.scheduling.calendar_schedule_generator import generate_daily_schedule

        date = None
        if target_date:
            date = datetime.strptime(target_date, '%Y-%m-%d')

        result = generate_daily_schedule(target_date=date)

        event_count = len(result.get('events', []))
        logger.info(f"Daily plan generated: {result.get('date')} with {event_count} events")

        return {
            'status': 'success',
            'date': result.get('date'),
            'event_count': event_count,
        }

    except Exception as e:
        logger.error(f"Daily plan generation failed: {e}")
        import traceback
        traceback.print_exc()

        # Retry on transient failures
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            logger.error("Daily plan generation failed after max retries")
            return {
                'status': 'failed',
                'error': str(e),
            }
