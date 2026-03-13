"""
Unified Scheduler -- centralised APScheduler wrapper for all timed tasks.

WHAT: Thread-safe singleton that wraps APScheduler's BackgroundScheduler,
      providing interval, cron, and one-shot job registration plus stats
      tracking (execution counts, errors, durations).

WHY:  Before this existed, proactive messages, memory cleanup, reminder checks,
      and delayed responses each ran their own timer. Consolidating into one
      scheduler avoids thread-leak bugs and makes it trivial to list/cancel jobs
      from the ops bot.

HOW:  APScheduler runs in a daemon thread. Jobs are registered via
      `add_interval_job`, `add_cron_job`, or `add_one_time_job`. An event
      listener tracks successes/failures in `self._stats`. All times use
      Pacific timezone.

Singleton: `get_unified_scheduler()` at module bottom (thread-safe via lock).
"""
import threading
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from src.utils.timezone_utils import now_pacific_naive
import pytz


class UnifiedScheduler:
    """
    Centralized scheduler for all timed tasks in the companion framework.

    Uses APScheduler for robust job scheduling with:
    - Interval triggers (every N minutes/hours)
    - Cron triggers (specific times/days)
    - One-time triggers (delayed execution)
    - Job persistence and monitoring
    """

    def __init__(self):
        # Pacific timezone for all scheduling
        self.pacific_tz = pytz.timezone('America/Los_Angeles')

        # Create background scheduler
        self.scheduler = BackgroundScheduler(timezone=self.pacific_tz)

        # Job statistics
        self.job_stats = {
            'total_executions': 0,
            'total_errors': 0,
            'last_error': None
        }
        self.stats_lock = threading.Lock()

        # Setup event listeners for monitoring
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)

        print("🔧 Unified Scheduler initialized")

    def _on_job_executed(self, event):
        """Track successful job executions"""
        with self.stats_lock:
            self.job_stats['total_executions'] += 1

    def _on_job_error(self, event):
        """Track job errors"""
        with self.stats_lock:
            self.job_stats['total_errors'] += 1
            self.job_stats['last_error'] = {
                'job_id': event.job_id,
                'exception': str(event.exception),
                'timestamp': datetime.now(self.pacific_tz).isoformat()
            }
            print(f"⚠️  Scheduler job error ({event.job_id}): {event.exception}")

    # =========================================================================
    # Interval Scheduling
    # =========================================================================

    def schedule_interval(
        self,
        job_id: str,
        func: Callable,
        minutes: Optional[int] = None,
        hours: Optional[int] = None,
        seconds: Optional[int] = None,
        replace_existing: bool = True,
        **kwargs
    ) -> bool:
        """
        Schedule a job to run at regular intervals.

        Args:
            job_id: Unique identifier for the job
            func: Function to execute
            minutes: Interval in minutes
            hours: Interval in hours
            seconds: Interval in seconds
            replace_existing: Whether to replace existing job with same ID
            **kwargs: Additional arguments to pass to func

        Returns:
            True if scheduled successfully
        """
        try:
            trigger = IntervalTrigger(
                minutes=minutes or 0,
                hours=hours or 0,
                seconds=seconds or 0,
                timezone=self.pacific_tz
            )

            self.scheduler.add_job(
                func=func,
                trigger=trigger,
                id=job_id,
                name=job_id,
                replace_existing=replace_existing,
                kwargs=kwargs
            )

            interval_desc = []
            if hours: interval_desc.append(f"{hours}h")
            if minutes: interval_desc.append(f"{minutes}m")
            if seconds: interval_desc.append(f"{seconds}s")

            print(f"✓ Scheduled '{job_id}' every {' '.join(interval_desc)}")
            return True

        except Exception as e:
            print(f"✗ Failed to schedule '{job_id}': {e}")
            return False

    def register_interval_job(
        self,
        job_id: str,
        func: Callable,
        minutes: Optional[int] = None,
        hours: Optional[int] = None,
        seconds: Optional[int] = None,
        description: str = None,
        **kwargs
    ) -> bool:
        """
        Alias for schedule_interval with a description parameter.
        Used by web_chat.py for registering jobs.
        """
        if description:
            print(f"📋 Registering job: {description}")
        return self.schedule_interval(
            job_id=job_id,
            func=func,
            minutes=minutes,
            hours=hours,
            seconds=seconds,
            **kwargs
        )

    # =========================================================================
    # Cron Scheduling
    # =========================================================================

    def schedule_cron(
        self,
        job_id: str,
        func: Callable,
        hour: Optional[int] = None,
        minute: Optional[int] = None,
        day_of_week: Optional[str] = None,
        replace_existing: bool = True,
        **kwargs
    ) -> bool:
        """
        Schedule a job using cron-style timing.

        Args:
            job_id: Unique identifier for the job
            func: Function to execute
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            day_of_week: Day(s) to run (mon,tue,wed,thu,fri,sat,sun)
            replace_existing: Whether to replace existing job with same ID
            **kwargs: Additional arguments to pass to func

        Returns:
            True if scheduled successfully
        """
        try:
            trigger = CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week=day_of_week,
                timezone=self.pacific_tz
            )

            self.scheduler.add_job(
                func=func,
                trigger=trigger,
                id=job_id,
                name=job_id,
                replace_existing=replace_existing,
                kwargs=kwargs
            )

            print(f"✓ Scheduled '{job_id}' at {hour:02d}:{minute:02d} Pacific")
            return True

        except Exception as e:
            print(f"✗ Failed to schedule '{job_id}': {e}")
            return False

    # =========================================================================
    # One-Time/Delayed Scheduling
    # =========================================================================

    def schedule_once(
        self,
        job_id: str,
        func: Callable,
        run_date: Optional[datetime] = None,
        delay_seconds: Optional[float] = None,
        replace_existing: bool = True,
        **kwargs
    ) -> bool:
        """
        Schedule a job to run once at a specific time or after a delay.

        Args:
            job_id: Unique identifier for the job
            func: Function to execute
            run_date: Specific datetime to run (naive, Pacific time)
            delay_seconds: OR delay in seconds from now
            replace_existing: Whether to replace existing job with same ID
            **kwargs: Additional arguments to pass to func

        Returns:
            True if scheduled successfully
        """
        try:
            if delay_seconds is not None:
                run_date = datetime.now(self.pacific_tz) + timedelta(seconds=delay_seconds)
            elif run_date is not None:
                # Convert naive datetime to Pacific timezone
                run_date = self.pacific_tz.localize(run_date)
            else:
                raise ValueError("Must specify either run_date or delay_seconds")

            trigger = DateTrigger(run_date=run_date, timezone=self.pacific_tz)

            self.scheduler.add_job(
                func=func,
                trigger=trigger,
                id=job_id,
                name=job_id,
                replace_existing=replace_existing,
                kwargs=kwargs
            )

            print(f"✓ Scheduled '{job_id}' for {run_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return True

        except Exception as e:
            print(f"✗ Failed to schedule '{job_id}': {e}")
            return False

    # =========================================================================
    # Job Management
    # =========================================================================

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a scheduled job.

        Args:
            job_id: Job identifier to remove

        Returns:
            True if removed successfully
        """
        try:
            self.scheduler.remove_job(job_id)
            print(f"✓ Removed job '{job_id}'")
            return True
        except Exception as e:
            print(f"✗ Failed to remove job '{job_id}': {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a job without removing it"""
        try:
            self.scheduler.pause_job(job_id)
            print(f"⏸  Paused job '{job_id}'")
            return True
        except Exception as e:
            print(f"✗ Failed to pause job '{job_id}': {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        try:
            self.scheduler.resume_job(job_id)
            print(f"▶️  Resumed job '{job_id}'")
            return True
        except Exception as e:
            print(f"✗ Failed to resume job '{job_id}': {e}")
            return False

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a scheduled job.

        Returns:
            Dict with job info or None if not found
        """
        job = self.scheduler.get_job(job_id)
        if not job:
            return None

        return {
            'id': job.id,
            'name': job.name,
            'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
            'trigger': str(job.trigger),
            'pending': job.pending
        }

    def list_jobs(self) -> list:
        """
        Get list of all scheduled jobs.

        Returns:
            List of job info dicts
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        return jobs

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            print("▶️  Unified Scheduler started")

            # Print scheduled jobs
            jobs = self.list_jobs()
            if jobs:
                print(f"   Active jobs: {len(jobs)}")
                for job in jobs:
                    next_run = job['next_run_time'] if job['next_run_time'] else 'N/A'
                    print(f"   - {job['id']}: next run {next_run}")

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            print("⏹  Unified Scheduler stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        with self.stats_lock:
            return {
                'total_executions': self.job_stats['total_executions'],
                'total_errors': self.job_stats['total_errors'],
                'last_error': self.job_stats['last_error'],
                'active_jobs': len(self.scheduler.get_jobs()),
                'running': self.scheduler.running
            }


# =========================================================================
# Global Singleton
# =========================================================================

_unified_scheduler_instance = None
_scheduler_lock = threading.Lock()


def get_unified_scheduler() -> UnifiedScheduler:
    """Get or create the global unified scheduler instance"""
    global _unified_scheduler_instance

    with _scheduler_lock:
        if _unified_scheduler_instance is None:
            _unified_scheduler_instance = UnifiedScheduler()
        return _unified_scheduler_instance
