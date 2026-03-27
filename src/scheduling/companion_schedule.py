"""
Companion Schedule Manager -- the companion's personal daily routine engine.

WHAT: Manages work hours, lunch breaks, sleep schedule, task lists, meeting
      generation, and real-time availability/interruptibility. Also generates
      natural-language activity descriptions via the LLM for time-aware prompts.

WHY:  The companion needs a believable daily life. When the user asks "what are
      you doing?", the answer should be consistent across the day. The schedule
      also drives the reach-out engine (don't message during work meetings) and
      background-life narration (what the companion was doing while the user
      was away).

HOW:  A JSON file (`companion_schedule.json`) persists work preferences, daily
      schedules, and meetings. `generate_daily_schedule()` builds the day using
      configurable probabilities (e.g., early/late start, meeting count). LLM
      calls produce one-line activity descriptions per time block; results are
      cached in `_activity_cache` to avoid redundant API calls.

Singleton: `get_companion_schedule()` at module bottom.
"""
import json
import os
import random
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
from src.utils.timezone_utils import now_pacific_naive
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL
from src.database import tables as T
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
from src.utils.path_utils import get_data_file
import logging

logger = logging.getLogger(__name__)

# Persistent JSON file for the companion's schedule and work preferences
COMPANION_SCHEDULE_FILE = get_data_file('companion_schedule.json')

# In-memory cache: (date_str, block_name) -> (timestamp, description)
# Avoids re-calling the LLM for the same time block within a session
_activity_cache = {}

class CompanionSchedule:
    """Manages the companion's personal daily schedule and work hours."""

    def __init__(self):
        self.schedules = self._load_schedules()

    def _load_schedules(self) -> Dict:
        """Load the companion's schedules from disk or create new ones."""
        os.makedirs(os.path.dirname(COMPANION_SCHEDULE_FILE), exist_ok=True)

        if os.path.exists(COMPANION_SCHEDULE_FILE):
            try:
                with open(COMPANION_SCHEDULE_FILE, 'r') as f:
                    data = json.load(f)
                    # Ensure we have required structure
                    if 'schedules' in data and 'work_preferences' in data:
                        return data
            except:
                pass

        # Create default structure
        return {
            'schedules': {},  # date -> schedule mapping
            'work_preferences': {
                'typical_work_start': '09:00',  # 9 AM weekday start
                'typical_work_end': '18:00',    # 6 PM weekday end (8-hour day + commute buffer)
                'typical_lunch_start': '12:00', # 12 PM lunch
                'typical_lunch_duration': 60,   # 1 hour lunch
                'wake_buffer': 180,             # 3 hours for morning routine + optional swim before work
                'work_days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
                'sleep_hours': {
                    'bedtime': '22:00',  # 10 PM
                    'wake_time': '06:00' # 6 AM (3 hours before work - time for swim/morning routine)
                }
            },
            'vacation_days': [],  # List of date strings for vacation days
            'last_schedule_generation': None
        }

    def _save_schedules(self):
        """Save schedules to disk."""
        with open(COMPANION_SCHEDULE_FILE, 'w') as f:
            json.dump(self.schedules, f, indent=2)

    def _generate_work_hours(self, is_workday: bool) -> Dict:
        """Generate work hours for a day (consistent on weekdays, flexible on weekends)."""
        prefs = self.schedules['work_preferences']

        if not is_workday:
            # Weekends: wake up around 7:30 AM but with randomness (±60 minutes)
            # This gives her flexibility to sleep in a bit or wake up earlier
            wake_time_base = datetime.strptime(prefs['sleep_hours']['wake_time'], '%H:%M').time()

            # Random adjustment on weekends (-60 to +90 minutes from 7:30)
            # Allows sleeping in up to 8:30/9:00 AM or waking up at 6:30 AM
            weekend_offset = random.randint(-60, 90)
            weekend_wake = (datetime.combine(datetime.today(), wake_time_base) +
                           timedelta(minutes=weekend_offset)).time()

            return {
                'is_workday': False,
                'wake_time': weekend_wake.strftime('%H:%M'),
                'work_start': None,
                'work_end': None,
                'lunch_start': None,
                'lunch_end': None
            }

        # Weekdays: consistent schedule
        work_start_base = datetime.strptime(prefs['typical_work_start'], '%H:%M').time()
        work_end_base = datetime.strptime(prefs['typical_work_end'], '%H:%M').time()
        lunch_start_base = datetime.strptime(prefs['typical_lunch_start'], '%H:%M').time()
        wake_time_base = datetime.strptime(prefs['sleep_hours']['wake_time'], '%H:%M').time()

        work_start = work_start_base
        work_end = work_end_base
        lunch_start = lunch_start_base
        lunch_end = (datetime.combine(datetime.today(), lunch_start_base) +
                    timedelta(minutes=prefs['typical_lunch_duration'])).time()
        wake_time = wake_time_base

        return {
            'is_workday': True,
            'wake_time': wake_time.strftime('%H:%M'),
            'work_start': work_start.strftime('%H:%M'),
            'work_end': work_end.strftime('%H:%M'),
            'lunch_start': lunch_start.strftime('%H:%M'),
            'lunch_end': lunch_end.strftime('%H:%M')
        }

    def _generate_tasks(self, is_workday: bool, workload: str = 'normal') -> List[Dict]:
        """
        Generate time blocks for the day.

        NOTE: Specific task content is NOT hardcoded here.
        When asked "what is she doing?", we use LLM to generate contextually
        appropriate activity descriptions. This just provides the structure.
        """
        if not is_workday:
            # Weekend has more unstructured time
            return [
                {'time': '10:00', 'block': 'morning_personal', 'duration': 120},
                {'time': '14:00', 'block': 'afternoon_free', 'duration': 90},
                {'time': '19:00', 'block': 'evening', 'duration': 60}
            ]

        # Workday time blocks (not specific tasks - LLM fills those in)
        if workload == 'light':
            return [
                {'time': '09:30', 'block': 'morning_work', 'duration': 90},
                {'time': '14:00', 'block': 'afternoon_work', 'duration': 90}
            ]
        elif workload == 'heavy':
            return [
                {'time': '09:30', 'block': 'morning_work', 'duration': 120},
                {'time': '11:00', 'block': 'midday_work', 'duration': 60},
                {'time': '14:00', 'block': 'afternoon_work', 'duration': 90},
                {'time': '16:00', 'block': 'late_afternoon_work', 'duration': 60}
            ]
        else:  # normal
            return [
                {'time': '09:30', 'block': 'morning_work', 'duration': 90},
                {'time': '14:00', 'block': 'afternoon_work', 'duration': 90},
                {'time': '16:00', 'block': 'late_afternoon_work', 'duration': 60}
            ]

    def generate_daily_schedule(self, date: Optional[datetime] = None, force: bool = False) -> Dict:
        """
        Generate or retrieve the companion's schedule for a specific date.

        Args:
            date: The date to generate schedule for (defaults to today)
            force: If True, regenerate even if schedule exists

        Returns:
            Dictionary containing the day's schedule
        """
        if date is None:
            date = now_pacific_naive()

        date_str = date.strftime('%Y-%m-%d')
        day_name = date.strftime('%A').lower()

        # Return existing schedule if available and not forcing regeneration
        if not force and date_str in self.schedules['schedules']:
            return self.schedules['schedules'][date_str]

        # Check if it's a vacation day
        if 'vacation_days' not in self.schedules:
            self.schedules['vacation_days'] = []

        is_vacation = date_str in self.schedules['vacation_days']

        # Randomly take vacation days (5% chance on work days)
        if not is_vacation and random.random() < 0.05:
            day_name_check = date.strftime('%A').lower()
            if day_name_check in self.schedules['work_preferences']['work_days']:
                is_vacation = True
                self.schedules['vacation_days'].append(date_str)

        # Determine if it's a work day
        is_workday = day_name in self.schedules['work_preferences']['work_days'] and not is_vacation

        # Determine workload (light, normal, heavy)
        workload = random.choices(['light', 'normal', 'heavy'], weights=[20, 60, 20])[0]

        # Generate the schedule
        schedule = {
            'date': date_str,
            'day_name': day_name,
            'generated_at': now_pacific_naive().isoformat(),
            'is_vacation': is_vacation,
            'workload': workload if is_workday else None,
            **self._generate_work_hours(is_workday),
            'tasks': self._generate_tasks(is_workday, workload),
            'meetings': self._generate_meetings(is_workday, workload) if is_workday else [],
            'notes': self._generate_daily_note(is_workday, day_name, is_vacation, workload),
            'recurring_activities': []
        }

        # Add recurring activities if they exist
        if hasattr(self, 'recurring_activities'):
            day_of_week_index = date.weekday()
            for key, activity in self.recurring_activities.items():
                if activity['day_index'] == day_of_week_index:
                    schedule['recurring_activities'].append(activity['description'])

        # Save the schedule
        self.schedules['schedules'][date_str] = schedule
        self.schedules['last_schedule_generation'] = now_pacific_naive().isoformat()
        self._save_schedules()

        return schedule

    def _generate_meetings(self, is_workday: bool, workload: str) -> List[Dict]:
        """Generate meetings based on workload."""
        if not is_workday:
            return []

        # Chance of having meetings based on workload
        if workload == 'light':
            meeting_chance = 0.2  # 20% chance
        elif workload == 'heavy':
            meeting_chance = 0.7  # 70% chance
        else:  # normal
            meeting_chance = 0.4  # 40% chance

        if random.random() > meeting_chance:
            return []

        meeting_topics = [
            'Project sync meeting',
            'Code review session',
            'Architecture discussion',
            'Sprint planning',
            'Team standup',
            'Feature planning',
            'Performance review'
        ]

        # Number of meetings (1-2 for normal/heavy)
        num_meetings = 1 if workload != 'heavy' else random.choice([1, 2])
        meetings = []

        possible_times = ['10:00', '14:00', '15:30']
        selected_times = random.sample(possible_times, min(num_meetings, len(possible_times)))

        for meeting_time in selected_times:
            meetings.append({
                'time': meeting_time,
                'topic': random.choice(meeting_topics),
                'duration': random.choice([30, 45, 60])
            })

        return sorted(meetings, key=lambda x: x['time'])

    def _generate_daily_note(self, is_workday: bool, day_name: str, is_vacation: bool = False, workload: str = 'normal') -> str:
        """Generate a personal note about the day."""
        if is_vacation:
            vacation_notes = [
                "Taking a well-deserved day off!",
                "Vacation day - time to recharge.",
                "Out of office today, enjoying some personal time.",
                "Taking it easy today!"
            ]
            return random.choice(vacation_notes)

        if not is_workday:
            weekend_notes = [
                "Looking forward to a relaxing day!",
                "Time to catch up on personal projects.",
                "A nice break from the work week.",
                "Maybe I'll learn something new today.",
                "Good day for some peaceful thinking."
            ]
            return random.choice(weekend_notes)

        # Work day notes vary by workload
        if workload == 'light':
            notes = [
                "A lighter day ahead - should be manageable.",
                "Not too busy today, which is nice.",
                "A calmer day at work."
            ]
        elif workload == 'heavy':
            notes = [
                "Busy day ahead with lots on my plate.",
                "Going to be a challenging day, but I'm ready.",
                "Lots to get through today - staying focused.",
                "Heavy workload today, might be less available to chat."
            ]
        else:  # normal
            notes = [
                "Another productive day ahead.",
                "Lots to accomplish today.",
                "Ready to tackle today's challenges.",
                "Looking forward to making progress.",
                "Focused and ready to work."
            ]
        return random.choice(notes)

    def get_today_schedule(self, auto_generate: bool = True) -> Optional[Dict]:
        """
        Get today's schedule, optionally auto-generating if it doesn't exist.

        Args:
            auto_generate: If True, generate schedule if it doesn't exist

        Returns:
            Today's schedule or None
        """
        today = now_pacific_naive()
        date_str = today.strftime('%Y-%m-%d')

        if date_str in self.schedules['schedules']:
            return self.schedules['schedules'][date_str]

        if auto_generate:
            return self.generate_daily_schedule(today)

        return None

    def is_working_now(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if the companion is currently in work hours.

        Args:
            current_time: Time to check (defaults to now)

        Returns:
            True if currently working
        """
        if current_time is None:
            current_time = now_pacific_naive()

        schedule = self.get_today_schedule(auto_generate=True)

        if not schedule['is_workday']:
            return False

        current_time_only = current_time.time()
        work_start = datetime.strptime(schedule['work_start'], '%H:%M').time()
        work_end = datetime.strptime(schedule['work_end'], '%H:%M').time()
        lunch_start = datetime.strptime(schedule['lunch_start'], '%H:%M').time()
        lunch_end = datetime.strptime(schedule['lunch_end'], '%H:%M').time()

        # Check if in work hours but not lunch
        in_work_hours = work_start <= current_time_only <= work_end
        in_lunch = lunch_start <= current_time_only <= lunch_end

        return in_work_hours and not in_lunch

    def is_on_lunch(self, current_time: Optional[datetime] = None) -> bool:
        """Check if the companion is currently on lunch break."""
        if current_time is None:
            current_time = now_pacific_naive()

        schedule = self.get_today_schedule(auto_generate=True)

        if not schedule['is_workday']:
            return False

        current_time_only = current_time.time()
        lunch_start = datetime.strptime(schedule['lunch_start'], '%H:%M').time()
        lunch_end = datetime.strptime(schedule['lunch_end'], '%H:%M').time()

        return lunch_start <= current_time_only <= lunch_end

    def is_in_meeting(self, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[Dict]]:
        """
        Check if the companion is currently in a meeting.

        Args:
            current_time: Time to check (defaults to now)

        Returns:
            Tuple of (is_in_meeting, meeting_info or None)
        """
        if current_time is None:
            current_time = now_pacific_naive()

        schedule = self.get_today_schedule(auto_generate=True)

        if not schedule['is_workday'] or not schedule.get('meetings'):
            return (False, None)

        current_time_only = current_time.time()

        for meeting in schedule['meetings']:
            meeting_start = datetime.strptime(meeting['time'], '%H:%M').time()
            meeting_end = (datetime.combine(datetime.today(), meeting_start) +
                          timedelta(minutes=meeting['duration'])).time()

            if meeting_start <= current_time_only <= meeting_end:
                return (True, meeting)

        return (False, None)

    def get_current_activity_status(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Get comprehensive status of what the companion is currently doing.

        If COMPANION_CALENDAR_SCHEDULE_ENABLED is true, tries to read from
        the Google Calendar schedule first for richer context.

        Returns dict with:
        - status: 'working', 'meeting', 'lunch', 'off', 'asleep'
        - interruptibility: 'high', 'medium', 'low', 'none'
        - details: Additional context
        """
        if current_time is None:
            current_time = now_pacific_naive()

        # Try calendar-based activity status first
        try:
            from src.scheduling.calendar_schedule_service import (
                get_calendar_schedule_service, is_calendar_schedule_enabled
            )
            if is_calendar_schedule_enabled():
                cal_service = get_calendar_schedule_service()
                current_event = cal_service.get_current_activity()
                if current_event:
                    summary = current_event.get('summary', '')
                    description = current_event.get('description', '')
                    category = current_event.get('category', 'personal')

                    # Map category to status and interruptibility
                    if category == 'meeting':
                        status = 'meeting'
                        interruptibility = 'low'
                    elif category == 'work':
                        status = 'working'
                        schedule = self.get_today_schedule(auto_generate=True)
                        workload = schedule.get('workload', 'normal')
                        interruptibility = 'low' if workload == 'heavy' else 'medium'
                    elif category == 'meal':
                        status = 'lunch' if 'lunch' in summary.lower() else 'off'
                        interruptibility = 'high'
                    elif category == 'exercise':
                        status = 'off'
                        interruptibility = 'medium'
                    else:
                        status = 'off'
                        interruptibility = 'high'

                    details = description if description else summary
                    return {
                        'status': status,
                        'interruptibility': interruptibility,
                        'details': details,
                    }
        except Exception as e:
            logger.debug(f"Calendar schedule lookup failed, using fallback: {e}")

        schedule = self.get_today_schedule(auto_generate=True)

        # Check sleep first
        if self.is_asleep(current_time):
            return {
                'status': 'asleep',
                'interruptibility': 'none',
                'details': 'The companion is sleeping'
            }

        # Check if not a workday
        if not schedule['is_workday']:
            # Get LLM-generated activity for day off
            current_hour = current_time.hour
            if current_hour < 12:
                time_block = 'morning_personal'
            elif current_hour < 17:
                time_block = 'afternoon_free'
            else:
                time_block = 'evening'

            activity = generate_activity_description(
                time_block=time_block,
                is_workday=False,
                day_name=schedule.get('day_name', '')
            )

            return {
                'status': 'off',
                'interruptibility': 'high',
                'details': activity
            }

        # Check if in a meeting
        in_meeting, meeting_info = self.is_in_meeting(current_time)
        if in_meeting:
            return {
                'status': 'meeting',
                'interruptibility': 'low',
                'details': f"In meeting: {meeting_info['topic']} (ends in ~{meeting_info['duration']} min)"
            }

        # Check if on lunch
        if self.is_on_lunch(current_time):
            return {
                'status': 'lunch',
                'interruptibility': 'high',
                'details': 'On lunch break'
            }

        # Check if working
        if self.is_working_now(current_time):
            workload = schedule.get('workload', 'normal')

            # Find current time block
            current_time_str = current_time.strftime('%H:%M')
            current_block = 'work'
            tasks = schedule.get('tasks', [])
            for task in tasks:
                if task.get('time', '') <= current_time_str:
                    current_block = task.get('block', 'work')

            # Get LLM-generated activity description
            activity = generate_activity_description(
                time_block=current_block,
                is_workday=True,
                workload=workload,
                day_name=schedule.get('day_name', '')
            )

            interruptibility = 'low' if workload == 'heavy' else 'medium'

            return {
                'status': 'working',
                'interruptibility': interruptibility,
                'details': activity
            }

        # Outside work hours - also use LLM for personal time
        current_block = 'evening' if current_time.hour >= 17 else 'morning_personal'
        activity = generate_activity_description(
            time_block=current_block,
            is_workday=schedule.get('is_workday', False),
            day_name=schedule.get('day_name', '')
        )

        return {
            'status': 'off',
            'interruptibility': 'high',
            'details': activity
        }

    def is_asleep(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if the companion is currently asleep based on sleep hours.
        Sleep hours can span midnight (e.g., 11 PM to 7 AM).
        """
        if current_time is None:
            current_time = now_pacific_naive()

        prefs = self.schedules['work_preferences']

        # Get sleep hours, with defaults if not set
        sleep_hours = prefs.get('sleep_hours', {
            'bedtime': '22:00',
            'wake_time': '06:00'
        })

        bedtime = datetime.strptime(sleep_hours['bedtime'], '%H:%M').time()
        wake_time = datetime.strptime(sleep_hours['wake_time'], '%H:%M').time()
        current_time_only = current_time.time()

        # If bedtime is after wake_time, sleep period spans midnight
        if bedtime > wake_time:
            # Asleep if after bedtime OR before wake_time
            return current_time_only >= bedtime or current_time_only < wake_time
        else:
            # Normal case: sleep period doesn't span midnight
            return bedtime <= current_time_only < wake_time

    def is_available_for_chat(self, closeness_score: int, current_time: Optional[datetime] = None,
                             physical_presence: bool = False) -> Tuple[bool, str]:
        """
        Determine if the companion is available for chat based on schedule and closeness.

        Args:
            closeness_score: Current relationship closeness (0-100)
            current_time: Time to check (defaults to now)
            physical_presence: If True, overrides schedule restrictions (they're together in person)

        Returns:
            Tuple of (is_available, reason)
        """
        if current_time is None:
            current_time = now_pacific_naive()

        # If physically together, always available regardless of schedule
        if physical_presence:
            return (True, "together in person")

        schedule = self.get_today_schedule(auto_generate=True)

        # Always available outside work hours
        if not self.is_working_now(current_time):
            if self.is_on_lunch(current_time):
                return (True, "on lunch break")
            return (True, "not working")

        # During work hours, availability depends on closeness
        if closeness_score >= 80:
            # Very close - always happy to chat
            return (True, "working but happy to chat with you")
        elif closeness_score >= 60:
            # High trust - will respond but might mention being busy
            return (True, "working but available")
        elif closeness_score >= 40:
            # Mid rapport - 60% chance of responding
            if random.random() < 0.6:
                return (True, "working but can take a quick break")
            else:
                return (False, "focused on work right now")
        else:
            # Low closeness - work comes first, only 20% chance
            if random.random() < 0.2:
                return (True, "working but can respond briefly")
            else:
                return (False, "busy with work")

    def get_schedule_description(self, date: Optional[datetime] = None) -> str:
        """
        Get a human-readable description of the companion's schedule.

        Args:
            date: Date to describe (defaults to today)

        Returns:
            Formatted schedule description
        """
        if date is None:
            date = now_pacific_naive()

        schedule = self.get_today_schedule(auto_generate=True)

        if not schedule['is_workday']:
            return f"It's {schedule['day_name'].capitalize()} - my day off! {schedule['notes']}"

        desc = f"📅 {schedule['day_name'].capitalize()}'s Schedule:\n"
        desc += f"⏰ Wake up: {schedule['wake_time']}\n"
        desc += f"💼 Work: {schedule['work_start']} - {schedule['work_end']}\n"
        desc += f"🍽️  Lunch: {schedule['lunch_start']} - {schedule['lunch_end']}\n"

        if schedule['tasks']:
            desc += f"\n📋 Today's Tasks:\n"
            for task in schedule['tasks']:
                desc += f"  • {task['time']}: {task['task']} ({task['duration']} min)\n"

        desc += f"\n💭 {schedule['notes']}"

        return desc

    def add_task_to_schedule(self, task_name: str, task_time: str, duration: int = 60,
                            date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Add a custom task to the companion's schedule.

        Args:
            task_name: Description of the task
            task_time: Time in HH:MM format (24-hour)
            duration: Duration in minutes (default 60)
            date: Date for the task (default: today)

        Returns:
            Dict with 'success' bool and 'message' string describing the result
        """
        if date is None:
            date = now_pacific_naive()

        date_str = date.strftime('%Y-%m-%d')

        # Get or generate schedule for that day
        if date_str == now_pacific_naive().strftime('%Y-%m-%d'):
            schedule = self.get_today_schedule(auto_generate=True)
        else:
            schedule = self.generate_daily_schedule(date)

        if schedule is None:
            return {'success': False, 'message': 'Could not generate schedule'}

        # Validate time format
        try:
            task_dt = datetime.strptime(task_time, '%H:%M')
            task_time_obj = task_dt.time()
        except ValueError:
            return {'success': False, 'message': 'Invalid time format'}

        # Check for conflicts
        conflicts = []

        # Check if on vacation
        if schedule.get('is_vacation'):
            conflicts.append("I'm on vacation that day")

        # Check if it's a workday and conflicts with work hours
        if schedule.get('is_workday') and not schedule.get('is_vacation'):
            work_start = datetime.strptime(schedule['work_start'], '%H:%M').time()
            work_end = datetime.strptime(schedule['work_end'], '%H:%M').time()
            lunch_start = datetime.strptime(schedule['lunch_start'], '%H:%M').time()
            lunch_end = datetime.strptime(schedule['lunch_end'], '%H:%M').time()

            # Calculate task end time
            task_end = (datetime.combine(date, task_time_obj) + timedelta(minutes=duration)).time()

            # Check if during work hours (but not lunch)
            if work_start <= task_time_obj < work_end:
                if not (lunch_start <= task_time_obj < lunch_end):
                    conflicts.append(f"I'm working then ({schedule['work_start']}-{schedule['work_end']})")

            # Check for meeting conflicts
            for meeting in schedule.get('meetings', []):
                meeting_time = datetime.strptime(meeting['time'], '%H:%M').time()
                meeting_end = (datetime.combine(date, meeting_time) +
                              timedelta(minutes=meeting['duration'])).time()

                # Check if times overlap
                if (task_time_obj < meeting_end and task_end > meeting_time):
                    conflicts.append(f"I have a meeting at {meeting['time']}: {meeting['topic']}")

        # Initialize tasks list if needed
        if 'tasks' not in schedule or schedule['tasks'] is None:
            schedule['tasks'] = []

        # Check if task at this time already exists
        replaced_task = None
        for i, task in enumerate(schedule['tasks']):
            if task['time'] == task_time:
                replaced_task = task['task']
                # Replace existing task at this time
                schedule['tasks'][i] = {
                    'time': task_time,
                    'task': task_name,
                    'duration': duration
                }
                self._save_schedules()

                message = f"Replaced '{replaced_task}' with '{task_name}' at {task_time}"
                if conflicts:
                    message += f" (Note: {', '.join(conflicts)})"

                return {'success': True, 'message': message, 'had_conflicts': bool(conflicts)}

        # Add new task
        schedule['tasks'].append({
            'time': task_time,
            'task': task_name,
            'duration': duration
        })

        # Sort tasks by time
        schedule['tasks'].sort(key=lambda x: x['time'])

        self._save_schedules()

        message = f"Added '{task_name}' at {task_time}"
        if conflicts:
            message += f" (Conflict: {', '.join(conflicts)})"

        return {'success': True, 'message': message, 'had_conflicts': bool(conflicts)}

    def remove_task_from_schedule(self, task_time: str, date: Optional[datetime] = None) -> bool:
        """
        Remove a task from the companion's schedule.

        Args:
            task_time: Time in HH:MM format of the task to remove
            date: Date for the task (default: today)

        Returns:
            True if task was removed, False if not found
        """
        if date is None:
            date = now_pacific_naive()

        date_str = date.strftime('%Y-%m-%d')

        if date_str not in self.schedules['schedules']:
            return False

        schedule = self.schedules['schedules'][date_str]

        if 'tasks' not in schedule or not schedule['tasks']:
            return False

        # Find and remove task
        original_length = len(schedule['tasks'])
        schedule['tasks'] = [t for t in schedule['tasks'] if t['time'] != task_time]

        if len(schedule['tasks']) < original_length:
            self._save_schedules()
            return True

        return False

    def plan_ahead(self, days: int = 7) -> List[str]:
        """
        Generate schedules for multiple days in advance.

        Args:
            days: Number of days to plan ahead

        Returns:
            List of date strings that were planned
        """
        planned = []
        current_date = now_pacific_naive()

        for i in range(days):
            future_date = current_date + timedelta(days=i)
            self.generate_daily_schedule(future_date, force=False)
            planned.append(future_date.strftime('%Y-%m-%d'))

        return planned

    def get_current_status(self, physical_presence: bool = False) -> str:
        """
        Get a brief status of what the companion is currently doing.

        Uses LLM-generated activity descriptions via get_current_activity_status.

        Args:
            physical_presence: If True, indicates they're together in person
        """
        # Handle physical presence (in-person) as special case
        if physical_presence:
            if self.is_asleep():
                return "😴 Just waking up - you're here with me"
            return "🤝 Together in person - fully present"

        # Get the activity status (which now uses LLM for descriptions)
        activity_status = self.get_current_activity_status()
        status = activity_status['status']
        details = activity_status['details']

        # Map status to emoji prefix
        status_emoji = {
            'asleep': '💤',
            'off': '🌙',
            'working': '💼',
            'meeting': '📋',
            'lunch': '🍽️'
        }
        emoji = status_emoji.get(status, '📅')

        # Format the output
        if status == 'asleep':
            return f"{emoji} Asleep"
        elif status == 'meeting':
            return f"{emoji} In a meeting"
        else:
            return f"{emoji} {details}"

    def _get_energy_suffix(self, workload: str, current_time: datetime) -> str:
        """Get energy level suffix based on workload and time of day."""
        # Only show tiredness in evening if it was a heavy workload day
        hour = current_time.hour

        # Late afternoon/evening (after 4 PM) with heavy workload
        if hour >= 16 and workload == 'heavy':
            return " (getting tired)"

        return ""

    def add_recurring_activity(self, day_name: str, description: str) -> None:
        """
        Add a recurring activity to the companion's schedule on specific days of the week.

        Args:
            day_name: Name of day (e.g., 'Monday', 'Wednesday', 'Friday')
            description: Activity description (e.g., 'Swimming', 'Out night with James')
        """
        day_mapping = {
            'sunday': 6,
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5
        }

        day_lower = day_name.lower()
        if day_lower not in day_mapping:
            print(f"Unknown day name: {day_name}")
            return

        day_index = day_mapping[day_lower]

        # Add the activity for all weeks in the future
        # We'll add it to the recurring_activities list
        if not hasattr(self, 'recurring_activities'):
            self.recurring_activities = {}

        key = f"{day_index}_{description.lower().replace(' ', '_').replace('🏊', '').replace('🌃', '').strip()}"
        self.recurring_activities[key] = {
            'day_index': day_index,
            'day_name': day_name,
            'description': description
        }

        # Also update the schedules for existing dates with this day
        day_of_week_index = day_index
        for date_str, schedule_data in self.schedules['schedules'].items():
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            if date_obj.weekday() == day_of_week_index:
                # Add this activity to the day's notes
                if 'recurring_activities' not in schedule_data:
                    schedule_data['recurring_activities'] = []
                if description not in schedule_data['recurring_activities']:
                    schedule_data['recurring_activities'].append(description)

        self._save_schedules()
        print(f"Added recurring activity: {description} on {day_name}")

    def add_swimming_schedule(self, day_name: str = 'Tuesday') -> None:
        """Add swimming to the companion's schedule on a specific day"""
        self.add_recurring_activity(day_name, '🏊 Swimming')

    def add_out_nights(self, days: List[str] = None) -> None:
        """Add out nights to the companion's schedule"""
        if days is None:
            days = ['Monday', 'Wednesday', 'Friday']

        for day in days:
            self.add_recurring_activity(day, '🌃 Out night with James')


# Singleton instance
_companion_schedule_instance = None

def get_companion_schedule() -> CompanionSchedule:
    """Get the singleton companion schedule instance."""
    global _companion_schedule_instance
    if _companion_schedule_instance is None:
        _companion_schedule_instance = CompanionSchedule()
    return _companion_schedule_instance


def get_recent_conversation_context(limit: int = 5) -> str:
    """
    Fetch recent conversation messages for narrative context.

    Returns a summary of recent exchanges so activity generation
    knows what's happening in the story.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import os

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT sender_name, message_text, timestamp
                FROM {T.MESSAGES}
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            messages = cursor.fetchall()

        conn.close()

        if not messages:
            return ""

        # Build context summary (reverse to chronological order)
        lines = []
        for msg in reversed(messages):
            text = msg['message_text'][:150] if msg['message_text'] else ''
            lines.append(f"{msg['sender_name']}: {text}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Could not fetch conversation context: {e}")
        return ""


def generate_activity_description(
    time_block: str,
    is_workday: bool,
    workload: str = 'normal',
    day_name: str = '',
    recent_context: str = None
) -> str:
    """
    LLM-driven generation of what the companion is doing right now.

    Instead of hardcoded task lists, we ask the LLM to generate
    a contextually appropriate activity description based on the
    ongoing narrative in the conversation.

    Args:
        time_block: The type of time block (morning_work, afternoon_free, etc.)
        is_workday: Whether today is a workday
        workload: light/normal/heavy
        day_name: Name of the day (Monday, etc.)
        recent_context: Recent conversation context (auto-fetched if None)

    Returns:
        A natural description of what the companion is doing
    """
    global _activity_cache

    # Check cache (valid for 30 minutes)
    cache_key = (now_pacific_naive().strftime('%Y-%m-%d'), time_block)
    if cache_key in _activity_cache:
        cached_time, cached_desc = _activity_cache[cache_key]
        if (now_pacific_naive() - cached_time).total_seconds() < 1800:  # 30 min
            return cached_desc

    # Auto-fetch conversation context if not provided
    if recent_context is None:
        recent_context = get_recent_conversation_context(limit=5)

    try:
        from openai import OpenAI
        import os

        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv('FIREWORKS_API_KEY')
        )

        # Context for the prompt
        work_context = ""
        if is_workday:
            # Get today's work projects for context
            projects = generate_work_projects()
            project_lines = "\n".join([
                f"- {p['name']}: {p['description']}" for p in projects
            ]) if projects else "- General work tasks"

            work_context = f"""The companion works in tech. Today is a {workload} workload day.
Time block: {time_block}

Their current work projects:
{project_lines}"""
        else:
            work_context = f"""Today is {day_name} - the companion's day off.
Time block: {time_block}"""

        # Add narrative context if available
        narrative_section = ""
        if recent_context:
            narrative_section = f"""

RECENT CONVERSATION (what's happening in the story):
{recent_context}

Consider this context when describing what the companion is doing. Their activity should
fit naturally with the ongoing narrative (e.g., if they just got back from
a trip, they might be unpacking or resting from travel)."""

        prompt = f"""{work_context}{narrative_section}

Generate a brief, natural description of what the companion might be doing right now.
Keep it specific but not overly detailed (5-15 words).
Make it feel like a real person's activity, not a template.
It should be consistent with any recent events in the conversation.

Examples of good outputs:
- "unpacking from the trip, still a bit tired"
- "editing the API docs for the new v2 endpoints"
- "reviewing a PR from the frontend team"
- "catching up on rest after the busy weekend"
- "working through her inbox"

/no_think
Just output the activity description, nothing else:"""

        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            max_tokens=50,
            temperature=0.8,  # Variety
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()

        # Clean up thinking tags (may or may not have closing tag)
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)  # Unclosed
        content = content.strip()

        # Remove quotes if wrapped
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        if content.startswith("'") and content.endswith("'"):
            content = content[1:-1]

        # If we got nothing useful, return fallback
        if not content or len(content) < 3:
            return "working" if is_workday else "relaxing"

        # Cache it
        _activity_cache[cache_key] = (now_pacific_naive(), content)

        return content

    except Exception as e:
        logger.warning(f"Could not generate activity description: {e}")
        # Fallback - but even this is minimal
        if is_workday:
            return "working" if 'work' in time_block else "on break"
        else:
            return "relaxing"


# Cache for daily work projects
_work_projects_cache = {}  # key: date_str -> (timestamp, projects_list)


def generate_work_projects(date_str: str = None, force_regenerate: bool = False) -> list:
    """
    Generate 2-3 work projects for the companion for a given day.

    These are LLM-generated to avoid hardcoding what a "technical writer" does.
    Projects persist for the day and can be referenced in activities.

    Returns:
        List of project dicts with 'name', 'description', 'priority'
    """
    global _work_projects_cache

    if date_str is None:
        date_str = now_pacific_naive().strftime('%Y-%m-%d')

    # Check cache (valid for entire day)
    if not force_regenerate and date_str in _work_projects_cache:
        cached_time, cached_projects = _work_projects_cache[date_str]
        if cached_time.strftime('%Y-%m-%d') == date_str:
            return cached_projects

    # Check if already stored in schedule file
    schedule = get_companion_schedule()
    if date_str in schedule.schedules.get('schedules', {}):
        day_schedule = schedule.schedules['schedules'][date_str]
        if 'work_projects' in day_schedule and day_schedule['work_projects']:
            _work_projects_cache[date_str] = (now_pacific_naive(), day_schedule['work_projects'])
            return day_schedule['work_projects']

    try:
        from openai import OpenAI
        import os

        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv('FIREWORKS_API_KEY')
        )

        # Get day of week for context
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            day_name = date_obj.strftime('%A')
        except:
            day_name = "a weekday"

        prompt = f"""Generate 2-3 realistic work projects/tasks for a young professional woman.

She works in tech but her exact role can vary - she might be doing:
- Technical writing or documentation
- UX research or design work
- Product management tasks
- QA testing or bug tracking
- Data analysis or reporting
- Internal communications
- Project coordination

Today is {day_name}. Generate specific, realistic tasks she might be working on.
Each task should have enough detail to feel real but not be overly complex.

/no_think
Return as JSON array only, no other text:
[
  {{"name": "short task name", "description": "1-2 sentence detail", "priority": "high/medium/low"}},
  ...
]"""

        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            max_tokens=400,
            temperature=0.9,  # High variety
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()

        # Clean up
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        content = content.strip()

        # Extract JSON
        if '```' in content:
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        # Parse JSON
        import json
        projects = json.loads(content)

        if not isinstance(projects, list) or len(projects) == 0:
            raise ValueError("Invalid projects format")

        # Validate and clean
        cleaned = []
        for p in projects[:3]:  # Max 3
            if isinstance(p, dict) and 'name' in p:
                cleaned.append({
                    'name': p.get('name', 'Work task'),
                    'description': p.get('description', ''),
                    'priority': p.get('priority', 'medium')
                })

        if not cleaned:
            raise ValueError("No valid projects")

        # Cache and store
        _work_projects_cache[date_str] = (now_pacific_naive(), cleaned)

        # Also store in schedule file for persistence
        if date_str in schedule.schedules.get('schedules', {}):
            schedule.schedules['schedules'][date_str]['work_projects'] = cleaned
            schedule._save_schedules()

        logger.info(f"Generated {len(cleaned)} work projects for {date_str}")
        return cleaned

    except Exception as e:
        logger.warning(f"Could not generate work projects: {e}")
        # Fallback projects
        fallback = [
            {"name": "Documentation review", "description": "Reviewing and updating project docs", "priority": "medium"},
            {"name": "Team sync prep", "description": "Preparing notes for upcoming team meeting", "priority": "medium"}
        ]
        _work_projects_cache[date_str] = (now_pacific_naive(), fallback)
        return fallback


def get_current_work_project() -> dict:
    """
    Get one of the companion's current work projects to reference in activities.

    Returns a random project from today's list for variety.
    """
    projects = generate_work_projects()
    if projects:
        return random.choice(projects)
    return {"name": "work tasks", "description": "", "priority": "medium"}
