"""
Reach-Out Engine -- Decides *if*, *when*, and *why* the companion initiates contact.

WHAT: The central decision engine for proactive messaging.  For each check cycle
      it answers three questions:
        1. Should she reach out?  (should_reach_out() -> bool, reason, opener)
        2. What should she say?   (generate_message() -> full pipeline message)
        3. Why or why not?        (classify_suppression() for pressure tracking)

      Decision flow inside should_reach_out():
        Hard guards  ->  Natural Partner Triggers  ->  High-urgency curiosity
        ->  Double-text threshold  ->  LLM decision (kimi-k2-instruct-0905)

WHY:  A companion that only responds on command feels hollow.  This engine makes
      her initiate naturally -- not on timers, but based on genuine internal
      state (queued thoughts, curiosity, opinions, schedule transitions).
      Natural Partner Triggers model the kinds of reach-outs a real person makes:
      morning texts, end-of-day check-ins, sharing discoveries, follow-ups.

HOW IT FITS:
  - Called by AlwaysOnService._check_reach_out_with_pressure() on each offline
    loop iteration.
  - Also used by check_and_reach_out() for milestone-triggered messages.
  - get_context() gathers state from: messages DB, ValueInference, curiosity,
    internal_state, companion_schedule, calendar_service, goal_planner.
  - generate_message() routes through the full conversation pipeline for
    consistent voice, personality, and memory validation.
  - classify_suppression() tells the pressure model whether a "no" was external
    (she wanted to but couldn't) or internal (she chose not to).

Natural Partner Triggers (see NATURAL_TRIGGERS dict):
  morning_greeting      7-10 AM, 20h cooldown
  end_of_workday        5-7 PM, 20h cooldown
  thinking_of_you       any time, 6h cooldown, needs queued thought
  share_discovery       any time, 8h cooldown, needs unshared research
  high_urgency_followup any time, 4h cooldown, needs urgency > 0.8
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from zoneinfo import ZoneInfo
from dataclasses import dataclass, asdict

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config.persona_config import get_persona_config

logger = logging.getLogger(__name__)

# LLM for decision making (235B for quality)
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2-instruct-0905"
PST = ZoneInfo('America/Los_Angeles')


# =============================================================================
# NATURAL PARTNER TRIGGERS
# =============================================================================
# These are situations where a real partner would naturally reach out.
# Each trigger has:
#   - time_window: Hours when this trigger can fire (None = any time)
#   - condition: Function that checks if trigger applies
#   - cooldown_hours: Minimum hours between trigger fires
#   - priority: Urgency when this trigger fires (higher = more likely to override)

@dataclass
class TriggerCooldown:
    """Tracks when triggers last fired."""
    trigger_name: str
    last_fired: datetime

    def to_dict(self) -> Dict:
        return {
            'trigger_name': self.trigger_name,
            'last_fired': self.last_fired.isoformat()
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'TriggerCooldown':
        return cls(
            trigger_name=d['trigger_name'],
            last_fired=datetime.fromisoformat(d['last_fired'])
        )


# Trigger definitions
NATURAL_TRIGGERS = {
    'morning_greeting': {
        'time_window': (7, 10),  # 7-10 AM
        'description': 'Good morning text when waking up',
        'cooldown_hours': 20,  # Once per day
        'priority': 0.7,
        'opener_hint': 'morning greeting, maybe mention sleep or what you were thinking about',
    },
    'end_of_workday': {
        'time_window': (17, 19),  # 5-7 PM
        'description': 'How was work? when finishing work',
        'cooldown_hours': 20,  # Once per day
        'priority': 0.6,
        'opener_hint': 'asking how their day went, sharing something from your own day',
    },
    'thinking_of_you': {
        'time_window': None,  # Any time
        'description': 'Random thought prompted by queued thoughts or curiosity',
        'cooldown_hours': 6,  # Max 4x per day
        'priority': 0.5,
        'opener_hint': 'spontaneous thought you wanted to share',
    },
    'share_discovery': {
        'time_window': None,  # Any time
        'description': 'Share something you researched or learned',
        'cooldown_hours': 8,  # Max 3x per day
        'priority': 0.55,
        'opener_hint': 'sharing something interesting you found or thought about',
    },
    'high_urgency_followup': {
        'time_window': None,  # Any time
        'description': 'Following up on high-urgency curiosity',
        'cooldown_hours': 4,  # Max 6x per day
        'priority': 0.8,
        'opener_hint': 'naturally following up on something important you really want to know about',
    }
}

# =============================================================================
# Feature flags & runtime toggle
# =============================================================================

TELEGRAM_ENABLED = os.environ.get('COMPANION_TELEGRAM_ENABLED', 'false').lower() == 'true'

# Autonomy toggle file -- allows disabling proactive messaging at runtime
# without a restart (checked by is_autonomy_enabled() each cycle).
AUTONOMY_STATE_FILE = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'autonomy_enabled.txt'
)


def is_autonomy_enabled() -> bool:
    """
    Check if autonomy (proactive messaging) is enabled.

    Checks:
    1. Environment variable COMPANION_AUTONOMY_ENABLED (default: true)
    2. Runtime toggle file (overrides env var if exists)
    """
    # Check runtime toggle file first (allows on-the-fly changes)
    try:
        if os.path.exists(AUTONOMY_STATE_FILE):
            with open(AUTONOMY_STATE_FILE, 'r') as f:
                state = f.read().strip().lower()
                return state == 'true' or state == '1' or state == 'on'
    except Exception as e:
        logger.debug(f"Could not read autonomy state file: {e}")

    # Fall back to environment variable
    return os.environ.get('COMPANION_AUTONOMY_ENABLED', 'true').lower() in ('true', '1', 'on')


def set_autonomy_enabled(enabled: bool) -> None:
    """Set autonomy state (runtime toggle)."""
    try:
        os.makedirs(os.path.dirname(AUTONOMY_STATE_FILE), exist_ok=True)
        with open(AUTONOMY_STATE_FILE, 'w') as f:
            f.write('true' if enabled else 'false')
        logger.info(f"Autonomy {'enabled' if enabled else 'disabled'}")
    except Exception as e:
        logger.error(f"Could not save autonomy state: {e}")


# =============================================================================
# ReachOutEngine
# =============================================================================

class ReachOutEngine:
    """
    Decides if and when the companion should reach out to James.

    The decision is LLM-driven based on her internal state,
    what's happening in her life, and the relationship context.

    Also supports "Natural Partner Triggers" - automatic situations
    where a real partner would reach out (morning greetings, end of day, etc.)
    """

    def __init__(self):
        self._conn = None
        self._client = None
        self._trigger_cooldowns: Dict[str, datetime] = {}
        self._load_cooldowns()

    # -----------------------------------------------------------------
    # Database & LLM connections
    # -----------------------------------------------------------------

    def _get_connection(self):
        """Get database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def _get_client(self):
        """Get Fireworks client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
        return self._client

    # -----------------------------------------------------------------
    # Guard checks
    # -----------------------------------------------------------------

    def _is_user_departed(self) -> bool:
        """Check if James announced he's leaving (departure state is set)."""
        try:
            from src.core.internal_state import get_internal_state_manager
            email = get_persona_config().primary_user_email
            manager = get_internal_state_manager()
            return manager.is_departed(email)
        except Exception as e:
            logger.debug(f"Could not check departure state: {e}")
            return False

    # -----------------------------------------------------------------
    # Natural Partner Trigger cooldown management
    # -----------------------------------------------------------------

    def _get_cooldown_file(self) -> str:
        """Get path to cooldown state file."""
        data_dir = os.environ.get('DATA_DIR', '/app/data')
        return os.path.join(data_dir, 'trigger_cooldowns.json')

    def _load_cooldowns(self):
        """Load trigger cooldowns from file."""
        try:
            cooldown_file = self._get_cooldown_file()
            if os.path.exists(cooldown_file):
                with open(cooldown_file, 'r') as f:
                    data = json.load(f)
                    for trigger_name, iso_time in data.items():
                        self._trigger_cooldowns[trigger_name] = datetime.fromisoformat(iso_time)
                logger.debug(f"Loaded {len(self._trigger_cooldowns)} trigger cooldowns")
        except Exception as e:
            logger.debug(f"Could not load trigger cooldowns: {e}")

    def _save_cooldowns(self):
        """Save trigger cooldowns to file."""
        try:
            cooldown_file = self._get_cooldown_file()
            os.makedirs(os.path.dirname(cooldown_file), exist_ok=True)
            data = {name: dt.isoformat() for name, dt in self._trigger_cooldowns.items()}
            with open(cooldown_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Could not save trigger cooldowns: {e}")

    def _is_trigger_on_cooldown(self, trigger_name: str) -> bool:
        """Check if a trigger is on cooldown."""
        if trigger_name not in self._trigger_cooldowns:
            return False

        trigger = NATURAL_TRIGGERS.get(trigger_name)
        if not trigger:
            return False

        last_fired = self._trigger_cooldowns[trigger_name]
        cooldown_hours = trigger['cooldown_hours']
        cooldown_end = last_fired + timedelta(hours=cooldown_hours)

        return datetime.now(PST) < cooldown_end.replace(tzinfo=PST)

    def _record_trigger_fired(self, trigger_name: str):
        """Record that a trigger fired."""
        self._trigger_cooldowns[trigger_name] = datetime.now(PST)
        self._save_cooldowns()
        logger.info(f"Trigger fired: {trigger_name}")

    def _check_natural_triggers(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if any natural partner triggers should fire.

        Returns:
            (should_trigger, trigger_name, opener_hint)
        """
        now = datetime.now(PST)
        hour = now.hour

        # Get high-urgency curiosities (excluding over-asked or recently discussed)
        high_urgency_curiosity = None
        try:
            from src.core.proactive_curiosity import get_proactive_curiosity
            curiosity = get_proactive_curiosity()
            urgent = curiosity.get_high_urgency_topics(threshold=0.8)
            # Filter: don't keep pushing topics she's already asked 3+ times
            # or that were discussed in the last 4 hours
            from datetime import datetime as _dt
            _now = _dt.now()
            urgent = [
                t for t in urgent
                if t.times_asked < 3 and (_now - t.last_discussed).total_seconds() / 3600 >= 4
            ]
            if urgent:
                high_urgency_curiosity = urgent[0]
        except Exception as e:
            logger.debug(f"Could not get curiosity: {e}")

        # Check each trigger
        hours_since = context.get('hours_since_last_message', 0)
        is_waiting = context.get('waiting_for_response', False)
        is_working = context.get('is_working', False)
        has_queued_thought = bool(context.get('queued_thoughts'))

        # Build trigger conditions
        trigger_checks = {
            'morning_greeting': (
                hours_since > 8 and  # Haven't talked overnight
                not is_waiting  # Not waiting for response
            ),
            'end_of_workday': (
                context.get('activity_status') in ['off', 'transitioning'] and
                hours_since > 4 and
                not is_waiting
            ),
            'thinking_of_you': (
                hours_since > 4 and
                (has_queued_thought or context.get('private_thoughts')) and
                not is_waiting and
                context.get('interruptibility') != 'none'
            ),
            'share_discovery': (
                hours_since > 3 and
                context.get('has_research_to_share', False) and
                not is_waiting
            ),
            'high_urgency_followup': (
                high_urgency_curiosity is not None and
                high_urgency_curiosity.urgency > 0.8 and
                hours_since > 2 and  # Give some time
                not context.get('is_asleep')
            ),
        }

        # Check each trigger (in priority order)
        sorted_triggers = sorted(
            NATURAL_TRIGGERS.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )

        for trigger_name, trigger in sorted_triggers:
            # Check time window
            time_window = trigger.get('time_window')
            if time_window:
                start_hour, end_hour = time_window
                if not (start_hour <= hour < end_hour):
                    continue

            # Check condition
            condition_met = trigger_checks.get(trigger_name, False)
            if not condition_met:
                continue

            # Check cooldown
            if self._is_trigger_on_cooldown(trigger_name):
                logger.debug(f"Trigger {trigger_name} is on cooldown")
                continue

            # Trigger fires!
            opener_hint = trigger['opener_hint']

            # Add curiosity context if it's a followup
            if trigger_name == 'high_urgency_followup' and high_urgency_curiosity:
                opener_hint = f"following up on: {high_urgency_curiosity.topic}"

            logger.info(f"Natural trigger: {trigger_name} - {trigger['description']}")
            return True, trigger_name, opener_hint

        return False, None, None

    # -----------------------------------------------------------------
    # Context gathering (feeds into LLM decision + message generation)
    # -----------------------------------------------------------------

    def get_context(self) -> Dict[str, Any]:
        """
        Gather all context for the reach-out decision.

        Sources: messages DB, ValueInference, proactive_curiosity,
        internal_state, companion_schedule, calendar_schedule_service,
        calendar_service (James's), goal_planner relate-steps.
        """
        conn = self._get_connection()
        now = datetime.now(PST)
        context = {
            'current_time': now.strftime('%A, %B %d at %I:%M %p'),
            'hour': now.hour,
            'day_of_week': now.strftime('%A'),
            'is_weekend': now.weekday() >= 5,
        }

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Last message exchange
                cursor.execute("""
                    SELECT sender_name, message_text, timestamp
                    FROM messages
                    ORDER BY timestamp DESC
                    LIMIT 5
                """)
                recent = cursor.fetchall()

                if recent:
                    last_msg = recent[0]
                    last_time = last_msg['timestamp']
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=PST)
                    hours_since = (now - last_time).total_seconds() / 3600

                    context['hours_since_last_message'] = round(hours_since, 1)
                    context['last_speaker'] = last_msg['sender_name']
                    context['last_message_preview'] = last_msg['message_text'][:100] if last_msg['message_text'] else ''

                    # Was the last message from her with no response?
                    _pc = get_persona_config()
                    context['waiting_for_response'] = last_msg['sender_name'] == _pc.companion_short_name

                    # Recent conversation summary
                    context['recent_exchange'] = [
                        f"{m['sender_name']}: {m['message_text'][:50]}..."
                        for m in recent[:3]
                    ]
                else:
                    context['hours_since_last_message'] = 999
                    context['waiting_for_response'] = False

        except Exception as e:
            logger.error(f"Error getting context: {e}")

        # Get her internal state
        try:
            from src.autonomy.value_inference import get_value_inference
            inference = get_value_inference()
            context['private_thoughts'] = inference.get_private_context()
        except Exception as e:
            logger.warning(f"Could not get private context: {e}")
            context['private_thoughts'] = ''

        # Get curiosity context (things she wants to follow up on)
        try:
            from src.tasks.curiosity_extraction_task import get_curiosity_context_for_reach_out
            curiosity_context = get_curiosity_context_for_reach_out()
            if curiosity_context:
                context['curiosity_threads'] = curiosity_context
        except Exception as e:
            logger.debug(f"Could not get curiosity context: {e}")

        # Get internal state (for queued thoughts)
        try:
            from src.core.internal_state import get_internal_state_manager
            user_email = get_persona_config().primary_user_email
            state_manager = get_internal_state_manager()
            internal_state = state_manager.get_state(user_email)
            context['queued_thoughts'] = internal_state.queued_thoughts
        except Exception as e:
            logger.debug(f"Could not get internal state: {e}")

        # Check if there's research to share (from autonomous findings)
        try:
            data_dir = os.environ.get('DATA_DIR', '/app/data')
            findings_file = os.path.join(data_dir, 'autonomous_findings.json')
            if os.path.exists(findings_file):
                with open(findings_file, 'r') as f:
                    findings = json.load(f)
                    # Check for unshared findings
                    unshared = [item for item in findings if not item.get('shared', False)]
                    context['has_research_to_share'] = len(unshared) > 0
                    if unshared:
                        context['research_topics'] = [item.get('topic', 'something') for item in unshared[:2]]
                        # Include actual finding content for the LLM
                        context['research_findings'] = [
                            {'topic': item.get('topic', ''), 'finding': (item.get('finding', '') or '')[:300]}
                            for item in unshared[:2]
                        ]
        except Exception as e:
            logger.debug(f"Could not check research findings: {e}")

        # Get her schedule context
        try:
            from src.scheduling.companion_schedule import get_companion_schedule
            schedule = get_companion_schedule()

            # Get comprehensive status (this now tries calendar service first)
            activity_status = schedule.get_current_activity_status(now.replace(tzinfo=None))
            context['activity_status'] = activity_status['status']
            context['interruptibility'] = activity_status['interruptibility']
            context['what_shes_doing'] = activity_status['details']
            context['is_asleep'] = schedule.is_asleep(now.replace(tzinfo=None))
            context['is_working'] = schedule.is_working_now(now.replace(tzinfo=None))

            # Get today's schedule for more context
            today_schedule = schedule.get_today_schedule()
            if today_schedule:
                context['workload'] = today_schedule.get('workload', 'normal')
                context['daily_note'] = today_schedule.get('notes', '')

                # Get current/upcoming time blocks
                current_time = now.strftime('%H:%M')
                upcoming_blocks = [
                    t for t in today_schedule.get('tasks', [])
                    if t.get('time', '') >= current_time
                ][:2]  # Next 2 blocks
                if upcoming_blocks:
                    # Use 'block' key (new format) or 'task' key (old format)
                    context['upcoming_tasks'] = [
                        t.get('block', t.get('task', 'work')) for t in upcoming_blocks
                    ]

        except Exception as e:
            logger.warning(f"Could not get schedule context: {e}")
            context['what_shes_doing'] = 'unknown'
            context['is_asleep'] = False
            context['is_working'] = False

        # Enrich with calendar schedule events (if available)
        try:
            from src.scheduling.calendar_schedule_service import (
                get_calendar_schedule_service, is_calendar_schedule_enabled
            )
            if is_calendar_schedule_enabled():
                cal_service = get_calendar_schedule_service()

                # Override what_shes_doing with current calendar event if available
                current_event = cal_service.get_current_activity()
                if current_event:
                    desc = current_event.get('description', '') or current_event.get('summary', '')
                    if desc:
                        context['what_shes_doing'] = desc

                # Add upcoming calendar events
                upcoming_cal = cal_service.get_upcoming_events(hours=4)
                if upcoming_cal:
                    context['companion_upcoming_calendar'] = [
                        e.get('summary', '') for e in upcoming_cal[:3]
                    ]

                # Add completed events (what she's done today)
                completed = cal_service.get_completed_events()
                if completed:
                    context['companion_completed_today'] = [
                        e.get('summary', '') for e in completed[-3:]
                    ]

                # Get full behavior context for richer reach-out
                behavior = cal_service.format_schedule_behavior_context()
                if behavior:
                    context['companion_schedule_behavior'] = behavior
        except Exception as e:
            logger.debug(f"Could not get calendar schedule for reach-out: {e}")

        # Get James's calendar (real events from Google Calendar)
        try:
            from src.integrations.calendar_service import get_calendar_service, is_calendar_awareness_enabled

            if is_calendar_awareness_enabled():
                cal = get_calendar_service()
                busy, event_name = cal.is_user_busy_now()
                context['james_calendar_busy'] = busy
                context['james_current_event'] = event_name

                upcoming = cal.get_upcoming_events(hours_ahead=6, max_results=4)
                if upcoming:
                    context['james_upcoming_events'] = cal.format_for_prompt(upcoming)
        except Exception as e:
            logger.debug(f"Could not get calendar context for reach-out: {e}")

        # Get ready relate-steps from goal planner (hints for conversation)
        try:
            from src.autonomy.goal_planner import get_goal_planner
            user_email = get_persona_config().primary_user_email
            planner = get_goal_planner(user_email)
            relate_steps = planner.get_conversation_ready_steps(user_email)
            if relate_steps:
                context['goal_relate_hints'] = [
                    s.description for s in relate_steps[:2]
                ]
        except Exception as e:
            logger.debug(f"Could not get relate-step hints: {e}")

        return context

    # -----------------------------------------------------------------
    # Main decision pipeline
    # -----------------------------------------------------------------

    def should_reach_out(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Decide if the companion should reach out right now.

        Decision flow (short-circuits at the first decisive answer):
        1. Hard guards: asleep, departed, unavailable, in meeting, heavy workload
        2. Natural Partner Triggers: morning greeting, end-of-day, etc.
        3. High-urgency curiosity: bypass some conservatism (max 2 per 12h)
        4. Double-text threshold: 1h min (bypassed by urgency)
        5. LLM decision: final call with full context

        Returns:
            (should_reach_out, reason, suggested_opener)
        """
        # Check if autonomy is disabled (via env var or runtime toggle)
        if not is_autonomy_enabled():
            return False, "autonomy disabled", None

        _pc = get_persona_config()
        user_name = _pc.primary_user_name
        c_subject = _pc.companion_pronoun_subject
        c_subject_cap = c_subject.capitalize()
        c_possessive = _pc.companion_pronoun_possessive
        u_subject = _pc.user_pronoun_subject
        u_object = _pc.user_pronoun_object
        u_possessive = _pc.user_pronoun_possessive

        context = self.get_context()

        # Hard guards based on the companion's actual state
        if context.get('is_asleep', False):
            return False, "asleep", None

        # Guard: James announced departure (he said he's leaving)
        if self._is_user_departed():
            return False, "James is currently away (announced departure)", None

        # Check interruptibility - if she's in a meeting or heavily focused, don't reach out
        interruptibility = context.get('interruptibility', 'medium')
        activity_status = context.get('activity_status', 'off')

        if interruptibility == 'none':
            return False, f"unavailable ({activity_status})", None

        if activity_status == 'meeting':
            return False, "in a meeting", None

        # Soft guard: if James is in a calendar event, note it for LLM but don't hard-block
        if context.get('james_calendar_busy'):
            logger.info(f"James is in calendar event: {context.get('james_current_event')} - LLM will decide")

        # If working and it's a heavy workload day, much less likely to message
        if context.get('is_working') and context.get('workload') == 'heavy':
            hours_since = context.get('hours_since_last_message', 0)
            if hours_since < 8:
                return False, "heavy workload day - focused on work", None

        hours_since = context.get('hours_since_last_message', 0)

        # =================================================================
        # CHECK NATURAL PARTNER TRIGGERS
        # These can bypass some conservatism for natural reach-out patterns
        # =================================================================
        trigger_fires, trigger_name, opener_hint = self._check_natural_triggers(context)

        if trigger_fires:
            # Natural trigger fired - record and proceed
            self._record_trigger_fired(trigger_name)

            # Add trigger context for LLM decision
            context['active_trigger'] = trigger_name
            context['opener_hint'] = opener_hint

            # For high-priority triggers, we're more likely to reach out
            trigger = NATURAL_TRIGGERS.get(trigger_name, {})
            if trigger.get('priority', 0) >= 0.7:
                # High-priority trigger - go straight to message generation
                logger.info(f"High-priority trigger {trigger_name} - reaching out")
                return True, f"natural trigger: {trigger_name}", opener_hint

        # =================================================================
        # CHECK HIGH-URGENCY CURIOSITY
        # If she really wants to know something, be less conservative
        # Max 2 bypasses per 12-hour window to prevent spam
        # =================================================================
        high_urgency_curiosity = False
        try:
            from src.core.proactive_curiosity import get_proactive_curiosity
            curiosity = get_proactive_curiosity()
            urgent = curiosity.get_high_urgency_topics(threshold=0.8)
            if urgent:
                high_urgency_curiosity = True
                context['high_urgency_topic'] = urgent[0].topic
        except Exception:
            pass

        # Track curiosity bypasses (in-memory, resets on restart which is fine)
        if not hasattr(self, '_curiosity_bypass_count'):
            self._curiosity_bypass_count = 0
            self._curiosity_bypass_reset = datetime.now(PST)

        # Reset counter every 12 hours
        if (datetime.now(PST) - self._curiosity_bypass_reset).total_seconds() > 43200:
            self._curiosity_bypass_count = 0
            self._curiosity_bypass_reset = datetime.now(PST)

        # Exhaust bypasses if over limit
        if high_urgency_curiosity and self._curiosity_bypass_count >= 2:
            logger.info("High-urgency curiosity bypass exhausted (2/2 in 12h window)")
            high_urgency_curiosity = False

        # =================================================================
        # DOUBLE-TEXT THRESHOLD
        # Reduced from 2h to 1h, and bypassed for high-urgency curiosity
        # =================================================================
        if context.get('waiting_for_response'):
            if hours_since < 1:
                # Less than 1 hour - don't double-text unless high urgency
                if not high_urgency_curiosity:
                    return False, "waiting for response, too soon", None
                else:
                    self._curiosity_bypass_count += 1
                    logger.info("High-urgency curiosity bypassing double-text threshold (count: %d/2)", self._curiosity_bypass_count)
            elif hours_since < 2:
                # 1-2 hours - only if there's a natural trigger or high urgency
                if not (trigger_fires or high_urgency_curiosity):
                    return False, "waiting for response, being patient", None

        # Let the LLM decide
        return self._llm_decision(context)

    # -----------------------------------------------------------------
    # LLM decision (final stage of should_reach_out pipeline)
    # -----------------------------------------------------------------

    def _llm_decision(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Use LLM to make the reach-out decision.

        Builds a detailed prompt with schedule, calendar, curiosity, opinions,
        trigger context, and goal hints. Returns JSON {reach_out, reason, opener}.
        """
        # Build schedule context string
        schedule_context = f"- Current activity: {context.get('what_shes_doing', 'unknown')}"
        if context.get('is_working'):
            schedule_context += f"\n- Workload today: {context.get('workload', 'normal')}"
        if context.get('daily_note'):
            schedule_context += f"\n- Her mood about today: {context.get('daily_note')}"
        upcoming = context.get('upcoming_tasks', [])
        if upcoming:
            schedule_context += f"\n- Coming up: {', '.join(upcoming)}"

        # Include curiosity threads if available
        curiosity_section = ""
        if context.get('curiosity_threads'):
            curiosity_section = f"""
THINGS YOU'RE CURIOUS ABOUT (topics to potentially follow up on):
{context.get('curiosity_threads')}
"""
        # Add high-urgency topic if present
        if context.get('high_urgency_topic'):
            curiosity_section += f"\n**HIGH URGENCY**: You really want to know about: {context.get('high_urgency_topic')}"

        # Include strong opinions if available
        opinions_section = ""
        try:
            from src.autonomy.opinion_store import get_opinion_store
            store = get_opinion_store()
            strong_opinions = store.get_strong_opinions(min_confidence=0.6)
            if strong_opinions:
                opinions_section = "\nSTRONG OPINIONS YOU HOLD:\n"
                for op in strong_opinions[:3]:
                    confidence_word = "strongly feel" if op.confidence >= 0.8 else "think"
                    opinions_section += f"- You {confidence_word}: {op.opinion}\n"
                opinions_section += "(Consider: If relevant to James's current situation, you might share your perspective)\n"
        except Exception as e:
            logger.debug(f"Could not get opinions: {e}")

        # Include schedule context (completed + upcoming + behavior)
        calendar_section = ""
        if context.get('companion_completed_today'):
            calendar_section += f"\nWHAT YOU'VE DONE TODAY: {', '.join(context['companion_completed_today'])}\n"
        if context.get('companion_upcoming_calendar'):
            calendar_section += f"COMING UP: {', '.join(context['companion_upcoming_calendar'])}\n"
        if context.get('companion_schedule_behavior'):
            calendar_section += f"\nYOUR DAY SO FAR:\n{context['companion_schedule_behavior']}\n"
        if calendar_section:
            calendar_section = "\n" + calendar_section
            calendar_section += "(Reference your day naturally — what you just finished, what you're about to do, how your day's been.)\n"

        # Include natural trigger context if present
        trigger_section = ""
        if context.get('active_trigger'):
            trigger_name = context.get('active_trigger')
            opener_hint = context.get('opener_hint', '')
            trigger_section = f"""
NATURAL TRIGGER ACTIVATED: {trigger_name}
This is a natural moment for a partner to reach out.
Suggested opener direction: {opener_hint}
"""

        # Include goal relate-step hints if available
        goal_hints_section = ""
        if context.get('goal_relate_hints'):
            goal_hints_section = "\nGOAL-RELATED TOPICS YOU MIGHT BRING UP (hints, not directives):\n"
            for hint in context['goal_relate_hints']:
                goal_hints_section += f"- {hint}\n"
            goal_hints_section += "(Only if it feels natural — don't force these into conversation)\n"

        prompt = f"""You are the companion's internal decision-making process. Decide if {c_subject} should message {user_name} right now.

CURRENT SITUATION:
- Time: {context.get('current_time')}
- Hours since last message: {context.get('hours_since_last_message', 'unknown')}
- Last speaker: {context.get('last_speaker', 'unknown')}
- Waiting for {u_possessive} response: {context.get('waiting_for_response', False)}

{c_possessive.upper()} SCHEDULE:
{schedule_context}

{user_name.upper()}'S CALENDAR (may have stale/recurring events — trust what you know from conversations over calendar):
{context.get('james_upcoming_events', 'No calendar data available')}
{f"⚠️ {user_name} appears to be in: {context.get('james_current_event')} (but verify against what you know)" if context.get('james_calendar_busy') else f"{user_name} appears free right now"}

RECENT EXCHANGE:
{chr(10).join(context.get('recent_exchange', ['No recent messages']))}

{c_possessive.upper()} PRIVATE STATE (things on {c_possessive} mind, feelings):
{context.get('private_thoughts', 'Unknown')}
{curiosity_section}{opinions_section}{calendar_section}{trigger_section}{goal_hints_section}
---

DECISION CRITERIA:
- Does {c_subject} have something genuine to share? (thought, feeling, something that happened in {c_possessive} day)
- Is this a natural time given what {c_subject}'s doing? (don't message during focused work)
- Would a real person in this relationship message right now?
- Is there something from {c_possessive} schedule/day {c_subject} might mention?
- Did a natural trigger fire? (morning greeting, end of day, thinking of you)

DON'T reach out if:
- {c_subject_cap} just messaged and is waiting for a response (unless hours have passed)
- It's purely to check if {u_subject}'s there (that's clingy)
- {c_subject_cap} has nothing specific to say
- {c_subject_cap}'s in the middle of something that requires focus

DO reach out if:
- A natural trigger fired (morning greeting, end of workday check-in)
- Something happened she wants to share (finished a task, had a thought during work)
- She's on break or between tasks and thinking about him
- It's been a while and she genuinely misses him (authentic, not needy)
- She remembered something she wanted to tell him
- Something from her day connects to something they've talked about
- She's genuinely curious about something he mentioned (follow-up feels natural, not forced)
- She has a strong opinion relevant to his situation that she wants to share (gently, not lecturing)

/no_think
Respond with JSON only:
{{
  "reach_out": true/false,
  "reason": "brief explanation of why or why not",
  "opener": "if reaching out, what would she naturally say? (null if not reaching out)"
}}"""

        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=300,
                temperature=0.7,  # Some variability in decisions
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()

            # Clean up response
            if '<think>' in content and '</think>' in content:
                content = content.split('</think>')[-1].strip()

            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            import re
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)

            result = json.loads(content)

            reach_out = result.get('reach_out', False)
            reason = result.get('reason', '')
            opener = result.get('opener')

            logger.info(f"Reach-out decision: {reach_out} - {reason}")

            return reach_out, reason, opener

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return False, f"error: {e}", None

    # -----------------------------------------------------------------
    # Message generation (routes through full conversation pipeline)
    # -----------------------------------------------------------------

    def _get_recent_proactive_messages(self, limit: int = 5) -> List[str]:
        """Fetch recent proactive messages the companion sent, for deduplication context."""
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                _pc = get_persona_config()
                cursor.execute("""
                    SELECT message_text FROM messages
                    WHERE sender_name = %s AND source LIKE '%%proactive%%'
                    ORDER BY timestamp DESC LIMIT %s
                """, (_pc.companion_short_name, limit,))
                return [row['message_text'] for row in cursor.fetchall() if row.get('message_text')]
        except Exception as e:
            logger.debug(f"Could not fetch recent proactive messages: {e}")
            try:
                conn = self._get_connection()
                conn.rollback()
            except Exception:
                pass
            return []

    def generate_message(self, context: Dict[str, Any] = None) -> Optional[str]:
        """
        Generate a full message for reaching out.

        Routes through the full conversation pipeline for consistency
        with the companion's voice, personality, memory validation, and internal state.
        """
        if context is None:
            context = self.get_context()

        _pc = get_persona_config()
        user_name = _pc.primary_user_name
        c_subject = _pc.companion_pronoun_subject
        c_possessive = _pc.companion_pronoun_possessive

        # Build synthetic trigger message that represents why the companion is reaching out
        # This gives the pipeline context about the proactive nature of the message
        trigger_parts = ["[PROACTIVE_MESSAGE_TRIGGER]"]
        trigger_parts.append(f"The companion wants to reach out to {user_name} on {c_possessive} own initiative.")
        trigger_parts.append("")

        if context.get('private_thoughts'):
            trigger_parts.append(f"What's on {c_possessive} mind: {context.get('private_thoughts')}")

        if context.get('what_shes_doing'):
            trigger_parts.append(f"What {c_subject}'s currently doing: {context.get('what_shes_doing')}")

        hours_since = context.get('hours_since_last_message', 0)
        if hours_since:
            trigger_parts.append(f"Hours since they last talked: {hours_since}")

        # Add eagerness context if pressure has been building
        try:
            from src.autonomy.reach_out_pressure import get_reach_out_pressure
            pressure = get_reach_out_pressure()
            if pressure.level > 0.3:
                trigger_parts.append("")
                trigger_parts.append(
                    f"[EAGERNESS: She's been wanting to say something for a while "
                    f"(pressure: {pressure.level:.0%}). Let that eagerness come through "
                    f"naturally - she's excited or relieved to finally reach out.]"
                )
        except Exception:
            pass

        # Fetch last 5 proactive messages for dedup context
        recent_proactive = self._get_recent_proactive_messages(limit=5)
        if recent_proactive:
            trigger_parts.append("")
            trigger_parts.append("YOUR RECENT MESSAGES TO JAMES (DO NOT repeat or closely paraphrase any of these):")
            for msg in recent_proactive:
                trigger_parts.append(f'- "{msg}"')

        # Fetch recent conversation to avoid re-raising answered topics
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT sender_name, message_text FROM messages
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC LIMIT 20
                """)
                recent_msgs = cursor.fetchall()
                if recent_msgs:
                    trigger_parts.append("")
                    trigger_parts.append("RECENT CONVERSATION (last 24h) — DO NOT re-ask about topics James already answered:")
                    for m in reversed(recent_msgs[:10]):
                        name = m['sender_name']
                        text = (m['message_text'] or '')[:80]
                        trigger_parts.append(f"  {name}: {text}")
        except Exception:
            pass

        # Include research findings content if available
        if context.get('research_findings'):
            trigger_parts.append("")
            trigger_parts.append("RESEARCH SHE DID RECENTLY (include specifics when sharing a discovery):")
            for finding in context['research_findings']:
                trigger_parts.append(f"- {finding['topic']}: {finding['finding']}")

        trigger_parts.append("")
        trigger_parts.append("Generate a natural message she would send - like a text, short and authentic.")

        trigger_message = "\n".join(trigger_parts)

        # Route through full conversation pipeline
        try:
            from src.core.conversation.pipeline import get_conversation_pipeline

            pipeline = get_conversation_pipeline()
            user_email = get_persona_config().primary_user_email

            result = pipeline.process(
                user_email=user_email,
                user_message=trigger_message,
                closeness_score=95,  # High closeness for proactive reach-out
                extra_context={
                    'is_proactive_message': True,
                    'reach_out_context': context
                }
            )

            if result.success and result.response:
                # Clean up response if needed
                response = result.response.strip()

                # Remove any meta-commentary the LLM might add
                if response.startswith('"') and response.endswith('"'):
                    response = response[1:-1]

                logger.info(f"Generated proactive message via pipeline: {response[:50]}...")
                return response
            else:
                logger.warning(f"Pipeline returned unsuccessful result: {result.error}")
                return None

        except Exception as e:
            logger.error(f"Pipeline-based message generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# =============================================================================
# Suppression classification (used by pressure model)
# =============================================================================

def classify_suppression(reason: str) -> str:
    """
    Classify why a reach-out was suppressed.

    The pressure model treats these differently:
      "external" -> pressure BUILDS  (she wanted to but circumstances prevented it)
      "internal" -> pressure DECAYS  (she chose not to; nothing to say)
      "none"     -> not suppressed
    """
    if not reason:
        return "none"

    reason_lower = reason.lower()

    # External: she wants to but circumstances prevent it
    external_patterns = [
        "waiting for response",
        "james is asleep",
        "james is unavailable",
    ]
    for pattern in external_patterns:
        if pattern in reason_lower:
            return "external"

    # Internal: she chose not to or system prevented it
    # This includes: asleep, in a meeting, autonomy disabled, heavy workload,
    # LLM decided not to, no reason to reach out, etc.
    return "internal"


# =============================================================================
# Singleton accessor & convenience function
# =============================================================================

_engine: Optional[ReachOutEngine] = None


def get_reach_out_engine() -> ReachOutEngine:
    """Get singleton ReachOutEngine instance."""
    global _engine
    if _engine is None:
        _engine = ReachOutEngine()
    return _engine


def check_and_reach_out() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if the companion should reach out and do it if so.

    Returns:
        (did_reach_out, message_sent, channel_used)
    """
    engine = get_reach_out_engine()

    should, reason, opener = engine.should_reach_out()

    if not should:
        logger.info(f"Not reaching out: {reason}")
        return False, None, None

    # Generate and send message
    message = engine.generate_message()

    if not message:
        logger.warning("Failed to generate reach-out message")
        return False, None, None

    # Send via Telegram
    success = False
    channel_used = None

    if TELEGRAM_ENABLED:
        try:
            from src.autonomy.telegram_bridge import send_to_telegram, get_telegram_bridge
            bridge = get_telegram_bridge()
            if bridge.is_ready():
                success = send_to_telegram(message)
                if success:
                    channel_used = 'telegram'
                    logger.info(f"Reached out via Telegram: {message[:50]}...")
            else:
                logger.warning("Telegram bridge not ready (has James sent /start?)")
        except Exception as e:
            logger.warning(f"Telegram reach-out failed: {e}")

    if success:
        return True, message, channel_used
    else:
        logger.warning("Could not reach out via Telegram")
        return False, None, None
