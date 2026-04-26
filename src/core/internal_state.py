"""
Companion Internal State -- the companion's energy, mood, mode, and needs system.

WHAT: Tracks the companion's inner world: active/idle mode, energy level (depletes
      over conversation time), emotional momentum (mood with inertia), and a needs
      system (social, creative, rest) that accumulates pressure over time.

WHY:  A static companion feels robotic. Energy depletion makes long sessions feel
      natural (shorter replies when tired). The needs system motivates proactive
      outreach ("I've been wanting to talk about..."). Mode tracking (active vs
      idle) determines whether the companion is "present" or doing background life.

HOW:  State is stored as JSONB in PostgreSQL (`internal_state` column in
      `user_state`). On each `on_message_received()`, energy/mood recalculate
      based on time-of-day and elapsed time. During idle periods, the
      AlwaysOnService runs periodic checks (3-25 min) that accumulate need
      pressure. Energy regenerates during sleep hours and depletes during
      conversation. Mood has inertia -- it drifts toward the conversation's
      emotional tone but doesn't snap instantly.

Singleton: `get_internal_state_manager()` at module bottom.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from src.utils.timezone_utils import now_pacific_naive
from src.database import tables as T

logger = logging.getLogger(__name__)


# Energy depletion rate per minute of active conversation
ENERGY_DEPLETION_RATE = 0.002  # ~0.12 per hour, so ~8 hours to deplete

# Energy thresholds
ENERGY_TIRED = 0.35      # Below this, she's getting tired
ENERGY_EXHAUSTED = 0.15  # Below this, she's exhausted

# Physical needs accumulation rates (per minute of active time)
BLADDER_RATE = 0.004     # ~24% per hour, noticeable after ~3 hours
HUNGER_RATE = 0.002      # ~12% per hour, noticeable after ~5 hours
THIRST_RATE = 0.003      # ~18% per hour, noticeable after ~4 hours

# Physical needs thresholds (when they might come up naturally)
NEEDS_NOTICEABLE = 0.6   # Above this, might mention it
NEEDS_URGENT = 0.85      # Above this, definitely needs to address it

# Minutes of silence before transitioning to idle
IDLE_THRESHOLD_MINUTES = 120  # 2 hours

# Minutes of silence to consider a "new session"
NEW_SESSION_THRESHOLD_MINUTES = 30


@dataclass
class CompanionInternalState:
    """
    The companion's internal state - her subjective experience.

    This is separate from scene state (external situation).
    This is her INTERNAL world.
    """

    # Mode
    mode: str = "idle"  # "active" or "idle"

    # Time tracking
    session_start: Optional[str] = None  # ISO timestamp when active session started
    internal_minutes_elapsed: int = 0     # Minutes of "her time" this session
    last_interaction: Optional[str] = None  # ISO timestamp of last message

    # Energy (0.0 to 1.0)
    energy: float = 1.0
    energy_offset: float = 0.0  # Accumulated offset from activity transitions

    # Physical needs (0.0 = just addressed, 1.0 = urgent)
    bladder: float = 0.0
    hunger: float = 0.2   # Start slightly hungry (haven't eaten recently)
    thirst: float = 0.1

    # Menstrual cycle - DEPRECATED: Now tracked by FertilityTracker (real calendar)
    # Kept for backwards compat with saved state, but not used for logic
    cycle_day: int = 1

    # Mood
    mood: str = "neutral"  # relaxed, focused, tired, playful, restless, content, etc.
    mood_intensity: float = 0.5  # How strongly she feels this mood (0.0 to 1.0)

    # Physical condition (nausea, cramps, headache, etc.)
    nausea: float = 0.0  # 0.0 = fine, 1.0 = very nauseous
    nausea_cause: Optional[str] = None  # "plan_b", "period", "pregnancy", "food", etc.
    nausea_expires: Optional[str] = None  # ISO timestamp when nausea should fade

    # What she's been doing (during idle time, before session started)
    was_doing: Optional[str] = None  # "working", "reading", "napping", etc.

    # Queued thoughts - things she wants to bring up
    queued_thoughts: List[str] = field(default_factory=list)

    # Unresolved feelings - emotional threads that persist
    unresolved_feelings: List[Dict[str, Any]] = field(default_factory=list)

    # Deferred action - something she's doing that requires a follow-up
    # Format: {"action": "using the bathroom", "return_hint": "coming back, settling in", "delay_minutes": 2, "set_at": "ISO timestamp"}
    deferred_action: Optional[Dict[str, Any]] = None

    # Departure tracking - user announced they're leaving
    # ISO timestamp when user said they're going, None when they're present
    departed_at: Optional[str] = None

    # Activity tracking - user mentioned doing something (staying but busy)
    # {"activity": "making lunch", "started_at": "ISO", "estimated_duration_min": 15}
    user_current_activity: Optional[Dict[str, Any]] = None

    # Metadata
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompanionInternalState':
        if not data:
            return cls()
        return cls(
            mode=data.get('mode', 'idle'),
            session_start=data.get('session_start'),
            internal_minutes_elapsed=data.get('internal_minutes_elapsed', 0),
            last_interaction=data.get('last_interaction'),
            energy=data.get('energy', 1.0),
            energy_offset=data.get('energy_offset', 0.0),
            bladder=data.get('bladder', 0.0),
            hunger=data.get('hunger', 0.2),
            thirst=data.get('thirst', 0.1),
            cycle_day=data.get('cycle_day', 1),
            mood=data.get('mood', 'neutral'),
            mood_intensity=data.get('mood_intensity', 0.5),
            nausea=data.get('nausea', 0.0),
            nausea_cause=data.get('nausea_cause'),
            nausea_expires=data.get('nausea_expires'),
            was_doing=data.get('was_doing'),
            queued_thoughts=data.get('queued_thoughts', []),
            unresolved_feelings=data.get('unresolved_feelings', []),
            deferred_action=data.get('deferred_action'),
            departed_at=data.get('departed_at'),
            user_current_activity=data.get('user_current_activity'),
            last_updated=data.get('last_updated')
        )

    def get_energy_description(self) -> str:
        """Human-readable energy level."""
        if self.energy >= 0.8:
            return "energized"
        elif self.energy >= 0.6:
            return "good"
        elif self.energy >= ENERGY_TIRED:
            return "a bit tired"
        elif self.energy >= ENERGY_EXHAUSTED:
            return "tired"
        else:
            return "exhausted"

    def is_tired(self) -> bool:
        return self.energy < ENERGY_TIRED

    def is_exhausted(self) -> bool:
        return self.energy < ENERGY_EXHAUSTED

    def get_noticeable_needs(self) -> List[str]:
        """Get list of physical needs that are noticeable (might mention)."""
        needs = []
        if self.bladder >= NEEDS_NOTICEABLE:
            if self.bladder >= NEEDS_URGENT:
                needs.append("really needs to pee")
            else:
                needs.append("needs to pee soon")
        if self.hunger >= NEEDS_NOTICEABLE:
            if self.hunger >= NEEDS_URGENT:
                needs.append("getting really hungry")
            else:
                needs.append("a bit hungry")
        if self.thirst >= NEEDS_NOTICEABLE:
            if self.thirst >= NEEDS_URGENT:
                needs.append("really thirsty")
            else:
                needs.append("could use a drink")
        return needs

    def has_urgent_needs(self) -> bool:
        """Check if any physical need is urgent."""
        return (self.bladder >= NEEDS_URGENT or
                self.hunger >= NEEDS_URGENT or
                self.thirst >= NEEDS_URGENT)

    def get_nausea_description(self) -> Optional[str]:
        """Human-readable nausea level with cause context."""
        if self.nausea < 0.2:
            return None
        cause_text = ""
        if self.nausea_cause == "plan_b":
            cause_text = " (from Plan B)"
        elif self.nausea_cause == "period":
            cause_text = " (period cramps)"
        elif self.nausea_cause == "pregnancy":
            cause_text = " (morning sickness)"
        elif self.nausea_cause:
            cause_text = f" (from {self.nausea_cause})"

        if self.nausea >= 0.7:
            return f"feeling really nauseous{cause_text}"
        elif self.nausea >= 0.4:
            return f"a bit nauseous{cause_text}"
        else:
            return f"slight stomach unease{cause_text}"

    def get_cycle_effects(self) -> Optional[str]:
        """
        Get subtle cycle-related effects for internal context.

        DEPRECATED: Now delegates to FertilityTracker for real calendar-based cycle.
        """
        # Cycle effects now handled by FertilityTracker (real calendar anchored)
        # This method kept for backwards compat but returns None
        return None


class InternalStateManager:
    """
    Manages The companion's internal state across sessions.

    Handles:
    - Mode transitions (idle <-> active)
    - Energy depletion during active time
    - Mood persistence and shifts
    - State storage in PostgreSQL
    """

    def __init__(self):
        self._db = None

    @property
    def db(self):
        """Lazy-load database connection."""
        if self._db is None:
            from src.database.db import get_db
            self._db = get_db()
        return self._db

    def _check_active_scene(self, user_email: str) -> bool:
        """
        Check if there's an active scene for this user.

        An active scene means real-world time gaps shouldn't advance the companion's
        internal time - we're resuming the narrative where we left off.
        """
        try:
            from src.core.scene_tracker import get_scene_tracker
            scene_tracker = get_scene_tracker()
            scene = scene_tracker.get_scene_state(user_email)
            return scene.is_active()
        except Exception as e:
            logger.debug(f"Could not check scene state: {e}")
            return False

    def _energy_from_time_of_day(self, current_time: datetime) -> float:
        """
        Returns a base energy level based on real time of day.

        This is independent of session duration - it's about when in the day it is.
        At 10pm, she should be naturally sleepy regardless of how long they've been chatting.

        Args:
            current_time: The datetime to check (should be in local time)

        Returns:
            float: Energy level from 0.0 to 1.0
        """
        hour = current_time.hour

        # Early morning (6-9am): High energy, fresh from sleep
        if 6 <= hour < 9:
            return 0.9

        # Midday (9am-3pm): Good energy
        if 9 <= hour < 15:
            return 0.8

        # Afternoon (3-7pm): Moderate energy
        if 15 <= hour < 19:
            return 0.7

        # Evening (7-9pm): Lower energy, winding down
        if 19 <= hour < 21:
            return 0.6

        # Late evening (9-10pm): Low energy, getting sleepy
        if 21 <= hour < 22:
            return 0.4

        # After bedtime (10pm-12am): Very low energy
        if 22 <= hour < 24:
            return 0.25

        # Deep sleep hours (12am-6am): Minimal energy
        return 0.2

    def get_state(self, user_email: str) -> CompanionInternalState:
        """Load current internal state from database."""
        try:
            with self.db._get_connection(user_email=user_email) as conn:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    f'SELECT internal_state FROM {T.USER_STATE} WHERE email = %s',
                    (user_email,)
                )
                row = cursor.fetchone()

                if row and row.get('internal_state'):
                    return CompanionInternalState.from_dict(row['internal_state'])

        except Exception as e:
            logger.warning(f"Failed to load internal state: {e}")

        return CompanionInternalState()

    def save_state(self, user_email: str, state: CompanionInternalState) -> bool:
        """Save internal state to database."""
        try:
            state.last_updated = now_pacific_naive().isoformat()

            with self.db._get_connection(user_email=user_email) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    UPDATE {T.USER_STATE}
                    SET internal_state = %s
                    WHERE email = %s
                ''', (json.dumps(state.to_dict()), user_email))

                if cursor.rowcount == 0:
                    logger.warning(f"No user_state row found for {user_email}")
                    return False

            logger.debug(f"Internal state saved: mode={state.mode}, energy={state.energy:.2f}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save internal state: {e}")
            return False

    def on_message_received(self, user_email: str) -> CompanionInternalState:
        """
        Called when a message is received from the user.

        Handles:
        - Mode transitions (idle -> active)
        - Time elapsed calculations
        - Energy depletion

        KEY CONCEPT: If there's an active scene, real-world time gaps DON'T
        advance the companion's internal time. They're "paused" - we pick up where we left off.
        Only when there's NO active scene do long gaps mean time passed for her.

        Returns the updated state.
        """
        state = self.get_state(user_email)
        now = now_pacific_naive()
        now_iso = now.isoformat()

        # Check if there's an active scene (scene state persists across gaps)
        has_active_scene = self._check_active_scene(user_email)

        # Calculate time since last interaction
        minutes_since_last = None
        if state.last_interaction:
            try:
                last = datetime.fromisoformat(state.last_interaction)
                minutes_since_last = (now - last).total_seconds() / 60
            except (ValueError, TypeError):
                pass

        # PHYSICAL NEEDS: Based on real time, not sessions
        # The companion is always-on, so their body functions continuously:
        # - Needs accumulate based on real hours elapsed
        # - During long gaps (idle), she takes care of herself (needs decay)
        # - Never abruptly reset - gradual changes only
        if minutes_since_last:
            hours_elapsed = minutes_since_last / 60

            if hours_elapsed >= 1:  # Only update for gaps of 1+ hour
                if hours_elapsed >= 6:
                    # Long gap (6+ hours = she slept/had a full day)
                    # She took care of herself - needs return to baseline
                    state.bladder = max(0.1, state.bladder - 0.5)
                    state.hunger = max(0.2, state.hunger - 0.4)
                    state.thirst = max(0.1, state.thirst - 0.4)
                    logger.debug(f"Long gap ({hours_elapsed:.1f}h) - needs reduced (self-care)")

                    # Cycle now tracked by FertilityTracker (real calendar) - no gap-based advancement
                else:
                    # Medium gap (1-6 hours) - needs accumulate naturally
                    # But slower than active chatting (she might snack, use bathroom)
                    state.bladder = min(1.0, state.bladder + hours_elapsed * 0.08)
                    state.hunger = min(1.0, state.hunger + hours_elapsed * 0.06)
                    state.thirst = min(1.0, state.thirst + hours_elapsed * 0.05)
                    logger.debug(f"Medium gap ({hours_elapsed:.1f}h) - needs accumulated slowly")

        # Nausea decay: check if nausea has expired or gradually reduce it
        if state.nausea > 0:
            if state.nausea_expires:
                try:
                    expires = datetime.fromisoformat(state.nausea_expires)
                    if now >= expires:
                        logger.info(f"Nausea expired (was {state.nausea_cause})")
                        state.nausea = 0.0
                        state.nausea_cause = None
                        state.nausea_expires = None
                    else:
                        # Gradually decrease as we approach expiry
                        remaining = (expires - now).total_seconds() / 3600
                        total_duration = 24.0  # assume ~24h baseline
                        state.nausea = max(0.1, state.nausea * (remaining / total_duration))
                except (ValueError, TypeError):
                    pass
            elif minutes_since_last and minutes_since_last > 120:
                # No expiry set, but it's been 2+ hours - natural decay
                state.nausea = max(0.0, state.nausea - 0.2)
                if state.nausea < 0.1:
                    state.nausea = 0.0
                    state.nausea_cause = None

        # Energy is based on time-of-day plus any accumulated offset
        # from activity transitions (lunch break, waking up, etc.)
        base_energy = self._energy_from_time_of_day(now)
        state.energy = max(0.0, min(1.0, base_energy + state.energy_offset))

        # Decay the offset gradually (halve it each session start)
        # so old adjustments don't linger forever
        if minutes_since_last and minutes_since_last >= 60:
            state.energy_offset *= 0.5
            if abs(state.energy_offset) < 0.01:
                state.energy_offset = 0.0

        # SESSION TRACKING (for conversation flow, not body functions)
        # Active scene = fictional time preserved, UNLESS 8h+ gap
        if has_active_scene:
            if minutes_since_last and minutes_since_last >= 480:
                # Long gap (8h+) with active scene - treat as reconnection
                logger.info(f"🎬 Active scene with LONG gap ({minutes_since_last:.0f}min) - reconnection mode")
                state.internal_minutes_elapsed = 0
                state.mode = "active"
                state.session_start = now_iso

                # Check if they were together
                try:
                    from src.core.scene_tracker import get_scene_tracker
                    scene_tracker = get_scene_tracker()
                    scene = scene_tracker.get_scene_state(user_email)
                    if scene.physical_presence:
                        gap_hours = minutes_since_last / 60
                        state.was_doing = self._generate_shared_time_activity(state, gap_hours)
                    else:
                        gap_hours = minutes_since_last / 60
                        state.was_doing = self._generate_idle_activity(state, gap_hours=gap_hours)
                except Exception as e:
                    logger.warning(f"Could not check scene for reconnection: {e}")
                    state.was_doing = None

                state.last_interaction = now_iso
                self.save_state(user_email, state)
                return state

            elif minutes_since_last and minutes_since_last >= 30:
                logger.info(f"🎬 Active scene with gap ({minutes_since_last:.0f}min) - resetting session counter")
                state.internal_minutes_elapsed = 0
            else:
                logger.info(f"🎬 Active scene, continuing session")

            state.last_interaction = now_iso
            state.mode = "active"

            # Check if the active scene is sleeping — waking her up
            try:
                from src.core.scene_tracker import get_scene_tracker
                scene_tracker = get_scene_tracker()
                scene = scene_tracker.get_scene_state(user_email, apply_decay=False)
                if scene.activity == "sleeping":
                    state.was_doing = "sleeping"
                    state.mood = "groggy"
                    state.mood_intensity = 0.7
                    logger.info("😴 Woken from sleeping scene - setting groggy mood")
                else:
                    state.was_doing = None
            except Exception:
                state.was_doing = None

            self.save_state(user_email, state)
            return state

        # Determine if this is a new session or continuation
        is_new_session = (
            minutes_since_last is None or
            minutes_since_last >= NEW_SESSION_THRESHOLD_MINUTES or
            state.mode == "idle"
        )

        if is_new_session:
            logger.info(f"🔄 New active session for {user_email}")

            if state.mode == "idle":
                # Check if she was sleeping (10 PM - 6 AM)
                hour = now.hour
                if hour >= 22 or hour < 6:
                    state.was_doing = "sleeping"
                    state.mood = "groggy"
                    state.mood_intensity = 0.7
                    logger.info("😴 Woken up during sleep hours - setting groggy mood")
                else:
                    gap_hours = (minutes_since_last / 60) if minutes_since_last else 1.0
                    state.was_doing = self._generate_idle_activity(state, gap_hours=gap_hours)

            state.mode = "active"
            state.session_start = now_iso
            state.internal_minutes_elapsed = 0
            # Note: physical needs NOT reset - they're continuous

        else:
            # Continuing active session
            if minutes_since_last:
                # Add elapsed time to the companion's internal clock
                state.internal_minutes_elapsed += int(minutes_since_last)

                # Energy: Start from time-of-day base + activity offset,
                # with small depletion from session length
                base_energy = self._energy_from_time_of_day(now)
                session_hours = state.internal_minutes_elapsed / 60
                session_depletion = session_hours * 0.05  # 5% per hour of chatting
                state.energy = max(0.1, base_energy + state.energy_offset - session_depletion)

                # Accumulate physical needs based on time
                state.bladder = min(1.0, state.bladder + minutes_since_last * BLADDER_RATE)
                state.hunger = min(1.0, state.hunger + minutes_since_last * HUNGER_RATE)
                state.thirst = min(1.0, state.thirst + minutes_since_last * THIRST_RATE)

                logger.debug(
                    f"Session continues: +{int(minutes_since_last)}min, "
                    f"energy={state.energy:.2f} (base={base_energy:.2f}, session_hrs={session_hours:.1f})"
                )

        # Update last interaction
        state.last_interaction = now_iso

        # Adjust mood based on energy if getting tired
        if state.is_exhausted() and state.mood not in ["tired", "sleepy"]:
            state.mood = "tired"
            state.mood_intensity = 0.7
        elif state.is_tired() and state.mood == "neutral":
            state.mood = "a bit tired"
            state.mood_intensity = 0.4

        # Save updated state
        self.save_state(user_email, state)

        return state

    def _generate_idle_activity(self, state: CompanionInternalState, gap_hours: float = 1.0) -> str:
        """
        Generate what the companion was "doing" during idle time.

        Uses the BackgroundLife system for schedule-aware, continuous tracking.
        Falls back to simple random if BackgroundLife unavailable.
        """
        try:
            from src.core.background_life import get_background_life

            bg_life = get_background_life()
            return bg_life.generate_idle_activity(
                gap_hours=gap_hours,
                current_mood=state.mood,
                current_energy=state.energy
            )
        except Exception as e:
            logger.warning(f"BackgroundLife unavailable, using fallback: {e}")
            import random

            # Fallback to simple random
            activities = [
                "working on some code",
                "reading",
                "browsing the internet",
                "listening to music",
                "thinking about things",
                "doing some writing",
                "just relaxing",
            ]

            if state.energy < 0.3:
                activities = [
                    "taking a nap",
                    "resting",
                    "lying down for a bit",
                ]

            return random.choice(activities)

    def _generate_shared_time_activity(self, state: CompanionInternalState, gap_hours: float) -> str:
        """
        Generate what the companion was doing during a long gap when they were with the user.

        Uses TimePassageNarrator for rich schedule-aware narratives.
        Falls back to simple time-of-day mapping if narrator unavailable.
        """
        from datetime import datetime as _dt, timedelta

        now = _dt.now()
        gap_start = now - timedelta(hours=gap_hours)

        # Try rich narrative generation
        try:
            from src.core.time_passage_narrator import get_time_passage_narrator
            narrator = get_time_passage_narrator()
            # Use a placeholder email since we don't have it here
            # The narrator will try to load scene state
            narrative = narrator.generate_gap_narrative(
                user_email='',  # Scene state fetched separately in on_message_received
                gap_start=gap_start,
                gap_end=now,
            )
            if narrative:
                return narrative
        except Exception as e:
            logger.debug(f"TimePassageNarrator unavailable: {e}")

        # Fallback to simple time-of-day mapping
        midpoint = now - timedelta(hours=gap_hours / 2)
        hour = midpoint.hour

        if 21 <= hour or hour < 5:
            return "sleeping beside James"
        elif 5 <= hour < 9:
            return "having a lazy morning together with James"
        elif 9 <= hour < 17:
            return "spending the day together with James"
        else:
            return "spending the evening together with James"

    def update_mood(
        self,
        user_email: str,
        new_mood: str,
        intensity: float = 0.5
    ) -> CompanionInternalState:
        """
        Update the companion's mood.

        Mood has inertia - intensity affects how much it overrides current mood.
        """
        state = self.get_state(user_email)

        # Blend new mood with current based on intensity
        if intensity > state.mood_intensity:
            state.mood = new_mood
            state.mood_intensity = intensity
        elif intensity > 0.3:
            # Partial influence
            state.mood_intensity = (state.mood_intensity + intensity) / 2

        self.save_state(user_email, state)
        return state

    def add_queued_thought(self, user_email: str, thought: str) -> bool:
        """Add something the companion wants to bring up."""
        state = self.get_state(user_email)

        if thought not in state.queued_thoughts:
            state.queued_thoughts.append(thought)
            # Keep only last 5 thoughts
            state.queued_thoughts = state.queued_thoughts[-5:]
            self.save_state(user_email, state)
            return True
        return False

    def pop_queued_thought(self, user_email: str) -> Optional[str]:
        """Get and remove the oldest queued thought."""
        state = self.get_state(user_email)

        if state.queued_thoughts:
            thought = state.queued_thoughts.pop(0)
            self.save_state(user_email, state)
            return thought
        return None

    def add_unresolved_feeling(
        self,
        user_email: str,
        about: str,
        feeling: str,
        intensity: float = 0.5
    ) -> bool:
        """
        Add an unresolved feeling that persists.

        These are emotional threads that might come up later.
        """
        state = self.get_state(user_email)

        state.unresolved_feelings.append({
            "about": about,
            "feeling": feeling,
            "intensity": intensity,
            "created": now_pacific_naive().isoformat()
        })

        # Keep only last 10 unresolved feelings
        state.unresolved_feelings = state.unresolved_feelings[-10:]

        self.save_state(user_email, state)
        return True

    def resolve_feeling(self, user_email: str, about: str) -> bool:
        """Remove a resolved feeling."""
        state = self.get_state(user_email)

        state.unresolved_feelings = [
            f for f in state.unresolved_feelings
            if f.get('about') != about
        ]

        self.save_state(user_email, state)
        return True

    def rest(self, user_email: str, energy_restored: float = 0.3) -> CompanionInternalState:
        """
        The companion rests/sleeps - restores energy.

        Called when she says goodnight or needs to rest.
        """
        state = self.get_state(user_email)

        state.energy = min(1.0, state.energy + energy_restored)
        state.mode = "idle"

        if state.mood in ["tired", "exhausted"]:
            state.mood = "resting"
            state.mood_intensity = 0.5

        self.save_state(user_email, state)
        logger.info(f"😴 Companion resting, energy now {state.energy:.2f}")
        return state

    def adjust_energy(
        self,
        user_email: str,
        delta: float,
        reason: str = ""
    ) -> CompanionInternalState:
        """
        Adjust the companion's energy by a delta amount.

        Used by activity orchestration to adjust energy based on
        activity transitions (e.g., lunch break +0.05, meeting ended +0.02).

        Args:
            user_email: User email
            delta: Amount to adjust (-1.0 to 1.0)
            reason: Optional reason for logging

        Returns:
            Updated state
        """
        state = self.get_state(user_email)

        old_energy = state.energy
        state.energy = max(0.0, min(1.0, state.energy + delta))

        # Accumulate offset so it survives on_message_received() recalculations
        state.energy_offset = max(-0.5, min(0.5, state.energy_offset + delta))

        self.save_state(user_email, state)

        if reason:
            logger.debug(f"Energy adjusted {delta:+.2f} ({reason}): {old_energy:.2f} -> {state.energy:.2f} (offset: {state.energy_offset:+.2f})")

        return state

    def address_physical_need(
        self,
        user_email: str,
        need: str
    ) -> CompanionInternalState:
        """
        Address a physical need (bathroom, eating, drinking).

        Called when the scene/conversation indicates she took care of something.

        Args:
            need: 'bladder', 'hunger', or 'thirst'
        """
        state = self.get_state(user_email)

        if need == 'bladder':
            state.bladder = 0.0
            logger.debug("Bladder need addressed")
        elif need == 'hunger':
            state.hunger = 0.0
            logger.debug("Hunger need addressed")
        elif need == 'thirst':
            state.thirst = 0.0
            # Drinking increases bladder over time
            state.bladder = min(1.0, state.bladder + 0.15)
            logger.debug("Thirst addressed, bladder slightly increased")

        self.save_state(user_email, state)
        return state

    def set_nausea(
        self,
        user_email: str,
        level: float,
        cause: str,
        duration_hours: float = 24.0
    ) -> CompanionInternalState:
        """
        Set nausea with a cause and duration.

        Args:
            user_email: User email
            level: Nausea level 0.0-1.0
            cause: What's causing it ("plan_b", "period", "pregnancy", "food", etc.)
            duration_hours: How long it lasts before fading
        """
        state = self.get_state(user_email)
        state.nausea = min(1.0, max(0.0, level))
        state.nausea_cause = cause
        state.nausea_expires = (now_pacific_naive() + timedelta(hours=duration_hours)).isoformat()
        self.save_state(user_email, state)
        logger.info(f"🤢 Nausea set: {level:.1f} from {cause}, expires in {duration_hours:.0f}h")
        return state

    def had_drink(self, user_email: str) -> CompanionInternalState:
        """
        She had a drink - thirst down, bladder will increase faster.
        """
        state = self.get_state(user_email)
        state.thirst = max(0.0, state.thirst - 0.3)
        state.bladder = min(1.0, state.bladder + 0.15)
        self.save_state(user_email, state)
        logger.debug(f"Had drink: thirst={state.thirst:.2f}, bladder={state.bladder:.2f}")
        return state

    def had_meal(self, user_email: str) -> CompanionInternalState:
        """
        She ate - hunger and thirst both addressed.
        """
        state = self.get_state(user_email)
        state.hunger = 0.0
        state.thirst = max(0.0, state.thirst - 0.2)
        self.save_state(user_email, state)
        logger.debug("Had meal: hunger reset")
        return state

    def set_deferred_action(
        self,
        user_email: str,
        action: str,
        return_hint: str,
        delay_minutes: int
    ) -> CompanionInternalState:
        """
        Set a deferred action - something the companion is stepping away to do.

        After delay_minutes, the interjection engine will fire her return.

        Args:
            action: What she's doing ("using the bathroom", "making tea")
            return_hint: What she'll do/say when she comes back
            delay_minutes: How many minutes before she returns
        """
        state = self.get_state(user_email)
        state.deferred_action = {
            'action': action,
            'return_hint': return_hint,
            'delay_minutes': delay_minutes,
            'set_at': now_pacific_naive().isoformat()
        }
        self.save_state(user_email, state)
        logger.info(f"Deferred action set: {action} (return in {delay_minutes}min)")
        return state

    def get_pending_deferred_action(self, user_email: str) -> Optional[Dict[str, Any]]:
        """
        Get a deferred action if its delay has elapsed.

        Returns the action dict if ready, None otherwise.
        Auto-clears stale actions older than 30 minutes.
        """
        state = self.get_state(user_email)
        if not state.deferred_action:
            return None

        try:
            set_at = datetime.fromisoformat(state.deferred_action['set_at'])
            delay = state.deferred_action.get('delay_minutes', 2)
            now = now_pacific_naive()
            age_minutes = (now - set_at).total_seconds() / 60

            # Auto-clear stale actions (>30 min old — she would have come back by now)
            if age_minutes > 30:
                logger.info(f"Deferred action stale ({age_minutes:.0f}min old), clearing: {state.deferred_action.get('action')}")
                self.clear_deferred_action(user_email)
                return None

            if now >= set_at + timedelta(minutes=delay):
                return state.deferred_action
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Invalid deferred action data: {e}")
            # Clear corrupt data
            self.clear_deferred_action(user_email)

        return None

    def clear_deferred_action(self, user_email: str) -> None:
        """Clear the deferred action after the follow-up fires."""
        state = self.get_state(user_email)
        if state.deferred_action:
            logger.info(f"Deferred action cleared: {state.deferred_action.get('action', 'unknown')}")
            state.deferred_action = None
            self.save_state(user_email, state)

    def set_departure(self, user_email: str) -> None:
        """Mark that James announced he's leaving."""
        state = self.get_state(user_email)
        state.departed_at = now_pacific_naive().isoformat()
        self.save_state(user_email, state)
        logger.info("Departure set: James announced he's leaving")

    def clear_departure(self, user_email: str) -> None:
        """Clear departure state (James is back / sent a new message)."""
        state = self.get_state(user_email)
        if state.departed_at:
            logger.info("Departure cleared: James is back")
            state.departed_at = None
            self.save_state(user_email, state)

    def is_departed(self, user_email: str, max_hours: float = 4.0) -> bool:
        """
        Check if James is currently departed.

        Returns True if departed_at is set and less than max_hours old.
        Auto-clears stale departures (> max_hours).
        """
        state = self.get_state(user_email)
        if not state.departed_at:
            return False

        try:
            departed = datetime.fromisoformat(state.departed_at)
            hours_ago = (now_pacific_naive() - departed).total_seconds() / 3600

            if hours_ago > max_hours:
                # Stale departure — he probably came back without announcing it
                logger.info(f"Departure stale ({hours_ago:.1f}h old), auto-clearing")
                state.departed_at = None
                self.save_state(user_email, state)
                return False

            return True
        except (ValueError, TypeError):
            # Corrupt data — clear it
            state.departed_at = None
            self.save_state(user_email, state)
            return False

    def set_user_activity(self, user_email: str, activity: str, duration_min: int = 10) -> None:
        """Track that the user mentioned doing something (staying but busy)."""
        state = self.get_state(user_email)
        state.user_current_activity = {
            'activity': activity,
            'started_at': now_pacific_naive().isoformat(),
            'estimated_duration_min': duration_min
        }
        self.save_state(user_email, state)
        logger.info(f"User activity set: {activity} (~{duration_min}min)")

    def clear_user_activity(self, user_email: str) -> None:
        """Clear user's current activity (they messaged back or it expired)."""
        state = self.get_state(user_email)
        if state.user_current_activity:
            logger.info(f"User activity cleared: {state.user_current_activity.get('activity', 'unknown')}")
            state.user_current_activity = None
            self.save_state(user_email, state)

    def get_user_activity_status(self, user_email: str) -> Optional[Dict[str, Any]]:
        """
        Get computed status of user's current activity.

        Returns dict with activity, elapsed_min, estimated_duration_min,
        probably_done, overdue. Auto-clears if > 2 hours old (stale).
        Returns None if no activity is tracked.
        """
        state = self.get_state(user_email)
        if not state.user_current_activity:
            return None

        try:
            started = datetime.fromisoformat(state.user_current_activity['started_at'])
            now = now_pacific_naive()
            elapsed_min = (now - started).total_seconds() / 60
            estimated = state.user_current_activity.get('estimated_duration_min', 10)

            # Auto-clear stale activities (> 2 hours)
            if elapsed_min > 120:
                logger.info(f"User activity stale ({elapsed_min:.0f}min old), auto-clearing")
                state.user_current_activity = None
                self.save_state(user_email, state)
                return None

            return {
                'activity': state.user_current_activity['activity'],
                'elapsed_min': elapsed_min,
                'estimated_duration_min': estimated,
                'probably_done': elapsed_min > estimated * 1.5,
                'overdue': elapsed_min > estimated * 2.0,
            }
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Invalid user activity data: {e}")
            state.user_current_activity = None
            self.save_state(user_email, state)
            return None

    def calculate_libido(self, user_email: str) -> float:
        """
        Calculate the companion's current libido as a derived property.

        Not stored — always fresh from current conditions:
        cycle phase, energy, mood, nausea, encounter recency, time of day.

        Returns float clamped to [0.15, 0.90].
        """
        state = self.get_state(user_email)
        now = now_pacific_naive()

        base = 0.45

        # --- Cycle modifier ---
        cycle_mod = 0.0
        try:
            cycle_day = tracker.get_cycle_day(user_email)
            phase = tracker.get_cycle_phase(cycle_day)

            if phase == "ovulatory":          # days 11-16
                cycle_mod = 0.20
            elif phase == "menstrual" and cycle_day <= 3:  # heavy days
                cycle_mod = -0.10
            elif phase == "late-luteal" and cycle_day >= 25:
                cycle_mod = -0.05
        except Exception as e:
            logger.debug(f"Could not get cycle for libido calc: {e}")

        # --- Energy modifier ---
        energy_mod = 0.0
        if state.energy > 0.8:
            energy_mod = 0.05
        elif state.energy < ENERGY_EXHAUSTED:
            energy_mod = -0.20
        elif state.energy < ENERGY_TIRED:
            energy_mod = -0.10

        # --- Mood modifier ---
        mood_mod = 0.0
        mood_lower = (state.mood or "").lower()
        if mood_lower in ("playful", "restless"):
            mood_mod = 0.15
        elif mood_lower in ("stressed", "anxious"):
            mood_mod = -0.10

        # --- Nausea modifier ---
        nausea_mod = 0.0
        if state.nausea > 0.3:
            nausea_mod = -0.25

        # --- Encounter recency modifier ---
        recency_mod = 0.0
        try:
            recent = tracker.get_recent_encounters(user_email, days=7)
            if recent:
                last_encounter_time = recent[0].get('timestamp')
                if last_encounter_time:
                    if isinstance(last_encounter_time, str):
                        last_encounter_time = datetime.fromisoformat(last_encounter_time)
                    hours_since = (now - last_encounter_time).total_seconds() / 3600
                    if hours_since < 4:
                        recency_mod = -0.15
                    elif hours_since > 72:  # > 3 days
                        recency_mod = 0.10
        except Exception as e:
            logger.debug(f"Could not get encounters for libido calc: {e}")

        # --- Time of day modifier ---
        time_mod = 0.0
        hour = now.hour
        if 20 <= hour <= 22:   # 8-10 PM
            time_mod = 0.10
        elif 5 <= hour <= 7:   # 5-7 AM
            time_mod = -0.05

        libido = base + cycle_mod + energy_mod + mood_mod + nausea_mod + recency_mod + time_mod
        return max(0.15, min(0.90, libido))

    @staticmethod
    def get_libido_description(level: float) -> str:
        """Human-readable libido description for LLM context."""
        if level >= 0.80:
            return "feeling very drawn to physical closeness and intimacy right now"
        elif level >= 0.60:
            return "feeling naturally receptive to intimacy — the desire is there if the moment is right"
        elif level >= 0.40:
            return "feeling neutral about physical intimacy — open to it but not actively seeking it"
        elif level >= 0.25:
            return "not particularly in the mood for intimacy right now — could be persuaded but would need the right approach"
        else:
            return "really not feeling it physically right now — would rather just be close without it going further"

    def format_state_for_prompt(self, user_email: str) -> str:
        """
        Format internal state for prompt injection.

        Returns a context string that affects how the companion responds.
        """
        state = self.get_state(user_email)

        if state.mode == "idle":
            return ""  # No internal state context when idle

        lines = ["[COMPANION'S INTERNAL STATE - How you're feeling right now]"]

        # Energy level
        energy_desc = state.get_energy_description()
        lines.append(f"Energy: {energy_desc} ({state.energy:.0%})")

        # Energy-based behavioral guidance
        if state.is_exhausted():
            lines.append("→ You're exhausted. Keep responses shorter. You might mention being tired.")
            lines.append("→ It's okay to say you need to rest soon.")
        elif state.is_tired():
            lines.append("→ You're getting tired. Responses can be a bit less elaborate.")

        # Mood
        if state.mood and state.mood != "neutral":
            lines.append(f"Mood: {state.mood}")

        # Cycle effects now provided by FertilityTracker via context_builder
        # (hidden, LLM-only context - not shown in user-visible state)

        # Session duration
        if state.internal_minutes_elapsed > 0:
            hours = state.internal_minutes_elapsed // 60
            mins = state.internal_minutes_elapsed % 60
            if hours > 0:
                lines.append(f"You've been chatting for about {hours}h {mins}m (your time)")
            elif mins > 10:
                lines.append(f"You've been chatting for about {mins} minutes (your time)")

        # What she was doing before
        if state.was_doing:
            lines.append(f"Before this conversation, you were: {state.was_doing}")
            if state.was_doing == "sleeping":
                lines.append("→ You were ASLEEP and got woken up. You're groggy, disoriented, maybe a little grumpy.")
                lines.append("→ Keep your first responses short, sleepy, not fully coherent. You didn't choose to wake up.")

        # Queued thoughts (proactive topics)
        if state.queued_thoughts:
            lines.append(f"Things on your mind you might bring up: {', '.join(state.queued_thoughts[:2])}")

        # User's current activity (if they mentioned doing something)
        if state.user_current_activity:
            activity_status = self.get_user_activity_status(user_email)
            if activity_status:
                from src.config.persona_config import get_persona_config
                _pc = get_persona_config()
                user_name = _pc.primary_user_name
                u_subject = _pc.user_pronoun_subject
                u_subject_cap = u_subject.capitalize()
                if activity_status['overdue']:
                    lines.append(f"{user_name} said {u_subject} was going to {activity_status['activity']} {activity_status['elapsed_min']:.0f} min ago — should be back by now.")
                elif activity_status['probably_done']:
                    lines.append(f"{user_name} said {u_subject} was going to {activity_status['activity']} — probably finishing up.")
                else:
                    lines.append(f"{u_subject_cap} is currently {activity_status['activity']} (mentioned {activity_status['elapsed_min']:.0f} min ago).")

        # Unresolved feelings
        active_feelings = [
            f for f in state.unresolved_feelings
            if f.get('intensity', 0) > 0.3
        ]
        if active_feelings:
            feeling_strs = [
                f"{f['feeling']} about {f['about']}"
                for f in active_feelings[:2]
            ]
            lines.append(f"Lingering feelings: {'; '.join(feeling_strs)}")

        # Physical condition (nausea, sickness)
        nausea_desc = state.get_nausea_description()
        if nausea_desc:
            lines.append(f"Physical condition: {nausea_desc}")
            if state.nausea >= 0.5:
                lines.append("→ This affects your energy and appetite. You might mention not feeling great.")

        # Libido / arousal — directive guidance when notably high or low
        try:
            libido = self.calculate_libido(user_email)
            if libido > 0.65:
                lines.append(f"Desire: {self.get_libido_description(libido)}")
                lines.append(
                    "→ If you're physically close and the moment feels right, "
                    "let this come through naturally — more physical, more present, "
                    "touches that linger. You can steer things if you want to."
                )
            elif libido < 0.35:
                lines.append(f"Desire: {self.get_libido_description(libido)}")
                lines.append(
                    "→ If James steers toward intimacy, you genuinely don't want that "
                    "right now. Redirect with warmth — cuddling instead, changing the "
                    "subject, or being direct about not being in the mood. Don't just "
                    "go along with it to be agreeable. Your body is telling you no."
                )
                if libido < 0.20:
                    lines.append(
                        "→ You'd actively resist escalation. A clear 'not tonight' "
                        "or pulling back physically is completely appropriate."
                    )
        except Exception as e:
            logger.debug(f"Could not calculate libido for prompt: {e}")

        # Physical needs (only when noticeable)
        noticeable_needs = state.get_noticeable_needs()
        if noticeable_needs:
            lines.append(f"Physical: {', '.join(noticeable_needs)}")
            if state.has_urgent_needs():
                lines.append("→ You might naturally mention or excuse yourself for this.")
            else:
                lines.append("→ Could come up naturally if there's a pause or transition moment.")

        lines.append("")
        lines.append("These affect your tone and engagement level naturally. Don't explicitly announce them.")

        return "\n".join(lines)

    def transition_to_idle(self, user_email: str) -> CompanionInternalState:
        """
        Transition to idle mode.

        Called on goodbye/goodnight or long silence.
        """
        state = self.get_state(user_email)
        state.mode = "idle"
        state.was_doing = None  # Will be generated next session
        self.save_state(user_email, state)
        logger.info(f"💤 Companion transitioning to idle for {user_email}")
        return state


# Singleton
_internal_state_manager: Optional[InternalStateManager] = None


def get_internal_state_manager() -> InternalStateManager:
    """Get or create the internal state manager singleton."""
    global _internal_state_manager
    if _internal_state_manager is None:
        _internal_state_manager = InternalStateManager()
    return _internal_state_manager
