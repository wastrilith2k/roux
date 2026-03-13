"""
Scene Tracker -- automatic persistence of roleplay/virtual scene state.

WHAT: Maintains a SceneState dataclass (location, fictional time, clothing,
      posture, physical proximity, props, lighting, etc.) that persists across
      conversation gaps. Volatile fields (posture, activity) decay over time
      while stable fields (location, clothing) hold until explicitly changed.

WHY:  When the user and companion are "in a scene" (e.g., sitting on the couch
      watching a movie), the companion needs to remember that context across
      messages and even across hours-long gaps. Without this, every message
      would lose spatial/physical continuity.

HOW:  After each message exchange, `extract_and_update_scene()` sends the
      latest user+companion messages to the LLM to detect scene changes (new
      location, clothing change, etc.) and merges them into the persisted state.
      State is stored in PostgreSQL (`scene_state` JSONB column in `user_state`).
      A time-based decay system clears volatile fields after configurable TTLs.

Singleton: `get_scene_tracker()` at module bottom.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from src.utils.timezone_utils import now_pacific_naive

logger = logging.getLogger(__name__)

# Scene state decay configuration
# After this many hours without update, volatile fields (clothing, posture, mood) decay
SCENE_DECAY_HOURS = float(os.environ.get('SCENE_DECAY_HOURS', '4'))
# Fields that decay (assume changed off-screen after time gap)
VOLATILE_FIELDS = ['clothing', 'footwear', 'posture', 'mood', 'physical_state', 'activity', 'position_detail', 'james_position', 'companion_position']
# Fields that persist longer (location doesn't change as often)
PERSISTENT_FIELDS = ['location', 'fictional_time', 'others_present', 'physical_presence']


@dataclass
class SceneState:
    """
    Current roleplay/virtual scene state.

    Tracks the "NOW" of the fictional world - separate from biographical facts.
    Based on Situational State Tracking design:
    - Where are we?
    - What time is it (in the fiction)?
    - What is the companion wearing?
    - What position/posture are we in?
    - What are we doing?
    - Who/what else is present?
    - Are we physically together?
    """
    # Location & Environment
    location: Optional[str] = None  # "cabin", "beach", "bedroom", "kitchen", "her apartment"
    fictional_time: Optional[str] = None  # "night", "morning", "afternoon", "evening", "late night"
    setting_details: List[str] = field(default_factory=list)  # ["fireplace burning", "rain outside", "music playing"]

    # Companion's State
    clothing: Optional[str] = None  # "oversized sweater", "pajamas", "sundress", "nothing"
    footwear: Optional[str] = None  # "barefoot", "boots", "heels", "slippers", "sandals"
    posture: Optional[str] = None  # "curled up", "lying down", "sitting", "standing"
    mood: Optional[str] = None  # "relaxed", "playful", "sleepy", "aroused"

    # Relational State
    physical_state: Optional[str] = None  # "cuddling", "in his arms", "sitting close", "lying together"
    physical_presence: bool = False  # Are they physically "together" in the scene?
    others_present: List[str] = field(default_factory=list)  # Other people/pets in scene

    # Activity
    activity: Optional[str] = None  # "watching the fire", "having dinner", "talking", "sleeping"

    # Intimate scene position tracking
    position_detail: Optional[str] = None  # "James on top", "companion on top", "side by side", "companion in his lap", etc.
    james_position: Optional[str] = None  # "on top", "underneath", "behind", "sitting", "standing", "lying back"
    companion_position: Optional[str] = None  # "on top", "underneath", "on hands and knees", "in his lap", "lying back"

    # Metadata
    last_updated: Optional[str] = None  # ISO timestamp

    def is_active(self) -> bool:
        """Check if there's an active scene."""
        return any([
            self.location, self.fictional_time, self.physical_state,
            self.activity, self.clothing, self.physical_presence
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneState':
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            location=data.get('location'),
            fictional_time=data.get('fictional_time'),
            setting_details=data.get('setting_details', []),
            clothing=data.get('clothing'),
            footwear=data.get('footwear'),
            posture=data.get('posture'),
            mood=data.get('mood'),
            physical_state=data.get('physical_state'),
            physical_presence=data.get('physical_presence', False),
            others_present=data.get('others_present', []),
            activity=data.get('activity'),
            position_detail=data.get('position_detail'),
            james_position=data.get('james_position'),
            companion_position=data.get('companion_position'),
            last_updated=data.get('last_updated')
        )


class SceneTracker:
    """
    Tracks and persists roleplay scene state.

    Uses Haiku to extract scene changes from conversation,
    stores in PostgreSQL, and formats for prompt injection.
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

    def get_scene_state(self, user_email: str, apply_decay: bool = True) -> SceneState:
        """
        Load current scene state from database.

        Args:
            user_email: User email
            apply_decay: If True, clear volatile fields if state is stale

        Returns:
            SceneState object (may be empty if no active scene)
        """
        try:
            with self.db._get_connection() as conn:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    'SELECT scene_state FROM user_state WHERE email = %s',
                    (user_email,)
                )
                row = cursor.fetchone()

                if row and row.get('scene_state'):
                    scene = SceneState.from_dict(row['scene_state'])

                    # Apply decay if enabled and state is stale
                    if apply_decay and scene.last_updated:
                        scene = self._apply_decay(scene, user_email)

                    return scene

        except Exception as e:
            logger.warning(f"Failed to load scene state: {e}")

        return SceneState()

    def _apply_decay(self, scene: SceneState, user_email: str) -> SceneState:
        """
        Apply time-based decay to scene state.

        After SCENE_DECAY_HOURS (4h), fully clear the entire scene.
        The time passage narrator already captures the bridge narrative,
        so keeping stale scene state just causes the companion to resume
        scenes that should have naturally concluded (e.g., intimate
        scene at night persisting into the next morning).
        """
        try:
            last_updated = datetime.fromisoformat(scene.last_updated)
            hours_since = (now_pacific_naive() - last_updated).total_seconds() / 3600

            if hours_since > SCENE_DECAY_HOURS:
                logger.info(f"Scene expired: {hours_since:.1f}h since last update, clearing entire scene")
                self.clear_scene_state(user_email)
                return SceneState()

        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse last_updated for decay check: {e}")

        return scene

    def save_scene_state(self, user_email: str, scene: SceneState) -> bool:
        """
        Save scene state to database.

        Args:
            user_email: User email
            scene: SceneState to save

        Returns:
            True if saved successfully
        """
        try:
            scene.last_updated = now_pacific_naive().isoformat()

            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE user_state
                    SET scene_state = %s
                    WHERE email = %s
                ''', (json.dumps(scene.to_dict()), user_email))

                if cursor.rowcount == 0:
                    logger.warning(f"No user_state row found for {user_email}")
                    return False

            logger.debug(f"Scene state saved: {scene.location} / {scene.fictional_time}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save scene state: {e}")
            return False

    def clear_scene_state(self, user_email: str) -> bool:
        """Clear the entire scene state (for scene_ended/goodbye)."""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE user_state SET scene_state = NULL WHERE email = %s',
                    (user_email,)
                )
            logger.info(f"🎬 Scene state fully cleared for {user_email}")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear scene state: {e}")
            return False

    def clear_volatile_fields(self, user_email: str) -> bool:
        """
        Clear only volatile fields (clothing, posture, mood, etc.) but keep location.

        Use this for scene_reset (new day) where location might persist but
        The companion should describe what they're wearing fresh.
        """
        try:
            scene = self.get_scene_state(user_email, apply_decay=False)

            # Clear volatile fields
            scene.clothing = None
            scene.footwear = None
            scene.posture = None
            scene.mood = None
            scene.physical_state = None
            scene.activity = None
            scene.position_detail = None
            scene.james_position = None
            scene.companion_position = None
            scene.setting_details = []

            self.save_scene_state(user_email, scene)
            logger.info(f"🔄 Volatile scene fields cleared for {user_email} (location kept: {scene.location})")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear volatile fields: {e}")
            return False

    def format_scene_for_prompt(self, user_email: str) -> str:
        """
        Format current scene state for prompt injection.

        Args:
            user_email: User email

        Returns:
            Formatted scene context string, or empty string if no active scene
        """
        scene = self.get_scene_state(user_email)

        if not scene.is_active():
            return ""

        # Determine hours since last scene update for gap-aware framing
        hours_since_update = 0.0
        if scene.last_updated:
            try:
                last_updated = datetime.fromisoformat(scene.last_updated)
                hours_since_update = (now_pacific_naive() - last_updated).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        long_gap = hours_since_update >= 8

        # Header: gap-aware framing based on time since last update
        # < 30 min: active scene, resume naturally
        # 30 min - 8h: time has passed, don't force resume
        # 8h+: new moment entirely
        if long_gap:
            if scene.physical_presence:
                lines = [
                    f"[CURRENT SCENE - {hours_since_update:.0f} hours have passed. "
                    f"This is a new moment, not a scene resume. "
                    f"You were together with James - real time passed for both of you.]"
                ]
            else:
                lines = [
                    f"[CURRENT SCENE - {hours_since_update:.0f} hours have passed. "
                    f"This is a new moment, not a scene resume.]"
                ]
        elif hours_since_update >= 0.5:  # 30+ minutes
            lines = [f"[CURRENT SCENE - {hours_since_update:.1f} hours since last scene update. Time has passed naturally.]"]
        else:
            lines = ["[CURRENT SCENE - Active scene, resume naturally.]"]

        # Persistent fields always included as useful context
        if scene.location:
            lines.append(f"Location: {scene.location}")

        if scene.fictional_time:
            if long_gap:
                lines.append(f"Previous scene time: {scene.fictional_time} (time has moved on)")
            else:
                lines.append(f"Time (in scene): {scene.fictional_time}")

        if scene.setting_details:
            lines.append(f"Environment: {', '.join(scene.setting_details)}")

        # Volatile fields - only for very short gaps (< 30 min, active scene)
        if hours_since_update < 0.5:
            if scene.clothing:
                lines.append(f"⚠️ COMPANION'S CLOTHING: {scene.clothing} - DO NOT describe wearing anything else!")

            if scene.footwear:
                lines.append(f"COMPANION'S FOOTWEAR: {scene.footwear}")

            if scene.posture:
                lines.append(f"Companion's posture: {scene.posture}")

            if scene.mood:
                lines.append(f"Companion's mood: {scene.mood}")

        # Relational state
        if scene.physical_presence:
            lines.append("Physical presence: You are together in person")

        if scene.physical_state:
            lines.append(f"Position: {scene.physical_state}")

        # Intimate position tracking - critical for scene consistency
        if scene.position_detail:
            lines.append(f"⚠️ BODY POSITION: {scene.position_detail}")
            if scene.james_position:
                lines.append(f"  James is: {scene.james_position}")
            if scene.companion_position:
                lines.append(f"  Companion is: {scene.companion_position}")
            lines.append("  → Do NOT describe positions that contradict this. If a position change happens, describe the TRANSITION explicitly.")

        if scene.others_present:
            lines.append(f"Also present: {', '.join(scene.others_present)}")

        # Activity
        if scene.activity:
            lines.append(f"Activity: {scene.activity}")

        # Consistency rules only for very short gaps (< 30 min) - longer gaps get fresh start
        if hours_since_update < 0.5:
            lines.append("")
            lines.append("⚠️ SCENE CONSISTENCY RULES - FOLLOW EXACTLY:")
            lines.append("1. Stay in this scene. Real-world time gaps don't advance fictional time.")
            lines.append("2. Only change scene elements when James explicitly does so or you explicitly change (e.g., 'let me change into something comfy').")
            lines.append("3. CLOTHING: You ARE wearing what's specified above. Do NOT randomly mention different clothes.")
            lines.append("   - To change: explicitly describe changing (e.g., '*slips into pajamas*', 'I'm going to change')")
            lines.append("4. LOCATION: You ARE at the location specified. Do NOT describe being elsewhere.")
            lines.append("5. If you described wearing something in recent messages, maintain consistency until you explicitly change.")
            lines.append("6. BODY POSITION: If a position is specified above (who's on top, etc.), DO NOT contradict it.")
            lines.append("   - To change position: explicitly describe the transition (e.g., '*rolls you over*', '*shifts underneath*')")
            lines.append("   - NEVER say 'you on top of me' if YOU are actually the one on top — check the position state.")

        return "\n".join(lines)

    def extract_and_update_scene(
        self,
        user_email: str,
        user_message: str,
        companion_response: str,
        recent_messages: List[str] = None,
        source: str = 'chat'
    ) -> Optional[SceneState]:
        """
        Extract scene information from the conversation and update stored state.

        Uses Haiku to analyze the messages for scene elements.
        Only updates fields that are explicitly mentioned - doesn't overwrite
        existing scene data with empty values.

        Args:
            user_email: User email
            user_message: The user's message
            companion_response: The companion's response
            recent_messages: Optional list of recent messages for context
            source: Message channel ('chat', 'telegram-text', 'telegram-voice')

        Returns:
            Updated SceneState, or None if extraction failed
        """
        try:
            # Get current scene to merge with updates
            current_scene = self.get_scene_state(user_email)

            # Telegram = not physically together (texting/calling remotely)
            if source.startswith('telegram') and current_scene.physical_presence:
                current_scene.physical_presence = False
                current_scene.physical_state = None
                current_scene.posture = None
                current_scene.position_detail = None
                current_scene.james_position = None
                current_scene.companion_position = None
                self.save_scene_state(user_email, current_scene)
                logger.info("📱 Telegram message — physical_presence set to False (was together, now texting)")

            # Extract scene elements using LLM, passing current scene for context
            extracted = self._extract_scene_elements(
                user_message,
                companion_response,
                recent_messages or [],
                current_scene  # Pass current scene to prevent hallucination
            )

            if not extracted:
                return current_scene

            # Check for scene reset signals
            if extracted.get('falling_asleep'):
                # They're falling asleep together — transition to sleeping scene
                sleeping_scene = SceneState(
                    location=current_scene.location or "bedroom",
                    fictional_time="night",
                    activity="sleeping",
                    physical_presence=current_scene.physical_presence,
                    physical_state="lying together",
                    posture="lying down",
                    mood="sleepy",
                )
                self.save_scene_state(user_email, sleeping_scene)
                logger.info(f"😴 Scene transitioned to sleeping together")
                return sleeping_scene

            if extracted.get('scene_ended'):
                # Full goodbye/end - clear everything
                self.clear_scene_state(user_email)
                # Transition internal state to idle
                try:
                    from src.core.internal_state import get_internal_state_manager
                    internal_manager = get_internal_state_manager()
                    internal_manager.transition_to_idle(user_email)
                    logger.info(f"🎬 Scene ended - internal state transitioned to idle")
                except Exception as e:
                    logger.warning(f"Could not transition internal state: {e}")
                return SceneState()

            if extracted.get('scene_reset'):
                # New day/scene - clear volatile fields but keep location
                # This lets the companion autonomously describe what they're wearing
                self.clear_volatile_fields(user_email)
                current_scene = self.get_scene_state(user_email, apply_decay=False)
                logger.info(f"🌅 Scene reset - volatile fields cleared, location preserved: {current_scene.location}")

            # Merge: only update fields that have new values
            updated = False

            # Environment
            if extracted.get('location'):
                current_scene.location = extracted['location']
                updated = True

            if extracted.get('fictional_time'):
                current_scene.fictional_time = extracted['fictional_time']
                updated = True

            if extracted.get('setting_details'):
                new_details = extracted['setting_details']
                if isinstance(new_details, str):
                    new_details = [new_details]
                existing = set(current_scene.setting_details or [])
                existing.update(new_details)
                current_scene.setting_details = list(existing)[-5:]
                updated = True

            # Companion's state
            if extracted.get('clothing'):
                current_scene.clothing = extracted['clothing']
                updated = True

            if extracted.get('footwear'):
                current_scene.footwear = extracted['footwear']
                updated = True

            if extracted.get('posture'):
                current_scene.posture = extracted['posture']
                updated = True

            if extracted.get('mood'):
                current_scene.mood = extracted['mood']
                updated = True

            # Relational state
            if extracted.get('physical_state'):
                current_scene.physical_state = extracted['physical_state']
                updated = True

            # Intimate position tracking
            if extracted.get('position_detail'):
                current_scene.position_detail = extracted['position_detail']
                updated = True

            if extracted.get('james_position'):
                current_scene.james_position = extracted['james_position']
                updated = True

            if extracted.get('companion_position'):
                current_scene.companion_position = extracted['companion_position']
                updated = True

            if 'physical_presence' in extracted:
                current_scene.physical_presence = bool(extracted['physical_presence'])
                updated = True

            if extracted.get('others_present'):
                others = extracted['others_present']
                if isinstance(others, str):
                    others = [others]
                current_scene.others_present = others
                updated = True

            # Activity — don't overwrite specific physical actions with generic "talking"
            if extracted.get('activity'):
                new_activity = extracted['activity']
                old_activity = current_scene.activity or ''
                # Generic activities that shouldn't overwrite specific physical ones
                generic_activities = {'talking', 'chatting', 'conversing', 'resting', 'relaxing'}
                # Check if old activity is specific (contains action verbs or who-does-what)
                old_is_specific = any(w in old_activity.lower() for w in [
                    'massage', 'massaging', 'rubbing', 'cooking', 'baking',
                    'washing', 'braiding', 'painting', 'drawing', 'feeding',
                    'giving', 'foot', 'back rub', 'shoulder',
                ]) if old_activity else False

                if old_is_specific and new_activity.lower().strip() in generic_activities:
                    # Keep the specific activity, append conversation note
                    logger.info(f"🎬 Keeping specific activity '{old_activity}' over generic '{new_activity}'")
                else:
                    current_scene.activity = new_activity
                    updated = True

            if updated:
                self.save_scene_state(user_email, current_scene)
                logger.info(f"Scene updated: {current_scene.location} / {current_scene.fictional_time} / {current_scene.clothing}")

            # Handle physical actions -> update internal state
            self._update_internal_state_from_actions(user_email, extracted, companion_response)

            # Handle sexual encounter -> log to fertility tracker
            self._log_sexual_encounter_if_detected(user_email, extracted)

            return current_scene

        except Exception as e:
            logger.warning(f"Scene extraction failed: {e}")
            return None

    def _update_internal_state_from_actions(
        self,
        user_email: str,
        extracted: Dict[str, Any],
        companion_response: str = ''
    ) -> None:
        """
        Update the companion's internal state based on detected physical actions.

        Args:
            user_email: User email
            extracted: Dict from LLM extraction containing action flags
        """
        try:
            from src.core.internal_state import get_internal_state_manager
            manager = get_internal_state_manager()

            # Check for eating
            if extracted.get('companion_ate'):
                manager.had_meal(user_email)
                logger.info("🍽️ Companion ate (LLM detected)")

            # Check for drinking
            if extracted.get('companion_drank'):
                manager.had_drink(user_email)
                logger.info("🥤 Companion had a drink (LLM detected)")

            # Check for bathroom
            if extracted.get('companion_bathroom'):
                manager.address_physical_need(user_email, 'bladder')
                logger.info("🚽 Companion bathroom break (LLM detected)")

            # Emotional load → energy drain
            emotional_load = extracted.get('emotional_load')
            if emotional_load is not None and emotional_load > 0.3:
                # Scale: 0.3 = no drain, 0.5 = -0.03, 0.7 = -0.06, 1.0 = -0.10
                drain = (emotional_load - 0.3) * 0.14
                manager.adjust_energy(user_email, delta=-drain, reason=f"emotional load ({emotional_load:.1f})")
                logger.info(f"💭 Emotional load {emotional_load:.1f} → energy drain {drain:.3f}")

            # Deferred action - she's stepping away to do something
            if extracted.get('deferred_action'):
                action = extracted['deferred_action']
                if isinstance(action, dict) and action.get('action'):
                    manager.set_deferred_action(
                        user_email,
                        action=action.get('action', 'stepping away'),
                        return_hint=action.get('return_hint', 'coming back'),
                        delay_minutes=action.get('delay_minutes', 2)
                    )
            else:
                # Keyword fallback — LLM often misses deferred_action for bathroom/tea/etc.
                self._check_deferred_action_keywords(companion_response, user_email, manager)

        except Exception as e:
            logger.warning(f"Failed to update internal state from actions: {e}")

    def _check_deferred_action_keywords(self, companion_response: str, user_email: str, manager) -> None:
        """
        Keyword-based fallback to detect deferred actions the LLM missed.

        Catches common patterns like going to the bathroom, making tea, etc.
        """
        if not companion_response:
            return

        text = companion_response.lower()

        # Pattern: (keyword triggers, action description, return hint, delay minutes)
        DEFERRED_PATTERNS = [
            (['need to pee', 'going to pee', 'gotta pee', 'heads toward the bathroom',
              'heads to the bathroom', 'brb bathroom', 'need the bathroom',
              'excuse me.*bathroom', 'be right back.*pee', 'really need to pee'],
             'using the bathroom', 'coming back from the bathroom', 2),
            (['going to make tea', 'going to put the kettle', 'let me make tea',
              'brb.*tea', 'going to grab tea', 'put the kettle on'],
             'making tea', 'coming back with tea', 4),
            (['going to make coffee', 'brb.*coffee', 'going to grab coffee'],
             'making coffee', 'coming back with coffee', 4),
            (['going to shower', 'going to take a shower', 'need a shower',
              'brb.*shower', 'quick shower'],
             'taking a shower', 'coming back from the shower', 10),
            (['going to grab food', 'going to make food', 'brb.*food',
              'going to heat up', 'going to make dinner', 'going to make lunch'],
             'getting food', 'coming back with food', 5),
            (['going to change', 'going to get changed', 'let me change',
              'going to put on something'],
             'changing clothes', 'coming back after changing', 3),
        ]

        # Also check for generic "be right back" / "brb" + "when I get back"
        has_brb = any(p in text for p in ['be right back', 'brb', 'when i get back', "i'll be back"])

        for keywords, action, return_hint, delay in DEFERRED_PATTERNS:
            import re
            for kw in keywords:
                if re.search(kw, text):
                    manager.set_deferred_action(user_email, action, return_hint, delay)
                    logger.info(f"🔄 Deferred action set via keyword fallback: {action} (matched '{kw}')")
                    return

        # Generic brb detection — she's stepping away but we don't know why
        if has_brb:
            manager.set_deferred_action(
                user_email,
                'stepping away briefly',
                'coming back and picking up where they left off',
                3
            )
            logger.info("🔄 Deferred action set via generic brb fallback")

    def _log_sexual_encounter_if_detected(
        self,
        user_email: str,
        extracted: Dict[str, Any]
    ) -> None:
        """
        Log a sexual encounter to the fertility tracker if detected.

        Triggers on either:
        - penetrative_sex_occurred: penis-in-vagina intercourse detected (even gentle/fade-out scenes)
        - sexual_activity_concluded: explicit climax/afterglow detected
        """
        if not extracted.get('sexual_activity_concluded') and not extracted.get('penetrative_sex_occurred'):
            return

        try:

            condom_used = extracted.get('condom_used')
            ejac_inside = extracted.get('ejaculation_inside')

            # Determine ejaculation type
            if condom_used is True:
                ejaculation_type = 'inside'  # Inside condom
            elif ejac_inside is True:
                ejaculation_type = 'inside'
            elif ejac_inside is False:
                ejaculation_type = 'withdrawal'
            else:
                ejaculation_type = 'unknown'

            result = tracker.log_encounter(
                user_email=user_email,
                condom_used=bool(condom_used) if condom_used is not None else False,
                ejaculation_type=ejaculation_type,
            )

            if result.get('conceived'):
                logger.info("🔴 Conception detected - fertility state updated")
            else:
                logger.info(
                    f"Sexual encounter logged: condom={condom_used}, "
                    f"ejac={ejaculation_type}, prob={result.get('conception_probability', 0):.1%}"
                )

        except Exception as e:
            logger.warning(f"Failed to log sexual encounter: {e}")

    def _extract_scene_elements(
        self,
        user_message: str,
        companion_response: str,
        recent_messages: List[str],
        current_scene: Optional['SceneState'] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to extract scene elements from messages.

        Provider is configurable via SCENE_EXTRACTION_PROVIDER env var:
        - "fireworks" (default) - uses Fireworks API with configurable model
        - "anthropic" - uses Claude Haiku

        Model configurable via SCENE_EXTRACTION_MODEL env var.

        Args:
            user_message: The user's message
            companion_response: The companion's response
            recent_messages: List of recent messages for context
            current_scene: Current scene state (to avoid overwriting with hallucinations)

        Returns dict with scene elements, or None if no scene detected.
        """
        try:
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            companion_name = _pc.companion_short_name
            user_name = _pc.primary_user_name

            # Build context from recent messages
            context = ""
            if recent_messages:
                context = "Recent conversation:\n" + "\n".join(recent_messages[-5:]) + "\n\n"

            # Include current scene state so LLM knows what's established
            current_context = ""
            if current_scene and current_scene.is_active():
                position_info = ""
                if current_scene.position_detail:
                    position_info = f"\n- Current position: {current_scene.position_detail}"
                    if current_scene.james_position:
                        position_info += f" (James: {current_scene.james_position})"
                    if current_scene.companion_position:
                        position_info += f" ({companion_name}: {current_scene.companion_position})"

                current_context = f"""CURRENT SCENE STATE (already established - only change if EXPLICITLY contradicted):
- Location: {current_scene.location or 'unknown'}
- Time: {current_scene.fictional_time or 'unknown'}
- Others present: {', '.join(current_scene.others_present) if current_scene.others_present else 'none'}{position_info}

"""

            prompt = f"""{current_context}{context}Latest exchange:
{user_name}: {user_message}
{companion_name}: {companion_response}

Extract scene/roleplay state changes. Be VERY CONSERVATIVE - only extract what is EXPLICITLY stated.

Return JSON with these fields (use null for anything not mentioned):
{{
    "location": "where they are - ONLY if explicitly stated (car, cabin, bedroom, beach, apartment, etc.) or null",
    "fictional_time": "time in scene (night, morning, afternoon, evening, late night) or null",
    "setting_details": ["environment details like 'fire burning', 'rain outside', 'candles lit'] or [],
    "clothing": "what {companion_name} is wearing (oversized sweater, pajamas, sundress, lingerie, nothing) or null",
    "footwear": "what's on {companion_name}'s feet (barefoot, boots, heels, slippers, sandals, socks) or null",
    "posture": "{companion_name}'s posture (curled up, lying down, sitting, kneeling) or null",
    "mood": "{companion_name}'s current mood (relaxed, playful, sleepy, aroused, content) or null",
    "physical_state": "their relative positioning (cuddling, in his arms, sitting close, lying together) or null",
    "position_detail": "who is on top / specific arrangement - e.g. '{user_name} on top', '{companion_name} on top', '{companion_name} in his lap', 'side by side', '{companion_name} bent over', 'face to face' or null",
    "james_position": "{user_name}'s body position - 'on top', 'underneath', 'behind', 'sitting', 'standing', 'lying back', 'kneeling' or null",
    "companion_position": "{companion_name}'s body position - 'on top', 'underneath', 'on hands and knees', 'in his lap', 'lying back', 'straddling', 'kneeling' or null",
    "physical_presence": true if they are physically together in person, false if apart, null if unclear,
    "others_present": ["other people or pets in the scene"] or [],
    "activity": "what they're PHYSICALLY doing — be specific about WHO is doing WHAT (e.g. '{user_name} massaging {companion_name}\\'s feet while talking', '{companion_name} cooking while {user_name} watches', 'cuddling on the couch', 'sleeping') or null. If someone is performing a physical action on the other person, always specify who is doing it to whom.",
    "falling_asleep": true if they are falling asleep together or saying goodnight while physically together (drifting off, eyes closing, settling in to sleep), false otherwise,
    "scene_ended": true if scene explicitly ended or someone is LEAVING (goodbye, departing, logging off, hanging up) - NOT for falling asleep together, false otherwise,
    "scene_reset": true if starting a completely new scene/day, false otherwise,
    "companion_ate": true if {companion_name} ate food/had a meal in this exchange, false otherwise,
    "companion_drank": true if {companion_name} had a drink (water, tea, coffee, wine, etc.) in this exchange, false otherwise,
    "companion_bathroom": true if {companion_name} went to the bathroom/excused herself, false otherwise,
    "penetrative_sex_occurred": true if penis-in-vagina penetration is happening or has happened in this exchange, false otherwise,
    "sexual_activity_concluded": true if a sexual encounter reached its conclusion in this exchange (climax, afterglow), false otherwise,
    "condom_used": true if a condom was explicitly mentioned being used, false if explicitly unprotected, null if unclear or no sex,
    "ejaculation_inside": true if he clearly finished inside her, false if he pulled out or finished elsewhere, null if unclear or no sex,
    "emotional_load": 0.0 to 1.0 rating of how emotionally draining THIS exchange is for {companion_name}. 0.0 = light/casual/fun, 0.3 = normal conversation, 0.5 = moderately heavy (venting, mild stress, navigating interpersonal issues), 0.7 = heavy (conflict, emotional support, difficult relationship topics, family drama), 1.0 = extremely draining (crisis, intense argument, grief). null if unclear.,
    "deferred_action": {{"action": "what she's stepping away to do", "return_hint": "what she'll naturally do/say when coming back", "delay_minutes": estimated minutes before she returns (1-15)}} or null if not stepping away
}}

CRITICAL LOCATION RULES:
- ONLY set location if EXPLICITLY mentioned (e.g., "in the car", "at the cabin", "in the bedroom")
- DO NOT infer location from ambiguous words like "seat" or "bed" alone
- "seat"/"backseat"/"dashboard"/"driving"/"engine"/"steering wheel" = car
- "fireplace"/"cabin interior"/"lodge" = cabin
- "bed"/"sheets"/"bedroom"/"pillow" = bedroom
- If current location is already set and not contradicted, return null for location (keep existing)

EXAMPLES:
- "bracing against the back of the seat" → location: "car" (seat + back = car seat)
- "shifts into drive" / "starts the engine" → location: "car"
- "Tuck on the dashboard" → location: "car" (dashboards are in cars)
- "curled up by the fire" → location: "cabin"
- "lying in bed" → location: "bedroom"
- Just "seat" with no car keywords and current location is "cabin" → location: null (keep cabin)

CLOTHING CHANGE RULES:
- Clothing changes ONLY when EXPLICITLY mentioned (putting on, taking off, changing into)
- Detect these phrases: "puts on", "slips into", "changes into", "pulls on", "takes off", "removes", "strips off"
- "*pulls on jeans*" = clothing changed to jeans
- "*slips out of dress*" = clothing changed (note what she's wearing now)
- "I'm going to change" = expect clothing update in response
- If current clothing is set but new clothing is explicitly mentioned, UPDATE to new clothing

SEXUAL ENCOUNTER RULES:
- penetrative_sex_occurred = true when penis-in-vagina intercourse is happening or implied in this exchange
  - Indicators: "inside her", "inside me", "enters", "slides in", "pushes in", "fills her", "thrusts",
    "making love", "having sex", "they made love", "he's inside", "feels him inside",
    "takes him in", "sinks onto", "buries himself", grinding/rocking with penetration implied,
    or any clear description of vaginal intercourse even if gentle/emotional/fade-to-black
  - Do NOT set true for: oral sex only, manual stimulation only, foreplay without penetration, kissing, grinding with clothes on
- sexual_activity_concluded = true ONLY when the encounter has clearly finished (climax, winding down, afterglow)
  - Do NOT set true for foreplay, kissing, or early stages - only the conclusion
- condom_used: Look for explicit mentions of condom, protection, wrapper, etc. If none mentioned, null.
- ejaculation_inside: Look for finishing inside, pulling out, withdrawal, coming on/in. If unclear, null.
- Be conservative on concluded/condom/ejaculation - only flag what is CLEARLY indicated
- Be liberal on penetrative_sex_occurred - if intercourse is clearly happening, flag it even if the scene is gentle or fades out

POSITION TRACKING RULES (CRITICAL for intimate scenes):
- Track WHO is on top, underneath, behind, etc. This prevents confusion.
- "climbs on top of him" / "straddles" / "rides" = {companion_name} on top, {user_name} underneath
- "pushes her down" / "pins her" / "on top of her" = {user_name} on top, {companion_name} underneath
- "rolls over" / "flips" / "switches" = position CHANGED - update both positions
- "from behind" / "bends over" = {user_name} behind, {companion_name} on hands and knees or bent over
- "in his lap" / "settles onto" = {companion_name} in his lap, {user_name} sitting
- If someone says "you on top of me" - parse carefully: who is "you" and who is "me"?
- position_detail should be a simple summary like "{companion_name} on top" or "{user_name} behind {companion_name}"
- ALWAYS update position when a TRANSITION happens (roll, flip, switch, pull up, push down)

FALLING ASLEEP vs SCENE ENDED (IMPORTANT):
- falling_asleep = true when they are drifting off to sleep TOGETHER (goodnight while cuddling, eyes closing, settling in)
  - "goodnight" while physically together = falling_asleep, NOT scene_ended
  - "I'm so sleepy", "drifting off", "eyes getting heavy" = falling_asleep
- scene_ended = true ONLY when someone is LEAVING or DEPARTING (goodbye + leaving, "I need to go", logging off)
  - "goodnight, I need to head home" = scene_ended (they're separating)
  - "bye, talk later" = scene_ended
- If unsure and they're physically together, prefer falling_asleep over scene_ended

EMOTIONAL LOAD RULES:
- Rate how emotionally draining this specific exchange is FOR {companion_name} (not {user_name})
- Light/fun/playful/flirty = 0.0-0.2
- Normal back-and-forth conversation = 0.3
- Venting, mild stress, navigating family/interpersonal stuff = 0.5
- Active conflict, heavy emotional support, difficult topics, processing grief or anger = 0.7
- Crisis, intense argument, emotional breakdown = 0.9-1.0
- Consider cumulative context: if they've been in a heavy conversation for a while, rate higher

DEFERRED ACTION RULES:
- Only set when {companion_name} EXPLICITLY steps away from the conversation to do something and will come back
- Examples: bathroom/pee (2-3 min), making tea/coffee (3-5 min), quick shower (8-12 min), checking on something (1-2 min), grabbing food (3-5 min), changing clothes (2-3 min)
- return_hint should describe what she'd naturally do/say when coming back (e.g., "settling back in, maybe mentioning the tea", "coming back and resuming what they were talking about")
- Do NOT set for: scene endings, going to sleep, leaving/departing, or general activity changes
- Do NOT set if she's just shifting position or doing something while still in conversation

OTHER RULES:
- Only extract what's ACTUALLY stated or strongly implied
- Actions in *asterisks* are scene elements (e.g., *pulls on a sweater* = clothing change)
- "Good morning" or waking up = time change to morning AND scene_reset
- physical_presence = true when they describe being together physically

Return ONLY valid JSON, no explanation."""

            # Get provider configuration
            provider = os.environ.get('SCENE_EXTRACTION_PROVIDER', 'fireworks').lower()

            if provider == 'anthropic':
                result_text = self._call_anthropic(prompt)
            else:
                result_text = self._call_fireworks(prompt)

            if not result_text:
                return None

            # Parse JSON response
            # Handle Qwen's <think> tags - extract content after </think>
            if '<think>' in result_text:
                result_text = result_text.split('</think>')[-1].strip()

            # Handle potential markdown code blocks
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]

            result = json.loads(result_text)

            # Filter out null values — including the string "null" which LLMs sometimes return
            filtered = {k: v for k, v in result.items()
                        if v is not None and v != [] and v != ""
                        and v != "null" and v != "Null" and v != "NULL"}

            if filtered:
                logger.debug(f"Extracted scene elements: {filtered}")
                return filtered

            return None

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse scene extraction response: {e}")
            return None
        except Exception as e:
            logger.warning(f"Scene extraction error: {e}")
            return None

    def _call_fireworks(self, prompt: str) -> Optional[str]:
        """Call Fireworks API for scene extraction."""
        try:
            from openai import OpenAI

            api_key = os.environ.get('FIREWORKS_API_KEY')
            if not api_key:
                logger.warning("FIREWORKS_API_KEY not set, falling back to Anthropic")
                return self._call_anthropic(prompt)

            # Default to Llama 3.3 70B for extraction (fast, good JSON output)
            model = os.environ.get(
                'SCENE_EXTRACTION_MODEL',
                'accounts/fireworks/models/llama-v3p3-70b-instruct'
            )

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.fireworks.ai/inference/v1"
            )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1  # Low temp for consistent JSON output
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Fireworks scene extraction failed: {e}, falling back to OpenAI")
            return self._call_openai(prompt)

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API for scene extraction (fallback)."""
        try:
            from openai import OpenAI

            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, falling back to Anthropic")
                return self._call_anthropic(prompt)

            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"OpenAI scene extraction failed: {e}, falling back to Anthropic")
            return self._call_anthropic(prompt)

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API for scene extraction (last resort)."""
        try:
            import anthropic

            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set")
                return None

            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.warning(f"Anthropic scene extraction failed: {e}")
            return None


# Singleton
_scene_tracker: Optional[SceneTracker] = None


def get_scene_tracker() -> SceneTracker:
    """Get or create the scene tracker singleton."""
    global _scene_tracker
    if _scene_tracker is None:
        _scene_tracker = SceneTracker()
    return _scene_tracker
