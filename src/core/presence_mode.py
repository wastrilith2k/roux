"""
Presence Mode — Companion Framework

WHAT: Manages the communication context between user and companion: whether they
      are physically co-located (in_person) or communicating remotely (texting).
WHY:  Without this, the companion has no way to distinguish between the user being
      physically present (narrative/roleplay context) and texting remotely. This leads
      to inappropriate responses like physical actions when the user is texting from work.
HOW:  PresenceMode is stored in the scene_state table's scene_data JSONB. The manager
      provides get/set/format methods. The context builder injects the mode into every
      conversation turn, and the pipeline adds behavioral constraints based on the mode.

Singleton: get_presence_mode_manager() at module bottom.
"""

import json
import logging
import threading
from enum import Enum
from typing import Optional

from src.database import tables as T

logger = logging.getLogger(__name__)


class PresenceMode(Enum):
    """Communication context between user and companion."""
    IN_PERSON = "in_person"    # Co-located; narrative/physical actions appropriate
    TEXTING = "texting"        # Remote; communicating through phone/messaging


# Default when no mode has been set
DEFAULT_PRESENCE_MODE = PresenceMode.IN_PERSON


class PresenceModeManager:
    """
    Manages presence mode state per user.

    Stores the mode in the scene_state table's scene_data JSONB column
    alongside the existing scene state, under the key 'presence_mode'.
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

    def get_presence_mode(self, user_email: str) -> PresenceMode:
        """
        Get the current presence mode for a user.

        Returns DEFAULT_PRESENCE_MODE if no mode has been set.
        """
        try:
            result = self.db.execute(
                f'SELECT scene_data FROM {T.SCENE_STATE} WHERE user_email = %s',
                (user_email,)
            )
            row = result.fetchone()

            if row and row.get('scene_data'):
                state = row['scene_data']
                if isinstance(state, str):
                    state = json.loads(state)
                mode_str = state.get('presence_mode')
                if mode_str:
                    try:
                        return PresenceMode(mode_str)
                    except ValueError:
                        logger.warning(f"Invalid presence mode in DB: {mode_str}")

        except Exception as e:
            logger.warning(f"Failed to load presence mode: {e}")

        return DEFAULT_PRESENCE_MODE

    def set_presence_mode(self, user_email: str, mode: PresenceMode) -> bool:
        """
        Set the presence mode for a user.

        Merges into the existing scene_data JSONB so other scene fields
        are preserved. Uses an atomic UPDATE with JSONB merge operator.

        Returns True if saved successfully.
        """
        try:
            result = self.db.execute(
                f"""UPDATE {T.SCENE_STATE}
                    SET scene_data = COALESCE(scene_data, '{{}}'::jsonb)
                                      || %s::jsonb
                    WHERE user_email = %s
                    RETURNING user_email
                """,
                (json.dumps({'presence_mode': mode.value}), user_email)
            )
            row = result.fetchone()
            if not row:
                logger.warning(f"No scene_state row found for {user_email}")
                return False

            logger.info(f"Presence mode set to {mode.value} for {user_email}")
            return True

        except Exception as e:
            logger.warning(f"Failed to save presence mode: {e}")
            return False

    def format_for_prompt(self, user_email: str) -> str:
        """
        Format the current presence mode for injection into the conversation context.

        Returns a short context string describing the communication mode.
        """
        mode = self.get_presence_mode(user_email)

        if mode == PresenceMode.TEXTING:
            return (
                "Communication mode: TEXTING (SMS/messaging)\n"
                "You and the user are NOT in the same physical space. "
                "You are communicating through text messages. "
                "Do NOT describe physical actions, touches, or shared surroundings."
            )
        else:
            return (
                "Communication mode: IN PERSON\n"
                "You and the user are physically together. "
                "Physical actions, shared environment, and narrative are appropriate."
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_manager: Optional[PresenceModeManager] = None
_manager_lock = threading.Lock()


def get_presence_mode_manager() -> PresenceModeManager:
    """Get or create the singleton PresenceModeManager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = PresenceModeManager()
    return _manager
