"""
VoiceStateManager - Per-chat voice mode preferences.

Stores whether each chat is in 'review' mode (default) or 'handsfree' mode.
Persisted to a JSON file in the data directory.
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(
    os.environ.get('DATA_DIR', '/app/data'),
    'telegram_voice_state.json'
)


class VoiceStateManager:
    """Manages per-chat voice mode preferences."""

    def __init__(self):
        self._state: dict = {}
        self._load()

    def _load(self):
        """Load state from disk."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    self._state = json.load(f)
                logger.info(f"Loaded voice state for {len(self._state)} chats")
        except Exception as e:
            logger.warning(f"Could not load voice state: {e}")
            self._state = {}

    def _save(self):
        """Save state to disk."""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump(self._state, f)
        except Exception as e:
            logger.error(f"Could not save voice state: {e}")

    def get_mode(self, chat_id: str) -> str:
        """Get voice mode for a chat. Returns 'review' or 'handsfree'."""
        return self._state.get(str(chat_id), {}).get('mode', 'review')

    def set_mode(self, chat_id: str, mode: str):
        """Set voice mode for a chat."""
        chat_id = str(chat_id)
        if chat_id not in self._state:
            self._state[chat_id] = {}
        self._state[chat_id]['mode'] = mode
        self._save()
        logger.info(f"Voice mode for {chat_id} set to '{mode}'")

    def is_handsfree(self, chat_id: str) -> bool:
        """Check if chat is in hands-free mode."""
        return self.get_mode(chat_id) == 'handsfree'

    def get_voice_reply(self, chat_id: str) -> bool:
        """Get voice reply setting for a chat. When True, text messages get voice responses."""
        return self._state.get(str(chat_id), {}).get('voice_reply', False)

    def set_voice_reply(self, chat_id: str, enabled: bool):
        """Set voice reply for a chat. When enabled, all replies are voice notes."""
        chat_id = str(chat_id)
        if chat_id not in self._state:
            self._state[chat_id] = {}
        self._state[chat_id]['voice_reply'] = enabled
        self._save()
        logger.info(f"Voice reply for {chat_id} set to {enabled}")


# Singleton
_manager: Optional[VoiceStateManager] = None


def get_voice_state_manager() -> VoiceStateManager:
    """Get singleton VoiceStateManager instance."""
    global _manager
    if _manager is None:
        _manager = VoiceStateManager()
    return _manager
