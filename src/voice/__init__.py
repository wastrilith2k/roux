"""
Voice module for the companion - Speech-to-Text and Text-to-Speech via Telegram voice notes.

Uses Deepgram Nova-2 for STT and Edge TTS for TTS (free, no API key).
"""

from src.voice.voice_service import VoiceService, get_voice_service
from src.voice.voice_state import VoiceStateManager, get_voice_state_manager

__all__ = [
    'VoiceService', 'get_voice_service',
    'VoiceStateManager', 'get_voice_state_manager',
]
