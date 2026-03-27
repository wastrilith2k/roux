"""
Configuration Module

Central configuration for the companion application.
"""

from .app_config import AppConfig
from .socketio_config import SocketIOConfig
from .logging_config import setup_logging
from .persona_config import get_persona_config, PersonaConfig
from .models import (
    FIREWORKS_DEFAULT_MODEL,
    FIREWORKS_FALLBACK_MODEL,
    FIREWORKS_BACKGROUND_MODEL,
    FIREWORKS_DEFAULT_MODEL_SHORT,
)

__all__ = [
    'AppConfig', 'SocketIOConfig', 'setup_logging',
    'get_persona_config', 'PersonaConfig',
    'FIREWORKS_DEFAULT_MODEL', 'FIREWORKS_FALLBACK_MODEL',
    'FIREWORKS_BACKGROUND_MODEL', 'FIREWORKS_DEFAULT_MODEL_SHORT',
]
