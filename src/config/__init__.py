"""
Configuration Module

Central configuration for the companion application.
"""

from .app_config import AppConfig
from .socketio_config import SocketIOConfig
from .logging_config import setup_logging
from .persona_config import get_persona_config, PersonaConfig

__all__ = ['AppConfig', 'SocketIOConfig', 'setup_logging', 'get_persona_config', 'PersonaConfig']
