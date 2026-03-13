"""
Core Commands Module - Unified command system for all interfaces

Commands are defined here and can be invoked from:
- WebSocket (web UI)
- Telegram
- CLI
- Any future interface

Each command returns a dict with:
- 'text': Human-readable response
- 'data': Structured data for programmatic use
- 'error': Error message if failed
"""

from .registry import CommandRegistry, get_command_registry
from .models_command import register_models_command
from .status_command import register_status_command
from .biographies_command import register_biographies_command

__all__ = [
    'CommandRegistry',
    'get_command_registry',
    'register_models_command',
    'register_status_command',
    'register_biographies_command',
]
