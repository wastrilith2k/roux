"""
/help command - Shows available commands
"""

from typing import Dict, Any


def handle_help_command(args: str, context: dict) -> dict:
    """Handle /help command."""
    # Import here to avoid circular import
    from .registry import get_command_registry

    registry = get_command_registry()

    # If args provided, show help for specific command
    if args:
        cmd_name = args.strip().lower().lstrip('/')
        # TODO: Show detailed help for specific command
        return registry.get_help()

    return registry.get_help()


def register_help_command(registry):
    """Register the /help command."""
    registry.register(
        name='help',
        handler=handle_help_command,
        description='Show available commands',
        aliases=['?', 'commands']
    )
