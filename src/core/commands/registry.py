"""
Command Registry - Central registry for all slash commands

Usage:
    from src.core.commands import get_command_registry

    registry = get_command_registry()
    result = registry.execute('/models')
    print(result['text'])
"""

import logging
from typing import Dict, Any, Callable, Optional, List

logger = logging.getLogger(__name__)


class CommandRegistry:
    """
    Central registry for slash commands.

    Commands are registered with:
    - name: The command name (without /)
    - handler: Function that takes (args: str, context: dict) -> dict
    - description: Brief description for help
    - aliases: Alternative names for the command
    """

    def __init__(self):
        self._commands: Dict[str, dict] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        handler: Callable[[str, dict], dict],
        description: str = "",
        aliases: List[str] = None
    ):
        """Register a command handler."""
        name = name.lower().lstrip('/')

        self._commands[name] = {
            'handler': handler,
            'description': description,
            'aliases': aliases or []
        }

        # Register aliases
        for alias in (aliases or []):
            self._aliases[alias.lower().lstrip('/')] = name

        logger.debug(f"Registered command: /{name}")

    def execute(
        self,
        command_str: str,
        context: Optional[dict] = None
    ) -> dict:
        """
        Execute a command.

        Args:
            command_str: Full command string (e.g., "/models" or "/help topics")
            context: Optional context dict with user_email, interface, etc.

        Returns:
            dict with 'text', 'data', and/or 'error'
        """
        context = context or {}

        # Parse command and args
        parts = command_str.strip().split(maxsplit=1)
        cmd_name = parts[0].lower().lstrip('/')
        args = parts[1] if len(parts) > 1 else ""

        # Resolve alias
        if cmd_name in self._aliases:
            cmd_name = self._aliases[cmd_name]

        # Check if command exists
        if cmd_name not in self._commands:
            return {
                'error': f"Unknown command: /{cmd_name}",
                'text': f"Unknown command: /{cmd_name}\nUse /help to see available commands."
            }

        # Execute handler
        try:
            handler = self._commands[cmd_name]['handler']
            return handler(args, context)
        except Exception as e:
            logger.error(f"Error executing /{cmd_name}: {e}")
            return {
                'error': str(e),
                'text': f"Error executing /{cmd_name}: {e}"
            }

    def is_command(self, message: str) -> bool:
        """Check if a message is a command."""
        if not message or not message.startswith('/'):
            return False

        cmd_name = message.strip().split()[0].lower().lstrip('/')
        return cmd_name in self._commands or cmd_name in self._aliases

    def get_help(self) -> dict:
        """Get help text for all commands."""
        lines = ["**Available Commands:**\n"]

        for name, info in sorted(self._commands.items()):
            desc = info['description'] or "No description"
            aliases = info['aliases']

            line = f"  /{name}"
            if aliases:
                line += f" (aliases: {', '.join('/' + a for a in aliases)})"
            line += f" - {desc}"
            lines.append(line)

        return {
            'text': '\n'.join(lines),
            'data': {
                'commands': [
                    {
                        'name': name,
                        'description': info['description'],
                        'aliases': info['aliases']
                    }
                    for name, info in self._commands.items()
                ]
            }
        }

    def list_commands(self) -> List[str]:
        """Get list of all command names."""
        return list(self._commands.keys())


# Singleton instance
_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get the global command registry (creates if needed)."""
    global _registry

    if _registry is None:
        _registry = CommandRegistry()
        _register_all_commands(_registry)

    return _registry


def _register_all_commands(registry: CommandRegistry):
    """Register all built-in commands."""
    from .models_command import register_models_command
    from .status_command import register_status_command
    from .help_command import register_help_command
    from .biographies_command import register_biographies_command
    from .relationships_command import register_relationships_command

    register_help_command(registry)
    register_models_command(registry)
    register_status_command(registry)
    register_biographies_command(registry)
    register_relationships_command(registry)
