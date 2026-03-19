"""
Slash Command Handler -- dispatches CLI commands to their implementations.

WHAT: Parses /command [args] input, dispatches to handler functions, and returns
      a CommandResult (handled, should_quit, optional message). Covers chat
      commands (/help, /clear, /status, /history, /mood, /see, /activity,
      /autopilot, /img, /state, /test, /quit) and fact-approval commands
      (/pending, /approve, /reject, /edit).

WHY:  Separates command logic from the main CLI loop and UI rendering, making
      it easy to add new commands without touching the input loop.

HOW:  A dict maps command names (including aliases like /h, /q, /cls) to async
      handler methods. `execute()` splits the input, looks up the handler, and
      calls it with any arguments. Each handler interacts with the CLI instance
      (self.cli) for state access and connection.send_* for backend requests.
"""

from typing import Optional, Callable, Awaitable, Dict, Any
from dataclasses import dataclass


@dataclass
class CommandResult:
    """Result of a command execution"""
    handled: bool  # Whether the command was recognized
    should_quit: bool = False  # Whether to exit the CLI
    message: Optional[str] = None  # Optional message to display


class CommandHandler:
    """Handles slash commands"""

    def __init__(self, cli_ref):
        """
        Initialize command handler

        Args:
            cli_ref: Reference to the CompanionCLI instance for accessing state
        """
        self.cli = cli_ref
        self.commands: Dict[str, Callable] = {
            'help': self.cmd_help,
            'h': self.cmd_help,
            '?': self.cmd_help,
            'clear': self.cmd_clear,
            'cls': self.cmd_clear,
            'status': self.cmd_status,
            'history': self.cmd_history,
            'hist': self.cmd_history,
            'mood': self.cmd_mood,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,
            'q': self.cmd_quit,
            'reconnect': self.cmd_reconnect,
            'timestamps': self.cmd_timestamps,
            'ts': self.cmd_timestamps,
            'test': self.cmd_test,
            'state': self.cmd_state,
            'img': self.cmd_img,
            'images': self.cmd_img,
            'see': self.cmd_see,
            'activity': self.cmd_activity,
            # Fact approval commands
            'pending': self.cmd_pending,
            'approve': self.cmd_approve,
            'reject': self.cmd_reject,
            'edit': self.cmd_edit_fact,
            # James autopilot commands
            'autopilot': self.cmd_autopilot,
            'where': self.cmd_autopilot,
            'james': self.cmd_autopilot,
            # Display settings
            'typewriter': self.cmd_typewriter,
            'tw': self.cmd_typewriter,
        }

    def is_command(self, text: str) -> bool:
        """Check if text is a slash command"""
        return text.startswith('/') and len(text) > 1

    async def execute(self, text: str) -> CommandResult:
        """
        Execute a slash command

        Args:
            text: Full command text including /

        Returns:
            CommandResult with execution status
        """
        if not self.is_command(text):
            return CommandResult(handled=False)

        # Parse command and arguments
        parts = text[1:].split(maxsplit=1)
        cmd_name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Look up and execute command
        handler = self.commands.get(cmd_name)
        if handler:
            return await handler(args)
        else:
            return CommandResult(
                handled=True,
                message=f"unknown command: /{cmd_name}"
            )

    async def cmd_help(self, args: str) -> CommandResult:
        """Show help"""
        self.cli.ui.print_help()
        return CommandResult(handled=True)

    async def cmd_clear(self, args: str) -> CommandResult:
        """Clear screen"""
        self.cli.ui.clear_screen()
        return CommandResult(handled=True)

    async def cmd_status(self, args: str) -> CommandResult:
        """Show connection status"""
        self.cli.ui.print_status(
            self.cli.connection.connected,
            self.cli.connection.backend_url
        )
        return CommandResult(handled=True)

    async def cmd_history(self, args: str) -> CommandResult:
        """Show message history"""
        count = 10
        if args:
            try:
                count = int(args)
            except ValueError:
                pass

        self.cli.ui.print_history(self.cli.message_history, count)
        return CommandResult(handled=True)

    async def cmd_mood(self, args: str) -> CommandResult:
        """Show the companion's current mood"""
        self.cli.ui.print_mood(self.cli.ui.last_mood)
        return CommandResult(handled=True)

    async def cmd_quit(self, args: str) -> CommandResult:
        """Exit the CLI"""
        return CommandResult(handled=True, should_quit=True)

    async def cmd_reconnect(self, args: str) -> CommandResult:
        """Reconnect to backend"""
        self.cli.ui.print_info("reconnecting...")
        await self.cli.connection.disconnect()

        if await self.cli.connection.connect():
            self.cli.ui.print_success("reconnected")
        else:
            self.cli.ui.print_error("reconnection failed")

        return CommandResult(handled=True)

    async def cmd_timestamps(self, args: str) -> CommandResult:
        """Toggle timestamp display"""
        self.cli.ui.show_timestamps = not self.cli.ui.show_timestamps
        state = "on" if self.cli.ui.show_timestamps else "off"
        self.cli.ui.print_info(f"timestamps: {state}")
        return CommandResult(handled=True)

    async def cmd_typewriter(self, args: str) -> CommandResult:
        """Toggle typewriter effect or set speed"""
        args = args.strip()
        if args in ('on', 'true', '1'):
            self.cli.ui.typewriter_enabled = True
            self.cli.ui.print_info("typewriter: on")
        elif args in ('off', 'false', '0'):
            self.cli.ui.typewriter_enabled = False
            self.cli.ui.print_info("typewriter: off")
        elif args:
            # Set speed (ms per char)
            try:
                ms = int(args)
                self.cli.ui.TYPEWRITER_DELAY = ms / 1000.0
                self.cli.ui.typewriter_enabled = True
                self.cli.ui.print_info(f"typewriter: {ms}ms per char")
            except ValueError:
                self.cli.ui.print_error("usage: /typewriter [on|off|<ms>]")
        else:
            self.cli.ui.typewriter_enabled = not self.cli.ui.typewriter_enabled
            state = "on" if self.cli.ui.typewriter_enabled else "off"
            self.cli.ui.print_info(f"typewriter: {state}")
        return CommandResult(handled=True)

    async def cmd_test(self, args: str) -> CommandResult:
        """
        Toggle test mode (dry run - messages not saved to database).

        Usage:
            /test on   - Enable test mode
            /test off  - Disable test mode
            /test      - Show current status

        When test mode is on, messages are processed but nothing is saved.
        The companion's responses will be prefixed with [TEST].
        """
        arg = args.strip().lower()

        if arg == 'on':
            self.cli.test_mode = True
            self.cli.ui.print_info("🧪 TEST MODE ON - messages will not be saved")
        elif arg == 'off':
            self.cli.test_mode = False
            self.cli.ui.print_info("✅ TEST MODE OFF - messages will be saved normally")
        else:
            status = "ON" if getattr(self.cli, 'test_mode', False) else "OFF"
            self.cli.ui.print_info(f"Test mode: {status}")
            self.cli.ui.print_info("Usage: /test on | /test off")

        return CommandResult(handled=True)

    async def cmd_state(self, args: str) -> CommandResult:
        """
        Show the companion's current scene and internal state.
        Debug command - doesn't affect conversation.
        """
        self.cli.ui.print_info("Fetching state...")
        await self.cli.connection.request_state()
        return CommandResult(handled=True)

    async def cmd_img(self, args: str) -> CommandResult:
        """
        Show recent generated image URLs.
        Usage: /img [count] - default 20
        """
        count = 20
        if args:
            try:
                count = int(args)
            except ValueError:
                pass

        self.cli.ui.print_info("Fetching images...")
        await self.cli.connection.request_images(count)
        return CommandResult(handled=True)

    async def cmd_see(self, args: str) -> CommandResult:
        """
        Show what the companion looks like right now (clothing, posture, etc.)
        from the current scene state.
        """
        self.cli.pending_see_command = True
        await self.cli.connection.request_state()
        return CommandResult(handled=True)

    async def cmd_activity(self, args: str) -> CommandResult:
        """
        Show the companion's current activity and today's schedule.
        """
        self.cli.ui.print_info("Fetching activities...")
        await self.cli.connection.request_activity()
        return CommandResult(handled=True)

    async def cmd_autopilot(self, args: str) -> CommandResult:
        """
        Show James's current autopilot status - what he's probably doing.
        This is what the companion knows about his routine.
        """
        self.cli.ui.print_info("Fetching autopilot status...")
        await self.cli.connection.request_autopilot()
        return CommandResult(handled=True)

    # ========================================================================
    # FACT APPROVAL COMMANDS
    # ========================================================================

    async def cmd_pending(self, args: str) -> CommandResult:
        """
        Show pending facts awaiting approval.
        """
        self.cli.ui.print_info("Fetching pending facts...")
        await self.cli.connection.request_pending_facts()
        return CommandResult(handled=True)

    async def cmd_approve(self, args: str) -> CommandResult:
        """
        Approve a pending fact.
        Usage: /approve [id] - approves fact with ID, or current pending fact if no ID
        """
        fact_id = None

        if args.strip():
            try:
                fact_id = int(args.strip())
            except ValueError:
                self.cli.ui.print_error("Invalid fact ID. Usage: /approve <id>")
                return CommandResult(handled=True)
        elif self.cli.pending_approval:
            fact_id = self.cli.pending_approval.get('pending_id')
        else:
            self.cli.ui.print_error("No pending fact. Use /pending to see all, or /approve <id>")
            return CommandResult(handled=True)

        if fact_id:
            self.cli.ui.print_info(f"Approving fact {fact_id}...")
            await self.cli.connection.approve_fact(fact_id)
            self.cli.pending_approval = None

        return CommandResult(handled=True)

    async def cmd_reject(self, args: str) -> CommandResult:
        """
        Reject a pending fact.
        Usage: /reject [id] [reason] - rejects fact with optional reason
        """
        parts = args.strip().split(maxsplit=1)
        fact_id = None
        reason = ""

        if parts:
            try:
                fact_id = int(parts[0])
                reason = parts[1] if len(parts) > 1 else ""
            except ValueError:
                # First arg isn't a number, maybe it's just a reason for current pending
                if self.cli.pending_approval:
                    fact_id = self.cli.pending_approval.get('pending_id')
                    reason = args.strip()
                else:
                    self.cli.ui.print_error("Invalid fact ID. Usage: /reject <id> [reason]")
                    return CommandResult(handled=True)
        elif self.cli.pending_approval:
            fact_id = self.cli.pending_approval.get('pending_id')
        else:
            self.cli.ui.print_error("No pending fact. Use /pending to see all, or /reject <id>")
            return CommandResult(handled=True)

        if fact_id:
            self.cli.ui.print_info(f"Rejecting fact {fact_id}...")
            await self.cli.connection.reject_fact(fact_id, reason)
            self.cli.pending_approval = None

        return CommandResult(handled=True)

    async def cmd_edit_fact(self, args: str) -> CommandResult:
        """
        Edit and approve a pending fact.
        Usage: /edit <new text> - edits current pending fact
               /edit <id> <new text> - edits specific fact
        """
        if not args.strip():
            self.cli.ui.print_error("Usage: /edit <new text> or /edit <id> <new text>")
            return CommandResult(handled=True)

        parts = args.strip().split(maxsplit=1)
        fact_id = None
        new_text = ""

        # Try to parse first arg as ID
        try:
            fact_id = int(parts[0])
            new_text = parts[1] if len(parts) > 1 else ""
        except ValueError:
            # First arg isn't a number, use current pending
            if self.cli.pending_approval:
                fact_id = self.cli.pending_approval.get('pending_id')
                new_text = args.strip()
            else:
                self.cli.ui.print_error("No pending fact. Use /edit <id> <new text>")
                return CommandResult(handled=True)

        if not new_text:
            self.cli.ui.print_error("Please provide the new fact text")
            return CommandResult(handled=True)

        if fact_id:
            self.cli.ui.print_info(f"Editing fact {fact_id}...")
            await self.cli.connection.edit_fact(fact_id, new_text)
            self.cli.pending_approval = None

        return CommandResult(handled=True)
