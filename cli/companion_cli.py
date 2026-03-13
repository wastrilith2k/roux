#!/usr/bin/env python3
"""
Companion CLI -- minimal terminal chat client.

WHAT: Async terminal application that connects to the companion backend via
      Socket.IO, sends messages, displays responses, and handles slash
      commands (/help, /state, /see, /activity, /test, /pending, /approve, etc.).

WHY:  Provides a developer-friendly chat interface without needing the web
      frontend. Useful for testing, debugging, and quick conversations.
      Also supports test mode (messages not saved to DB) and fact approval
      workflow.

HOW:  Uses python-socketio (async client) to connect to the Flask-SocketIO
      backend. User input via prompt_toolkit (multi-line, arrow keys, history).
      Incoming messages are routed through `message_handler()` which dispatches
      by type (response, thinking, error, history, state, images, activity,
      fact_approval, etc.). Slash commands are handled by CommandHandler.
"""

import asyncio
import os
import sys
from typing import Optional, List, Dict
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from connection import CompanionConnection
from ui import CliUI
from commands import CommandHandler


class CompanionCLI:
    """Main CLI application"""

    def __init__(self, backend_url: str = "http://localhost:5000", api_key: Optional[str] = None):
        """
        Initialize CLI application

        Args:
            backend_url: WebSocket URL of backend
            api_key: Authentication API key
        """
        self.connection = CompanionConnection(api_key=api_key, backend_url=backend_url)
        self.ui = CliUI()
        self.commands = CommandHandler(self)
        self.running = False
        self.message_history: List[Dict] = []
        self.waiting_for_response = False
        self.test_mode = False  # When True, messages are not saved to database
        self.pending_see_command = False  # Track /see command for simplified output
        self.pending_approval = None  # Current pending fact awaiting approval

    def add_to_history(self, role: str, content: str, metadata: Optional[dict] = None):
        """Add message to local history"""
        self.message_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        })

    async def message_handler(self, message: dict):
        """Handle messages from backend"""
        msg_type = message.get("type", "unknown")

        if msg_type == "response" or msg_type == "message":
            content = message.get("content", "")
            metadata = message.get("metadata", {})
            mood = metadata.get("emotion") if metadata else None

            self.ui.clear_thinking()

            # Prefix with [TEST] if in test mode
            if self.test_mode:
                self.ui.print_dim("[TEST] ", end="")

            self.ui.print_companion(content, mood=mood)

            self.add_to_history('companion', content, metadata)
            self.waiting_for_response = False

        elif msg_type == "thinking":
            status = message.get("status", "thinking")
            self.ui.print_thinking(status)

        elif msg_type == "error":
            self.ui.clear_thinking()
            self.ui.print_error(message.get("message", "unknown error"))
            self.waiting_for_response = False

        elif msg_type == "info":
            self.ui.clear_thinking()
            self.ui.print_info(message.get("message", ""))

        elif msg_type == "history":
            # Load history into local cache
            messages = message.get("messages", [])
            if messages:
                self.ui.print_dim("recent:")
                for msg in messages:
                    self.add_to_history(
                        msg.get('role', 'unknown'),
                        msg.get('content', ''),
                        None
                    )
                    # Display in history format (sanitize all content)
                    from ui import sanitize_output
                    role = msg.get('role', 'unknown')
                    content = sanitize_output(msg.get('content', ''))
                    if role == 'user':
                        self.ui.print_dim(f"  you: {content}")
                    else:
                        self.ui.print_dim(f"  {content}")
                print()

        elif msg_type == "images":
            # Display image URLs
            self.ui.clear_thinking()
            images = message.get("images", [])
            self.ui.print_images(images)

        elif msg_type == "activity":
            # Display activity info
            self.ui.clear_thinking()
            current_status = message.get("current_status") or {}
            activities = message.get("activities", [])
            self.ui.print_activity(current_status, activities)

        elif msg_type == "autopilot":
            # Display James autopilot status
            self.ui.clear_thinking()
            self.ui.print_autopilot(message)

        elif msg_type == "fact_approval_request":
            # Incoming approval request - prompt user
            self.ui.print_fact_approval_request(message)
            # Store for later action
            self.pending_approval = message

        elif msg_type == "pending_facts":
            # List of pending facts
            self.ui.print_pending_facts(message.get("facts", []))

        elif msg_type in ("fact_approved", "fact_rejected", "fact_edited"):
            # Confirmation of approval action
            self.ui.print_fact_action_result(msg_type, message)

        elif msg_type == "state":
            # Display scene and internal state
            self.ui.clear_thinking()
            scene = message.get("scene_state") or {}
            internal = message.get("internal_state") or {}

            # If /see command, just show the simplified scene view
            if self.pending_see_command:
                self.pending_see_command = False
                self.ui.print_scene(scene)
                return

            # Full /state output
            print()
            self.ui.print_info("=== SCENE STATE ===")
            if scene:
                for key, value in scene.items():
                    if value is not None and value != [] and value != "":
                        self.ui.print_dim(f"  {key}: {value}")
            else:
                self.ui.print_dim("  (no scene state)")

            print()
            self.ui.print_info("=== INTERNAL STATE ===")
            if internal:
                for key, value in internal.items():
                    if value is not None and value != [] and value != "":
                        self.ui.print_dim(f"  {key}: {value}")
            else:
                self.ui.print_dim("  (no internal state)")

            print()
            relationship = message.get("relationship_state") or {}
            self.ui.print_info("=== RELATIONSHIP STATE ===")
            if relationship:
                closeness = relationship.get("closeness", 0)
                trust = relationship.get("trust", 0)
                wounds = relationship.get("wounds", [])
                pos = relationship.get("positive_interactions", 0)
                neg = relationship.get("negative_interactions", 1)
                ratio = pos / neg if neg > 0 else pos

                # Calculate health
                combined = (closeness + trust) / 2
                if combined > 0.85:
                    health = "flourishing"
                elif combined > 0.70:
                    health = "healthy"
                elif combined > 0.50:
                    health = "strained"
                elif combined > 0.30:
                    health = "in_crisis"
                else:
                    health = "at_breaking_point"

                self.ui.print_dim(f"  closeness: {closeness:.0%}")
                self.ui.print_dim(f"  trust: {trust:.0%}")
                self.ui.print_dim(f"  health: {health}")
                self.ui.print_dim(f"  wounds: {len(wounds)}")
                self.ui.print_dim(f"  ratio: {ratio:.1f}:1 (pos:neg)")
            else:
                self.ui.print_dim("  (no relationship state)")
            print()

    async def run(self):
        """Main CLI loop"""
        self.ui.print_welcome()

        # Connect to backend
        self.ui.print_dim("connecting...")
        if not await self.connection.connect():
            self.ui.print_error("failed to connect to backend")
            sys.exit(1)

        self.ui.print_success("connected")
        self.running = True

        # Start listening for messages in background
        listen_task = asyncio.create_task(
            self.connection.listen(self.message_handler)
        )

        # Request recent history
        await self.connection.request_history(5)
        await asyncio.sleep(0.5)  # Brief pause for history to load

        # Main input loop
        while self.running:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.ui.print_prompt
                )

                if user_input is None:
                    # EOF or interrupt
                    break

                if not user_input:
                    # Empty input, just show prompt again
                    continue

                # Check for slash commands
                if self.commands.is_command(user_input):
                    result = await self.commands.execute(user_input)
                    if result.should_quit:
                        break
                    if result.message:
                        self.ui.print_info(result.message)
                    continue

                # Regular message - send to the companion
                self.add_to_history('user', user_input)
                self.ui.print_thinking()
                self.waiting_for_response = True

                # Use 'test' message_type if in test mode (won't save to database)
                msg_type = 'test' if self.test_mode else 'chat'
                sent = await self.connection.send_message(user_input, message_type=msg_type)
                if not sent:
                    self.ui.clear_thinking()
                    self.ui.print_error("failed to send message")
                    self.waiting_for_response = False

            except KeyboardInterrupt:
                print()
                break
            except Exception as e:
                self.ui.print_error(str(e))

        # Cleanup
        self.running = False
        listen_task.cancel()
        await self.connection.disconnect()
        print()


async def main():
    """Main entry point"""
    backend_url = os.environ.get("COMPANION_BACKEND_URL", "http://agent-service:5000")
    api_key = os.environ.get("COMPANION_API_KEY")

    cli = CompanionCLI(backend_url=backend_url, api_key=api_key)
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
