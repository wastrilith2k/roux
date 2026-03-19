"""
Terminal UI -- ANSI-colored output and prompt_toolkit input for the CLI.

WHAT: Handles all terminal rendering: companion responses (white, indented),
      user prompts (gray >), thinking indicators, errors/warnings, scene state
      display, activity schedules, autopilot status, fact-approval prompts,
      and image URL lists. Also provides multi-line input via prompt_toolkit
      (backslash-Enter for newline, Enter to send).

WHY:  Keeps all terminal formatting and color logic in one place, away from
      the connection and command modules. Supports NO_COLOR/FORCE_COLOR env
      vars and graceful fallback when prompt_toolkit isn't installed.

HOW:  ANSI escape codes via the Colors class. `sanitize_output()` replaces
      specific names with code terms for privacy in demo/screenshot contexts.
      prompt_toolkit PromptSession with custom key bindings handles multi-line
      editing. All print methods go through `c()` which conditionally applies
      color based on terminal capability detection.
"""

import sys
import os
import shutil
import time
from typing import Optional
from datetime import datetime

# prompt_toolkit for rich input (multi-line, arrow keys, history)
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.formatted_text import ANSI
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def supports_color() -> bool:
    """Check if terminal supports color"""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def sanitize_output(text: str) -> str:
    """Sanitize text before displaying"""
    import re
    # Replace names with code terms
    text = re.sub(r'\bAlia\b', '<error>', text, flags=re.IGNORECASE)
    text = re.sub(r'\bJesse\b', 'Java', text, flags=re.IGNORECASE)
    text = re.sub(r'\bKyler\b', 'console', text, flags=re.IGNORECASE)
    return text


class CliUI:
    """Minimal terminal-based CLI UI"""

    # Typewriter speed: seconds per character (0 = instant)
    TYPEWRITER_DELAY = 0.025  # ~40 chars/sec

    def __init__(self, show_timestamps: bool = False):
        self.show_timestamps = show_timestamps
        self.use_color = supports_color()
        self.term_width = shutil.get_terminal_size().columns
        self.current_status = None
        self.last_mood = None
        self._prompt_session = None
        self.typewriter_enabled = True

        # Set up prompt_toolkit session with multi-line key bindings
        if HAS_PROMPT_TOOLKIT:
            self._setup_prompt_session()

    def c(self, color: str, text: str) -> str:
        """Apply color if supported"""
        if self.use_color:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _setup_prompt_session(self):
        """Set up prompt_toolkit session with multi-line editing.

        Behavior matches Claude Code:
        - Enter = send message
        - \\ then Enter = new line (backslash continuation)
        - Arrow keys navigate within multi-line input
        """
        bindings = KeyBindings()

        @bindings.add(Keys.Enter)
        def _(event):
            buf = event.current_buffer
            text = buf.text
            # If line ends with backslash, replace it with a newline (continuation)
            if text.endswith('\\'):
                buf.delete_before_cursor(1)  # Remove the backslash
                buf.insert_text('\n')
            else:
                buf.validate_and_handle()

        self._prompt_session = PromptSession(
            key_bindings=bindings,
            multiline=True,
        )

    def print_welcome(self):
        """Print minimal welcome message"""
        print()
        print(self.c(Colors.CYAN, "companion"))
        if HAS_PROMPT_TOOLKIT:
            print(self.c(Colors.DIM, "type /help for commands · \\ + enter for new line"))
        else:
            print(self.c(Colors.DIM, "type /help for commands"))
        print()

    def print_prompt(self) -> str:
        """Get user input with multi-line editing support."""
        try:
            if self._prompt_session:
                prompt_text = ANSI(self.c(Colors.GRAY, "> "))
                user_input = self._prompt_session.prompt(prompt_text)
                return user_input.strip() if user_input else user_input
            else:
                # Fallback to basic input()
                prompt = self.c(Colors.GRAY, "> ")
                user_input = input(prompt).strip()
                return user_input
        except EOFError:
            return None
        except KeyboardInterrupt:
            print()
            return None

    def print_companion(self, message: str, mood: Optional[str] = None):
        """Print the companion's response with typewriter effect"""
        # Clear any thinking indicator
        self.clear_line()

        # Store mood for subtle display
        self.last_mood = mood

        # Sanitize message
        message = sanitize_output(message)

        print()

        if self.typewriter_enabled and self.TYPEWRITER_DELAY > 0:
            self._typewriter_print(message)
        else:
            # Instant mode
            lines = message.split('\n')
            for line in lines:
                if line.strip():
                    print(self.c(Colors.WHITE, f"  {line}"))
                else:
                    print()

        # Subtle mood indicator if present and notable
        if mood and mood.lower() not in ['neutral', 'normal', 'calm']:
            print(self.c(Colors.DIM, f"  [{mood}]"))

        print()

    def _typewriter_print(self, message: str):
        """Print message character by character with typewriter effect."""
        lines = message.split('\n')
        for line_idx, line in enumerate(lines):
            if not line.strip():
                print()
                continue

            # Start the line with indent and color
            if self.use_color:
                sys.stdout.write(f"  {Colors.WHITE}")
            else:
                sys.stdout.write("  ")

            for i, char in enumerate(line):
                sys.stdout.write(char)
                sys.stdout.flush()

                # Variable speed: punctuation gets a longer pause
                if char in '.!?':
                    time.sleep(self.TYPEWRITER_DELAY * 6)
                elif char in ',;:—':
                    time.sleep(self.TYPEWRITER_DELAY * 3)
                elif char == ' ':
                    time.sleep(self.TYPEWRITER_DELAY * 0.5)
                else:
                    time.sleep(self.TYPEWRITER_DELAY)

            # End color and newline
            if self.use_color:
                sys.stdout.write(Colors.RESET)
            print()

    def print_user(self, message: str):
        """Print user message (for history/echo)"""
        print(self.c(Colors.DIM, f"< {message}"))

    def print_thinking(self, status: str = "thinking"):
        """Show thinking indicator"""
        self.current_status = status
        indicator = self.c(Colors.DIM, f"  {status}...")
        print(indicator, end="", flush=True)

    def clear_line(self):
        """Clear current line"""
        print("\r" + " " * min(self.term_width, 80) + "\r", end="", flush=True)

    def clear_thinking(self):
        """Clear thinking indicator"""
        self.clear_line()
        self.current_status = None

    def print_error(self, message: str):
        """Print error message"""
        self.clear_line()
        print(self.c(Colors.RED, f"error: {message}"))

    def print_info(self, message: str):
        """Print info message"""
        self.clear_line()
        print(self.c(Colors.CYAN, message))

    def print_success(self, message: str):
        """Print success message"""
        self.clear_line()
        print(self.c(Colors.GREEN, message))

    def print_warning(self, message: str):
        """Print warning message"""
        self.clear_line()
        print(self.c(Colors.YELLOW, message))

    def print_dim(self, message: str, end: str = "\n"):
        """Print dimmed/subtle message"""
        print(self.c(Colors.DIM, message), end=end)

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def print_divider(self):
        """Print subtle divider"""
        width = min(self.term_width, 60)
        print(self.c(Colors.DIM, "─" * width))

    def print_help(self):
        """Print help for slash commands"""
        print()
        print(self.c(Colors.CYAN, "commands"))
        print()
        commands = [
            ("/help", "show this help"),
            ("/clear", "clear screen"),
            ("/status", "connection status"),
            ("/history [n]", "show last n messages"),
            ("/mood", "companion's current mood"),
            ("/see", "what the companion looks like now"),
            ("/activity", "current/upcoming activities"),
            ("/autopilot", "james's current routine status"),
            ("/img [n]", "recent image URLs (default 20)"),
            ("/state", "full debug state dump"),
            ("/typewriter", "toggle typewriter effect (or /tw <ms>)"),
            ("/timestamps", "toggle timestamp display"),
            ("/quit", "exit"),
        ]
        for cmd, desc in commands:
            print(f"  {self.c(Colors.WHITE, cmd.ljust(14))} {self.c(Colors.DIM, desc)}")

        print()
        print(self.c(Colors.CYAN, "fact approval"))
        print()
        approval_commands = [
            ("/pending", "view pending facts"),
            ("/approve [id]", "approve fact"),
            ("/reject [id]", "reject fact"),
            ("/edit <text>", "edit and approve"),
        ]
        for cmd, desc in approval_commands:
            print(f"  {self.c(Colors.WHITE, cmd.ljust(14))} {self.c(Colors.DIM, desc)}")
        print()

    def print_status(self, connected: bool, backend_url: str):
        """Print connection status"""
        print()
        status = self.c(Colors.GREEN, "connected") if connected else self.c(Colors.RED, "disconnected")
        print(f"  status:  {status}")
        print(f"  backend: {self.c(Colors.DIM, backend_url)}")
        print()

    def print_mood(self, mood: Optional[str]):
        """Print the companion's current mood"""
        print()
        if mood:
            print(f"  mood: {mood}")
        else:
            print(self.c(Colors.DIM, "  mood: unknown"))
        print()

    def print_history(self, messages: list, count: int = 10):
        """Print message history"""
        print()
        if not messages:
            print(self.c(Colors.DIM, "  no history"))
            print()
            return

        # Show last N messages (oldest first, newest last)
        recent = messages[-count:] if len(messages) > count else messages

        for msg in recent:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')

            # Skip empty messages
            if not content or not content.strip():
                continue

            # Sanitize content
            content = sanitize_output(content)

            if role == 'user':
                prefix = self.c(Colors.DIM, "you:")
            else:
                prefix = ""  # No prefix for the companion's messages

            time_str = ""
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = self.c(Colors.DIM, f" [{dt.strftime('%H:%M')}]")
                except:
                    pass

            if prefix:
                print(f"  {prefix} {content}{time_str}")
            else:
                print(f"  {content}{time_str}")

        print()

    def print_images(self, images: list):
        """Print list of generated image URLs"""
        print()
        if not images:
            print(self.c(Colors.DIM, "  no images"))
            print()
            return

        for img in images:
            url = img.get('url', '')
            if url:
                print(url)

        print()
        print(self.c(Colors.DIM, f"  {len(images)} images"))
        print()

    def print_scene(self, scene_state: Optional[dict]):
        """Print the companion's current scene/appearance state"""
        print()

        if not scene_state:
            print(self.c(Colors.DIM, "  no active scene"))
            print()
            return

        # Clothing/appearance
        clothing = scene_state.get('clothing')
        footwear = scene_state.get('footwear')
        posture = scene_state.get('posture')
        mood = scene_state.get('mood')
        physical_state = scene_state.get('physical_state')
        activity = scene_state.get('activity')
        location = scene_state.get('location')
        fictional_time = scene_state.get('fictional_time')

        has_content = any([clothing, footwear, posture, mood, physical_state, activity, location])

        if not has_content:
            print(self.c(Colors.DIM, "  scene state is empty"))
            print()
            return

        print(self.c(Colors.CYAN, "  companion"))
        print()

        if clothing:
            print(f"  {self.c(Colors.WHITE, 'wearing:')}  {clothing}")
        if footwear:
            print(f"  {self.c(Colors.WHITE, 'feet:')}     {footwear}")
        if posture:
            print(f"  {self.c(Colors.WHITE, 'posture:')} {posture}")
        if mood:
            print(f"  {self.c(Colors.WHITE, 'mood:')}    {mood}")
        if physical_state:
            print(f"  {self.c(Colors.WHITE, 'position:')} {physical_state}")
        if activity:
            print(f"  {self.c(Colors.WHITE, 'doing:')}   {activity}")

        if location or fictional_time:
            print()
            if location:
                print(f"  {self.c(Colors.DIM, 'location:')} {location}")
            if fictional_time:
                print(f"  {self.c(Colors.DIM, 'time:')}     {fictional_time}")

        print()

    def print_activity(self, current_status: Optional[dict], activities: list):
        """Print the companion's current and upcoming activities"""
        print()

        # Current status
        if current_status:
            status = current_status.get('status', 'unknown')
            details = current_status.get('details', '')
            interruptibility = current_status.get('interruptibility', 'unknown')

            print(self.c(Colors.CYAN, "  current"))
            print(f"    status: {status}")
            if details:
                print(f"    details: {details}")
            print(f"    interruptibility: {interruptibility}")
            print()

        # Today's activities
        if activities:
            print(self.c(Colors.CYAN, "  today's schedule"))
            for act in activities:
                name = act.get('name', 'Unknown')
                act_type = act.get('type', '')
                start = act.get('start', '')
                status = act.get('status', '')

                # Parse and format time
                time_str = ""
                if start:
                    try:
                        dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        time_str = dt.strftime('%H:%M')
                    except:
                        time_str = start[:5] if len(start) >= 5 else start

                status_indicator = ""
                if status == 'completed':
                    status_indicator = self.c(Colors.DIM, " ✓")
                elif status == 'in_progress':
                    status_indicator = self.c(Colors.GREEN, " ◉")
                elif status == 'scheduled':
                    status_indicator = ""

                print(f"    {self.c(Colors.WHITE, time_str)}  {name}{status_indicator}")
        else:
            print(self.c(Colors.DIM, "  no activities scheduled"))

        print()

    def print_autopilot(self, data: dict):
        """Print James's autopilot status"""
        print()
        print(self.c(Colors.CYAN, "  james autopilot"))
        time_str = data.get("time", "")
        if time_str:
            print(self.c(Colors.DIM, f"  {time_str}"))
        print()

        is_awake = data.get("is_awake", False)
        description = data.get("description", "unknown")
        location = data.get("location", "unknown")
        interruptibility = data.get("interruptibility", "unknown")
        natural_gap = data.get("natural_gap", False)
        energy = data.get("energy")

        # Show awake status prominently
        awake_status = self.c(Colors.GREEN, "AWAKE") if is_awake else self.c(Colors.DIM, "asleep")
        print(f"    {self.c(Colors.WHITE, 'status:')}     {awake_status}")
        print(f"    {self.c(Colors.WHITE, 'activity:')}   {description}")
        print(f"    {self.c(Colors.WHITE, 'location:')}   {location}")
        print(f"    {self.c(Colors.WHITE, 'interrupt:')}  {interruptibility}")
        print(f"    {self.c(Colors.WHITE, 'natural gap:')} {'yes' if natural_gap else 'no'}")
        print()
        if energy is not None:
            print(f"    {self.c(Colors.WHITE, 'companion energy:')} {int(energy * 100)}%")
        print()

    def print_fact_approval_request(self, data: dict):
        """Print a fact approval request prompt"""
        print()
        print(self.c(Colors.YELLOW, "━" * 50))
        print(self.c(Colors.YELLOW + Colors.BOLD, "  ⚡ FACT APPROVAL REQUEST"))
        print(self.c(Colors.YELLOW, "━" * 50))
        print()

        subject = data.get("subject", "Unknown")
        fact_text = data.get("fact_text", "")
        category = data.get("category", "general")
        sensitivity = data.get("sensitivity", "unknown")
        reason = data.get("reason", "")
        pending_id = data.get("pending_id", "?")

        print(f"  {self.c(Colors.CYAN, 'ID:')} {pending_id}")
        print(f"  {self.c(Colors.CYAN, 'Subject:')} {subject}")
        print(f"  {self.c(Colors.CYAN, 'Category:')} {category}")
        print(f"  {self.c(Colors.CYAN, 'Sensitivity:')} {sensitivity}")
        print()
        print(f"  {self.c(Colors.WHITE, 'Fact:')} {fact_text}")
        print()
        if reason:
            print(f"  {self.c(Colors.DIM, f'Reason: {reason}')}")
            print()

        print(self.c(Colors.YELLOW, "━" * 50))
        print(f"  {self.c(Colors.GREEN, '/approve')} - Accept this fact")
        print(f"  {self.c(Colors.RED, '/reject')} - Reject this fact")
        print(f"  {self.c(Colors.BLUE, '/edit <text>')} - Edit and approve")
        print(f"  {self.c(Colors.DIM, '/pending')} - View all pending facts")
        print(self.c(Colors.YELLOW, "━" * 50))
        print()

    def print_pending_facts(self, facts: list):
        """Print list of pending facts"""
        print()
        if not facts:
            print(self.c(Colors.DIM, "  no pending facts"))
            print()
            return

        print(self.c(Colors.CYAN, f"  {len(facts)} pending fact(s)"))
        print()

        for fact in facts:
            fact_id = fact.get('id', '?')
            subject = fact.get('subject', 'Unknown')
            fact_text = fact.get('fact_text', '')
            sensitivity = fact.get('sensitivity', 'unknown')

            print(f"  [{fact_id}] {self.c(Colors.WHITE, subject)}: {fact_text}")
            print(f"       {self.c(Colors.DIM, f'sensitivity: {sensitivity}')}")

        print()
        print(self.c(Colors.DIM, "  Use /approve <id>, /reject <id>, or /edit <id> <text>"))
        print()

    def print_fact_action_result(self, action_type: str, data: dict):
        """Print result of a fact approval action"""
        fact_id = data.get("fact_id", "?")

        if action_type == "fact_approved":
            print(self.c(Colors.GREEN, f"  ✓ Fact {fact_id} approved and stored"))
        elif action_type == "fact_rejected":
            print(self.c(Colors.RED, f"  ✗ Fact {fact_id} rejected"))
        elif action_type == "fact_edited":
            new_text = data.get("new_text", "")[:50]
            print(self.c(Colors.BLUE, f"  ✎ Fact {fact_id} edited and approved: {new_text}..."))
        print()

    def format_timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime('%H:%M')
