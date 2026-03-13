"""
Ops Bot Command Handler - Interactive ops via Telegram

Handles commands from James and can spawn Claude Code for fixes.

NOTE: The actual bot runs in a separate subprocess to avoid
eventlet/asyncio conflicts. This module just manages the subprocess.
"""

import os
import subprocess
import logging
from typing import Optional
from threading import Thread

logger = logging.getLogger(__name__)


class OpsBotHandler:
    """
    Manager for the ops bot subprocess.

    The actual Telegram bot runs in a separate Python process
    to avoid conflicts with eventlet's monkey patching.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[Thread] = None

    def start(self) -> bool:
        """Start the ops bot as a subprocess."""
        token = os.environ.get('TELEGRAM_OPS_BOT_TOKEN')
        if not token:
            logger.warning("TELEGRAM_OPS_BOT_TOKEN not set - ops bot disabled")
            return False

        # Path to the runner script
        runner_path = os.path.join(
            os.path.dirname(__file__),
            'ops_bot_runner.py'
        )

        if not os.path.exists(runner_path):
            logger.error(f"Ops bot runner not found: {runner_path}")
            return False

        def monitor():
            """Monitor subprocess and log any issues."""
            try:
                # Start the subprocess
                self._process = subprocess.Popen(
                    ['python', runner_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ.copy()
                )

                logger.info(f"Ops bot subprocess started (PID: {self._process.pid})")

                # Monitor output
                if self._process.stdout:
                    for line in self._process.stdout:
                        line = line.strip()
                        if line:
                            logger.info(f"[ops-bot] {line}")

                # Process ended
                returncode = self._process.wait()
                if returncode != 0:
                    logger.error(f"Ops bot subprocess exited with code {returncode}")
                else:
                    logger.info("Ops bot subprocess exited normally")

            except Exception as e:
                logger.error(f"Ops bot subprocess error: {e}")

        self._monitor_thread = Thread(target=monitor, daemon=True, name="ops-bot-monitor")
        self._monitor_thread.start()
        logger.info("Ops bot command handler started (subprocess)")
        return True

    def stop(self):
        """Stop the ops bot subprocess."""
        if self._process:
            logger.info("Stopping ops bot subprocess...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def is_running(self) -> bool:
        """Check if the subprocess is running."""
        return self._process is not None and self._process.poll() is None


# Singleton
_handler: Optional[OpsBotHandler] = None


def get_ops_bot_handler() -> OpsBotHandler:
    """Get singleton handler instance."""
    global _handler
    if _handler is None:
        _handler = OpsBotHandler()
    return _handler


def start_ops_bot() -> bool:
    """Start the ops bot command handler."""
    handler = get_ops_bot_handler()
    return handler.start()
