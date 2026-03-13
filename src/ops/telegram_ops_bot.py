"""
Telegram Ops Bot -- one-way notification sender for system alerts.

WHAT: Sends formatted Telegram messages for system events: health-check failures,
      maintenance summaries, cost alerts, deployment status, and error spikes.

WHY:  The companion's own Telegram channel is for relationship conversation.
      Ops alerts go to a separate bot/chat so system noise doesn't interrupt
      the user experience. This class is the write side; the interactive
      command side lives in ops_bot_runner.py.

HOW:  Uses python-telegram-bot's Bot class directly (not the Updater). Provides
      both sync (`send_sync`) and async (`send`) methods. Chat ID is loaded from
      TELEGRAM_OPS_CHAT_ID env var or persisted to a local file on first use.
      Formatted reports use Telegram's MarkdownV2 for readability.

Singleton: `get_ops_telegram_bot()` at module bottom.
"""

import os
import asyncio
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Separate bot token and chat ID for ops notifications
OPS_BOT_TOKEN = os.environ.get('TELEGRAM_OPS_BOT_TOKEN')
OPS_CHAT_ID = os.environ.get('TELEGRAM_OPS_CHAT_ID')


class OpsTelegramBot:
    """
    Simple Telegram bot for ops notifications.

    Unlike the companion's bridge, this is one-way (notifications only).
    """

    def __init__(self):
        self._bot = None
        self._chat_id = OPS_CHAT_ID

    def _get_bot(self):
        """Lazy-load the bot."""
        if self._bot is None:
            if not OPS_BOT_TOKEN:
                raise ValueError("TELEGRAM_OPS_BOT_TOKEN not set")
            try:
                from telegram import Bot
                self._bot = Bot(token=OPS_BOT_TOKEN)
            except ImportError:
                raise ImportError("python-telegram-bot not installed")
        return self._bot

    def _load_chat_id(self):
        """Load chat ID from file if not in env."""
        if self._chat_id:
            return
        try:
            chat_id_file = os.path.join(
                os.environ.get('DATA_DIR', '/app/data'),
                'ops_telegram_chat_id.txt'
            )
            if os.path.exists(chat_id_file):
                with open(chat_id_file, 'r') as f:
                    self._chat_id = f.read().strip()
                logger.info(f"Loaded ops chat_id: {self._chat_id}")
        except Exception as e:
            logger.warning(f"Could not load ops chat_id: {e}")

    def save_chat_id(self, chat_id: str):
        """Save chat ID for future use."""
        self._chat_id = chat_id
        try:
            chat_id_file = os.path.join(
                os.environ.get('DATA_DIR', '/app/data'),
                'ops_telegram_chat_id.txt'
            )
            with open(chat_id_file, 'w') as f:
                f.write(str(chat_id))
            logger.info(f"Saved ops chat_id to {chat_id_file}")
        except Exception as e:
            logger.warning(f"Could not save ops chat_id: {e}")

    async def send_message_async(self, message: str, parse_mode: str = None) -> bool:
        """Send a message asynchronously."""
        self._load_chat_id()

        if not self._chat_id:
            logger.error("Ops chat_id not set - run setup first")
            return False

        try:
            bot = self._get_bot()
            await bot.send_message(
                chat_id=int(self._chat_id),
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"Sent ops message: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send ops message: {e}")
            return False

    def send_message(self, message: str, parse_mode: str = None) -> bool:
        """Send a message synchronously."""
        try:
            return asyncio.run(self.send_message_async(message, parse_mode))
        except RuntimeError:
            # Already in an event loop
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(self.send_message_async(message, parse_mode))
            return loop.run_until_complete(future)

    def send_maintenance_summary(self, summary: dict) -> bool:
        """
        Send a formatted maintenance summary.

        Args:
            summary: Dict with keys:
                - timestamp: When the check ran
                - total_checks: Number of checks
                - passed: Number passed
                - failed: Number failed
                - results: Dict of check_name -> {passed, message}
                - failures: List of failed check names
                - backup_taken: Whether backup was taken
                - backup_path: Path to backup if taken
        """
        lines = []

        # Header
        timestamp = summary.get('timestamp', datetime.now().isoformat())
        lines.append(f"🔧 COMPANION MAINTENANCE REPORT")
        lines.append(f"📅 {timestamp}")
        lines.append("")

        # Backup status (if issues were found)
        if summary.get('backup_taken'):
            lines.append(f"💾 BACKUP TAKEN: {summary.get('backup_path', 'yes')}")
            lines.append("")

        # Overall status
        passed = summary.get('passed', 0)
        total = summary.get('total_checks', 0)
        failed = summary.get('failed', 0)

        if failed == 0:
            lines.append(f"✅ ALL SYSTEMS HEALTHY ({passed}/{total})")
        else:
            lines.append(f"⚠️ ISSUES FOUND: {failed}/{total} checks failed")
        lines.append("")

        # Individual results
        lines.append("─" * 30)
        results = summary.get('results', {})
        for name, data in results.items():
            status = "✅" if data.get('passed') else "❌"
            msg = data.get('message', '')
            lines.append(f"{status} {name}")
            if msg:
                lines.append(f"   {msg}")

        lines.append("─" * 30)

        # Footer
        if summary.get('fixes_made'):
            lines.append("")
            lines.append(f"🔨 FIXES APPLIED: {summary.get('fixes_made')}")

        if summary.get('commit_made'):
            lines.append(f"📝 COMMIT: {summary.get('commit_hash', 'yes')}")

        message = "\n".join(lines)
        return self.send_message(message)


# Singleton
_ops_bot: Optional[OpsTelegramBot] = None


def get_ops_bot() -> OpsTelegramBot:
    """Get singleton ops bot instance."""
    global _ops_bot
    if _ops_bot is None:
        _ops_bot = OpsTelegramBot()
    return _ops_bot


def send_ops_notification(message: str) -> bool:
    """Convenience function to send an ops notification."""
    bot = get_ops_bot()
    return bot.send_message(message)


def send_maintenance_report(summary: dict) -> bool:
    """Convenience function to send a maintenance report."""
    bot = get_ops_bot()
    return bot.send_maintenance_summary(summary)


# CLI for setup
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("Ops Telegram Bot Setup")
        print("=" * 40)
        print()
        print("1. Create a new bot via @BotFather on Telegram")
        print("2. Copy the bot token")
        print("3. Set environment variable: TELEGRAM_OPS_BOT_TOKEN=<token>")
        print("4. Message your new bot with /start")
        print("5. Run this script again with 'test' to send a test message")
        print()
        print("To get your chat ID, message the bot and check:")
        print("  https://api.telegram.org/bot<TOKEN>/getUpdates")
        print()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Sending test message...")
        result = send_ops_notification("🔧 Test message from Companion Ops Bot")
        print(f"Result: {'Success' if result else 'Failed'}")
    else:
        print("Usage:")
        print("  python telegram_ops_bot.py setup  - Show setup instructions")
        print("  python telegram_ops_bot.py test   - Send test message")
