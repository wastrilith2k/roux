#!/usr/bin/env python3
"""
Ops Bot Runner -- standalone Telegram bot for system administration.

WHAT: Interactive Telegram bot with slash-commands for live system management:
      /status, /maintenance, /backup, /deploy, /restart, /logs, /fix (spawns
      Claude for auto-repair), /costs, /companion (send message as companion),
      /models (list LLM models), /core (show config).

WHY:  The main Flask process uses eventlet, which conflicts with python-telegram-bot's
      asyncio event loop. Running the ops bot in a separate process avoids that.
      It also provides a secure remote control plane over Telegram -- critical
      since the companion server has no SSH-accessible web admin UI.

HOW:  Launched as a subprocess from the main app. Authenticates every command
      against TELEGRAM_OPS_CHAT_ID to restrict access to the owner. Commands
      shell out to scripts in /root/companion/scripts/ for backup, deploy, etc.
"""

import os
import sys
import asyncio
import subprocess
import logging
from datetime import datetime

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Auth - only allow the configured chat ID
AUTHORIZED_CHAT_ID = os.environ.get('TELEGRAM_OPS_CHAT_ID')
TOKEN = os.environ.get('TELEGRAM_OPS_BOT_TOKEN')


def is_authorized(chat_id: int) -> bool:
    """Check if the chat ID is authorized."""
    if not AUTHORIZED_CHAT_ID:
        return False
    return str(chat_id) == str(AUTHORIZED_CHAT_ID)


async def cmd_start(update, context):
    """Handle /start command."""
    from telegram import Update
    chat_id = update.effective_chat.id

    if not is_authorized(chat_id):
        await update.message.reply_text("Unauthorized.")
        return

    await update.message.reply_text(
        "🔧 Companion Ops Bot\n\n"
        "System:\n"
        "/status - Quick health check\n"
        "/maintenance - Full maintenance run\n"
        "/backup - Trigger full backup\n"
        "/deploy - Pull & rebuild\n"
        "/restart <service> - Restart container\n"
        "/logs [service] - Get recent logs\n\n"
        "Companion:\n"
        "/companion - Companion system status\n"
        "/models - LLM models in use\n"
        "/costs - Recent API costs\n"
        "/core <cmd> - Run any core command\n\n"
        "AI Fix:\n"
        "/fix <issue> - Have Claude fix something\n\n"
        "Or just describe an issue and I'll pass it to Claude."
    )


async def cmd_help(update, context):
    """Handle /help command."""
    await cmd_start(update, context)


async def cmd_status(update, context):
    """Handle /status - quick health check."""
    if not is_authorized(update.effective_chat.id):
        return

    await update.message.reply_text("Running health check...")

    try:
        from src.diagnostics.system_health import run_diagnostics
        results = run_diagnostics(report_to_sentry=False)

        lines = [f"📊 Status: {results['passed']}/{results['total_checks']} passing\n"]

        for name, data in results['results'].items():
            status = "✅" if data['passed'] else "❌"
            lines.append(f"{status} {name}")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_backup(update, context):
    """Handle /backup - trigger full backup."""
    if not is_authorized(update.effective_chat.id):
        return

    await update.message.reply_text("Starting backup...")

    try:
        result = subprocess.run(
            ["bash", "-c", "cd /root/companion && ./scripts/full-backup.sh"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            backup_path = lines[-1] if lines else "unknown"
            await update.message.reply_text(f"✅ Backup complete:\n{backup_path}")
        else:
            await update.message.reply_text(f"❌ Backup failed:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        await update.message.reply_text("❌ Backup timed out")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_maintenance(update, context):
    """Handle /maintenance - run full maintenance."""
    if not is_authorized(update.effective_chat.id):
        return

    await update.message.reply_text("Running maintenance check...")

    try:
        from src.ops.maintenance_runner import run_maintenance
        results = run_maintenance(send_notification=False, backup_on_issues=True)

        lines = [f"🔧 Maintenance: {results['passed']}/{results['total_checks']} passing"]

        if results.get('backup_taken'):
            lines.append(f"💾 Backup: {results.get('backup_path', 'yes')}")

        lines.append("")
        for name, data in results['results'].items():
            status = "✅" if data['passed'] else "❌"
            lines.append(f"{status} {name}")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_fix(update, context):
    """Handle /fix <issue> - spawn Claude to fix."""
    if not is_authorized(update.effective_chat.id):
        return

    issue = ' '.join(context.args) if context.args else None

    if not issue:
        await update.message.reply_text(
            "Usage: /fix <describe the issue>\n"
            "Example: /fix the scheduler isn't running"
        )
        return

    await update.message.reply_text(f"🤖 Spawning Claude to fix: {issue}\n\nThis may take a few minutes...")

    try:
        result = await run_claude_fix(issue)

        # Truncate if too long for Telegram
        if len(result) > 4000:
            result = result[:3900] + "\n\n... (truncated)"

        await update.message.reply_text(f"Claude's response:\n\n{result}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error running Claude: {e}")


async def run_claude_fix(issue: str) -> str:
    """Run Claude Code to fix an issue."""
    prompt = f"""You are the Companion Maintenance Agent. An issue was reported via Telegram:

ISSUE: {issue}

Your tasks:
1. Investigate the issue
2. Check relevant logs and diagnostics
3. Fix the issue if possible
4. Be conservative - don't break things
5. Commit changes if you make fixes (prefix with "maintenance:")

Start by investigating the reported issue.
"""

    process = await asyncio.create_subprocess_exec(
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        prompt,
        cwd="/root/companion",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await asyncio.wait_for(
        process.communicate(),
        timeout=300  # 5 minute timeout
    )

    output = stdout.decode() if stdout else ""
    if stderr:
        output += f"\n\nStderr: {stderr.decode()}"

    return output or "No output from Claude"


async def cmd_logs(update, context):
    """Handle /logs [service] - get recent logs."""
    if not is_authorized(update.effective_chat.id):
        return

    service = context.args[0] if context.args else "companion-backend"

    try:
        result = subprocess.run(
            ["docker", "logs", service, "--tail", "50"],
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout or result.stderr or "No logs"

        # Truncate if too long
        if len(output) > 4000:
            output = output[-3900:] + "\n\n(showing last 3900 chars)"

        await update.message.reply_text(f"📋 Logs for {service}:\n\n{output}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_restart(update, context):
    """Handle /restart <service> - restart a container."""
    if not is_authorized(update.effective_chat.id):
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /restart <service>\n"
            "Example: /restart companion-backend"
        )
        return

    service = context.args[0]

    # Safety check - only allow known services
    allowed = ['companion-backend', 'companion-redis', 'companion-neo4j', 'companion-code-executor', 'companion-postgres']
    if service not in allowed:
        await update.message.reply_text(
            f"Unknown service. Allowed: {', '.join(allowed)}"
        )
        return

    await update.message.reply_text(f"Restarting {service}...")

    try:
        result = subprocess.run(
            ["docker", "restart", service],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            await update.message.reply_text(f"✅ {service} restarted")
        else:
            await update.message.reply_text(f"❌ Failed: {result.stderr}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_deploy(update, context):
    """Handle /deploy - pull latest code and rebuild."""
    if not is_authorized(update.effective_chat.id):
        return

    await update.message.reply_text("🚀 Starting deploy...\n\n1/3 Pulling latest code...")

    try:
        # Git pull
        result = subprocess.run(
            ["git", "pull"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/root/companion"
        )

        if result.returncode != 0:
            await update.message.reply_text(f"❌ Git pull failed:\n{result.stderr}")
            return

        await update.message.reply_text(f"✅ Git pull done\n\n2/3 Rebuilding container...")

        # Rebuild
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "--build", "--force-recreate", "agent-service"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/root/companion"
        )

        if result.returncode != 0:
            await update.message.reply_text(f"❌ Build failed:\n{result.stderr[-1000:]}")
            return

        await update.message.reply_text("✅ Deploy complete!\n\n3/3 Container rebuilt and running.")

    except subprocess.TimeoutExpired:
        await update.message.reply_text("❌ Deploy timed out")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_costs(update, context):
    """Handle /costs - show recent API costs."""
    if not is_authorized(update.effective_chat.id):
        return

    await update.message.reply_text("📊 Fetching cost data...")

    try:
        from src.core.cost_tracker import get_cost_summary

        summary = get_cost_summary()

        lines = ["💰 API Costs\n"]

        if summary.get('today'):
            lines.append(f"Today: ${summary['today']:.4f}")
        if summary.get('week'):
            lines.append(f"This week: ${summary['week']:.4f}")
        if summary.get('month'):
            lines.append(f"This month: ${summary['month']:.4f}")

        if summary.get('by_model'):
            lines.append("\nBy model:")
            for model, cost in summary['by_model'].items():
                lines.append(f"  {model}: ${cost:.4f}")

        await update.message.reply_text("\n".join(lines))
    except ImportError:
        await update.message.reply_text("Cost tracker not available")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_companion(update, context):
    """Handle /companion - Companion-specific system status."""
    if not is_authorized(update.effective_chat.id):
        return

    await update.message.reply_text("💜 Checking companion systems...")

    lines = ["💜 Companion Status\n"]

    # Check autonomy
    try:
        from src.autonomy.reach_out_engine import is_autonomy_enabled
        autonomy = "✅ Enabled" if is_autonomy_enabled() else "⏸️ Disabled"
        lines.append(f"Autonomy: {autonomy}")
    except:
        lines.append("Autonomy: ❓ Unknown")

    # Check user autopilot
    try:
        from src.core.user_context import is_user_autopilot_enabled
        autopilot = "✅ Enabled" if is_user_autopilot_enabled() else "⏸️ Disabled"
        lines.append(f"User Autopilot: {autopilot}")
    except:
        lines.append("User Autopilot: ❓ Unknown")

    # Check value inference
    try:
        from src.autonomy.value_inference import get_value_inference
        vi = get_value_inference()
        stats = vi.get_stats()
        if stats:
            categories = len(stats)
            latest = max(s.get('last_updated') for s in stats.values() if s.get('last_updated'))
            if latest:
                age = (datetime.now() - latest).days
                lines.append(f"Value Inference: {categories} categories, {age}d old")
            else:
                lines.append(f"Value Inference: {categories} categories")
        else:
            lines.append("Value Inference: No data")
    except:
        lines.append("Value Inference: ❓ Unknown")

    # Check recent messages
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT COUNT(*) as count, MAX(timestamp) as latest
                FROM messages
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            result = cur.fetchone()
        conn.close()

        count = result['count'] if result else 0
        lines.append(f"Messages (24h): {count}")
    except:
        lines.append("Messages: ❓ Unknown")

    # Check Telegram bridge
    try:
        from src.autonomy.telegram_bridge import get_telegram_bridge
        bridge = get_telegram_bridge()
        if bridge.is_ready():
            lines.append(f"Telegram: ✅ Ready")
        else:
            lines.append(f"Telegram: ⚠️ Not ready")
    except:
        lines.append("Telegram: ❓ Unknown")

    # Check scheduler jobs
    try:
        from src.scheduling.unified_scheduler import get_unified_scheduler
        scheduler = get_unified_scheduler()
        jobs = scheduler.list_jobs()
        lines.append(f"Scheduled Jobs: {len(jobs)}")
    except:
        lines.append("Scheduled Jobs: ❓ Unknown")

    await update.message.reply_text("\n".join(lines))


async def cmd_models(update, context):
    """Handle /models - show LLM models in use (via core commands)."""
    if not is_authorized(update.effective_chat.id):
        return

    try:
        from src.core.commands import get_command_registry
        registry = get_command_registry()
        result = registry.execute('/models', context={'interface': 'telegram'})

        # Telegram uses different markdown, convert
        text = result.get('text', 'No response')
        # Convert **bold** to Telegram's *bold* for MarkdownV2
        text = text.replace('**', '*')

        await update.message.reply_text(text, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_core(update, context):
    """Handle /core <command> - run any core command."""
    if not is_authorized(update.effective_chat.id):
        return

    # Get the command from args
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /core <command>\n\n"
            "Examples:\n"
            "  /core models\n"
            "  /core status\n"
            "  /core help"
        )
        return

    # Build full command string
    cmd_str = '/' + ' '.join(args)

    try:
        from src.core.commands import get_command_registry
        registry = get_command_registry()
        result = registry.execute(cmd_str, context={'interface': 'telegram'})

        text = result.get('text', 'No response')
        text = text.replace('**', '*')

        await update.message.reply_text(text, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def handle_message(update, context):
    """Handle plain text messages - treat as fix requests."""
    if not is_authorized(update.effective_chat.id):
        return

    message = update.message.text

    await update.message.reply_text(
        f"Got it. Want me to have Claude look into this?\n\n"
        f"Reply /fix {message}\n\n"
        f"Or use /status for a quick health check."
    )


def main():
    """Main entry point."""
    if not TOKEN:
        logger.error("TELEGRAM_OPS_BOT_TOKEN not set")
        sys.exit(1)

    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    # Build application
    app = Application.builder().token(TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("backup", cmd_backup))
    app.add_handler(CommandHandler("maintenance", cmd_maintenance))
    app.add_handler(CommandHandler("fix", cmd_fix))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("restart", cmd_restart))
    app.add_handler(CommandHandler("deploy", cmd_deploy))
    app.add_handler(CommandHandler("costs", cmd_costs))
    app.add_handler(CommandHandler("companion", cmd_companion))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("core", cmd_core))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Ops bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
