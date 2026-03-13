"""
/status command - Shows system status overview
"""

import os
from datetime import datetime
from typing import Dict, Any
from zoneinfo import ZoneInfo


def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status.

    Returns dict with:
    - autonomy: Autonomy system status
    - services: Service health
    - memory: Memory system stats
    - time: Current time info
    """
    status = {
        'time': {},
        'autonomy': {},
        'telegram': {},
        'services': {},
        'memory': {}
    }

    # Current time
    try:
        tz = ZoneInfo('America/Los_Angeles')
        now = datetime.now(tz)
        status['time'] = {
            'current': now.strftime('%A %I:%M %p %Z'),
            'date': now.strftime('%B %d, %Y')
        }
    except Exception as e:
        status['time'] = {'error': str(e)}

    # Autonomy status
    try:
        from src.autonomy.reach_out_engine import is_autonomy_enabled
        status['autonomy'] = {
            'enabled': is_autonomy_enabled(),
            'telegram_enabled': os.environ.get('COMPANION_TELEGRAM_ENABLED', 'false').lower() == 'true'
        }
    except Exception as e:
        status['autonomy'] = {'error': str(e)}

    # User autopilot
    try:
        from src.core.user_context import is_user_autopilot_enabled
        status['user_autopilot'] = {
            'enabled': is_user_autopilot_enabled()
        }
    except Exception as e:
        status['user_autopilot'] = {'error': str(e)}

    # Memory stats
    try:
        from src.database.db import get_db
        db = get_db()

        with db._get_connection() as conn:
            cursor = conn.cursor()

            # Message count
            cursor.execute("SELECT COUNT(*) FROM messages")
            msg_count = cursor.fetchone()[0]

            # Messages with embeddings
            cursor.execute("SELECT COUNT(*) FROM messages WHERE embedding_vec IS NOT NULL")
            embedded_count = cursor.fetchone()[0]

            # Fact count
            cursor.execute("SELECT COUNT(*) FROM facts")
            fact_count = cursor.fetchone()[0]

            cursor.close()

        status['memory'] = {
            'messages': msg_count,
            'embedded': embedded_count,
            'embedding_coverage': f"{(embedded_count/msg_count*100):.1f}%" if msg_count > 0 else "N/A",
            'facts': fact_count
        }
    except Exception as e:
        status['memory'] = {'error': str(e)}

    # LLM provider
    status['llm'] = {
        'provider': os.environ.get('LLM_PROVIDER', 'fireworks'),
        'model': os.environ.get('FIREWORKS_MODEL', 'kimi-k2-instruct-0905').split('/')[-1]
    }

    return status


def format_status_text(status: Dict[str, Any]) -> str:
    """Format status as readable text."""
    lines = ["**System Status:**\n"]

    # Time
    if 'time' in status and 'current' not in status['time'].get('error', ''):
        lines.append(f"**Time:** {status['time'].get('current', 'Unknown')}")
        lines.append(f"**Date:** {status['time'].get('date', 'Unknown')}")

    # LLM
    if 'llm' in status:
        lines.append(f"\n**LLM:** {status['llm'].get('provider', 'unknown')} / {status['llm'].get('model', 'unknown')}")

    # Autonomy
    if 'autonomy' in status and 'error' not in status['autonomy']:
        auto_status = "" if status['autonomy'].get('enabled') else ""
        tg_status = "" if status['autonomy'].get('telegram_enabled') else ""
        lines.append(f"\n**Autonomy:** {auto_status} Reach-out | {tg_status} Telegram")

    # User autopilot
    if 'user_autopilot' in status and 'error' not in status['user_autopilot']:
        ap_status = "" if status['user_autopilot'].get('enabled') else ""
        lines.append(f"**User Autopilot:** {ap_status}")

    # Memory
    if 'memory' in status and 'error' not in status['memory']:
        mem = status['memory']
        lines.append(f"\n**Memory:**")
        lines.append(f"  • Messages: {mem.get('messages', 0):,}")
        lines.append(f"  • Embedded: {mem.get('embedded', 0):,} ({mem.get('embedding_coverage', 'N/A')})")
        lines.append(f"  • Facts: {mem.get('facts', 0):,}")

    return '\n'.join(lines)


def handle_status_command(args: str, context: dict) -> dict:
    """Handle /status command."""
    status = get_system_status()

    return {
        'text': format_status_text(status),
        'data': status
    }


def register_status_command(registry):
    """Register the /status command."""
    registry.register(
        name='status',
        handler=handle_status_command,
        description='Show system status overview',
        aliases=['sys', 'info']
    )
