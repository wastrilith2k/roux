#!/usr/bin/env python3
"""
Admin CLI — Companion Framework

Unified admin tool for monitoring, debugging, and managing the companion framework.

Usage:
    python scripts/admin.py status
    python scripts/admin.py errors [--companion kai] [--module fact_store] [--limit 20]
    python scripts/admin.py costs [--companion kai] [--days 7]
    python scripts/admin.py companions
    python scripts/admin.py companion kai
    python scripts/admin.py memory kai [--search "jesse"]
    python scripts/admin.py goals kai
    python scripts/admin.py db
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def find_project_root() -> Path:
    candidates = [Path(__file__).parent.parent, Path('/app'), Path.cwd()]
    for p in candidates:
        if (p / 'data').exists():
            return p
    raise RuntimeError("Cannot find project root")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    """Get a raw psycopg2 connection."""
    from src.database.connection import get_connection
    return get_connection()


def query(sql, params=None, fetchall=True):
    """Execute a query and return results as list of dicts."""
    from psycopg2.extras import RealDictCursor
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())
            if fetchall:
                return cur.fetchall()
            return cur.fetchone()
    finally:
        conn.close()


def query_scalar(sql, params=None):
    """Execute a query and return a single value."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def section(text):
    print(f"\n--- {text} ---")


def ok(text):
    print(f"  [OK] {text}")


def warn(text):
    print(f"  [!!] {text}")


def info(text):
    print(f"  {text}")


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args):
    """System health check."""
    header("System Status")

    # Database
    section("PostgreSQL")
    try:
        count = query_scalar("SELECT COUNT(*) FROM messages")
        ok(f"Connected — {count} total messages")
    except Exception as e:
        warn(f"Connection failed: {e}")

    # Redis
    section("Redis")
    try:
        import redis
        r = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
        r.ping()
        ok("Connected")
    except Exception as e:
        warn(f"Connection failed: {e}")

    # Companions
    section("Companions")
    try:
        import yaml
        with open(PROJECT_ROOT / 'data' / 'companions.yaml') as f:
            companions = yaml.safe_load(f)
        for cid, cfg in companions.get('companions', {}).items():
            enabled = cfg.get('enabled', False)
            try:
                msg_count = query_scalar(
                    "SELECT COUNT(*) FROM messages WHERE companion_id = %s", (cid,)
                )
                fact_count = query_scalar(
                    "SELECT COUNT(*) FROM facts WHERE archived_at IS NULL AND companion_id = %s", (cid,)
                ) or 0
            except Exception:
                msg_count = '?'
                fact_count = '?'
            status = "enabled" if enabled else "disabled"
            info(f"  {cid}: {status} | {msg_count} messages | {fact_count} facts")
    except Exception as e:
        warn(f"Could not load companions: {e}")

    # Recent errors
    section("Recent Errors (24h)")
    try:
        from src.utils.error_tracker import get_error_summary
        summary = get_error_summary(days=1)
        if summary['total'] == 0:
            ok("No errors in the last 24 hours")
        else:
            warn(f"{summary['total']} errors in the last 24 hours")
            for module, count in sorted(summary['by_module'].items(), key=lambda x: -x[1])[:5]:
                info(f"    {module}: {count}")
    except Exception as e:
        info(f"  Error tracker unavailable: {e}")

    # Costs today
    section("Costs Today")
    try:
        from src.core.cost_tracker import get_cost_tracker
        tracker = get_cost_tracker()
        today = tracker.get_daily_summary()
        cost = today.get('total_cost', 0)
        calls = today.get('total_calls', 0)
        ok(f"${cost:.4f} across {calls} calls")
    except Exception as e:
        info(f"  Cost tracker unavailable: {e}")


# ---------------------------------------------------------------------------
# Subcommand: errors
# ---------------------------------------------------------------------------

def cmd_errors(args):
    """Show recent errors."""
    header(f"Errors{' for ' + args.companion if args.companion else ''}")

    try:
        from src.utils.error_tracker import get_recent_errors, get_error_summary

        # Summary first
        summary = get_error_summary(
            companion_id=args.companion,
            days=args.days if hasattr(args, 'days') else 7,
        )
        info(f"Total errors (last {getattr(args, 'days', 7)}d): {summary['total']}")
        if summary['by_module']:
            section("By Module")
            for module, count in sorted(summary['by_module'].items(), key=lambda x: -x[1]):
                info(f"  {module}: {count}")

        # Recent errors
        section(f"Recent Errors (limit {args.limit})")
        errors = get_recent_errors(
            companion_id=args.companion,
            module=args.module if hasattr(args, 'module') else None,
            limit=args.limit,
        )
        if not errors:
            ok("No errors found")
            return

        for err in errors:
            ts = err.get('timestamp', '?')
            if hasattr(ts, 'strftime'):
                ts = ts.strftime('%m-%d %H:%M')
            module = err.get('module', '?')
            etype = err.get('error_type', '?')
            msg = err.get('error_message', '')[:100]
            cid = err.get('companion_id', '')
            prefix = f"[{cid}] " if cid else ""
            print(f"  {ts} {prefix}{module}.{etype}: {msg}")

    except Exception as e:
        warn(f"Error tracker unavailable: {e}")
        info("Falling back to log files...")
        _show_log_errors(args.limit)


def _show_log_errors(limit):
    """Fallback: grep log files for errors."""
    log_dir = PROJECT_ROOT / 'logs'
    if not log_dir.exists():
        info("  No logs/ directory found")
        return
    import subprocess
    for log_file in sorted(log_dir.glob('*.log'), reverse=True)[:3]:
        try:
            result = subprocess.run(
                ['grep', '-i', 'error\\|critical', str(log_file)],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')[-limit:]
            for line in lines:
                if line.strip():
                    info(f"  {line.strip()[:120]}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Subcommand: costs
# ---------------------------------------------------------------------------

def cmd_costs(args):
    """Show API costs."""
    header(f"API Costs{' for ' + args.companion if args.companion else ''}")

    try:
        from src.core.cost_tracker import CostTracker

        cid = args.companion if args.companion else None
        tracker = CostTracker(companion_id=cid)

        # Today
        today = tracker.get_daily_summary()
        section("Today")
        info(f"  Cost: ${today.get('total_cost', 0):.4f}")
        info(f"  Calls: {today.get('total_calls', 0)}")
        info(f"  Tokens: {today.get('input_tokens', 0):,} in / {today.get('output_tokens', 0):,} out")

        # Weekly
        week = tracker.get_weekly_summary()
        section(f"Last 7 Days")
        info(f"  Cost: ${week.get('total_cost', 0):.4f}")
        info(f"  Calls: {week.get('total_calls', 0)}")

        # Daily breakdown
        section("Daily Breakdown")
        for day in week.get('daily', []):
            date = day.get('date', '?')
            cost = day.get('total_cost', 0)
            calls = day.get('total_calls', 0)
            if calls > 0:
                info(f"  {date}: ${cost:.4f} ({calls} calls)")

        # Recent calls by model
        section("Recent Calls by Model")
        recent = tracker.get_recent_calls(100)
        by_model = {}
        for call in recent:
            model = call.get('model', 'unknown')
            if '/' in model:
                model = model.split('/')[-1]
            entry = by_model.setdefault(model, {'cost': 0, 'calls': 0})
            entry['cost'] += call.get('cost_usd', 0)
            entry['calls'] += 1

        for model, data in sorted(by_model.items(), key=lambda x: -x[1]['cost']):
            info(f"  {model}: ${data['cost']:.4f} ({data['calls']} calls)")

    except Exception as e:
        warn(f"Cost tracker unavailable: {e}")


# ---------------------------------------------------------------------------
# Subcommand: companions
# ---------------------------------------------------------------------------

def cmd_companions(args):
    """List all companions with stats."""
    header("Companions")

    try:
        import yaml
        with open(PROJECT_ROOT / 'data' / 'companions.yaml') as f:
            companions = yaml.safe_load(f)

        for cid, cfg in companions.get('companions', {}).items():
            section(cid.upper())
            enabled = cfg.get('enabled', False)
            info(f"  Status: {'enabled' if enabled else 'disabled'}")
            info(f"  Config: {cfg.get('persona_config', 'N/A')}")

            try:
                msg_count = query_scalar(
                    "SELECT COUNT(*) FROM messages WHERE companion_id = %s", (cid,)
                )
                info(f"  Messages: {msg_count}")
            except Exception:
                info(f"  Messages: (unavailable)")

            try:
                fact_count = query_scalar(
                    "SELECT COUNT(*) FROM facts WHERE archived_at IS NULL AND companion_id = %s",
                    (cid,)
                ) or 0
                archived = query_scalar(
                    "SELECT COUNT(*) FROM facts WHERE archived_at IS NOT NULL AND companion_id = %s",
                    (cid,)
                ) or 0
                info(f"  Facts: {fact_count} active, {archived} archived")
            except Exception:
                info(f"  Facts: (unavailable)")

            try:
                opinion_count = query_scalar(
                    "SELECT COUNT(*) FROM opinions WHERE companion_id = %s", (cid,)
                ) or 0
                info(f"  Opinions: {opinion_count}")
            except Exception:
                pass

            try:
                from src.utils.error_tracker import get_error_summary
                summary = get_error_summary(companion_id=cid, days=1)
                if summary['total'] > 0:
                    warn(f"  Errors (24h): {summary['total']}")
                else:
                    ok(f"  Errors (24h): 0")
            except Exception:
                pass

    except Exception as e:
        warn(f"Could not load companions: {e}")


# ---------------------------------------------------------------------------
# Subcommand: companion <id>
# ---------------------------------------------------------------------------

def cmd_companion(args):
    """Deep status for a specific companion."""
    cid = args.id
    header(f"Companion: {cid}")

    # Messages
    section("Messages")
    try:
        total = query_scalar("SELECT COUNT(*) FROM messages WHERE companion_id = %s", (cid,))
        recent = query_scalar(
            "SELECT COUNT(*) FROM messages WHERE companion_id = %s AND created_at > NOW() - INTERVAL '24 hours'",
            (cid,)
        )
        info(f"  Total: {total}")
        info(f"  Last 24h: {recent}")
    except Exception as e:
        warn(f"  Messages unavailable: {e}")

    # Facts
    section("Facts")
    try:
        active = query_scalar(
            "SELECT COUNT(*) FROM facts WHERE companion_id = %s AND archived_at IS NULL", (cid,)
        ) or 0
        high_imp = query_scalar(
            "SELECT COUNT(*) FROM facts WHERE companion_id = %s AND archived_at IS NULL AND importance >= 7",
            (cid,)
        ) or 0
        info(f"  Active: {active} ({high_imp} high-importance)")
    except Exception as e:
        info(f"  Facts unavailable: {e}")

    # Opinions
    section("Opinions")
    try:
        opinions = query(
            "SELECT topic, confidence FROM opinions WHERE companion_id = %s ORDER BY updated_at DESC LIMIT 5",
            (cid,)
        )
        for op in opinions:
            info(f"  {op['topic']} (confidence: {op['confidence']:.2f})")
        if not opinions:
            info("  No opinions yet")
    except Exception:
        info("  Opinions unavailable")

    # Goals
    section("Active Goals")
    try:
        goals = query(
            "SELECT description, mode, status FROM companion_goals WHERE companion_id = %s AND status = 'active' ORDER BY created_at DESC LIMIT 5",
            (cid,)
        )
        for g in goals:
            info(f"  [{g['mode']}] {g['description'][:60]}")
        if not goals:
            info("  No active goals")
    except Exception:
        info("  Goals unavailable")

    # Curiosities
    section("Active Curiosities")
    try:
        curiosities = query(
            "SELECT topic, urgency FROM curiosity_threads WHERE companion_id = %s AND resolved = false ORDER BY urgency DESC LIMIT 5",
            (cid,)
        )
        for c in curiosities:
            info(f"  {c['topic']} (urgency: {c['urgency']:.2f})")
        if not curiosities:
            info("  No active curiosities")
    except Exception:
        info("  Curiosities unavailable")

    # Recent errors
    section("Recent Errors")
    try:
        from src.utils.error_tracker import get_error_summary
        summary = get_error_summary(companion_id=cid, days=1)
        if summary['total'] == 0:
            ok("No errors in the last 24 hours")
        else:
            warn(f"{summary['total']} errors in the last 24 hours")
            for module, count in sorted(summary['by_module'].items(), key=lambda x: -x[1])[:5]:
                info(f"    {module}: {count}")
    except Exception:
        info("  Error tracker unavailable")

    # Costs
    section("Costs Today")
    try:
        from src.core.cost_tracker import CostTracker
        tracker = CostTracker(companion_id=cid)
        today = tracker.get_daily_summary()
        info(f"  ${today.get('total_cost', 0):.4f} across {today.get('total_calls', 0)} calls")
    except Exception:
        info("  Cost tracker unavailable")


# ---------------------------------------------------------------------------
# Subcommand: memory <companion_id>
# ---------------------------------------------------------------------------

def cmd_memory(args):
    """Memory stats for a companion."""
    cid = args.id
    header(f"Memory: {cid}")

    # Facts
    section("Fact Store")
    try:
        active = query_scalar(
            "SELECT COUNT(*) FROM facts WHERE companion_id = %s AND archived_at IS NULL", (cid,)
        ) or 0
        archived = query_scalar(
            "SELECT COUNT(*) FROM facts WHERE companion_id = %s AND archived_at IS NOT NULL", (cid,)
        ) or 0
        avg_conf = query_scalar(
            "SELECT AVG(confidence) FROM facts WHERE companion_id = %s AND archived_at IS NULL", (cid,)
        ) or 0
        info(f"  Active: {active}")
        info(f"  Archived: {archived}")
        info(f"  Avg confidence: {avg_conf:.2f}")
    except Exception as e:
        info(f"  Facts unavailable: {e}")

    # Episodes
    section("Episodes")
    try:
        episode_count = query_scalar(
            "SELECT COUNT(*) FROM episodes WHERE companion_id = %s", (cid,)
        ) or 0
        info(f"  Total: {episode_count}")
    except Exception:
        info("  Episodes unavailable")

    # Observations
    section("Observations")
    try:
        obs_count = query_scalar(
            "SELECT COUNT(*) FROM observations WHERE companion_id = %s", (cid,)
        ) or 0
        info(f"  Total: {obs_count}")
    except Exception:
        info("  Observations unavailable")

    # Search
    if args.search:
        section(f"Search: '{args.search}'")
        try:
            results = query(
                """SELECT subject, predicate, object, confidence, importance
                   FROM facts
                   WHERE companion_id = %s AND archived_at IS NULL
                   AND (subject ILIKE %s OR object ILIKE %s OR predicate ILIKE %s)
                   ORDER BY importance DESC
                   LIMIT 10""",
                (cid, f'%{args.search}%', f'%{args.search}%', f'%{args.search}%')
            )
            for r in results:
                conf = r.get('confidence', 0)
                imp = r.get('importance', 0)
                info(f"  [{imp}] {r['subject']} {r['predicate']} {r['object'][:60]} (conf: {conf:.2f})")
            if not results:
                info("  No matching facts found")
        except Exception as e:
            warn(f"  Search failed: {e}")


# ---------------------------------------------------------------------------
# Subcommand: goals <companion_id>
# ---------------------------------------------------------------------------

def cmd_goals(args):
    """Show active goals for a companion."""
    cid = args.id
    header(f"Goals: {cid}")

    try:
        goals = query(
            """SELECT id, description, mode, status, created_at
               FROM companion_goals
               WHERE companion_id = %s
               ORDER BY
                   CASE status WHEN 'active' THEN 0 WHEN 'paused' THEN 1 ELSE 2 END,
                   created_at DESC
               LIMIT 10""",
            (cid,)
        )

        for g in goals:
            status_marker = {'active': '[*]', 'completed': '[v]', 'paused': '[-]'}.get(g['status'], '[?]')
            date = g['created_at']
            if hasattr(date, 'strftime'):
                date = date.strftime('%m-%d')
            print(f"  {status_marker} [{g['mode']}] {g['description'][:70]} ({date})")

            # Show steps
            try:
                steps = query(
                    """SELECT action_type, description, status
                       FROM goal_steps
                       WHERE goal_id = %s
                       ORDER BY step_order""",
                    (g['id'],)
                )
                for s in steps:
                    step_marker = {'completed': 'v', 'in_progress': '>', 'pending': ' ', 'blocked': 'x'}.get(s['status'], '?')
                    print(f"      [{step_marker}] {s['action_type']}: {s['description'][:55]}")
            except Exception:
                pass

        if not goals:
            info("No goals found")

    except Exception as e:
        warn(f"Goals unavailable: {e}")


# ---------------------------------------------------------------------------
# Subcommand: db
# ---------------------------------------------------------------------------

def cmd_db(args):
    """Database stats."""
    header("Database")

    section("Table Sizes")
    try:
        tables = query("""
            SELECT schemaname, relname as table_name,
                   n_live_tup as row_count,
                   pg_size_pretty(pg_total_relation_size(relid)) as total_size
            FROM pg_stat_user_tables
            ORDER BY n_live_tup DESC
            LIMIT 20
        """)
        for t in tables:
            info(f"  {t['table_name']}: {t['row_count']:,} rows ({t['total_size']})")
    except Exception as e:
        warn(f"Could not get table stats: {e}")

    section("Active Connections")
    try:
        conns = query_scalar("""
            SELECT COUNT(*) FROM pg_stat_activity
            WHERE datname = current_database() AND state = 'active'
        """)
        total = query_scalar("""
            SELECT COUNT(*) FROM pg_stat_activity
            WHERE datname = current_database()
        """)
        info(f"  Active: {conns}, Total: {total}")
    except Exception as e:
        warn(f"Could not get connection stats: {e}")

    section("Database Size")
    try:
        size = query_scalar("SELECT pg_size_pretty(pg_database_size(current_database()))")
        info(f"  {size}")
    except Exception as e:
        warn(f"Could not get database size: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Companion Framework Admin CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/admin.py status
    python scripts/admin.py errors --companion kai --limit 20
    python scripts/admin.py costs --companion kai
    python scripts/admin.py companion kai
    python scripts/admin.py memory kai --search "jesse"
    python scripts/admin.py goals kai
    python scripts/admin.py db
        """,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # status
    subparsers.add_parser('status', help='System health check')

    # errors
    p_errors = subparsers.add_parser('errors', help='Show recent errors')
    p_errors.add_argument('--companion', '-c', help='Filter by companion ID')
    p_errors.add_argument('--module', '-m', help='Filter by module')
    p_errors.add_argument('--limit', '-l', type=int, default=20, help='Max errors to show')
    p_errors.add_argument('--days', '-d', type=int, default=7, help='Days to look back')

    # costs
    p_costs = subparsers.add_parser('costs', help='Show API costs')
    p_costs.add_argument('--companion', '-c', help='Filter by companion ID')
    p_costs.add_argument('--days', '-d', type=int, default=7, help='Days to look back')

    # companions
    subparsers.add_parser('companions', help='List all companions with stats')

    # companion <id>
    p_comp = subparsers.add_parser('companion', help='Deep status for a specific companion')
    p_comp.add_argument('id', help='Companion ID')

    # memory <id>
    p_mem = subparsers.add_parser('memory', help='Memory stats for a companion')
    p_mem.add_argument('id', help='Companion ID')
    p_mem.add_argument('--search', '-s', help='Search facts by keyword')

    # goals <id>
    p_goals = subparsers.add_parser('goals', help='Show goals for a companion')
    p_goals.add_argument('id', help='Companion ID')

    # db
    subparsers.add_parser('db', help='Database stats')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        'status': cmd_status,
        'errors': cmd_errors,
        'costs': cmd_costs,
        'companions': cmd_companions,
        'companion': cmd_companion,
        'memory': cmd_memory,
        'goals': cmd_goals,
        'db': cmd_db,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
