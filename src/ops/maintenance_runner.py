"""
Maintenance Runner -- automated health check with backup and notification.

WHAT: Runs SystemHealth diagnostics across all subsystems (Postgres, Redis,
      Neo4j, Fireworks, Telegram, scheduler, fact extraction, etc.). If any
      check fails, triggers a full backup via shell script. Sends a formatted
      summary to the ops Telegram bot regardless of outcome.

WHY:  Nightly (or on-demand) health checks catch silent failures early --
      e.g., Celery stopped extracting facts, Redis went down, Neo4j disk full.
      The automatic backup on failure ensures we have a restore point.

HOW:  CLI entry point via argparse. Imports SystemHealth from diagnostics,
      runs all checks, optionally calls /root/companion/scripts/full-backup.sh,
      and posts results to Telegram via OpsTelegramBot.
"""

import os
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def run_backup() -> Dict[str, Any]:
    """
    Run full system backup.

    Returns:
        Dict with backup_taken, backup_path, error
    """
    try:
        result = subprocess.run(
            ["/root/companion/scripts/full-backup.sh"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            # Last line of output is the backup path
            lines = result.stdout.strip().split('\n')
            backup_path = lines[-1] if lines else "unknown"
            return {
                "backup_taken": True,
                "backup_path": backup_path,
                "error": None
            }
        else:
            return {
                "backup_taken": False,
                "backup_path": None,
                "error": result.stderr or "Backup failed"
            }
    except subprocess.TimeoutExpired:
        return {
            "backup_taken": False,
            "backup_path": None,
            "error": "Backup timed out"
        }
    except Exception as e:
        return {
            "backup_taken": False,
            "backup_path": None,
            "error": str(e)
        }


def run_maintenance(
    send_notification: bool = True,
    backup_on_issues: bool = True,
    report_to_sentry: bool = True
) -> Dict[str, Any]:
    """
    Run full maintenance check.

    Args:
        send_notification: Send summary to ops Telegram
        backup_on_issues: Take backup if any issues found
        report_to_sentry: Report failures to Sentry

    Returns:
        Full summary dict
    """
    timestamp = datetime.now().isoformat()
    summary = {
        "timestamp": timestamp,
        "total_checks": 0,
        "passed": 0,
        "failed": 0,
        "results": {},
        "failures": [],
        "backup_taken": False,
        "backup_path": None,
        "fixes_made": None,
        "commit_made": False,
    }

    # Run diagnostics
    print(f"[{timestamp}] Running system diagnostics...")

    try:
        from src.diagnostics.system_health import SystemHealth

        health = SystemHealth()
        diag_results = health.run_all_checks(report_failures=report_to_sentry)

        summary["total_checks"] = diag_results.get("total_checks", 0)
        summary["passed"] = diag_results.get("passed", 0)
        summary["failed"] = diag_results.get("failed", 0)
        summary["results"] = diag_results.get("results", {})
        summary["failures"] = diag_results.get("failures", [])

    except ImportError as e:
        print(f"Error: Could not import diagnostics: {e}")
        summary["results"]["Diagnostics Import"] = {
            "passed": False,
            "message": str(e)
        }
        summary["failed"] = 1
        summary["total_checks"] = 1
        summary["failures"] = ["Diagnostics Import"]

    except Exception as e:
        print(f"Error running diagnostics: {e}")
        summary["results"]["Diagnostics"] = {
            "passed": False,
            "message": str(e)
        }
        summary["failed"] = 1
        summary["total_checks"] = 1
        summary["failures"] = ["Diagnostics"]

    # If issues found, take backup FIRST
    if summary["failed"] > 0 and backup_on_issues:
        print(f"Issues found ({summary['failed']}), taking backup...")
        backup_result = run_backup()
        summary["backup_taken"] = backup_result["backup_taken"]
        summary["backup_path"] = backup_result["backup_path"]
        if backup_result["error"]:
            print(f"Backup error: {backup_result['error']}")

    # Print summary to console
    print("\n" + "=" * 50)
    print("MAINTENANCE SUMMARY")
    print("=" * 50)
    for name, data in summary["results"].items():
        status = "✅" if data.get("passed") else "❌"
        print(f"{status} {name}: {data.get('message', '')}")
    print("-" * 50)
    print(f"Total: {summary['passed']}/{summary['total_checks']} passed")
    if summary["backup_taken"]:
        print(f"Backup: {summary['backup_path']}")
    print("=" * 50 + "\n")

    # Send notification
    if send_notification:
        try:
            from src.ops.telegram_ops_bot import send_maintenance_report
            send_maintenance_report(summary)
            print("Notification sent to ops Telegram")
        except Exception as e:
            print(f"Could not send notification: {e}")

    return summary


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Companion Maintenance Runner")
    parser.add_argument("--no-notify", action="store_true",
                        help="Don't send Telegram notification")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't take backup even if issues found")
    parser.add_argument("--no-sentry", action="store_true",
                        help="Don't report to Sentry")

    args = parser.parse_args()

    result = run_maintenance(
        send_notification=not args.no_notify,
        backup_on_issues=not args.no_backup,
        report_to_sentry=not args.no_sentry
    )

    # Exit with error code if issues found
    if result["failed"] > 0:
        exit(1)
    exit(0)


if __name__ == "__main__":
    main()
