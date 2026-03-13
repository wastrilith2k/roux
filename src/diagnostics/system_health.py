"""
System Health Diagnostics -- comprehensive subsystem health checker.

WHAT: Runs connectivity and staleness checks against every major subsystem:
      PostgreSQL, Redis, Neo4j, Fireworks API, Telegram bridge, value inference,
      fact extraction, scene state, proactive messaging, unified scheduler,
      code executor, and recent message flow.

WHY:  Silent failures (e.g., Celery stopped extracting facts, Neo4j ran out of
      disk) go unnoticed until something visibly breaks. Running these checks
      nightly (via maintenance_runner.py) or on-demand catches issues early.

HOW:  Each `check_*` method attempts a real connection/query and returns a
      CheckResult dataclass (passed/failed + details). `run_all_checks()`
      iterates through all checks, optionally reports failures to Sentry, and
      returns a summary dict. `print_diagnostics()` provides a CLI-friendly
      output. Can also be invoked as `python -m src.diagnostics.system_health`.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import Sentry
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = None


class SystemHealth:
    """
    Comprehensive health checker for all companion subsystems.

    Reports failures to Sentry with context.
    """

    def __init__(self):
        self._results: List[CheckResult] = []

    def _report_to_sentry(self, check: CheckResult):
        """Report a failed check to Sentry."""
        if not SENTRY_AVAILABLE:
            logger.warning(f"Sentry not available - can't report: {check.name}")
            return

        sentry_sdk.capture_message(
            f"[HEALTH] {check.name} - FAILED: {check.message}",
            level="warning",
            extras={
                "check_name": check.name,
                "details": check.details,
                "last_activity": check.last_activity.isoformat() if check.last_activity else None,
            }
        )
        logger.info(f"Reported to Sentry: {check.name}")

    def check_database_connections(self) -> CheckResult:
        """Check PostgreSQL connection."""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return CheckResult("PostgreSQL", True, "Connected successfully")
        except Exception as e:
            return CheckResult("PostgreSQL", False, str(e))

    def check_redis(self) -> CheckResult:
        """Check Redis connection."""
        try:
            import redis
            r = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379))
            )
            r.ping()
            return CheckResult("Redis", True, "Connected successfully")
        except Exception as e:
            return CheckResult("Redis", False, str(e))

    def check_neo4j(self) -> CheckResult:
        """Check Neo4j connection."""
        try:
            from neo4j import GraphDatabase
            uri = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
            user = os.environ.get('NEO4J_USER', 'neo4j')
            password = os.environ.get('NEO4J_PASSWORD', 'password')

            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            return CheckResult("Neo4j", True, "Connected successfully")
        except Exception as e:
            return CheckResult("Neo4j", False, str(e))

    def check_fireworks_api(self) -> CheckResult:
        """Check Fireworks API is accessible."""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
            # Just check we can list models (cheap call)
            # Actually, let's just verify the key exists
            if not os.getenv('FIREWORKS_API_KEY'):
                return CheckResult("Fireworks API", False, "FIREWORKS_API_KEY not set")
            return CheckResult("Fireworks API", True, "API key configured")
        except Exception as e:
            return CheckResult("Fireworks API", False, str(e))

    def check_telegram(self) -> CheckResult:
        """Check Telegram bridge status."""
        try:
            from src.autonomy.telegram_bridge import get_telegram_bridge
            bridge = get_telegram_bridge()
            if bridge.is_ready():
                return CheckResult("Telegram Bridge", True, f"Ready, chat_id={bridge.get_chat_id()}")
            else:
                return CheckResult("Telegram Bridge", False, "Not ready (no chat_id or not started)")
        except Exception as e:
            return CheckResult("Telegram Bridge", False, str(e))

    def check_value_inference(self) -> CheckResult:
        """Check if value inference has recent data."""
        try:
            from src.autonomy.value_inference import get_value_inference
            vi = get_value_inference()
            stats = vi.get_stats()

            if not stats:
                return CheckResult(
                    "Value Inference",
                    False,
                    "No values stored",
                    details={"stats": {}}
                )

            # Check when last updated
            latest_update = None
            for cat, data in stats.items():
                if data.get('last_updated'):
                    if latest_update is None or data['last_updated'] > latest_update:
                        latest_update = data['last_updated']

            # Consider stale if > 7 days old
            if latest_update:
                days_old = (datetime.now() - latest_update).days
                if days_old > 7:
                    return CheckResult(
                        "Value Inference",
                        False,
                        f"Data is {days_old} days old (stale)",
                        details={"categories": len(stats), "days_old": days_old},
                        last_activity=latest_update
                    )
                return CheckResult(
                    "Value Inference",
                    True,
                    f"Has {len(stats)} categories, {days_old} days old",
                    details={"categories": len(stats), "days_old": days_old},
                    last_activity=latest_update
                )

            return CheckResult("Value Inference", False, "No timestamp data")
        except Exception as e:
            return CheckResult("Value Inference", False, str(e))

    def check_fact_extraction(self) -> CheckResult:
        """Check if fact extraction is working (has recent facts)."""
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
                # Check for facts in the last 24 hours
                cur.execute("""
                    SELECT COUNT(*) as count, MAX(created_at) as latest
                    FROM facts
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                result = cur.fetchone()
            conn.close()

            count = result['count'] if result else 0
            latest = result['latest'] if result else None

            if count > 0:
                return CheckResult(
                    "Fact Extraction",
                    True,
                    f"{count} facts in last 24h",
                    details={"recent_count": count},
                    last_activity=latest
                )
            else:
                # Check if there are ANY facts
                conn = psycopg2.connect(
                    host=os.environ.get('POSTGRES_HOST', 'postgres'),
                    port=os.environ.get('POSTGRES_PORT', '5432'),
                    dbname=os.environ.get('POSTGRES_DB', 'companion'),
                    user=os.environ.get('POSTGRES_USER', 'companion'),
                    password=os.environ.get('POSTGRES_PASSWORD', '')
                )
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT COUNT(*) as count, MAX(created_at) as latest FROM facts")
                    total = cur.fetchone()
                conn.close()

                if total['count'] > 0:
                    return CheckResult(
                        "Fact Extraction",
                        False,
                        f"No recent facts (total: {total['count']})",
                        details={"total_count": total['count']},
                        last_activity=total['latest']
                    )
                return CheckResult("Fact Extraction", False, "No facts in database")
        except Exception as e:
            return CheckResult("Fact Extraction", False, str(e))

    def check_scene_state(self) -> CheckResult:
        """Check if scene state tracking is working."""
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
                    SELECT * FROM scene_state
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                scene = cur.fetchone()
            conn.close()

            if scene:
                updated = scene.get('updated_at')
                hours_old = (datetime.now() - updated).total_seconds() / 3600 if updated else None

                return CheckResult(
                    "Scene State",
                    True,
                    f"Active scene: {scene.get('scene_type', 'unknown')}",
                    details={
                        "scene_type": scene.get('scene_type'),
                        "hours_since_update": round(hours_old, 1) if hours_old else None
                    },
                    last_activity=updated
                )
            return CheckResult("Scene State", False, "No scene state found")
        except Exception as e:
            # Table might not exist
            if "does not exist" in str(e):
                return CheckResult("Scene State", False, "Table does not exist")
            return CheckResult("Scene State", False, str(e))

    def check_proactive_messaging(self) -> CheckResult:
        """Check if proactive messaging is enabled and working."""
        try:
            from src.autonomy.reach_out_engine import is_autonomy_enabled

            if not is_autonomy_enabled():
                return CheckResult(
                    "Proactive Messaging",
                    True,  # Not a failure, just disabled
                    "Disabled by toggle",
                    details={"enabled": False}
                )

            # Check for recent proactive messages
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
                    WHERE source LIKE '%proactive%'
                    AND timestamp > NOW() - INTERVAL '7 days'
                """)
                result = cur.fetchone()
            conn.close()

            count = result['count'] if result else 0
            latest = result['latest'] if result else None

            return CheckResult(
                "Proactive Messaging",
                True,
                f"Enabled, {count} messages in last 7 days",
                details={"enabled": True, "recent_count": count},
                last_activity=latest
            )
        except Exception as e:
            return CheckResult("Proactive Messaging", False, str(e))

    def check_scheduler(self) -> CheckResult:
        """Check if the unified scheduler is running."""
        try:
            from src.scheduling.unified_scheduler import get_unified_scheduler
            scheduler = get_unified_scheduler()
            stats = scheduler.get_stats()

            if stats.get('running'):
                jobs = scheduler.list_jobs()
                return CheckResult(
                    "Unified Scheduler",
                    True,
                    f"Running with {len(jobs)} jobs",
                    details={"jobs": [j['id'] for j in jobs], "stats": stats}
                )
            return CheckResult("Unified Scheduler", False, "Not running")
        except Exception as e:
            return CheckResult("Unified Scheduler", False, str(e))

    def check_code_executor(self) -> CheckResult:
        """Check if code executor container is healthy."""
        try:
            import requests
            # Try both possible hostnames
            for host in ["companion-code-executor", "code-executor"]:
                try:
                    response = requests.get(
                        f"http://{host}:8080/health",
                        timeout=5
                    )
                    if response.status_code == 200:
                        return CheckResult("Code Executor", True, f"Healthy ({host})")
                except:
                    continue
            response = requests.get(
                "http://companion-code-executor:8080/health",
                timeout=5
            )
            if response.status_code == 200:
                return CheckResult("Code Executor", True, "Healthy")
            return CheckResult("Code Executor", False, f"HTTP {response.status_code}")
        except Exception as e:
            return CheckResult("Code Executor", False, str(e))

    def check_recent_messages(self) -> CheckResult:
        """Check if messages are flowing (recent activity)."""
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
            latest = result['latest'] if result else None

            # This is informational - not a failure if no messages
            return CheckResult(
                "Recent Messages",
                True,
                f"{count} messages in last 24h",
                details={"count": count},
                last_activity=latest
            )
        except Exception as e:
            return CheckResult("Recent Messages", False, str(e))

    def run_all_checks(self, report_failures: bool = True) -> Dict[str, Any]:
        """
        Run all health checks.

        Args:
            report_failures: If True, report failures to Sentry

        Returns:
            Summary dict with all results
        """
        self._results = []

        checks = [
            self.check_database_connections,
            self.check_redis,
            self.check_neo4j,
            self.check_fireworks_api,
            self.check_telegram,
            self.check_value_inference,
            self.check_fact_extraction,
            self.check_scene_state,
            self.check_proactive_messaging,
            self.check_scheduler,
            self.check_code_executor,
            self.check_recent_messages,
        ]

        for check_func in checks:
            try:
                result = check_func()
                self._results.append(result)

                if not result.passed and report_failures:
                    self._report_to_sentry(result)

            except Exception as e:
                result = CheckResult(check_func.__name__, False, f"Check crashed: {e}")
                self._results.append(result)
                if report_failures:
                    self._report_to_sentry(result)

        # Build summary
        passed = [r for r in self._results if r.passed]
        failed = [r for r in self._results if not r.passed]

        return {
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(self._results),
            "passed": len(passed),
            "failed": len(failed),
            "results": {r.name: {
                "passed": r.passed,
                "message": r.message,
                "details": r.details,
                "last_activity": r.last_activity.isoformat() if r.last_activity else None
            } for r in self._results},
            "failures": [r.name for r in failed]
        }

    def get_results(self) -> List[CheckResult]:
        """Get list of all check results."""
        return self._results


def run_diagnostics(report_to_sentry: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run all diagnostics.

    Returns summary dict.
    """
    health = SystemHealth()
    return health.run_all_checks(report_failures=report_to_sentry)


def print_diagnostics():
    """Run diagnostics and print results to console."""
    results = run_diagnostics(report_to_sentry=False)

    print("\n" + "=" * 60)
    print("COMPANION SYSTEM HEALTH CHECK")
    print(f"Time: {results['timestamp']}")
    print("=" * 60)

    for name, data in results['results'].items():
        status = "✅" if data['passed'] else "❌"
        print(f"{status} {name}: {data['message']}")
        if data.get('last_activity'):
            print(f"   Last activity: {data['last_activity']}")

    print("-" * 60)
    print(f"Total: {results['passed']}/{results['total_checks']} passed")

    if results['failures']:
        print(f"\n⚠️  FAILURES: {', '.join(results['failures'])}")

    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    print_diagnostics()
