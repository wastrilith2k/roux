"""
Unified Cost Tracking Service -- per-service spend tracking and budget alerts.

WHAT: SQLite-backed tracker for API costs across six services: Fireworks,
      OpenAI, RunComfy, Twilio, Hedra, and Google. Records each API call with
      token counts and dollar amounts, aggregates daily/monthly summaries,
      projects month-end spend, and generates budget-threshold alerts.

WHY:  The companion makes hundreds of API calls per day across multiple
      providers. Without tracking, a runaway loop or misconfigured prompt
      could blow through the monthly budget in hours. The cost dashboard and
      alerts make overspend visible immediately.

HOW:  SQLite database (separate from main Postgres) with one row per API call.
      `update_*` methods record calls with pre-calculated costs using the
      pricing constants below. `get_cost_summary()` aggregates by service
      and compares against per-service budgets. `get_historical_data()` powers
      the admin chart. CSV export via cost_routes.py.

Singleton: `get_cost_tracker()` at module bottom.
"""
import sqlite3
import os
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import calendar

from src.database import tables as T

# Cost constants (per service pricing as of October 2025)
FIREWORKS_INPUT_COST_PER_1M = 0.90  # $0.90 per 1M input tokens
FIREWORKS_OUTPUT_COST_PER_1M = 0.90  # $0.90 per 1M output tokens

OPENAI_GPT4O_MINI_INPUT_COST_PER_1M = 0.15  # $0.15 per 1M input tokens
OPENAI_GPT4O_MINI_OUTPUT_COST_PER_1M = 0.60  # $0.60 per 1M output tokens
OPENAI_REALTIME_AUDIO_INPUT_COST_PER_MIN = 0.06  # $0.06 per minute
OPENAI_REALTIME_AUDIO_OUTPUT_COST_PER_MIN = 0.24  # $0.24 per minute
OPENAI_TTS_COST_PER_1M_CHARS = 15.00  # $15 per 1M characters

HEDRA_COST_PER_MINUTE = 0.05  # $0.05 per minute of video

RUNCOMFY_PERSONAL_WORKFLOW_COST = 0.10  # $0.10 per image (companion portraits)
RUNCOMFY_GENERAL_WORKFLOW_COST = 0.08  # $0.08 per image (general scenes)

TWILIO_SMS_OUTBOUND_COST = 0.10  # $0.10 per outbound SMS
TWILIO_PHONE_RENTAL_MONTHLY = 1.00  # $1.00 per month for phone number

GOOGLE_API_COST = 0.00  # Currently free for personal use

ZEP_COST_PER_CREDIT = 0.00125  # $0.00125 per credit ($25/20k credits on Flex plan)
ZEP_EPISODE_BASE_SIZE = 350  # Base episode size in bytes (1 credit)

@dataclass
class ServiceCost:
    """Individual service cost breakdown"""
    service_name: str
    display_name: str
    icon: str
    cost_today: float
    cost_month: float
    budget_month: float
    usage_details: Dict[str, Any]
    links: Dict[str, str]

    @property
    def percentage_used(self) -> float:
        """Calculate percentage of budget used"""
        if self.budget_month == 0:
            return 0.0
        return (self.cost_month / self.budget_month) * 100

    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget"""
        return max(0, self.budget_month - self.cost_month)

@dataclass
class CostSummary:
    """Complete cost summary across all services"""
    total_today: float
    total_month: float
    total_budget: float
    projected_month_end: float
    services: List[ServiceCost]
    alerts: List[Dict[str, Any]]

    @property
    def percentage_used(self) -> float:
        """Calculate percentage of total budget used"""
        if self.total_budget == 0:
            return 0.0
        return (self.total_month / self.total_budget) * 100

    @property
    def days_in_month(self) -> int:
        """Get number of days in current month"""
        today = date.today()
        return calendar.monthrange(today.year, today.month)[1]

    @property
    def day_of_month(self) -> int:
        """Get current day of month"""
        return date.today().day


class CostTracker:
    """Central cost tracking service"""

    def __init__(self, db_path: str = None):
        # Auto-detect path based on environment
        if db_path is None:
            if os.path.exists('/app/data'):
                # Running in Docker
                db_path = '/app/data/companion.db'
            else:
                # Running on host - use relative path from this file
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                db_path = os.path.join(project_root, 'data', 'companion.db')

        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path, timeout=10.0)  # 10 second timeout for concurrent access
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        # Enable write-ahead logging for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def _init_database(self):
        """Initialize database tables from schema file"""
        # Use relative path from this file
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(src_dir, 'database', 'cost_tracking_schema.sql')

        if not os.path.exists(schema_path):
            print(f"⚠️  Cost tracking schema not found at {schema_path}")
            return

        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.executescript(schema_sql)
            conn.commit()
            conn.close()
            print("✅ Cost tracking database initialized")
        except Exception as e:
            print(f"❌ Failed to initialize cost tracking database: {e}")

    # ========================================================================
    # TRACKING METHODS (called when API calls are made)
    # ========================================================================

    def track_fireworks_call(self, user_id: str, prompt_tokens: int,
                            completion_tokens: int, model: str = 'llama-v3p1-70b-instruct',
                            response_time_ms: int = None, error: bool = False) -> float:
        """Track a Fireworks.ai API call and return cost"""
        cost = self._calculate_fireworks_cost(prompt_tokens, completion_tokens)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {T.FIREWORKS_USAGE}
            (user_id, model, prompt_tokens, completion_tokens, total_tokens, cost_usd, response_time_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, model, prompt_tokens, completion_tokens,
              prompt_tokens + completion_tokens, cost, response_time_ms, int(error)))
        conn.commit()
        conn.close()

        if not error:
            self._update_daily_summary(user_id, 'fireworks', cost, calls=1,
                                      tokens_in=prompt_tokens, tokens_out=completion_tokens)

        return cost

    def track_openai_call(self, user_id: str, prompt_tokens: int = 0,
                         completion_tokens: int = 0, audio_seconds: float = 0,
                         characters: int = 0, service_type: str = 'tool_detection',
                         model: str = 'gpt-4o-mini', error: bool = False) -> float:
        """Track an OpenAI API call and return cost"""
        cost = self._calculate_openai_cost(prompt_tokens, completion_tokens,
                                           audio_seconds, characters, service_type)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {T.OPENAI_USAGE}
            (user_id, model, service_type, prompt_tokens, completion_tokens,
             audio_seconds, characters, cost_usd, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, model, service_type, prompt_tokens, completion_tokens,
              audio_seconds, characters, cost, int(error)))
        conn.commit()
        conn.close()

        if not error:
            self._update_daily_summary(user_id, 'openai', cost, calls=1,
                                      tokens_in=prompt_tokens, tokens_out=completion_tokens,
                                      voice_minutes=audio_seconds / 60.0)

        return cost

    def track_runcomfy_generation(self, user_id: str, workflow_id: str,
                                  workflow_type: str, status: str,
                                  prompt: str = None, image_url: str = None,
                                  generation_time_seconds: int = None,
                                  error_message: str = None) -> float:
        """Track a RunComfy image generation and return cost"""
        cost = self._calculate_runcomfy_cost(workflow_type) if status == 'success' else 0.0

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {T.RUNCOMFY_USAGE}
            (user_id, workflow_id, workflow_type, prompt, image_url,
             generation_time_seconds, cost_usd, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, workflow_id, workflow_type, prompt, image_url,
              generation_time_seconds, cost, status, error_message))
        conn.commit()
        conn.close()

        if status == 'success':
            self._update_daily_summary(user_id, 'runcomfy', cost, images=1)

        return cost

    def track_twilio_sms(self, user_id: str, message_sid: str, direction: str,
                        to_number: str = None, from_number: str = None,
                        message_body: str = None, status: str = 'sent',
                        error_code: str = None) -> float:
        """Track a Twilio SMS and return cost"""
        cost = TWILIO_SMS_OUTBOUND_COST if direction == 'outbound' else 0.0

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {T.TWILIO_USAGE}
            (user_id, message_sid, direction, to_number, from_number,
             message_body, status, cost_usd, error_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, message_sid, direction, to_number, from_number,
              message_body, status, cost, error_code))
        conn.commit()
        conn.close()

        if direction == 'outbound':
            self._update_daily_summary(user_id, 'twilio', cost, sms_sent=1)
        else:
            self._update_daily_summary(user_id, 'twilio', 0, sms_received=1)

        return cost

    def track_hedra_generation(self, user_id: str, job_id: str, duration_seconds: int,
                              status: str, message_id: str = None,
                              audio_input_path: str = None, portrait_image_path: str = None,
                              video_output_path: str = None, error_message: str = None) -> float:
        """Track a Hedra avatar video generation and return cost"""
        cost = self._calculate_hedra_cost(duration_seconds) if status == 'completed' else 0.0

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {T.HEDRA_USAGE}
            (user_id, job_id, message_id, audio_input_path, portrait_image_path,
             video_output_path, duration_seconds, cost_usd, status, error_message,
             completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, job_id, message_id, audio_input_path, portrait_image_path,
              video_output_path, duration_seconds, cost, status, error_message,
              datetime.now() if status == 'completed' else None))
        conn.commit()
        conn.close()

        if status == 'completed':
            self._update_daily_summary(user_id, 'hedra', cost, videos=1,
                                      hedra_minutes=duration_seconds / 60.0)

        return cost

    def track_google_api_call(self, user_id: str, service: str, endpoint: str = None,
                             method: str = 'GET', quota_cost: int = 1,
                             response_time_ms: int = None, error: bool = False):
        """Track a Google API call (currently free, but track for quota monitoring)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {T.GOOGLE_API_USAGE}
            (user_id, service, endpoint, method, quota_cost, response_time_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, service, endpoint, method, quota_cost, response_time_ms, int(error)))
        conn.commit()
        conn.close()

        if not error:
            self._update_daily_summary(user_id, 'google', 0.0, google_calls=1,
                                      google_quota=quota_cost)

    # ========================================================================
    # COST CALCULATION HELPERS
    # ========================================================================

    def _calculate_fireworks_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate Fireworks.ai cost"""
        input_cost = (prompt_tokens / 1_000_000) * FIREWORKS_INPUT_COST_PER_1M
        output_cost = (completion_tokens / 1_000_000) * FIREWORKS_OUTPUT_COST_PER_1M
        return input_cost + output_cost

    def _calculate_openai_cost(self, prompt_tokens: int, completion_tokens: int,
                              audio_seconds: float, characters: int, service_type: str) -> float:
        """Calculate OpenAI cost based on service type"""
        if service_type == 'realtime_voice':
            # Realtime Voice API charges per minute for input and output
            minutes = audio_seconds / 60.0
            return (minutes * OPENAI_REALTIME_AUDIO_INPUT_COST_PER_MIN) + \
                   (minutes * OPENAI_REALTIME_AUDIO_OUTPUT_COST_PER_MIN)
        elif service_type == 'tts':
            # TTS charges per character
            return (characters / 1_000_000) * OPENAI_TTS_COST_PER_1M_CHARS
        else:
            # Text models (chat, tool_detection) charge per token
            input_cost = (prompt_tokens / 1_000_000) * OPENAI_GPT4O_MINI_INPUT_COST_PER_1M
            output_cost = (completion_tokens / 1_000_000) * OPENAI_GPT4O_MINI_OUTPUT_COST_PER_1M
            return input_cost + output_cost

    def _calculate_runcomfy_cost(self, workflow_type: str) -> float:
        """Calculate RunComfy cost"""
        if workflow_type == 'personal':
            return RUNCOMFY_PERSONAL_WORKFLOW_COST
        elif workflow_type == 'general':
            return RUNCOMFY_GENERAL_WORKFLOW_COST
        else:
            return 0.10  # Default

    def _calculate_hedra_cost(self, duration_seconds: int) -> float:
        """Calculate Hedra cost"""
        minutes = duration_seconds / 60.0
        return minutes * HEDRA_COST_PER_MINUTE

    # ========================================================================
    # DAILY SUMMARY AGGREGATION
    # ========================================================================

    def _update_daily_summary(self, user_id: str, service: str, cost: float, **kwargs):
        """Update or create daily summary entry"""
        today = date.today().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if entry exists for today
        cursor.execute(f"""
            SELECT id FROM {T.DAILY_COST_SUMMARY}
            WHERE date = ? AND user_id = ?
        """, (today, user_id))

        row = cursor.fetchone()

        if row:
            # Update existing entry
            update_parts = []
            update_values = []

            if service == 'fireworks':
                update_parts.append("fireworks_calls = fireworks_calls + ?")
                update_values.append(kwargs.get('calls', 0))
                update_parts.append("fireworks_tokens_in = fireworks_tokens_in + ?")
                update_values.append(kwargs.get('tokens_in', 0))
                update_parts.append("fireworks_tokens_out = fireworks_tokens_out + ?")
                update_values.append(kwargs.get('tokens_out', 0))
                update_parts.append("fireworks_cost_usd = fireworks_cost_usd + ?")
                update_values.append(cost)

            elif service == 'openai':
                update_parts.append("openai_calls = openai_calls + ?")
                update_values.append(kwargs.get('calls', 0))
                update_parts.append("openai_tokens_in = openai_tokens_in + ?")
                update_values.append(kwargs.get('tokens_in', 0))
                update_parts.append("openai_tokens_out = openai_tokens_out + ?")
                update_values.append(kwargs.get('tokens_out', 0))
                update_parts.append("openai_voice_minutes = openai_voice_minutes + ?")
                update_values.append(kwargs.get('voice_minutes', 0))
                update_parts.append("openai_cost_usd = openai_cost_usd + ?")
                update_values.append(cost)

            elif service == 'hedra':
                update_parts.append("hedra_videos = hedra_videos + ?")
                update_values.append(kwargs.get('videos', 0))
                update_parts.append("hedra_minutes = hedra_minutes + ?")
                update_values.append(kwargs.get('hedra_minutes', 0))
                update_parts.append("hedra_cost_usd = hedra_cost_usd + ?")
                update_values.append(cost)

            elif service == 'runcomfy':
                update_parts.append("runcomfy_images = runcomfy_images + ?")
                update_values.append(kwargs.get('images', 0))
                update_parts.append("runcomfy_cost_usd = runcomfy_cost_usd + ?")
                update_values.append(cost)

            elif service == 'twilio':
                update_parts.append("twilio_sms_sent = twilio_sms_sent + ?")
                update_values.append(kwargs.get('sms_sent', 0))
                update_parts.append("twilio_sms_received = twilio_sms_received + ?")
                update_values.append(kwargs.get('sms_received', 0))
                update_parts.append("twilio_cost_usd = twilio_cost_usd + ?")
                update_values.append(cost)

            elif service == 'google':
                update_parts.append("google_api_calls = google_api_calls + ?")
                update_values.append(kwargs.get('google_calls', 0))
                update_parts.append("google_quota_used = google_quota_used + ?")
                update_values.append(kwargs.get('google_quota', 0))
                update_parts.append("google_cost_usd = google_cost_usd + ?")
                update_values.append(cost)

            # Always update total and timestamp
            update_parts.append("total_cost_usd = total_cost_usd + ?")
            update_values.append(cost)
            update_parts.append("updated_at = ?")
            update_values.append(datetime.now().isoformat())

            update_values.extend([today, user_id])

            sql = f"""
                UPDATE {T.DAILY_COST_SUMMARY}
                SET {', '.join(update_parts)}
                WHERE date = ? AND user_id = ?
            """
            cursor.execute(sql, update_values)

        else:
            # Create new entry
            insert_data = {
                'date': today,
                'user_id': user_id,
                'fireworks_calls': 0, 'fireworks_tokens_in': 0, 'fireworks_tokens_out': 0, 'fireworks_cost_usd': 0,
                'openai_calls': 0, 'openai_tokens_in': 0, 'openai_tokens_out': 0, 'openai_voice_minutes': 0, 'openai_cost_usd': 0,
                'hedra_videos': 0, 'hedra_minutes': 0, 'hedra_cost_usd': 0,
                'runcomfy_images': 0, 'runcomfy_cost_usd': 0,
                'twilio_sms_sent': 0, 'twilio_sms_received': 0, 'twilio_cost_usd': 0,
                'google_api_calls': 0, 'google_quota_used': 0, 'google_cost_usd': 0,
                'total_cost_usd': 0
            }

            if service == 'fireworks':
                insert_data['fireworks_calls'] = kwargs.get('calls', 0)
                insert_data['fireworks_tokens_in'] = kwargs.get('tokens_in', 0)
                insert_data['fireworks_tokens_out'] = kwargs.get('tokens_out', 0)
                insert_data['fireworks_cost_usd'] = cost
            elif service == 'openai':
                insert_data['openai_calls'] = kwargs.get('calls', 0)
                insert_data['openai_tokens_in'] = kwargs.get('tokens_in', 0)
                insert_data['openai_tokens_out'] = kwargs.get('tokens_out', 0)
                insert_data['openai_voice_minutes'] = kwargs.get('voice_minutes', 0)
                insert_data['openai_cost_usd'] = cost
            elif service == 'hedra':
                insert_data['hedra_videos'] = kwargs.get('videos', 0)
                insert_data['hedra_minutes'] = kwargs.get('hedra_minutes', 0)
                insert_data['hedra_cost_usd'] = cost
            elif service == 'runcomfy':
                insert_data['runcomfy_images'] = kwargs.get('images', 0)
                insert_data['runcomfy_cost_usd'] = cost
            elif service == 'twilio':
                insert_data['twilio_sms_sent'] = kwargs.get('sms_sent', 0)
                insert_data['twilio_sms_received'] = kwargs.get('sms_received', 0)
                insert_data['twilio_cost_usd'] = cost
            elif service == 'google':
                insert_data['google_api_calls'] = kwargs.get('google_calls', 0)
                insert_data['google_quota_used'] = kwargs.get('google_quota', 0)
                insert_data['google_cost_usd'] = cost

            insert_data['total_cost_usd'] = cost

            columns = ', '.join(insert_data.keys())
            placeholders = ', '.join(['?' for _ in insert_data])

            cursor.execute(f"""
                INSERT INTO {T.DAILY_COST_SUMMARY} ({columns})
                VALUES ({placeholders})
            """, list(insert_data.values()))

        conn.commit()
        conn.close()

    # ========================================================================
    # QUERY METHODS (for dashboard)
    # ========================================================================

    def get_cost_summary(self, user_id: str) -> CostSummary:
        """Get complete cost summary for user"""
        today = date.today()

        # Get today's costs
        costs_today = self._get_costs_for_date(user_id, today)

        # Get month's costs
        costs_month = self._get_costs_for_month(user_id, today.year, today.month)

        # Get budgets
        budgets = self._get_budgets(user_id)

        # Calculate projection
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        day_of_month = today.day
        avg_daily = costs_month['total'] / day_of_month if day_of_month > 0 else 0
        projected = avg_daily * days_in_month

        # Build service breakdown
        services = self._build_service_breakdown(user_id, costs_today, costs_month, budgets)

        # Get alerts
        alerts = self._check_budget_alerts(user_id, costs_month, budgets)

        return CostSummary(
            total_today=costs_today['total'],
            total_month=costs_month['total'],
            total_budget=budgets['total'],
            projected_month_end=projected,
            services=services,
            alerts=alerts
        )

    def _get_costs_for_date(self, user_id: str, target_date: date) -> Dict[str, float]:
        """Get costs for a specific date"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT
                fireworks_cost_usd, openai_cost_usd, hedra_cost_usd,
                runcomfy_cost_usd, twilio_cost_usd, google_cost_usd, total_cost_usd
            FROM {T.DAILY_COST_SUMMARY}
            WHERE user_id = ? AND date = ?
        """, (user_id, target_date.isoformat()))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'fireworks': row['fireworks_cost_usd'],
                'openai': row['openai_cost_usd'],
                'hedra': row['hedra_cost_usd'],
                'runcomfy': row['runcomfy_cost_usd'],
                'twilio': row['twilio_cost_usd'],
                'google': row['google_cost_usd'],
                'total': row['total_cost_usd']
            }
        else:
            return {
                'fireworks': 0.0, 'openai': 0.0, 'hedra': 0.0,
                'runcomfy': 0.0, 'twilio': 0.0, 'google': 0.0, 'total': 0.0
            }

    def _get_costs_for_month(self, user_id: str, year: int, month: int) -> Dict[str, float]:
        """Get costs for a specific month"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT
                SUM(fireworks_cost_usd) as fireworks,
                SUM(openai_cost_usd) as openai,
                SUM(hedra_cost_usd) as hedra,
                SUM(runcomfy_cost_usd) as runcomfy,
                SUM(twilio_cost_usd) as twilio,
                SUM(google_cost_usd) as google,
                SUM(total_cost_usd) as total
            FROM {T.DAILY_COST_SUMMARY}
            WHERE user_id = ? AND strftime('%Y-%m', date) = ?
        """, (user_id, f"{year:04d}-{month:02d}"))

        row = cursor.fetchone()
        conn.close()

        if row and row['total']:
            return {
                'fireworks': row['fireworks'] or 0.0,
                'openai': row['openai'] or 0.0,
                'hedra': row['hedra'] or 0.0,
                'runcomfy': row['runcomfy'] or 0.0,
                'twilio': row['twilio'] or 0.0,
                'google': row['google'] or 0.0,
                'total': row['total'] or 0.0
            }
        else:
            return {
                'fireworks': 0.0, 'openai': 0.0, 'hedra': 0.0,
                'runcomfy': 0.0, 'twilio': 0.0, 'google': 0.0, 'total': 0.0
            }

    def _get_budgets(self, user_id: str) -> Dict[str, float]:
        """Get budget limits for user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT
                fireworks_monthly_limit_usd, openai_monthly_limit_usd,
                hedra_monthly_limit_usd, runcomfy_monthly_limit_usd,
                twilio_monthly_limit_usd, google_monthly_limit_usd,
                total_monthly_limit_usd
            FROM {T.BUDGET_LIMITS}
            WHERE user_id = ?
        """, (user_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'fireworks': row['fireworks_monthly_limit_usd'],
                'openai': row['openai_monthly_limit_usd'],
                'hedra': row['hedra_monthly_limit_usd'],
                'runcomfy': row['runcomfy_monthly_limit_usd'],
                'twilio': row['twilio_monthly_limit_usd'],
                'google': row['google_monthly_limit_usd'],
                'total': row['total_monthly_limit_usd']
            }
        else:
            # Return defaults
            return {
                'fireworks': 150.0, 'openai': 50.0, 'hedra': 100.0,
                'runcomfy': 25.0, 'twilio': 15.0, 'google': 0.0, 'total': 750.0
            }

    def _get_service_links(self) -> Dict[str, Dict[str, str]]:
        """Get third-party service billing links"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT service_name, billing_url, usage_url, api_dashboard_url, support_url
            FROM {T.SERVICE_LINKS}
        """)

        links = {}
        for row in cursor.fetchall():
            links[row['service_name']] = {
                'billing': row['billing_url'],
                'usage': row['usage_url'],
                'dashboard': row['api_dashboard_url'],
                'support': row['support_url']
            }

        conn.close()
        return links

    def _build_service_breakdown(self, user_id: str, costs_today: Dict,
                                 costs_month: Dict, budgets: Dict) -> List[ServiceCost]:
        """Build detailed service breakdown"""
        service_links = self._get_service_links()

        # Get detailed usage stats for each service
        conn = self._get_connection()
        cursor = conn.cursor()

        # Fireworks details
        cursor.execute(f"""
            SELECT
                COUNT(*) as calls,
                SUM(prompt_tokens) as tokens_in,
                SUM(completion_tokens) as tokens_out
            FROM {T.FIREWORKS_USAGE}
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
        """, (user_id,))
        fw_row = cursor.fetchone()

        # OpenAI details
        cursor.execute(f"""
            SELECT
                COUNT(*) as calls,
                SUM(prompt_tokens) as tokens_in,
                SUM(completion_tokens) as tokens_out
            FROM {T.OPENAI_USAGE}
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
        """, (user_id,))
        openai_row = cursor.fetchone()

        # RunComfy details
        cursor.execute(f"""
            SELECT
                COUNT(*) as total_images,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM {T.RUNCOMFY_USAGE}
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
        """, (user_id,))
        runcomfy_row = cursor.fetchone()

        # Twilio details
        cursor.execute(f"""
            SELECT
                SUM(CASE WHEN direction = 'outbound' THEN 1 ELSE 0 END) as sent,
                SUM(CASE WHEN direction = 'inbound' THEN 1 ELSE 0 END) as received
            FROM {T.TWILIO_USAGE}
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
        """, (user_id,))
        twilio_row = cursor.fetchone()

        # Google details
        cursor.execute(f"""
            SELECT
                COUNT(*) as total_calls,
                SUM(quota_cost) as quota_used,
                SUM(CASE WHEN service = 'gmail' THEN 1 ELSE 0 END) as gmail_calls,
                SUM(CASE WHEN service = 'calendar' THEN 1 ELSE 0 END) as calendar_calls,
                SUM(CASE WHEN service = 'drive' THEN 1 ELSE 0 END) as drive_calls
            FROM {T.GOOGLE_API_USAGE}
            WHERE user_id = ? AND DATE(timestamp) = DATE('now')
        """, (user_id,))
        google_row = cursor.fetchone()

        conn.close()

        services = []

        # Fireworks.ai
        services.append(ServiceCost(
            service_name='fireworks',
            display_name='Fireworks.ai',
            icon='🔥',
            cost_today=costs_today.get('fireworks', 0.0),
            cost_month=costs_month.get('fireworks', 0.0),
            budget_month=budgets.get('fireworks', 150.0),
            usage_details={
                'api_calls': fw_row['calls'] or 0,
                'input_tokens': f"{(fw_row['tokens_in'] or 0) / 1_000_000:.2f}M",
                'output_tokens': f"{(fw_row['tokens_out'] or 0) / 1_000_000:.2f}M",
                'avg_cost_per_call': f"${(costs_month.get('fireworks', 0.0) / fw_row['calls']):.4f}" if fw_row['calls'] else '$0.0000'
            },
            links=service_links.get('fireworks', {})
        ))

        # OpenAI
        services.append(ServiceCost(
            service_name='openai',
            display_name='OpenAI',
            icon='🤖',
            cost_today=costs_today.get('openai', 0.0),
            cost_month=costs_month.get('openai', 0.0),
            budget_month=budgets.get('openai', 50.0),
            usage_details={
                'api_calls': openai_row['calls'] or 0,
                'input_tokens': f"{(openai_row['tokens_in'] or 0) / 1_000_000:.2f}M",
                'output_tokens': f"{(openai_row['tokens_out'] or 0) / 1_000_000:.2f}M",
                'service_type': 'Tool detection (GPT-4o-mini)'
            },
            links=service_links.get('openai', {})
        ))

        # Hedra (not yet implemented)
        services.append(ServiceCost(
            service_name='hedra',
            display_name='Hedra',
            icon='🎭',
            cost_today=costs_today.get('hedra', 0.0),
            cost_month=costs_month.get('hedra', 0.0),
            budget_month=budgets.get('hedra', 100.0),
            usage_details={
                'status': '⚠️ Not yet configured',
                'estimated_cost_per_minute': '$0.05'
            },
            links=service_links.get('hedra', {})
        ))

        # RunComfy
        services.append(ServiceCost(
            service_name='runcomfy',
            display_name='RunComfy',
            icon='🎨',
            cost_today=costs_today.get('runcomfy', 0.0),
            cost_month=costs_month.get('runcomfy', 0.0),
            budget_month=budgets.get('runcomfy', 25.0),
            usage_details={
                'total_images': runcomfy_row['total_images'] or 0,
                'successful': runcomfy_row['successful'] or 0,
                'failed': runcomfy_row['failed'] or 0,
                'avg_cost_per_image': f"${(costs_month.get('runcomfy', 0.0) / runcomfy_row['successful']):.2f}" if runcomfy_row['successful'] else '$0.00'
            },
            links=service_links.get('runcomfy', {})
        ))

        # Twilio
        services.append(ServiceCost(
            service_name='twilio',
            display_name='Twilio',
            icon='📱',
            cost_today=costs_today.get('twilio', 0.0),
            cost_month=costs_month.get('twilio', 0.0),
            budget_month=budgets.get('twilio', 15.0),
            usage_details={
                'sms_sent': twilio_row['sent'] or 0,
                'sms_received': twilio_row['received'] or 0,
                'phone_rental': '$1.00/month',
                'rate': '$0.10 per outbound SMS'
            },
            links=service_links.get('twilio', {})
        ))

        # Google Workspace
        services.append(ServiceCost(
            service_name='google',
            display_name='Google Workspace',
            icon='📧',
            cost_today=costs_today.get('google', 0.0),
            cost_month=costs_month.get('google', 0.0),
            budget_month=budgets.get('google', 0.0),
            usage_details={
                'status': '✅ Free tier',
                'total_calls_today': google_row['total_calls'] or 0,
                'quota_used_today': f"{google_row['quota_used'] or 0}/10,000",
                'gmail_calls': google_row['gmail_calls'] or 0,
                'calendar_calls': google_row['calendar_calls'] or 0
            },
            links=service_links.get('google', {})
        ))

        return services

    def _check_budget_alerts(self, user_id: str, costs_month: Dict, budgets: Dict) -> List[Dict]:
        """Check for budget threshold violations and return alerts"""
        alerts = []

        # Check each service
        for service in ['fireworks', 'openai', 'hedra', 'runcomfy', 'twilio']:
            cost = costs_month.get(service, 0.0)
            budget = budgets.get(service, 0.0)

            if budget == 0:
                continue

            percentage = (cost / budget) * 100

            if percentage >= 90:
                alerts.append({
                    'type': 'critical',
                    'service': service,
                    'message': f"{service.capitalize()} has reached {percentage:.0f}% of budget (${cost:.2f} / ${budget:.2f})",
                    'percentage': percentage
                })
            elif percentage >= 75:
                alerts.append({
                    'type': 'warning',
                    'service': service,
                    'message': f"{service.capitalize()} has reached {percentage:.0f}% of budget (${cost:.2f} / ${budget:.2f})",
                    'percentage': percentage
                })

        # Check total budget
        total_cost = costs_month.get('total', 0.0)
        total_budget = budgets.get('total', 0.0)
        if total_budget > 0:
            percentage = (total_cost / total_budget) * 100

            if percentage >= 90:
                alerts.append({
                    'type': 'critical',
                    'service': 'total',
                    'message': f"Total monthly spend has reached {percentage:.0f}% of budget (${total_cost:.2f} / ${total_budget:.2f})",
                    'percentage': percentage
                })
            elif percentage >= 75:
                alerts.append({
                    'type': 'warning',
                    'service': 'total',
                    'message': f"Total monthly spend has reached {percentage:.0f}% of budget (${total_cost:.2f} / ${total_budget:.2f})",
                    'percentage': percentage
                })

        return alerts

    def get_historical_data(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get historical cost data for charting"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT
                date, total_cost_usd,
                fireworks_cost_usd, openai_cost_usd, hedra_cost_usd,
                runcomfy_cost_usd, twilio_cost_usd, google_cost_usd
            FROM {T.DAILY_COST_SUMMARY}
            WHERE user_id = ? AND date >= DATE('now', ?)
            ORDER BY date ASC
        """, (user_id, f'-{days} days'))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # ========================================================================
    # BUDGET ENFORCEMENT
    # ========================================================================

    def check_budget(self, user_id: str, service: str = None) -> Dict[str, Any]:
        """Check whether the user is within budget for a service (or overall).

        Args:
            user_id: User identifier (email).
            service: Optional service name ('fireworks', 'openai', etc.).
                     If None, checks total budget.

        Returns:
            Dict with keys:
              - allowed (bool): True if under budget
              - budget (float): Monthly budget limit
              - spent (float): Amount spent this month
              - remaining (float): Budget remaining
              - percentage (float): Percentage of budget used
        """
        today = date.today()
        costs_month = self._get_costs_for_month(user_id, today.year, today.month)
        budgets = self._get_budgets(user_id)

        key = service if service else 'total'
        spent = costs_month.get(key, 0.0)
        budget = budgets.get(key, 0.0)

        if budget <= 0:
            return {
                'allowed': True,
                'budget': budget,
                'spent': spent,
                'remaining': 0.0,
                'percentage': 0.0,
            }

        remaining = max(0.0, budget - spent)
        percentage = (spent / budget) * 100

        return {
            'allowed': remaining > 0,
            'budget': budget,
            'spent': spent,
            'remaining': remaining,
            'percentage': percentage,
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

import threading as _threading
_cost_tracker_instance = None
_cost_tracker_lock = _threading.Lock()

def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance (singleton, thread-safe)"""
    global _cost_tracker_instance
    if _cost_tracker_instance is None:
        with _cost_tracker_lock:
            # Double-check locking pattern
            if _cost_tracker_instance is None:
                _cost_tracker_instance = CostTracker()
                print("--- Cost Tracker initialized ---")
    return _cost_tracker_instance
