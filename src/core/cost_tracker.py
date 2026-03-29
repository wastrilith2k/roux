"""
Cost tracking for LLM API calls.

Tracks token usage and costs per model for monitoring and optimization.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import redis

logger = logging.getLogger(__name__)

# Costs per 1M tokens (as of Jan 2025)
MODEL_COSTS = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Fireworks (approximate)
    "kimi-k2p5": {"input": 0.22, "output": 0.88},
    "deepseek-v3": {"input": 0.56, "output": 1.68},
    "llama-v3p1-70b-instruct": {"input": 0.20, "output": 0.20},
}


@dataclass
class APICall:
    """Record of a single API call."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    purpose: str  # e.g., "tool_call", "main_response", "embedding"


class CostTracker:
    """Track API costs across all LLM calls."""

    def __init__(self, redis_url: str = None, companion_id: str = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._redis = None
        cid = companion_id or os.environ.get("COMPANION_ID", "default")
        self.key_prefix = f"companion:{cid}:costs:"

    @property
    def redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis

    def _get_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a call."""
        # Normalize model name
        model_key = model.lower()
        for key in MODEL_COSTS:
            if key in model_key:
                costs = MODEL_COSTS[key]
                return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
        # Unknown model - estimate
        return (input_tokens * 1.0 + output_tokens * 3.0) / 1_000_000

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str = "main_response"
    ) -> float:
        """
        Record an API call and return the cost.

        Args:
            model: Model name/ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            purpose: What this call was for (tool_call, main_response, etc.)

        Returns:
            Cost in USD for this call
        """
        cost = self._get_cost(model, input_tokens, output_tokens)

        call = APICall(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            purpose=purpose
        )

        try:
            # Store in Redis list (keep last 1000 calls)
            self.redis.lpush(f"{self.key_prefix}calls", json.dumps(asdict(call)))
            self.redis.ltrim(f"{self.key_prefix}calls", 0, 999)

            # Update daily totals
            today = datetime.now().strftime("%Y-%m-%d")
            self.redis.hincrbyfloat(f"{self.key_prefix}daily:{today}", "total_cost", cost)
            self.redis.hincrby(f"{self.key_prefix}daily:{today}", "total_calls", 1)
            self.redis.hincrby(f"{self.key_prefix}daily:{today}", "input_tokens", input_tokens)
            self.redis.hincrby(f"{self.key_prefix}daily:{today}", "output_tokens", output_tokens)

            # Expire daily keys after 30 days
            self.redis.expire(f"{self.key_prefix}daily:{today}", 60 * 60 * 24 * 30)

        except Exception as e:
            logger.warning(f"Failed to record cost: {e}")

        return cost

    def get_daily_summary(self, date: str = None) -> Dict:
        """
        Get cost summary for a specific day.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Dict with total_cost, total_calls, input_tokens, output_tokens
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            data = self.redis.hgetall(f"{self.key_prefix}daily:{date}")
            if not data:
                return {"date": date, "total_cost": 0, "total_calls": 0, "input_tokens": 0, "output_tokens": 0}

            return {
                "date": date,
                "total_cost": float(data.get(b"total_cost", 0)),
                "total_calls": int(data.get(b"total_calls", 0)),
                "input_tokens": int(data.get(b"input_tokens", 0)),
                "output_tokens": int(data.get(b"output_tokens", 0)),
            }
        except Exception as e:
            logger.warning(f"Failed to get daily summary: {e}")
            return {"date": date, "total_cost": 0, "total_calls": 0, "error": str(e)}

    def get_weekly_summary(self) -> Dict:
        """Get cost summary for the last 7 days."""
        summaries = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            summaries.append(self.get_daily_summary(date))

        total_cost = sum(s.get("total_cost", 0) for s in summaries)
        total_calls = sum(s.get("total_calls", 0) for s in summaries)
        total_input = sum(s.get("input_tokens", 0) for s in summaries)
        total_output = sum(s.get("output_tokens", 0) for s in summaries)

        return {
            "period": "last_7_days",
            "total_cost": total_cost,
            "total_calls": total_calls,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "daily": summaries
        }

    def get_recent_calls(self, limit: int = 10) -> list:
        """Get the most recent API calls."""
        try:
            calls = self.redis.lrange(f"{self.key_prefix}calls", 0, limit - 1)
            return [json.loads(c) for c in calls]
        except Exception as e:
            logger.warning(f"Failed to get recent calls: {e}")
            return []


# Singleton instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def get_cost_summary() -> Dict:
    """
    Get a summary of API costs for ops reporting.

    Returns:
        Dict with today, week, month costs and breakdown by model
    """
    tracker = get_cost_tracker()

    today = tracker.get_daily_summary()
    week = tracker.get_weekly_summary()

    # Get recent calls to build model breakdown
    recent_calls = tracker.get_recent_calls(100)
    by_model: Dict[str, float] = {}
    for call in recent_calls:
        model = call.get('model', 'unknown')
        # Simplify model name
        if '/' in model:
            model = model.split('/')[-1]
        by_model[model] = by_model.get(model, 0) + call.get('cost_usd', 0)

    return {
        'today': today.get('total_cost', 0),
        'week': week.get('total_cost', 0),
        'month': None,  # Would need 30-day aggregation
        'by_model': by_model,
        'calls_today': today.get('total_calls', 0),
        'calls_week': week.get('total_calls', 0),
    }
