"""
Action Budget Gate -- Anti-pushiness mechanism for goal-directed work.

WHAT: Limits the number of autonomous "doing" actions (research, preparation,
      content creation) the companion takes per calendar day.  This is NOT
      about limiting messages -- that is handled by reach-out pressure.

WHY:  A person who relentlessly works toward goals is exhausting.  Real people
      have goals but also just *exist*.  The budget enforces that balance:
      the companion can only do a few goal-directed tasks per day, and must
      wait at least MIN_ACTION_GAP_HOURS between them.

HOW IT FITS:
  - Called by the autonomous action task before executing any goal step.
  - Uses Redis for ephemeral daily counters (auto-expire after 24 h).
  - Budget is reduced by: active being-goals, low energy, long activity
    streaks, and weekends.
  - A single high-urgency bypass is allowed per day to avoid blocking
    something truly important.

Storage: Redis (ephemeral, resets daily).
"""

import os
import json
import logging
from datetime import datetime, date, timedelta
from typing import Tuple, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# ---- Configuration constants ----
BASE_DAILY_BUDGET = 4        # default "doing" actions per day
MIN_ACTION_GAP_HOURS = 2     # minimum gap between consecutive actions


# =============================================================================
# Redis helper
# =============================================================================

def _get_redis():
    """Get a Redis connection (import deferred to avoid startup cost)."""
    import redis
    return redis.Redis(
        host=os.environ.get('REDIS_HOST', 'redis'),
        port=int(os.environ.get('REDIS_PORT', '6379')),
        db=int(os.environ.get('REDIS_DB', '0')),
        decode_responses=True
    )


# =============================================================================
# ActionBudget
# =============================================================================

class ActionBudget:
    """Tracks and gates goal-directed actions to prevent pushy behavior."""

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email

    # ---- Key helpers (namespaced by companion ID + user + date) ----

    def _key(self, suffix: str) -> str:
        """Redis key scoped to today."""
        today = date.today().isoformat()
        cid = os.environ.get("COMPANION_ID", "default")
        return f"companion:{cid}:action_budget:{self.user_email}:{today}:{suffix}"

    def _global_key(self, suffix: str) -> str:
        """Redis key NOT scoped to today (for cross-day tracking)."""
        cid = os.environ.get("COMPANION_ID", "default")
        return f"companion:{cid}:action_budget:{self.user_email}:{suffix}"

    # -----------------------------------------------------------------
    # Budget calculation
    # -----------------------------------------------------------------

    def get_remaining_budget(self) -> int:
        """How many more doing-actions the companion should take today."""
        try:
            r = _get_redis()
            used = int(r.get(self._key('used')) or 0)
            max_budget = self._compute_max_budget()
            return max(0, max_budget - used)
        except Exception as e:
            logger.debug(f"Could not get budget from Redis: {e}")
            return BASE_DAILY_BUDGET  # fail open -- allow actions on Redis failure

    def _compute_max_budget(self) -> int:
        """
        Compute today's max budget considering modifiers.

        Modifiers (each can subtract from the base budget):
          - Active being-goals (suppress/being steps) reduce by count
          - Low companion energy (<0.4) reduces by 1
          - 3+ consecutive active days reduces by 1
          - Weekends reduce by 1
        Floor: always at least 1 action allowed.
        """
        budget = BASE_DAILY_BUDGET

        # Modifier: active being-goals reduce budget
        try:
            from src.autonomy.goal_planner import get_goal_planner
            planner = get_goal_planner(self.user_email)
            suppressions = planner.get_being_suppressions(self.user_email)
            if suppressions:
                budget -= len(suppressions)
                logger.debug(f"Being-goals reducing budget by {len(suppressions)}")
        except Exception:
            pass

        # Modifier: low energy reduces budget
        try:
            from src.core.internal_state import get_internal_state_manager
            state_manager = get_internal_state_manager()
            state = state_manager.get_state(self.user_email)
            if state.energy < 0.4:
                budget -= 1
                logger.debug("Low energy reducing budget by 1")
        except Exception:
            pass

        # Modifier: consecutive active days (3+ = fatigue penalty)
        try:
            r = _get_redis()
            streak = int(r.get(self._global_key('action_streak')) or 0)
            if streak >= 3:
                budget -= 1
                logger.debug(f"Action streak ({streak} days) reducing budget by 1")
        except Exception:
            pass

        # Modifier: weekends = relax
        now = datetime.now(PST)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            budget -= 1
            logger.debug("Weekend reducing budget by 1")

        return max(1, budget)  # always allow at least one action

    # -----------------------------------------------------------------
    # Gate: should the companion act?
    # -----------------------------------------------------------------

    def should_act(self, step=None) -> Tuple[bool, str]:
        """
        Check if this action should proceed.

        Returns (should_proceed, reason).
        """
        remaining = self.get_remaining_budget()

        if remaining <= 0:
            # Allow a single high-urgency bypass per day
            if step and self._is_high_urgency(step):
                return True, "high-urgency bypass (budget exhausted but urgent)"
            return False, f"daily action budget exhausted (0 remaining)"

        # Enforce minimum time gap between actions
        gap_ok, gap_reason = self._check_action_gap()
        if not gap_ok:
            return False, gap_reason

        # Being/suppress/relate steps are free -- they don't cost budget
        if step and step.action_type in ('suppress', 'being', 'relate'):
            return True, "non-budget action type"

        return True, f"budget allows ({remaining} remaining)"

    def _is_high_urgency(self, step) -> bool:
        """
        Check if a step has high urgency that can bypass budget (once/day).

        Reads step.parameters['urgency'] > 0.8 and ensures only one bypass
        per calendar day via a Redis flag.
        """
        try:
            r = _get_redis()
            bypass_used = int(r.get(self._key('urgency_bypass')) or 0)
            if bypass_used > 0:
                return False  # only one bypass per day

            params = step.parameters or {}
            urgency = params.get('urgency', 0)
            return urgency > 0.8
        except Exception:
            return False

    def _check_action_gap(self) -> Tuple[bool, str]:
        """Enforce minimum time gap between actions."""
        try:
            r = _get_redis()
            last_time_str = r.get(self._global_key('last_action_time'))
            if last_time_str:
                last_time = datetime.fromisoformat(last_time_str)
                elapsed = (datetime.now(PST) - last_time).total_seconds() / 3600
                if elapsed < MIN_ACTION_GAP_HOURS:
                    return False, f"too soon since last action ({elapsed:.1f}h < {MIN_ACTION_GAP_HOURS}h minimum)"
        except Exception:
            pass
        return True, "gap OK"

    # -----------------------------------------------------------------
    # Recording actions
    # -----------------------------------------------------------------

    def record_action(self, step=None):
        """Record that an action was taken -- decrement today's budget."""
        try:
            r = _get_redis()
            now = datetime.now(PST)

            # Increment daily counter
            key = self._key('used')
            r.incr(key)
            r.expire(key, 86400)  # auto-expire after 24 h

            # Record timestamp for gap enforcement
            r.set(
                self._global_key('last_action_time'),
                now.isoformat(),
                ex=86400
            )

            # Track high-urgency bypass usage
            if step and self._is_high_urgency(step):
                bypass_key = self._key('urgency_bypass')
                r.set(bypass_key, '1', ex=86400)

            # Update consecutive-active-day streak
            self._update_streak()

            used = int(r.get(key) or 0)
            logger.info(f"Action recorded (used: {used}, max: {self._compute_max_budget()})")

        except Exception as e:
            logger.debug(f"Could not record action in Redis: {e}")

    def _update_streak(self):
        """
        Update consecutive active days streak.

        If the last action was yesterday, increment; otherwise reset to 1.
        """
        try:
            r = _get_redis()
            streak_key = self._global_key('action_streak')
            last_date_key = self._global_key('last_action_date')

            today = date.today().isoformat()
            last_date = r.get(last_date_key)

            if last_date == today:
                return  # already counted today

            yesterday = (date.today() - timedelta(days=1)).isoformat()
            if last_date == yesterday:
                r.incr(streak_key)  # consecutive day
            else:
                r.set(streak_key, '1')  # gap in activity -- reset

            r.set(last_date_key, today, ex=86400 * 7)
            r.expire(streak_key, 86400 * 7)

        except Exception:
            pass

    # -----------------------------------------------------------------
    # Prompt context for LLM
    # -----------------------------------------------------------------

    def get_budget_context_for_prompt(self) -> str:
        """Generate a human-readable string about budget state for LLM prompts."""
        try:
            remaining = self.get_remaining_budget()
            max_budget = self._compute_max_budget()

            r = _get_redis()
            used = int(r.get(self._key('used')) or 0)
            streak = int(r.get(self._global_key('action_streak')) or 0)

            lines = []

            if remaining == 0:
                lines.append("[ACTION BUDGET: Exhausted for today. Take it easy — focus on being present, not productive.]")
            elif remaining <= 1:
                lines.append(f"[ACTION BUDGET: Low ({remaining}/{max_budget} remaining). Save energy for something important.]")
            elif used > 0:
                lines.append(f"[ACTION BUDGET: {remaining}/{max_budget} actions remaining today (used {used} so far).]")

            if streak >= 3:
                lines.append(f"[You've been busy for {streak} days straight. Consider a being-goal or downtime.]")

            # Mention active being-goals if any
            try:
                from src.autonomy.goal_planner import get_goal_planner
                planner = get_goal_planner(self.user_email)
                suppressions = planner.get_being_suppressions(self.user_email)
                if suppressions:
                    descs = [s.description[:60] for s in suppressions[:2]]
                    lines.append(f"[BEING MODE: Active — {'; '.join(descs)}]")
            except Exception:
                pass

            return "\n".join(lines) if lines else ""

        except Exception as e:
            logger.debug(f"Could not build budget context: {e}")
            return ""

    # -----------------------------------------------------------------
    # Testing utility
    # -----------------------------------------------------------------

    def reset_daily(self):
        """Manually reset daily budget (for testing)."""
        try:
            r = _get_redis()
            r.delete(self._key('used'))
            r.delete(self._key('urgency_bypass'))
        except Exception:
            pass


# =============================================================================
# Singleton accessor
# =============================================================================

_budget: Optional[ActionBudget] = None


def get_action_budget(user_email: str = None) -> ActionBudget:
    """Get singleton ActionBudget instance."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    global _budget
    if _budget is None or _budget.user_email != user_email:
        _budget = ActionBudget(user_email)
    return _budget
