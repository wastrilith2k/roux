"""
Outcome Tracker -- Tracks and analyzes the results of autonomous actions.

WHAT: Three levels of outcome tracking:
      1. Immediate (action-level): Did the web search return results?
      2. Conversational (relationship-level): Did the user engage when the
         companion shared a finding?
      3. Strategic (goal-level): What action types tend to succeed?

WHY:  Without feedback the companion can't learn what works.  Outcome data
      feeds back into goal planning -- if "research" steps consistently get
      "good" reception, the planner can favor them.

HOW IT FITS:
  - Immediate outcomes are recorded by the autonomous action task right after
    executing a step via ToolRouter.
  - Conversational outcomes are analyzed after the companion shares something
    and the user responds (called from message processing).
  - Strategic insights are computed on-the-fly (no separate table) when the
    planner or reach-out engine needs context.

Storage: outcomes live on the companion_goal_steps.outcome / outcome_quality columns.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


# =============================================================================
# OutcomeTracker
# =============================================================================

class OutcomeTracker:
    """Tracks action outcomes and provides feedback for future planning."""

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email
        self._conn = None

    def _get_connection(self):
        """Get database connection (lazy, auto-reconnect)."""
        import psycopg2
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    # -----------------------------------------------------------------
    # Level 1: Immediate action outcomes
    # -----------------------------------------------------------------

    def record_action_outcome(
        self,
        step_id: str,
        immediate_result: str,
        quality: str = 'neutral'
    ):
        """
        Record what happened right after an action executed.

        Delegates to GoalPlanner.complete_step() which persists the outcome
        on the step row and updates parent goal progress.

        Args:
            step_id: The goal step ID.
            immediate_result: Description of what happened.
            quality: "good", "neutral", or "poor".
        """
        from src.autonomy.goal_planner import get_goal_planner

        planner = get_goal_planner(self.user_email)
        planner.complete_step(step_id, immediate_result, quality)
        logger.info(f"Recorded outcome for step {step_id}: {quality}")

    # -----------------------------------------------------------------
    # Level 2: Conversational outcomes (user reaction analysis)
    # -----------------------------------------------------------------

    def analyze_conversation_outcome(
        self,
        step_id: str,
        messages: List[Dict]
    ) -> Optional[str]:
        """
        Analyze whether a goal-related topic was well-received in conversation.

        Called after the companion shares a finding or asks a goal-related
        question.  Uses a fast LLM call to classify the user's response.

        Args:
            step_id: The goal step that was addressed.
            messages: Recent messages after the sharing moment.

        Returns "good", "neutral", "poor", or None on error.
        """
        if not messages:
            return None

        try:
            from src.llm.provider_factory import generate_sync

            msgs_text = "\n".join([
                f"{m.get('sender', 'unknown')}: {m.get('text', '')[:200]}"
                for m in messages[:6]
            ])

            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            user_name = _pc.primary_user_name
            u_subject = _pc.user_pronoun_subject
            prompt = f"""Analyze whether {user_name} engaged positively when the companion brought up a topic.

MESSAGES AFTER COMPANION SHARED:
{msgs_text}

How did {user_name} respond?
- "good": {u_subject.capitalize()} engaged, asked follow-ups, showed interest, or appreciated it
- "neutral": {u_subject.capitalize()} acknowledged but didn't really engage further
- "poor": {u_subject.capitalize()} seemed disinterested, changed the subject, or was dismissive

Respond with ONLY one word: good, neutral, or poor"""

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=10
            )

            if response:
                quality = response.strip().lower()
                if quality in ('good', 'neutral', 'poor'):
                    self._update_step_conversation_outcome(step_id, quality)
                    return quality

        except Exception as e:
            logger.debug(f"Conversation outcome analysis failed: {e}")

        return None

    def _update_step_conversation_outcome(self, step_id: str, quality: str):
        """Append conversation quality annotation to an existing step outcome."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE companion_goal_steps
                    SET outcome_quality = %s,
                        outcome = COALESCE(outcome, '') || ' [Conversation: ' || %s || ']'
                    WHERE id = %s
                """, (quality, quality, step_id))
            conn.commit()
        except Exception as e:
            logger.debug(f"Error updating conversation outcome: {e}")
            if self._conn:
                self._conn.rollback()

    # -----------------------------------------------------------------
    # Level 3: Strategic insights (computed on-the-fly)
    # -----------------------------------------------------------------

    def get_strategy_insights(self, goal_id: str = None) -> str:
        """
        Compute strategy insights from past step outcomes.

        Aggregates outcome_quality by action_type to produce simple success
        rates.  No separate storage -- computed fresh each time.

        Args:
            goal_id: If given, scope to one goal.  Otherwise, all recent.

        Returns a formatted string for LLM context, or "" if no data.
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if goal_id:
                    cursor.execute("""
                        SELECT s.action_type, s.outcome_quality, s.description,
                               g.category, g.goal_mode
                        FROM companion_goal_steps s
                        JOIN companion_goals g ON s.goal_id = g.id
                        WHERE s.goal_id = %s AND s.status = 'completed'
                        ORDER BY s.completed_at DESC
                        LIMIT 10
                    """, (goal_id,))
                else:
                    cursor.execute("""
                        SELECT s.action_type, s.outcome_quality, s.description,
                               g.category, g.goal_mode
                        FROM companion_goal_steps s
                        JOIN companion_goals g ON s.goal_id = g.id
                        WHERE s.user_email = %s AND s.status = 'completed'
                        AND s.outcome_quality IS NOT NULL
                        ORDER BY s.completed_at DESC
                        LIMIT 20
                    """, (self.user_email,))

                rows = cursor.fetchall()

            if not rows:
                return ""

            # Aggregate counts by action type
            by_type: Dict[str, Dict[str, int]] = {}
            for action_type, quality, desc, category, mode in rows:
                if action_type not in by_type:
                    by_type[action_type] = {'good': 0, 'neutral': 0, 'poor': 0, 'total': 0}
                by_type[action_type]['total'] += 1
                if quality in ('good', 'neutral', 'poor'):
                    by_type[action_type][quality] += 1

            # Format into readable lines (skip types with <2 data points)
            lines = ["[STRATEGY INSIGHTS from past goal actions]"]
            for atype, counts in by_type.items():
                total = counts['total']
                if total < 2:
                    continue
                good_rate = counts['good'] / total if total > 0 else 0
                lines.append(
                    f"- {atype}: {counts['good']}/{total} positive "
                    f"({good_rate:.0%} success rate)"
                )

            return "\n".join(lines) if len(lines) > 1 else ""

        except Exception as e:
            logger.debug(f"Error getting strategy insights: {e}")
            return ""

    # -----------------------------------------------------------------
    # Recent outcomes for context
    # -----------------------------------------------------------------

    def get_recent_outcomes(self, limit: int = 5) -> List[Dict]:
        """Get the most recent completed step outcomes for LLM context."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT s.action_type, s.description, s.outcome, s.outcome_quality,
                           s.completed_at, g.goal
                    FROM companion_goal_steps s
                    JOIN companion_goals g ON s.goal_id = g.id
                    WHERE s.user_email = %s
                    AND s.status = 'completed'
                    AND s.outcome IS NOT NULL
                    ORDER BY s.completed_at DESC
                    LIMIT %s
                """, (self.user_email, limit))

                return [
                    {
                        'action_type': r[0],
                        'description': r[1],
                        'outcome': r[2],
                        'quality': r[3],
                        'completed_at': r[4].isoformat() if r[4] else '',
                        'goal': r[5]
                    }
                    for r in cursor.fetchall()
                ]

        except Exception as e:
            logger.debug(f"Error getting recent outcomes: {e}")
            return []


# =============================================================================
# Singleton accessor
# =============================================================================

_tracker: Optional[OutcomeTracker] = None


def get_outcome_tracker(user_email: str = None) -> OutcomeTracker:
    """Get singleton OutcomeTracker instance."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    global _tracker
    if _tracker is None or _tracker.user_email != user_email:
        _tracker = OutcomeTracker(user_email)
    return _tracker
