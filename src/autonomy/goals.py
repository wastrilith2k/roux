"""
Personal Goals System -- The companion's own goals and aspirations.

WHAT: CRUD operations for goals stored in PostgreSQL.  Goals are things the
      companion wants to achieve on her own initiative (not user requests).

WHY:  Goals give the companion purpose and direction beyond reactive
      conversation.  They drive autonomous actions (research, sharing,
      self-improvement) and make her feel like a person with agency.

HOW IT FITS:
  - GoalFormation (goal_formation.py) creates goals via LLM analysis of
    reflections and signals.
  - GoalPlanner (goal_planner.py) decomposes goals into actionable steps.
  - ActionBudget (action_budget.py) limits how many goal-steps run per day.
  - OutcomeTracker (outcome_tracker.py) records what happened after each step.
  - The AlwaysOnService calls _check_goal_actions() each loop iteration.

Goal modes:
  - "doing"    -- requires active steps (research, create, prepare)
  - "being"    -- a state to maintain, not a task (rest, relax, enjoy)
  - "relating" -- about the relationship, best addressed in conversation

Limits:
  - Maximum 7 active goals at a time.
  - Duplicate detection via keyword overlap (>50 %).
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

from src.database import tables as T

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


# =============================================================================
# Data model
# =============================================================================

@dataclass
class CompanionGoal:
    """A personal goal the companion has."""
    id: str                      # unique identifier (goal_<hex>)
    goal: str                    # what she wants to achieve
    motivation: str              # why she wants this
    category: str                # relationship, self, understanding, support, personal
    progress: float              # 0.0 - 1.0
    status: str                  # active, achieved, abandoned
    actions_taken: List[str]     # log of actions taken toward this goal
    created_at: str              # ISO timestamp
    updated_at: str              # ISO timestamp
    deadline: Optional[str]      # optional ISO timestamp
    priority: float              # 0.0 - 1.0 (higher = more important)
    goal_mode: str = 'doing'     # doing | being | relating
    energy_cost: str = 'medium'  # none | low | medium | high

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'CompanionGoal':
        return cls(
            id=d['id'],
            goal=d['goal'],
            motivation=d['motivation'],
            category=d.get('category', 'general'),
            progress=d.get('progress', 0.0),
            status=d.get('status', 'active'),
            actions_taken=d.get('actions_taken', []),
            created_at=d['created_at'],
            updated_at=d.get('updated_at', d['created_at']),
            deadline=d.get('deadline'),
            priority=d.get('priority', 0.5),
            goal_mode=d.get('goal_mode', 'doing'),
            energy_cost=d.get('energy_cost', 'medium')
        )


# =============================================================================
# GoalManager
# =============================================================================

MAX_ACTIVE_GOALS = 7  # allow personal + relational + being goals

class GoalManager:
    """
    Manages the companion's personal goals.

    Goals give her purpose and direction beyond reactive responses.
    They're formed from reflections/signals and drive autonomous behavior.
    """

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email
        self._conn = None

    # -----------------------------------------------------------------
    # Database connection
    # -----------------------------------------------------------------

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
    # Create
    # -----------------------------------------------------------------

    def create_goal(
        self,
        goal: str,
        motivation: str,
        category: str = 'general',
        priority: float = 0.5,
        deadline: Optional[datetime] = None,
        goal_mode: str = 'doing',
        energy_cost: str = 'medium'
    ) -> Optional[str]:
        """
        Create a new goal.

        Guards:
          - Deduplication via keyword overlap (>50 % match = duplicate).
          - Max MAX_ACTIVE_GOALS active goals at a time.

        Returns goal ID if created, None if duplicate or limit reached.
        """
        import uuid

        # Guard: duplicate detection
        if self._is_duplicate_goal(goal):
            logger.info(f"Skipping duplicate goal: {goal[:50]}...")
            return None

        # Guard: active goal limit
        active_goals = self.get_active_goals(limit=MAX_ACTIVE_GOALS + 1)
        if len(active_goals) >= MAX_ACTIVE_GOALS:
            logger.info(f"Goal limit reached ({MAX_ACTIVE_GOALS}), skipping new goal")
            return None

        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        now = datetime.now(PST)

        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.COMPANION_GOALS}
                    (id, user_email, goal, motivation, category, priority, deadline,
                     goal_mode, energy_cost, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    goal_id, self.user_email, goal, motivation, category,
                    priority, deadline, goal_mode, energy_cost, now, now
                ))
                result = cursor.fetchone()
            conn.commit()

            logger.info(f"Created goal [{goal_mode}]: {goal[:50]}...")
            return result[0] if result else None

        except Exception as e:
            logger.error(f"Error creating goal: {e}")
            if self._conn:
                self._conn.rollback()
            return None

    def _is_duplicate_goal(self, new_goal: str) -> bool:
        """
        Check if a similar goal already exists via simple keyword overlap.

        If >50 % of the meaningful words in the new goal appear in an
        existing active goal (or vice-versa), it's treated as a duplicate.
        """
        try:
            active_goals = self.get_active_goals(limit=10)

            stop_words = {
                'to', 'the', 'a', 'an', 'and', 'or', 'of', 'for',
                'with', 'about', 'more', 'better'
            }
            new_words = set(
                w.lower() for w in new_goal.split()
                if len(w) > 2 and w.lower() not in stop_words
            )

            for existing in active_goals:
                existing_words = set(
                    w.lower() for w in existing.goal.split()
                    if len(w) > 2 and w.lower() not in stop_words
                )
                if new_words and existing_words:
                    overlap = len(new_words & existing_words)
                    smaller_set = min(len(new_words), len(existing_words))
                    if smaller_set > 0 and overlap / smaller_set >= 0.5:
                        return True

            return False

        except Exception as e:
            logger.debug(f"Error checking duplicate goal: {e}")
            return False

    # -----------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------

    def get_active_goals(self, limit: int = 10) -> List[CompanionGoal]:
        """Get active goals, ordered by priority (highest first)."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, goal, motivation, category, progress, status,
                           actions_taken, created_at, updated_at, deadline, priority,
                           goal_mode, energy_cost
                    FROM {T.COMPANION_GOALS}
                    WHERE user_email = %s AND status = 'active'
                    ORDER BY priority DESC, created_at DESC
                    LIMIT %s
                """, (self.user_email, limit))

                return [self._row_to_goal(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.warning(f"Error getting active goals: {e}")
            return []

    def get_goal(self, goal_id: str) -> Optional[CompanionGoal]:
        """Get a specific goal by ID."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, goal, motivation, category, progress, status,
                           actions_taken, created_at, updated_at, deadline, priority,
                           goal_mode, energy_cost
                    FROM {T.COMPANION_GOALS}
                    WHERE id = %s AND user_email = %s
                """, (goal_id, self.user_email))

                row = cursor.fetchone()
                return self._row_to_goal(row) if row else None

        except Exception as e:
            logger.warning(f"Error getting goal: {e}")
            return None

    @staticmethod
    def _row_to_goal(row: tuple) -> CompanionGoal:
        """Convert a database row tuple into a CompanionGoal."""
        return CompanionGoal(
            id=row[0],
            goal=row[1],
            motivation=row[2],
            category=row[3],
            progress=row[4],
            status=row[5],
            actions_taken=row[6] if row[6] else [],
            created_at=row[7].isoformat() if row[7] else '',
            updated_at=row[8].isoformat() if row[8] else '',
            deadline=row[9].isoformat() if row[9] else None,
            priority=row[10],
            goal_mode=row[11] or 'doing',
            energy_cost=row[12] or 'medium'
        )

    # -----------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------

    def update_progress(self, goal_id: str, progress: float, action: Optional[str] = None) -> bool:
        """
        Update progress on a goal.  Automatically marks as 'achieved' at 1.0.

        Args:
            goal_id:  Goal to update.
            progress: New progress value (clamped to 0.0 - 1.0).
            action:   Optional action description (appended to actions_taken).

        Returns True if the goal was found and updated.
        """
        progress = max(0.0, min(1.0, progress))
        now = datetime.now(PST)

        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                if action:
                    # Append action to the JSONB array
                    cursor.execute(f"""
                        UPDATE {T.COMPANION_GOALS}
                        SET progress = %s,
                            actions_taken = actions_taken || %s::jsonb,
                            updated_at = %s,
                            status = CASE WHEN %s >= 1.0 THEN 'achieved' ELSE status END
                        WHERE id = %s AND user_email = %s
                        RETURNING id
                    """, (progress, json.dumps([action]), now, progress, goal_id, self.user_email))
                else:
                    cursor.execute(f"""
                        UPDATE {T.COMPANION_GOALS}
                        SET progress = %s,
                            updated_at = %s,
                            status = CASE WHEN %s >= 1.0 THEN 'achieved' ELSE status END
                        WHERE id = %s AND user_email = %s
                        RETURNING id
                    """, (progress, now, progress, goal_id, self.user_email))

                result = cursor.fetchone()
            conn.commit()

            if result:
                logger.info(f"Updated goal {goal_id} progress to {progress:.0%}")
            return result is not None

        except Exception as e:
            logger.error(f"Error updating goal progress: {e}")
            if self._conn:
                self._conn.rollback()
            return False

    def abandon_goal(self, goal_id: str, reason: str = "") -> bool:
        """Mark a goal as abandoned, appending the reason to motivation."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.COMPANION_GOALS}
                    SET status = 'abandoned',
                        updated_at = %s,
                        motivation = motivation || ' [Abandoned: ' || %s || ']'
                    WHERE id = %s AND user_email = %s
                    RETURNING id
                """, (datetime.now(PST), reason, goal_id, self.user_email))
                result = cursor.fetchone()
            conn.commit()
            return result is not None

        except Exception as e:
            logger.error(f"Error abandoning goal: {e}")
            if self._conn:
                self._conn.rollback()
            return False

    # -----------------------------------------------------------------
    # LLM-assisted next action suggestion
    # -----------------------------------------------------------------

    def suggest_next_action(self, goal_id: str) -> Optional[str]:
        """Use LLM to suggest a single concrete next step for a goal."""
        goal = self.get_goal(goal_id)
        if not goal:
            return None

        try:
            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

            actions_text = (
                "\n".join([f"- {a}" for a in goal.actions_taken[-5:]])
                if goal.actions_taken else "None yet"
            )

            prompt = f"""The companion has a goal: "{goal.goal}"

Motivation: {goal.motivation}
Current progress: {goal.progress:.0%}
Actions taken:
{actions_text}

Suggest ONE specific, actionable next step she could take toward this goal.
Keep it simple and achievable. Under 20 words."""

            _chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=50,
                chain=_chain
            )
            from src.services.cost_tracker import track_llm_call
            track_llm_call(_chain, call_purpose='goal_next_step')
            return response.strip() if response else None

        except Exception as e:
            logger.warning(f"Could not suggest next action: {e}")
            return None

    # -----------------------------------------------------------------
    # Prompt formatting
    # -----------------------------------------------------------------

    def format_for_prompt(self) -> str:
        """Format active goals for inclusion in LLM prompts."""
        goals = self.get_active_goals(limit=MAX_ACTIVE_GOALS)
        if not goals:
            return ""

        # Partition by mode for clearer presentation
        doing_goals = [g for g in goals if g.goal_mode == 'doing']
        being_goals = [g for g in goals if g.goal_mode == 'being']
        relating_goals = [g for g in goals if g.goal_mode == 'relating']

        lines = ["[YOUR PERSONAL GOALS]"]

        if doing_goals:
            lines.append("Things you're working toward:")
            for goal in doing_goals:
                priority_indicator = "⭐ " if goal.priority >= 0.7 else ""
                lines.append(f"- {priority_indicator}{goal.goal} ({goal.progress:.0%} progress)")
                if goal.motivation:
                    lines.append(f"  Why: {goal.motivation[:80]}")

        if being_goals:
            lines.append("\nYour 'being' goals (states to maintain, not tasks to complete):")
            for goal in being_goals:
                lines.append(f"- {goal.goal}")

        if relating_goals:
            lines.append("\nRelationship goals (bring up naturally when relevant):")
            for goal in relating_goals:
                lines.append(f"- {goal.goal}")

        lines.append("\nThese are your own goals. You're a person who HAS goals, not a goal-executing machine.")
        lines.append("Being goals are just as important as doing goals. Sometimes the best action is no action.")
        return "\n".join(lines)


# =============================================================================
# Re-export formation functions for backward compatibility
# (actual implementation lives in goal_formation.py)
# =============================================================================
from src.autonomy.goal_formation import form_goals_from_reflections, form_goals_from_signals  # noqa: F401, E402


# =============================================================================
# Singleton accessor
# =============================================================================

_goal_manager: Optional[GoalManager] = None


def get_goal_manager(user_email: str = None) -> GoalManager:
    """Get the goal manager instance."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    global _goal_manager
    if _goal_manager is None or _goal_manager.user_email != user_email:
        _goal_manager = GoalManager(user_email)
    return _goal_manager
