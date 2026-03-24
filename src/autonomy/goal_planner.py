"""
Goal Planner -- Decomposes goals into actionable steps and manages step lifecycle.

WHAT: When a goal is created (or periodically for unplanned goals), the LLM
      decomposes it into 1-5 concrete GoalSteps stored in companion_goal_steps.

WHY:  A goal like "learn about James's new project" is too abstract to act on.
      The planner turns it into concrete steps: research, prepare talking
      points, bring it up in conversation, etc.

HOW IT FITS:
  - GoalManager (goals.py) owns the goals; GoalPlanner owns the steps.
  - The AlwaysOnService calls _check_goal_actions() which uses
    get_ready_steps() to find executable steps.
  - ToolRouter (tool_router.py) executes each step.
  - OutcomeTracker (outcome_tracker.py) records what happened.
  - ActionBudget (action_budget.py) gates how many steps run per day.

Step lifecycle: pending -> ready -> in_progress -> completed / blocked / skipped

Step types:
  - research:  web search via Tavily
  - relate:    conversation topic (brought up during chat, not background)
  - suppress:  avoid action (for "being" goals) -- reduces action budget
  - prepare:   LLM content generation
  - create:    Google Docs creation
  - being:     exist in state (for "being" goals)
  - notify:    send email via Gmail

Wait conditions:
  - None:              ready immediately once dependencies are met
  - "conversation":    ready only during active chat (skipped in background)
  - "time:<hours>":    ready after N hours from creation
"""

import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

from src.database import tables as T

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


# =============================================================================
# Data model
# =============================================================================

@dataclass
class GoalStep:
    """A single actionable step toward achieving a goal."""
    id: str
    goal_id: str
    user_email: str
    step_number: int
    description: str
    action_type: str          # research | relate | suppress | prepare | create | being | notify
    parameters: Dict          # topic, tool, timing constraints, etc.
    status: str               # pending | ready | in_progress | completed | skipped | blocked
    depends_on: List[str]     # step IDs that must complete first
    wait_for: Optional[str]   # None | 'conversation' | 'time:<hours>' | 'event:<type>'
    outcome: Optional[str]
    outcome_quality: Optional[str]  # good | neutral | poor
    created_at: str
    completed_at: Optional[str]
    attempts: int
    max_attempts: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: tuple) -> 'GoalStep':
        """Construct from a database row (column order must match SELECT)."""
        return cls(
            id=row[0],
            goal_id=row[1],
            user_email=row[2],
            step_number=row[3],
            description=row[4],
            action_type=row[5],
            parameters=row[6] if row[6] else {},
            status=row[7],
            depends_on=row[8] if row[8] else [],
            wait_for=row[9],
            outcome=row[10],
            outcome_quality=row[11],
            created_at=row[12].isoformat() if row[12] else '',
            completed_at=row[13].isoformat() if row[13] else None,
            attempts=row[14] or 0,
            max_attempts=row[15] or 3
        )


# =============================================================================
# GoalPlanner
# =============================================================================

class GoalPlanner:
    """Decomposes goals into concrete steps and manages step lifecycle."""

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email
        self._conn = None

    # -----------------------------------------------------------------
    # Database & LLM connections
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

    def _get_fireworks_client(self):
        """
        Get Fireworks client directly.

        Uses direct Fireworks API (not the resilient chain) to avoid
        reasoning-model chain-of-thought issues with JSON output.
        """
        import openai
        api_key = os.environ.get('FIREWORKS_API_KEY')
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )

    # -----------------------------------------------------------------
    # Goal decomposition (LLM)
    # -----------------------------------------------------------------

    def decompose_goal(self, goal) -> List[GoalStep]:
        """
        LLM decomposes a goal into 1-5 concrete steps.

        Rules enforced by the prompt:
          - "being" mode goals -> suppress/being steps only
          - "relating" mode    -> primarily "relate" steps
          - "doing" mode       -> any step type

        Returns the created GoalSteps (already persisted to DB).
        """
        goal_mode = getattr(goal, 'goal_mode', 'doing')
        energy_cost = getattr(goal, 'energy_cost', 'medium')

        prompt = f"""/no_think
Decompose this goal into 1-5 concrete steps. Respond with ONLY a JSON array.

GOAL: {goal.goal}
MOTIVATION: {goal.motivation}
CATEGORY: {goal.category}
MODE: {goal_mode}
ENERGY COST: {energy_cost}

STEP TYPES: research, relate (conversation with James), suppress (avoid action - for "being" goals), prepare, create, being (exist in state - for "being" goals), notify

RULES:
- "being" mode goals → suppress/being steps only
- "relating" mode goals → primarily "relate" steps
- "doing" mode goals → any step type
- depends_on uses step numbers, wait_for can be null, "conversation", or "time:4" (hours)
- Fewer steps is better

JSON array format:
[{{"step_number": 1, "description": "...", "action_type": "research", "parameters": {{"topic": "..."}}, "depends_on": [], "wait_for": null}}]

Respond with ONLY the JSON array:"""

        try:
            client = self._get_fireworks_client()
            response = client.chat.completions.create(
                model="accounts/fireworks/models/kimi-k2-instruct-0905",
                max_tokens=1500,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = (response.choices[0].message.content or '').strip()
            if not response_text:
                return []

            # ---- Clean LLM output ----
            # Strip thinking tags from reasoning models
            if '<think>' in response_text and '</think>' in response_text:
                response_text = response_text.split('</think>')[-1].strip()

            # Strip markdown code fences
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                json_lines = [l for l in lines if not l.startswith('```')]
                response_text = '\n'.join(json_lines)

            # Extract JSON array even if surrounded by text
            import re
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                response_text = json_match.group(0)

            steps_data = json.loads(response_text)
            if not isinstance(steps_data, list):
                return []

            # ---- Build GoalStep objects ----
            steps = []
            step_ids = {}  # step_number -> generated step_id (for dependency resolution)

            for s in steps_data[:5]:  # hard cap at 5 steps
                step_id = f"step_{uuid.uuid4().hex[:8]}"
                step_num = s.get('step_number', len(steps) + 1)
                step_ids[step_num] = step_id

                # Resolve depends_on from step *numbers* to step *IDs*
                depends_on_nums = s.get('depends_on', [])
                depends_on_ids = [step_ids[n] for n in depends_on_nums if n in step_ids]

                step = GoalStep(
                    id=step_id,
                    goal_id=goal.id,
                    user_email=self.user_email,
                    step_number=step_num,
                    description=s.get('description', ''),
                    action_type=s.get('action_type', 'research'),
                    parameters=s.get('parameters', {}),
                    status='pending',
                    depends_on=depends_on_ids,
                    wait_for=s.get('wait_for'),
                    outcome=None,
                    outcome_quality=None,
                    created_at=datetime.now(PST).isoformat(),
                    completed_at=None,
                    attempts=0,
                    max_attempts=3
                )
                steps.append(step)

            # Persist to database
            self._store_steps(steps)
            logger.info(f"Decomposed goal '{goal.goal[:40]}...' into {len(steps)} steps")
            return steps

        except json.JSONDecodeError as e:
            logger.debug(f"Could not parse goal decomposition: {e}")
            return []
        except Exception as e:
            logger.warning(f"Goal decomposition failed: {e}")
            return []

    def _store_steps(self, steps: List[GoalStep]):
        """Persist GoalSteps to companion_goal_steps table."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                for step in steps:
                    cursor.execute(f"""
                        INSERT INTO {T.COMPANION_GOAL_STEPS}
                        (id, goal_id, user_email, step_number, description,
                         action_type, parameters, status, depends_on, wait_for,
                         attempts, max_attempts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        step.id, step.goal_id, step.user_email, step.step_number,
                        step.description, step.action_type,
                        json.dumps(step.parameters), step.status,
                        json.dumps(step.depends_on), step.wait_for,
                        step.attempts, step.max_attempts
                    ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing goal steps: {e}")
            if self._conn:
                self._conn.rollback()

    # -----------------------------------------------------------------
    # Step queries
    # -----------------------------------------------------------------

    def get_ready_steps(self, user_email: str = None) -> List[GoalStep]:
        """
        Get steps whose dependencies are met and wait conditions satisfied.

        Steps waiting for 'conversation' are excluded here (they're handled
        during active chat by get_conversation_ready_steps()).
        """
        email = user_email or self.user_email
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT s.id, s.goal_id, s.user_email, s.step_number, s.description,
                           s.action_type, s.parameters, s.status, s.depends_on, s.wait_for,
                           s.outcome, s.outcome_quality, s.created_at, s.completed_at,
                           s.attempts, s.max_attempts
                    FROM {T.COMPANION_GOAL_STEPS} s
                    JOIN {T.COMPANION_GOALS} g ON s.goal_id = g.id
                    WHERE s.user_email = %s
                    AND s.status IN ('pending', 'ready')
                    AND g.status = 'active'
                    AND s.attempts < s.max_attempts
                    ORDER BY s.step_number ASC
                """, (email,))
                rows = cursor.fetchall()

            all_steps = [GoalStep.from_row(r) for r in rows]
            completed_ids = self._get_completed_step_ids(email)

            ready = []
            for step in all_steps:
                # Check all dependencies are completed
                if not all(dep_id in completed_ids for dep_id in step.depends_on):
                    continue

                # Check wait conditions
                if step.wait_for:
                    if step.wait_for == 'conversation':
                        continue  # only ready during active chat
                    elif step.wait_for.startswith('time:'):
                        hours = int(step.wait_for.split(':')[1])
                        created = datetime.fromisoformat(step.created_at)
                        if datetime.now(PST) - created < timedelta(hours=hours):
                            continue  # time hasn't elapsed yet

                ready.append(step)

            return ready

        except Exception as e:
            logger.warning(f"Error getting ready steps: {e}")
            return []

    def get_conversation_ready_steps(self, user_email: str = None) -> List[GoalStep]:
        """
        Get 'relate' steps that are ready and waiting for conversation.

        These are topics the companion wants to bring up during active chat,
        not background tasks.
        """
        email = user_email or self.user_email
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT s.id, s.goal_id, s.user_email, s.step_number, s.description,
                           s.action_type, s.parameters, s.status, s.depends_on, s.wait_for,
                           s.outcome, s.outcome_quality, s.created_at, s.completed_at,
                           s.attempts, s.max_attempts
                    FROM {T.COMPANION_GOAL_STEPS} s
                    JOIN {T.COMPANION_GOALS} g ON s.goal_id = g.id
                    WHERE s.user_email = %s
                    AND s.action_type = 'relate'
                    AND s.status IN ('pending', 'ready')
                    AND g.status = 'active'
                    ORDER BY s.step_number ASC
                    LIMIT 3
                """, (email,))
                rows = cursor.fetchall()

            completed_ids = self._get_completed_step_ids(email)
            steps = [GoalStep.from_row(r) for r in rows]

            # Only include steps whose dependencies are all completed
            return [s for s in steps if all(d in completed_ids for d in s.depends_on)]

        except Exception as e:
            logger.debug(f"Error getting conversation-ready steps: {e}")
            return []

    def _get_completed_step_ids(self, user_email: str) -> set:
        """Get the set of IDs of all completed steps for a user."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id FROM {T.COMPANION_GOAL_STEPS}
                    WHERE user_email = %s AND status = 'completed'
                """, (user_email,))
                return {row[0] for row in cursor.fetchall()}
        except Exception:
            return set()

    def get_being_suppressions(self, user_email: str = None) -> List[GoalStep]:
        """
        Get active 'suppress' / 'being' steps.

        These reduce the action budget -- the companion is deliberately
        choosing NOT to act (e.g., a "take it easy" being-goal).
        """
        email = user_email or self.user_email
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT s.id, s.goal_id, s.user_email, s.step_number, s.description,
                           s.action_type, s.parameters, s.status, s.depends_on, s.wait_for,
                           s.outcome, s.outcome_quality, s.created_at, s.completed_at,
                           s.attempts, s.max_attempts
                    FROM {T.COMPANION_GOAL_STEPS} s
                    JOIN {T.COMPANION_GOALS} g ON s.goal_id = g.id
                    WHERE s.user_email = %s
                    AND s.action_type IN ('suppress', 'being')
                    AND s.status IN ('pending', 'ready', 'in_progress')
                    AND g.status = 'active'
                """, (email,))
                return [GoalStep.from_row(r) for r in cursor.fetchall()]

        except Exception as e:
            logger.debug(f"Error getting being suppressions: {e}")
            return []

    def get_unplanned_goals(self, user_email: str = None) -> List:
        """Get active goals that don't have any steps yet (need decomposition)."""
        email = user_email or self.user_email
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT g.id, g.goal, g.motivation, g.category, g.progress, g.status,
                           g.actions_taken, g.created_at, g.updated_at, g.deadline, g.priority,
                           g.goal_mode, g.energy_cost
                    FROM {T.COMPANION_GOALS} g
                    LEFT JOIN {T.COMPANION_GOAL_STEPS} s ON g.id = s.goal_id
                    WHERE g.user_email = %s
                    AND g.status = 'active'
                    AND s.id IS NULL
                """, (email,))
                rows = cursor.fetchall()

            from src.autonomy.goals import CompanionGoal
            return [CompanionGoal(
                id=row[0], goal=row[1], motivation=row[2],
                category=row[3], progress=row[4], status=row[5],
                actions_taken=row[6] or [],
                created_at=row[7].isoformat() if row[7] else '',
                updated_at=row[8].isoformat() if row[8] else '',
                deadline=row[9].isoformat() if row[9] else None,
                priority=row[10],
                goal_mode=row[11] or 'doing',
                energy_cost=row[12] or 'medium'
            ) for row in rows]

        except Exception as e:
            logger.warning(f"Error getting unplanned goals: {e}")
            return []

    def get_steps_for_goal(self, goal_id: str) -> List[GoalStep]:
        """Get all steps for a specific goal, ordered by step number."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, goal_id, user_email, step_number, description,
                           action_type, parameters, status, depends_on, wait_for,
                           outcome, outcome_quality, created_at, completed_at,
                           attempts, max_attempts
                    FROM {T.COMPANION_GOAL_STEPS}
                    WHERE goal_id = %s
                    ORDER BY step_number ASC
                """, (goal_id,))
                return [GoalStep.from_row(r) for r in cursor.fetchall()]

        except Exception as e:
            logger.debug(f"Error getting steps for goal: {e}")
            return []

    # -----------------------------------------------------------------
    # Step mutations
    # -----------------------------------------------------------------

    def complete_step(self, step_id: str, outcome: str, quality: str = 'neutral'):
        """
        Mark a step as completed with its outcome, then update parent goal progress.
        """
        now = datetime.now(PST)
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.COMPANION_GOAL_STEPS}
                    SET status = 'completed',
                        outcome = %s,
                        outcome_quality = %s,
                        completed_at = %s
                    WHERE id = %s
                    RETURNING goal_id
                """, (outcome, quality, now, step_id))
                result = cursor.fetchone()
            conn.commit()

            if result:
                goal_id = result[0]
                self._check_goal_progress(goal_id)
                logger.info(f"Completed step {step_id} (quality: {quality})")

        except Exception as e:
            logger.error(f"Error completing step: {e}")
            if self._conn:
                self._conn.rollback()

    def increment_step_attempt(self, step_id: str):
        """
        Increment attempt count for a step.

        If attempts reach max_attempts, the step is automatically marked as 'blocked'.
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.COMPANION_GOAL_STEPS}
                    SET attempts = attempts + 1,
                        status = CASE
                            WHEN attempts + 1 >= max_attempts THEN 'blocked'
                            ELSE status
                        END
                    WHERE id = %s
                """, (step_id,))
            conn.commit()
        except Exception as e:
            logger.debug(f"Error incrementing step attempt: {e}")
            if self._conn:
                self._conn.rollback()

    def _check_goal_progress(self, goal_id: str):
        """
        Recompute parent goal progress from step completion ratio.

        progress = (completed + skipped) / total_steps
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'skipped') as skipped
                    FROM {T.COMPANION_GOAL_STEPS}
                    WHERE goal_id = %s
                """, (goal_id,))

                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return

                total, completed, skipped = row
                progress = (completed + skipped) / total

                cursor.execute(f"""
                    UPDATE {T.COMPANION_GOALS}
                    SET progress = %s, updated_at = %s
                    WHERE id = %s
                """, (progress, datetime.now(PST), goal_id))
            conn.commit()

        except Exception as e:
            logger.debug(f"Error checking goal progress: {e}")
            if self._conn:
                self._conn.rollback()


# =============================================================================
# Singleton accessor
# =============================================================================

_planner: Optional[GoalPlanner] = None


def get_goal_planner(user_email: str = None) -> GoalPlanner:
    """Get singleton GoalPlanner instance."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    global _planner
    if _planner is None or _planner.user_email != user_email:
        _planner = GoalPlanner(user_email)
    return _planner
