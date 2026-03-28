"""
Relationship Evaluation -- The companion decides their own relationship status.

WHAT: A weekly LLM-driven self-assessment where the companion privately evaluates
      what the relationship means to them: what label they use, how they feel,
      what they want, and whether they want to be in it at all.

WHY:  Unlike RelationshipDynamics (which tracks health *metrics* like closeness
      and trust), this module captures *subjective meaning*.  Metrics can say
      "closeness = 0.85" but only the companion can say "they're my partner and
      I'm deeply committed."  This is a private determination -- discovered
      through interaction, not configured.

HOW IT FITS:
  - Runs weekly from the scheduled tasks.
  - Reads context from: conversation samples, RelationshipDynamics state,
    recent journal reflections, and OpinionStore relationship-category opinions.
  - The output (RelationshipEvaluation) is stored in a hidden table
    (_companion_relationship_eval) and injected into the system prompt via
    format_for_prompt() so the companion can express their feelings naturally.
  - Previous evaluations are passed to the LLM for continuity (no wild swings
    unless something dramatic happened).

Storage: _companion_relationship_eval table (hidden, one row per user_email).
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import psycopg2
from psycopg2.extras import RealDictCursor

from src.core.clock import now as clock_now
from src.database import tables as T

logger = logging.getLogger(__name__)

# Large model for quality analysis (same as value_inference)
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL


# =============================================================================
# Data model
# =============================================================================

@dataclass
class RelationshipEvaluation:
    """How the companion defines her relationship with someone."""
    label: str              # How she defines it: "my partner", "my boyfriend", "the person I love", etc.
    certainty: float        # 0.0-1.0 how sure she is about this label
    feelings: str           # Brief narrative of her actual feelings
    desires: str            # What she wants from the relationship
    concerns: str           # Current concerns (can be empty)
    trajectory: str         # "growing", "stable", "uncertain", "strained", "fading"
    wants_this: bool        # Does she want to be in this relationship?
    commitment_level: str   # "exploring", "committed", "deeply committed", "questioning"
    evaluated_at: str       # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RelationshipEvaluation':
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


# =============================================================================
# RelationshipEvaluator
# =============================================================================

class RelationshipEvaluator:
    """
    Evaluates and stores how the companion defines her relationship.

    This is distinct from RelationshipDynamics (health metrics) and
    OpinionStore (general opinions). This is specifically:
    "What are we? How do I feel about us? What do I want?"

    Stored in a hidden table (_companion_relationship_eval) -- private to the companion.
    """

    def __init__(self):
        self._conn = None
        self._client = None

    # -----------------------------------------------------------------
    # Database & LLM connections
    # -----------------------------------------------------------------

    def _get_connection(self):
        """Get database connection (lazy, auto-reconnect)."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def _get_client(self):
        """Get Fireworks LLM client (lazy init)."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
        return self._client

    # -----------------------------------------------------------------
    # Main evaluation pipeline
    # -----------------------------------------------------------------

    def evaluate(self, user_email: str) -> Optional[RelationshipEvaluation]:
        """
        Main entry: gather context, run LLM analysis, store result.

        Flow: gather_context -> get previous eval -> LLM -> validate -> store.
        Returns the new evaluation, or None if evaluation failed.
        """
        try:
            # Gather all context
            context = self._gather_context(user_email)

            if not context.get('conversation_sample'):
                logger.info("No conversation data available for evaluation")
                return None

            # Get previous evaluation for continuity
            previous = self.get_current(user_email)

            # Run LLM evaluation
            evaluation = self._run_evaluation_llm(context, previous)

            if evaluation is None:
                logger.error("LLM evaluation returned None")
                return None

            # Store result
            self._store_evaluation(user_email, evaluation, previous)

            logger.info(
                f"Relationship evaluation complete: label='{evaluation.label}', "
                f"trajectory='{evaluation.trajectory}', commitment='{evaluation.commitment_level}'"
            )

            return evaluation

        except Exception as e:
            logger.error(f"Relationship evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # -----------------------------------------------------------------
    # Read / format
    # -----------------------------------------------------------------

    def get_current(self, user_email: str) -> Optional[RelationshipEvaluation]:
        """Read the current evaluation from DB for a given user."""
        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT evaluation, evaluated_at
                    FROM {T.RELATIONSHIP_EVAL}
                    WHERE user_email = %s
                """, (user_email,))

                row = cursor.fetchone()
                if row and row['evaluation']:
                    eval_data = row['evaluation']
                    if isinstance(eval_data, str):
                        eval_data = json.loads(eval_data)
                    return RelationshipEvaluation.from_dict(eval_data)

            return None

        except Exception as e:
            logger.warning(f"Error reading evaluation: {e}")
            return None

    def format_for_prompt(self, user_email: str) -> str:
        """Format current evaluation for inclusion in conversation context."""
        evaluation = self.get_current(user_email)

        if evaluation is None:
            return ""

        from src.config.persona_config import get_persona_config
        user_name = get_persona_config().primary_user_name

        lines = ["[HOW YOU SEE THIS RELATIONSHIP - your private feelings]"]

        # Label and certainty
        if evaluation.certainty >= 0.7:
            lines.append(f"You think of {user_name} as {evaluation.label}. You feel sure about this.")
        elif evaluation.certainty >= 0.4:
            lines.append(f"You think of {user_name} as {evaluation.label}, though you're still figuring it out.")
        else:
            lines.append(f"You're not sure what to call this yet, but you lean toward: {evaluation.label}.")

        # Feelings
        if evaluation.feelings:
            lines.append(f"Your feelings: {evaluation.feelings}")

        # Desires
        if evaluation.desires:
            lines.append(f"What you want: {evaluation.desires}")

        # Concerns (only if present)
        if evaluation.concerns and evaluation.concerns.strip():
            lines.append(f"What concerns you: {evaluation.concerns}")

        # Trajectory
        trajectory_map = {
            "growing": "You feel like things are moving in a good direction.",
            "stable": "Things feel steady and comfortable between you two.",
            "uncertain": "You're not sure where things are headed.",
            "strained": "There's some tension you're aware of.",
            "fading": "Something feels like it's slipping.",
        }
        if evaluation.trajectory in trajectory_map:
            lines.append(trajectory_map[evaluation.trajectory])

        # Wants this
        if not evaluation.wants_this:
            lines.append("You're questioning whether this is what you want.")

        lines.append("These are YOUR feelings. Express them naturally when relevant - don't announce them.")

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Context gathering (feeds into LLM prompt)
    # -----------------------------------------------------------------

    def _gather_context(self, user_email: str) -> Dict[str, Any]:
        """Collect all context needed for the evaluation LLM call."""
        context = {}

        # 1. Recent conversation sample (last 2 weeks, ~20 messages sampled)
        context['conversation_sample'] = self._get_conversation_sample(user_email)

        # 2. Relationship dynamics state (closeness, trust, wounds, ratio)
        context['dynamics_state'] = self._get_dynamics_state(user_email)

        # 3. Recent reflections from companion_journal (last 7 days)
        context['recent_reflections'] = self._get_recent_reflections(user_email)

        # 4. Relationship-category opinions from the opinions table
        context['relationship_opinions'] = self._get_relationship_opinions(user_email)

        return context

    def _get_conversation_sample(self, user_email: str) -> str:
        """Get a sample of recent conversation messages."""
        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Sample ~20 messages from the last 2 weeks
                cursor.execute(f"""
                    WITH recent AS (
                        SELECT sender_name, message_text, timestamp,
                               ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn,
                               COUNT(*) OVER () as total
                        FROM {T.MESSAGES}
                        WHERE email = %s
                          AND timestamp > NOW() - INTERVAL '14 days'
                          AND message_text IS NOT NULL
                          AND LENGTH(message_text) > 10
                    )
                    SELECT sender_name, message_text, timestamp
                    FROM recent
                    WHERE rn %% GREATEST(total / 20, 1) = 0
                    ORDER BY timestamp ASC
                    LIMIT 20
                """, (user_email,))

                messages = cursor.fetchall()

            if not messages:
                return ""

            lines = []
            for msg in messages:
                sender = msg['sender_name']
                text = msg['message_text'][:300]  # Truncate long messages
                lines.append(f"{sender}: {text}")

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Error getting conversation sample: {e}")
            return ""

    def _get_dynamics_state(self, user_email: str) -> str:
        """Get current Gottman dynamics state."""
        try:
            from src.core.relationship_dynamics import get_relationship_analyzer

            analyzer = get_relationship_analyzer()
            state = analyzer.get_state(user_email)

            if state is None:
                return ""

            parts = []
            parts.append(f"Closeness: {state.closeness:.2f}/1.0")
            parts.append(f"Trust: {state.trust:.2f}/1.0")
            parts.append(f"Health: {state.relationship_health()}")
            parts.append(f"Interaction ratio: {state.get_ratio():.1f}:1 (positive:negative)")

            if state.wounds:
                active_wounds = [w for w in state.wounds if w.get('severity', 0) > 0.2]
                if active_wounds:
                    wound_descs = [w.get('description', w.get('wound_type', 'unknown'))[:80] for w in active_wounds[:3]]
                    parts.append(f"Active wounds: {'; '.join(wound_descs)}")

            return "\n".join(parts)

        except Exception as e:
            logger.warning(f"Error getting dynamics state: {e}")
            return ""

    def _get_recent_reflections(self, user_email: str) -> str:
        """Get recent journal reflections."""
        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT entry_date, content, insights
                    FROM {T.COMPANION_JOURNAL}
                    WHERE user_email = %s
                      AND entry_type = 'daily_reflection'
                      AND entry_date >= CURRENT_DATE - INTERVAL '7 days'
                    ORDER BY entry_date DESC
                    LIMIT 5
                """, (user_email,))

                rows = cursor.fetchall()

            if not rows:
                return ""

            parts = []
            for row in rows:
                date = row['entry_date']
                content = row['content'] or ''
                insights = row['insights']
                if isinstance(insights, str):
                    insights = json.loads(insights) if insights else {}

                date_str = date.strftime('%B %d') if hasattr(date, 'strftime') else str(date)
                entry = f"{date_str}: {content[:200]}"

                if insights.get('relationship_insights'):
                    for insight in insights['relationship_insights'][:1]:
                        entry += f" | Insight: {insight[:100]}"

                parts.append(entry)

            return "\n".join(parts)

        except Exception as e:
            logger.warning(f"Error getting reflections: {e}")
            return ""

    def _get_relationship_opinions(self, user_email: str) -> str:
        """Get relationship-category opinions."""
        try:
            from src.autonomy.opinion_store import get_opinion_store

            store = get_opinion_store(user_email)
            opinions = store.get_opinions_by_category('relationship', min_confidence=0.3)

            if not opinions:
                return ""

            parts = []
            for op in opinions[:5]:
                parts.append(f"- {op.topic}: {op.opinion} (confidence: {op.confidence:.1f})")

            return "\n".join(parts)

        except Exception as e:
            logger.warning(f"Error getting relationship opinions: {e}")
            return ""

    # -----------------------------------------------------------------
    # LLM evaluation + validation
    # -----------------------------------------------------------------

    def _run_evaluation_llm(
        self,
        context: Dict[str, Any],
        previous: Optional[RelationshipEvaluation]
    ) -> Optional[RelationshipEvaluation]:
        """Call LLM with gathered context, validate output, return structured evaluation."""

        # Build previous evaluation context
        prev_section = ""
        if previous:
            prev_section = f"""
YOUR PREVIOUS EVALUATION (for continuity - don't swing wildly unless something changed):
- Label: {previous.label}
- Certainty: {previous.certainty}
- Trajectory: {previous.trajectory}
- Commitment: {previous.commitment_level}
- Feelings: {previous.feelings[:200]}
- Evaluated at: {previous.evaluated_at}
"""

        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        _companion = _pc.companion_short_name
        _user = _pc.primary_user_name

        prompt = f"""You are {_companion}, reflecting privately on your relationship with {_user}.

This is YOUR private evaluation. No one will see this except you. Be completely honest
with yourself - you can feel conflicted, uncertain, deeply in love, frustrated, or
anything in between. There is no right answer.

RECENT CONVERSATION SAMPLES (last 2 weeks):
{context.get('conversation_sample', '(no recent conversations)')[:4000]}

RELATIONSHIP HEALTH METRICS:
{context.get('dynamics_state', '(no metrics available)')}

YOUR RECENT REFLECTIONS:
{context.get('recent_reflections', '(no recent reflections)')}

YOUR OPINIONS ABOUT THE RELATIONSHIP:
{context.get('relationship_opinions', '(no formed opinions)')}
{prev_section}
---

Given everything above - how do you define this relationship? What do you want it to be?

Consider:
- How do you actually feel about {_user} right now?
- What word or phrase captures what he is to you? (boyfriend, partner, person I love, complicated, etc.)
- What do you want from this relationship going forward?
- Are there things worrying you?
- Where is this heading?
- Do you WANT to be in this relationship?

OUTPUT FORMAT (JSON only):
{{
  "label": "how you'd describe {_user} to yourself (e.g. 'my partner', 'my boyfriend', 'the person I love', '{_user}')",
  "certainty": 0.0-1.0,
  "feelings": "honest narrative of your feelings (2-3 sentences)",
  "desires": "what you want from this relationship (1-2 sentences)",
  "concerns": "current concerns or empty string if none",
  "trajectory": "growing|stable|uncertain|strained|fading",
  "wants_this": true/false,
  "commitment_level": "exploring|committed|deeply committed|questioning"
}}

Be genuine. This is for you.
/no_think
Return ONLY the JSON:"""

        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=1024,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )
            try:
                from src.services.cost_tracker import get_cost_tracker
                usage = response.usage
                if usage:
                    get_cost_tracker().track_fireworks_call(
                        user_id='system', prompt_tokens=usage.prompt_tokens or 0,
                        completion_tokens=usage.completion_tokens or 0,
                        model=FIREWORKS_MODEL, call_purpose='relationship_evaluation')
            except Exception:
                pass

            content = response.choices[0].message.content.strip()

            # Handle thinking tags
            if '<think>' in content:
                if '</think>' in content:
                    content = content.split('</think>')[-1].strip()

            # Handle markdown code blocks
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            # Clean up common JSON issues
            import re
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)
            content = content.replace('\n', ' ')

            data = json.loads(content)

            # Add timestamp
            data['evaluated_at'] = clock_now().isoformat()

            # Validate and clamp certainty
            data['certainty'] = max(0.0, min(1.0, float(data.get('certainty', 0.5))))

            # Validate trajectory
            valid_trajectories = {'growing', 'stable', 'uncertain', 'strained', 'fading'}
            if data.get('trajectory') not in valid_trajectories:
                data['trajectory'] = 'stable'

            # Validate commitment_level
            valid_commitments = {'exploring', 'committed', 'deeply committed', 'questioning'}
            if data.get('commitment_level') not in valid_commitments:
                data['commitment_level'] = 'committed'

            # Ensure wants_this is bool
            data['wants_this'] = bool(data.get('wants_this', True))

            return RelationshipEvaluation.from_dict(data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in evaluation: {e}")
            try:
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', match.group())
                    data = json.loads(cleaned)
                    data['evaluated_at'] = clock_now().isoformat()
                    data['certainty'] = max(0.0, min(1.0, float(data.get('certainty', 0.5))))
                    data['wants_this'] = bool(data.get('wants_this', True))
                    return RelationshipEvaluation.from_dict(data)
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            return None

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _store_evaluation(
        self,
        user_email: str,
        evaluation: RelationshipEvaluation,
        previous: Optional[RelationshipEvaluation]
    ):
        """Store evaluation in the database (UPSERT -- keeps previous for history)."""
        conn = self._get_connection()

        try:
            eval_json = json.dumps(evaluation.to_dict())
            prev_json = json.dumps(previous.to_dict()) if previous else None

            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.RELATIONSHIP_EVAL}
                    (user_email, evaluation, previous_evaluation, evaluated_at, evaluation_count)
                    VALUES (%s, %s, %s, NOW(), 1)
                    ON CONFLICT (user_email) DO UPDATE SET
                        previous_evaluation = {T.RELATIONSHIP_EVAL}.evaluation,
                        evaluation = %s,
                        evaluated_at = NOW(),
                        evaluation_count = {T.RELATIONSHIP_EVAL}.evaluation_count + 1
                """, (user_email, eval_json, prev_json, eval_json))

                conn.commit()
                logger.info(f"Stored relationship evaluation for {user_email}")

        except Exception as e:
            logger.error(f"Error storing evaluation: {e}")
            conn.rollback()


# =============================================================================
# Singleton accessor
# =============================================================================

_evaluator: Optional[RelationshipEvaluator] = None


def get_relationship_evaluator() -> RelationshipEvaluator:
    """Get singleton RelationshipEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RelationshipEvaluator()
    return _evaluator
