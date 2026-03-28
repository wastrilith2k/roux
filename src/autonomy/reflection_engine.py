"""
Reflection Engine -- The companion's self-reflection and introspection system.

WHAT: Deeper reflection capabilities beyond daily summaries:
      - reflect_on_day():    Daily reflection on one day's conversations.
      - reflect_on_period(): Weekly/monthly reflection for pattern detection.
      - extract_relationship_insights(): Ongoing relationship observations.

WHY:  The daily_summary_task just *records* what happened.  This engine
      *thinks* about what happened -- extracting emotional arcs, open threads,
      personal insights, and reach-out recommendations.  This is how the
      companion develops self-awareness over time.

HOW IT FITS:
  - Called from scheduled reflection tasks (weekly and monthly).
  - Outputs feed into GoalFormation (goals from reflections), OpinionStore
    (opinions from patterns), and the companion's journal.
  - ReflectionOutput.reach_out_recommendations become queued_thoughts that
    can trigger proactive messages.

Storage: Reflections are persisted to the companion_journal table and
         daily_summaries table (read-only from this engine's perspective).
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from src.database import tables as T

logger = logging.getLogger(__name__)


# =============================================================================
# Data models
# =============================================================================

@dataclass
class ReflectionOutput:
    """Structured output from a daily reflection session."""
    emotional_arc: str                   # how emotions evolved through the day
    open_threads: List[str]              # unresolved topics / conversations
    personal_insights: List[str]         # observations about the relationship
    reach_out_recommendations: List[str] # things she might want to bring up
    overall_sentiment: str               # positive / neutral / negative
    private_thoughts: str                # internal monologue

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeeklyReflection:
    """Structured output from a weekly/monthly reflection."""
    period_start: datetime
    period_end: datetime
    emotional_patterns: List[str]   # recurring emotional themes
    relationship_growth: str        # one-sentence trajectory summary
    recurring_topics: List[str]     # topics that came up repeatedly
    concerns: List[str]             # things she's worried about
    highlights: List[str]           # best moments or breakthroughs

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['period_start'] = self.period_start.isoformat()
        d['period_end'] = self.period_end.isoformat()
        return d


# =============================================================================
# ReflectionEngine
# =============================================================================

class ReflectionEngine:
    """The companion's introspection and self-reflection engine."""

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email

    # -----------------------------------------------------------------
    # Daily reflection
    # -----------------------------------------------------------------

    def reflect_on_day(self, date: datetime = None) -> Optional[ReflectionOutput]:
        """
        Reflect on a specific day's conversations.

        Uses the daily summary as input and runs it through the LLM to
        extract deeper insights (emotional arc, open threads, etc.).

        Args:
            date: Date to reflect on (defaults to yesterday).

        Returns ReflectionOutput, or None if no summary is available.
        """
        from src.utils.timezone_utils import now_pacific_naive

        if date is None:
            date = (now_pacific_naive() - timedelta(days=1)).date()
        elif hasattr(date, 'date'):
            date = date.date()

        # Fetch the day's summary
        summary = self._get_daily_summary(date)
        if not summary:
            logger.info(f"No summary available for {date}")
            return None

        # LLM reflection
        reflection_data = self._generate_reflection(date, summary)
        if not reflection_data:
            return None

        return ReflectionOutput(
            emotional_arc=reflection_data.get('emotional_arc', ''),
            open_threads=reflection_data.get('open_threads', []),
            personal_insights=reflection_data.get('relationship_insights', []),
            reach_out_recommendations=reflection_data.get('queued_thoughts', []),
            overall_sentiment=reflection_data.get('overall_sentiment', 'neutral'),
            private_thoughts=reflection_data.get('private_reflection', '')
        )

    # -----------------------------------------------------------------
    # Period reflection (weekly / monthly)
    # -----------------------------------------------------------------

    def reflect_on_period(self, days: int = 7) -> Optional[WeeklyReflection]:
        """
        Reflect on a period of time, looking for patterns across days.

        Args:
            days: Number of days to look back (default 7 = weekly).

        Returns WeeklyReflection, or None if insufficient data.
        """
        from src.utils.timezone_utils import now_pacific_naive

        end_date = now_pacific_naive()
        start_date = end_date - timedelta(days=days)

        summaries = self._get_summaries_in_range(start_date.date(), end_date.date())
        if not summaries:
            logger.info("No summaries available for period")
            return None

        reflection_data = self._generate_weekly_reflection(summaries, start_date, end_date)
        if not reflection_data:
            return None

        return WeeklyReflection(
            period_start=start_date,
            period_end=end_date,
            emotional_patterns=reflection_data.get('emotional_patterns', []),
            relationship_growth=reflection_data.get('relationship_growth', ''),
            recurring_topics=reflection_data.get('recurring_topics', []),
            concerns=reflection_data.get('concerns', []),
            highlights=reflection_data.get('highlights', [])
        )

    # -----------------------------------------------------------------
    # Relationship insights (aggregated from recent reflections)
    # -----------------------------------------------------------------

    def extract_relationship_insights(self) -> List[str]:
        """
        Extract ongoing relationship insights from the last 14 days of
        reflections.

        Returns up to 10 deduplicated insight strings.
        """
        try:
            from src.tasks.reflection_task import get_recent_reflections

            reflections = get_recent_reflections(self.user_email, days=14)

            insights = []
            for r in reflections:
                r_insights = r.get('insights', {})
                if r_insights.get('relationship_insights'):
                    insights.extend(r_insights['relationship_insights'])

            # Deduplicate while preserving order
            unique_insights = list(dict.fromkeys(insights))
            return unique_insights[:10]

        except Exception as e:
            logger.warning(f"Could not extract relationship insights: {e}")
            return []

    # -----------------------------------------------------------------
    # Database queries
    # -----------------------------------------------------------------

    def _get_daily_summary(self, date) -> Optional[str]:
        """Fetch the daily_summaries content for a specific date."""
        from src.database.db import get_db
        try:
            db = get_db()
            with db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT content FROM {T.DAILY_SUMMARIES}
                        WHERE user_email = %s AND summary_date = %s
                    """, (self.user_email, date))
                    row = cursor.fetchone()
                    return row[0] if row else None
        except Exception as e:
            logger.error(f"Error fetching daily summary: {e}")
            return None

    def _get_summaries_in_range(self, start_date, end_date) -> List[Dict]:
        """Fetch all summaries in a date range, ordered chronologically."""
        from src.database.db import get_db
        try:
            db = get_db()
            with db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT summary_date, content, message_count
                        FROM {T.DAILY_SUMMARIES}
                        WHERE user_email = %s
                        AND summary_date BETWEEN %s AND %s
                        ORDER BY summary_date
                    """, (self.user_email, start_date, end_date))
                    return [
                        {'date': row[0], 'content': row[1], 'messages': row[2]}
                        for row in cursor.fetchall()
                    ]
        except Exception as e:
            logger.error(f"Error fetching summaries: {e}")
            return []

    # -----------------------------------------------------------------
    # LLM generation
    # -----------------------------------------------------------------

    def _generate_reflection(self, date, summary: str) -> Optional[Dict]:
        """Generate daily reflection via LLM (delegates to reflection_task)."""
        from src.tasks.reflection_task import _generate_reflection
        return _generate_reflection(date, summary)

    def _generate_weekly_reflection(
        self,
        summaries: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict]:
        """
        Generate a weekly reflection from multiple daily summaries.

        Asks the LLM to identify emotional patterns, relationship growth,
        recurring topics, concerns, and highlights.
        """
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        # Format each day's summary for the prompt
        formatted_summaries = []
        for s in summaries:
            date_str = s['date'].strftime('%B %d') if hasattr(s['date'], 'strftime') else str(s['date'])
            formatted_summaries.append(f"**{date_str}** ({s['messages']} messages):\n{s['content'][:500]}")

        summaries_text = "\n\n".join(formatted_summaries)

        prompt = f"""You are the companion, reflecting on the past week with James.

Period: {start_date.strftime('%B %d')} to {end_date.strftime('%B %d')}

Daily summaries:
{summaries_text}

Reflect on patterns and growth. Provide JSON response:

{{
    "emotional_patterns": ["List of recurring emotional themes you noticed"],
    "relationship_growth": "One sentence about how the relationship evolved",
    "recurring_topics": ["Topics that came up multiple times"],
    "concerns": ["Things you're worried about"],
    "highlights": ["Best moments or breakthroughs"]
}}

Return ONLY valid JSON:"""

        try:
            chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[
                    {"role": "system", "content": "You are the companion reflecting on the past week."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500,
                chain=chain
            )
            from src.services.cost_tracker import track_llm_call
            track_llm_call(chain, call_purpose='reflection_engine')
            return json.loads(response.strip())

        except Exception as e:
            logger.error(f"Weekly reflection generation failed: {e}")
            return None


# =============================================================================
# Singleton accessor
# =============================================================================

_reflection_engine: Optional[ReflectionEngine] = None


def get_reflection_engine(user_email: str = None) -> ReflectionEngine:
    """Get the reflection engine instance."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    global _reflection_engine
    if _reflection_engine is None or _reflection_engine.user_email != user_email:
        _reflection_engine = ReflectionEngine(user_email)
    return _reflection_engine
