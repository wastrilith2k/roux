"""
Retrieval Feedback - Track and learn from retrieval quality over time.

Stores per-message records of what the retrieval agent planned and executed,
then backfills engagement/resonance outcomes from interaction tracking.
Aggregated stats feed back into the retrieval agent's planning prompts so
it can learn which sources and query patterns actually produce good responses.
"""

import os
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.database import tables as T

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFeedback:
    """A record of what was retrieved for a given message and how it performed."""

    message_id: int
    retrieval_plan: Dict[str, Any]
    sources_used: List[str]
    queries_run: List[str]
    memories_retrieved: List[Dict[str, Any]]
    active_tip_ids: List[str] = field(default_factory=list)
    outcome_engagement: Optional[str] = None   # e.g. 'enthusiastic', 'engaged', 'deflected'
    outcome_resonance: Optional[float] = None  # 0.0–1.0
    id: Optional[int] = None
    created_at: Optional[datetime] = None


def format_source_performance(stats: Dict[str, Dict[str, Any]]) -> str:
    """
    Format per-source performance stats for injection into the retrieval agent prompt.

    Args:
        stats: mapping of source_name -> {'uses': int, 'engagement_rate': float, 'avg_resonance': float}

    Returns:
        Human-readable summary string, or "" if stats is empty.
    """
    if not stats:
        return ""

    lines = ["[SOURCE PERFORMANCE - based on recent history]"]
    for source, data in sorted(stats.items()):
        uses = data.get('uses', 0)
        eng = data.get('engagement_rate', 0.0)
        res = data.get('avg_resonance', 0.0)
        eng_pct = f"{eng * 100:.0f}%"
        res_pct = f"{res * 100:.0f}%"
        lines.append(f"  {source}: {uses} uses | engagement {eng_pct} | resonance {res_pct}")

    return "\n".join(lines)


def format_retrieval_insights(patterns: List[Dict[str, Any]]) -> str:
    """
    Format query pattern insights for injection into the retrieval agent prompt.

    Args:
        patterns: list of {'pattern': str, 'engagement_rate': float, 'count': int}

    Returns:
        Human-readable summary string, or "" if patterns is empty.
    """
    if not patterns:
        return ""

    lines = ["[QUERY PATTERN INSIGHTS - what kinds of retrieval work best]"]
    for p in patterns:
        pattern = p.get('pattern', '')
        eng = p.get('engagement_rate', 0.0)
        count = p.get('count', 0)
        eng_pct = f"{eng * 100:.0f}%"
        lines.append(f"  '{pattern}': {eng_pct} engagement ({count} samples)")

    return "\n".join(lines)


class RetrievalFeedbackStore:
    """Storage and aggregation for retrieval feedback records."""

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email
        self._conn = None

    def _get_connection(self):
        import psycopg2
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
            try:
                from src.database.schema_manager import set_search_path
                set_search_path(self._conn, self.user_email)
            except Exception:
                pass
        return self._conn

    def record_retrieval(self, feedback: RetrievalFeedback) -> Optional[int]:
        """
        Save a retrieval feedback record to the database.

        Returns the new record ID, or None on failure.
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.RETRIEVAL_FEEDBACK} (
                        message_id,
                        retrieval_plan,
                        sources_used,
                        queries_run,
                        memories_retrieved,
                        active_tip_ids
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    feedback.message_id,
                    json.dumps(feedback.retrieval_plan),
                    feedback.sources_used,
                    feedback.queries_run,
                    json.dumps(feedback.memories_retrieved),
                    feedback.active_tip_ids,
                ))
                row = cursor.fetchone()
                conn.commit()
                record_id = row[0] if row else None
                logger.info(f"Recorded retrieval feedback for message {feedback.message_id}")
                return record_id

        except Exception as e:
            logger.error(f"Failed to record retrieval feedback: {e}")
            if self._conn:
                self._conn.rollback()
            return None

    def backfill_outcome(
        self,
        message_id: int,
        engagement: str,
        resonance: float,
    ) -> List[str]:
        """
        Update the engagement/resonance outcome for a previously recorded retrieval.

        Returns the active_tip_ids that were active during this retrieval
        (so the caller can update strategy tip stats).
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.RETRIEVAL_FEEDBACK}
                    SET outcome_engagement = %s,
                        outcome_resonance = %s
                    WHERE message_id = %s
                    RETURNING active_tip_ids
                """, (engagement, resonance, message_id))
                row = cursor.fetchone()
                conn.commit()
                if row:
                    return row[0] or []
                return []

        except Exception as e:
            logger.error(f"Failed to backfill retrieval outcome for message {message_id}: {e}")
            if self._conn:
                self._conn.rollback()
            return []

    def get_source_performance(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate per-source engagement rates over the last N days.

        Uses SQL unnest(sources_used) to expand the array and join with outcomes.

        Returns:
            {source_name: {'uses': int, 'engagement_rate': float, 'avg_resonance': float}}
        """
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT
                        source,
                        COUNT(*) AS uses,
                        AVG(CASE
                            WHEN outcome_engagement IN ('enthusiastic', 'engaged') THEN 1.0
                            WHEN outcome_engagement IN ('deflected', 'ignored') THEN 0.0
                            ELSE 0.5
                        END) AS engagement_rate,
                        AVG(COALESCE(outcome_resonance, 0.0)) AS avg_resonance
                    FROM {T.RETRIEVAL_FEEDBACK},
                         UNNEST(sources_used) AS source
                    WHERE created_at >= NOW() - INTERVAL '{days} days'
                      AND outcome_engagement IS NOT NULL
                    GROUP BY source
                    ORDER BY engagement_rate DESC
                """)
                rows = cursor.fetchall()

            result = {}
            for row in rows:
                source, uses, eng_rate, avg_res = row
                result[source] = {
                    'uses': int(uses),
                    'engagement_rate': float(eng_rate or 0.0),
                    'avg_resonance': float(avg_res or 0.0),
                }
            return result

        except Exception as e:
            logger.error(f"Failed to get source performance: {e}")
            return {}


# Singleton
_store: Optional[RetrievalFeedbackStore] = None


def get_retrieval_feedback_store(user_email: str = None) -> RetrievalFeedbackStore:
    global _store
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    if _store is None or _store.user_email != user_email:
        _store = RetrievalFeedbackStore(user_email)
    return _store
