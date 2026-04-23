"""
Strategy Tips - Experiential learning from conversation episodes.

Stores structured situational awareness extracted from episodes and
interaction outcomes. Tips are descriptive (what was observed), not
prescriptive (what to do). They give the companion social awareness
built from experience.

Three tip types:
- strategy: patterns that created genuine connection
- recovery: how to handle things when they go wrong
- insight: observed dynamics without prescription

Design principle: authenticity over optimization. Tips track both
user engagement AND companion authenticity. A "principled" tip
(low engagement, high authenticity) is valuable.
"""

import os
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from zoneinfo import ZoneInfo

from src.database import tables as T

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# Labels used when injecting tips into the prompt context
_TYPE_LABELS = {
    'strategy': 'Observation',
    'recovery': 'Recovery note',
    'insight': 'Pattern',
}


@dataclass
class StrategyTip:
    """A single experiential tip extracted from conversation history."""

    tip_type: str  # 'strategy', 'recovery', 'insight'
    content: str
    trigger_condition: str
    source_type: str  # 'episode', 'interaction_outcome', 'reflection'
    context_category: str = ''
    emotional_context: str = ''
    source_ids: List[str] = field(default_factory=list)
    times_used: int = 0
    times_helpful: int = 0
    times_unhelpful: int = 0
    authenticity_avg: float = 0.7
    engagement_avg: float = 0.0
    is_principled: bool = False
    id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None

    @property
    def value_score(self) -> float:
        """Compute value score: times_helpful / (times_used + 1)."""
        return self.times_helpful / (self.times_used + 1)

    def format_for_context(self) -> str:
        """Format this tip for injection into prompt context."""
        label = _TYPE_LABELS.get(self.tip_type, 'Note')
        n = len(self.source_ids)
        plural = 'situation' if n == 1 else 'situations'
        return f"- {label}: {self.content}\n  (Based on {n} similar {plural})"


class StrategyTipStore:
    """Storage and retrieval for strategy tips."""

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

    def save_tip(self, tip: StrategyTip) -> Optional[str]:
        """Save a strategy tip to the database. Returns the tip ID."""
        try:
            # Generate embedding for the tip content
            embedding = self._get_embedding(tip.content + ' ' + tip.trigger_condition)

            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.STRATEGY_TIPS} (
                        tip_type, content, trigger_condition,
                        context_category, emotional_context,
                        source_type, source_ids,
                        is_principled, authenticity_avg,
                        embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    tip.tip_type, tip.content, tip.trigger_condition,
                    tip.context_category, tip.emotional_context,
                    tip.source_type, tip.source_ids,
                    tip.is_principled, tip.authenticity_avg,
                    embedding,
                ))
                row = cursor.fetchone()
                conn.commit()
                tip_id = str(row[0]) if row else None
                logger.info(f"Saved {tip.tip_type} tip: {tip.content[:60]}...")
                return tip_id

        except Exception as e:
            logger.error(f"Failed to save strategy tip: {e}")
            if self._conn:
                self._conn.rollback()
            return None

    def retrieve_relevant_tips(
        self,
        query_text: str,
        context_category: str = None,
        emotional_context: str = None,
        limit: int = 3,
    ) -> List[StrategyTip]:
        """
        Retrieve relevant tips using hybrid scoring.

        Score = 0.5 * cosine_similarity + 0.3 * value_score
              + 0.1 * recency + 0.1 * authenticity
        """
        try:
            embedding = self._get_embedding(query_text)
            if not embedding:
                return []

            conn = self._get_connection()
            with conn.cursor() as cursor:
                # pgvector cosine distance: 1 - (embedding <=> query) = similarity
                # We select extra candidates and re-rank in Python for the hybrid score
                params = [embedding]
                where_clauses = ["retired_at IS NULL", "embedding IS NOT NULL"]

                if context_category:
                    where_clauses.append("(context_category = %s OR context_category IS NULL OR context_category = '')")
                    params.append(context_category)

                where_sql = " AND ".join(where_clauses)

                cursor.execute(f"""
                    SELECT id, tip_type, content, trigger_condition,
                           context_category, emotional_context,
                           source_type, source_ids,
                           times_used, times_helpful, times_unhelpful,
                           authenticity_avg, engagement_avg, is_principled,
                           created_at, updated_at,
                           1 - (embedding <=> %s::vector) AS cosine_sim
                    FROM {T.STRATEGY_TIPS}
                    WHERE {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, params + [embedding, limit * 3])

                rows = cursor.fetchall()

            if not rows:
                return []

            # Re-rank with hybrid scoring
            now = datetime.now(PST)
            scored = []
            for row in rows:
                tip = StrategyTip(
                    id=str(row[0]),
                    tip_type=row[1],
                    content=row[2],
                    trigger_condition=row[3],
                    context_category=row[4] or '',
                    emotional_context=row[5] or '',
                    source_type=row[6],
                    source_ids=row[7] or [],
                    times_used=row[8] or 0,
                    times_helpful=row[9] or 0,
                    times_unhelpful=row[10] or 0,
                    authenticity_avg=row[11] or 0.7,
                    engagement_avg=row[12] or 0.0,
                    is_principled=row[13] or False,
                    created_at=row[14],
                    updated_at=row[15],
                )
                cosine_sim = row[16] or 0.0

                # Recency score: 1.0 for today, decaying over 30 days
                age_days = (now - (tip.updated_at or tip.created_at or now).replace(
                    tzinfo=PST if (tip.updated_at or tip.created_at or now).tzinfo is None else (tip.updated_at or tip.created_at or now).tzinfo
                )).total_seconds() / 86400
                recency = max(0.0, 1.0 - (age_days / 30.0))

                score = (
                    0.5 * cosine_sim
                    + 0.3 * tip.value_score
                    + 0.1 * recency
                    + 0.1 * (tip.authenticity_avg or 0.7)
                )
                scored.append((score, tip))

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Ensure tip type diversity: at most 2 of same type
            result = []
            type_counts: Dict[str, int] = {}
            for score, tip in scored:
                tc = type_counts.get(tip.tip_type, 0)
                if tc < 2:
                    result.append(tip)
                    type_counts[tip.tip_type] = tc + 1
                if len(result) >= limit:
                    break

            return result

        except Exception as e:
            logger.error(f"Failed to retrieve strategy tips: {e}")
            return []

    def update_tip_outcome(
        self,
        tip_id: str,
        engagement: str,
        authenticity: float = None,
    ):
        """Update a tip's usage counters after it was active in a conversation."""
        try:
            conn = self._get_connection()
            helpful = engagement in ('enthusiastic', 'engaged')
            unhelpful = engagement in ('deflected', 'ignored')

            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.STRATEGY_TIPS}
                    SET times_used = times_used + 1,
                        times_helpful = times_helpful + CASE WHEN %s THEN 1 ELSE 0 END,
                        times_unhelpful = times_unhelpful + CASE WHEN %s THEN 1 ELSE 0 END,
                        authenticity_avg = CASE
                            WHEN %s IS NOT NULL THEN
                                (authenticity_avg * times_used + %s) / (times_used + 1)
                            ELSE authenticity_avg
                        END,
                        engagement_avg = CASE
                            WHEN engagement_avg = 0 AND times_used = 0 THEN
                                CASE WHEN %s THEN 1.0 WHEN %s THEN 0.0 ELSE 0.5 END
                            ELSE
                                (engagement_avg * times_used +
                                 CASE WHEN %s THEN 1.0 WHEN %s THEN 0.0 ELSE 0.5 END
                                ) / (times_used + 1)
                        END,
                        updated_at = NOW()
                    WHERE id = %s
                """, (
                    helpful, unhelpful,
                    authenticity, authenticity,
                    helpful, unhelpful,
                    helpful, unhelpful,
                    tip_id,
                ))
            conn.commit()

        except Exception as e:
            logger.error(f"Failed to update tip outcome: {e}")
            if self._conn:
                self._conn.rollback()

    def retire_tip(self, tip_id: str):
        """Soft-delete a tip by setting retired_at."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.STRATEGY_TIPS}
                    SET retired_at = NOW()
                    WHERE id = %s
                """, (tip_id,))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to retire tip: {e}")
            if self._conn:
                self._conn.rollback()

    def get_active_tip_count(self) -> int:
        """Count active (non-retired) tips."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {T.STRATEGY_TIPS}
                    WHERE retired_at IS NULL
                """)
                return cursor.fetchone()[0]
        except Exception:
            return 0

    def format_tips_for_context(self, tips: List[StrategyTip]) -> str:
        """Format a list of tips for injection into the prompt."""
        if not tips:
            return ""
        lines = ["[SITUATIONAL AWARENESS - from past similar conversations]"]
        for tip in tips:
            lines.append(tip.format_for_context())
        return "\n".join(lines)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate an embedding for tip content."""
        try:
            from src.memory.embeddings import generate_embedding
            return generate_embedding(text)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None


# Singleton
_store: Optional[StrategyTipStore] = None


def get_strategy_tip_store(user_email: str = None) -> StrategyTipStore:
    global _store
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    if _store is None or _store.user_email != user_email:
        _store = StrategyTipStore(user_email)
    return _store
