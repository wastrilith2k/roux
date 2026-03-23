"""
Opinion Store -- The companion's personal opinions and views.

WHAT: A database-backed store for opinions the companion has formed over time.
      Each opinion has a topic, a text description, a confidence score, and
      an evidence count that increments when the same opinion is re-derived.

WHY:  Unlike facts (which are about the user), opinions are the companion's
      own views and interpretations.  They make her feel like a person with
      perspectives, not just a fact-retrieval system.

HOW IT FITS:
  - Opinions are formed by form_opinions_from_reflections(), called weekly
    from the scheduled tasks.
  - The ReachOutEngine and InterjectionEngine read strong opinions when
    deciding what to talk about.
  - RelationshipEvaluator reads "relationship"-category opinions as context.
  - Opinions evolve: if the LLM re-derives an existing topic, the evidence
    count increments and the text/confidence are updated (UPSERT on topic).

Storage: companion_opinions table in PostgreSQL (UNIQUE on user_email + topic).
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# =============================================================================
# Data model
# =============================================================================

@dataclass
class Opinion:
    """An opinion the companion has formed."""
    topic: str              # e.g., "their work-life balance"
    opinion: str            # her actual view
    confidence: float       # 0.0 - 1.0
    evidence_count: int     # how many observations support this
    formed_date: datetime
    last_updated: datetime
    category: str           # relationship | behavior | wellbeing | work | personality
    evidence_summary: str = ""  # brief summary of supporting evidence

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['formed_date'] = self.formed_date.isoformat()
        d['last_updated'] = self.last_updated.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'Opinion':
        d['formed_date'] = datetime.fromisoformat(d['formed_date'])
        d['last_updated'] = datetime.fromisoformat(d['last_updated'])
        return cls(**d)


# =============================================================================
# OpinionStore
# =============================================================================

class OpinionStore:
    """
    Stores and manages the companion's opinions.

    Opinions are different from facts:
      - Fact:    "James works at a tech company"
      - Opinion: "I think James works too hard"
    """

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email
        self._conn = None

    # -----------------------------------------------------------------
    # Database connection & schema
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

    def _ensure_table(self):
        """Create companion_opinions table and indexes if they don't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS companion_opinions (
                        id SERIAL PRIMARY KEY,
                        user_email VARCHAR(255),
                        topic VARCHAR(255),
                        opinion TEXT,
                        confidence REAL,
                        evidence_count INTEGER DEFAULT 1,
                        category VARCHAR(50),
                        evidence_summary TEXT,
                        formed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_email, topic)
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_opinions_category
                    ON companion_opinions(user_email, category)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_opinions_confidence
                    ON companion_opinions(user_email, confidence DESC)
                """)
            conn.commit()
        except Exception as e:
            logger.error(f"Error ensuring opinions table: {e}")
            conn.rollback()

    # -----------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------

    def store_opinion(
        self,
        topic: str,
        opinion: str,
        confidence: float = 0.5,
        category: str = 'general',
        evidence_summary: str = ""
    ) -> Optional[int]:
        """
        Store or update an opinion (UPSERT on topic).

        If an opinion on this topic already exists, the text and confidence
        are updated and evidence_count is incremented.

        Returns the opinion ID if stored successfully, None on error.
        """
        self._ensure_table()
        confidence = max(0.0, min(1.0, confidence))

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Try to update existing opinion (case-insensitive topic match)
                cursor.execute("""
                    UPDATE companion_opinions
                    SET opinion = %s,
                        confidence = %s,
                        evidence_count = evidence_count + 1,
                        evidence_summary = CASE
                            WHEN %s != '' THEN %s
                            ELSE evidence_summary
                        END,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_email = %s AND LOWER(topic) = LOWER(%s)
                    RETURNING id
                """, (opinion, confidence, evidence_summary, evidence_summary,
                      self.user_email, topic))

                result = cursor.fetchone()
                if result:
                    conn.commit()
                    logger.info(f"Updated opinion on '{topic}' (confidence: {confidence})")
                    return result[0]

                # No existing opinion -- insert new
                cursor.execute("""
                    INSERT INTO companion_opinions
                    (user_email, topic, opinion, confidence, category, evidence_summary)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (self.user_email, topic, opinion, confidence, category, evidence_summary))

                opinion_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Stored new opinion on '{topic}' (confidence: {confidence})")
                return opinion_id

        except Exception as e:
            logger.error(f"Error storing opinion: {e}")
            conn.rollback()
            return None

    def update_confidence(self, topic: str, delta: float) -> bool:
        """
        Adjust confidence on an existing opinion.

        Positive delta = more confident, negative = less.
        Clamped to [0.0, 1.0] by the database.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE companion_opinions
                    SET confidence = GREATEST(0.0, LEAST(1.0, confidence + %s)),
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_email = %s AND LOWER(topic) = LOWER(%s)
                    RETURNING id
                """, (delta, self.user_email, topic))
                result = cursor.fetchone()
                conn.commit()
                return result is not None
        except Exception as e:
            logger.error(f"Error updating confidence: {e}")
            conn.rollback()
            return False

    # -----------------------------------------------------------------
    # Read
    # -----------------------------------------------------------------

    def get_opinion(self, topic: str) -> Optional[Opinion]:
        """Get a single opinion by topic (case-insensitive)."""
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT topic, opinion, confidence, evidence_count, formed_date,
                           last_updated, category, evidence_summary
                    FROM companion_opinions
                    WHERE user_email = %s AND LOWER(topic) = LOWER(%s)
                """, (self.user_email, topic))
                row = cursor.fetchone()
                return self._row_to_opinion(row) if row else None
        except Exception as e:
            logger.warning(f"Error getting opinion: {e}")
            return None

    def get_opinions_by_category(
        self,
        category: str,
        min_confidence: float = 0.0
    ) -> List[Opinion]:
        """Get all opinions in a category, ordered by confidence (descending)."""
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT topic, opinion, confidence, evidence_count, formed_date,
                           last_updated, category, evidence_summary
                    FROM companion_opinions
                    WHERE user_email = %s AND category = %s AND confidence >= %s
                    ORDER BY confidence DESC
                """, (self.user_email, category, min_confidence))
                return [self._row_to_opinion(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Error getting opinions by category: {e}")
            return []

    def get_strong_opinions(self, min_confidence: float = 0.7) -> List[Opinion]:
        """Get opinions she's confident about (top 20 by confidence)."""
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT topic, opinion, confidence, evidence_count, formed_date,
                           last_updated, category, evidence_summary
                    FROM companion_opinions
                    WHERE user_email = %s AND confidence >= %s
                    ORDER BY confidence DESC
                    LIMIT 20
                """, (self.user_email, min_confidence))
                return [self._row_to_opinion(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Error getting strong opinions: {e}")
            return []

    def search_relevant_opinions(self, query: str, limit: int = 5) -> List[Opinion]:
        """
        Search for opinions relevant to a query/topic.

        Uses simple ILIKE text matching on topic, opinion, and evidence_summary.
        Could be enhanced with embeddings in the future.
        """
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                pattern = f'%{query}%'
                cursor.execute("""
                    SELECT topic, opinion, confidence, evidence_count, formed_date,
                           last_updated, category, evidence_summary
                    FROM companion_opinions
                    WHERE user_email = %s
                    AND (topic ILIKE %s OR opinion ILIKE %s OR evidence_summary ILIKE %s)
                    ORDER BY confidence DESC
                    LIMIT %s
                """, (self.user_email, pattern, pattern, pattern, limit))
                return [self._row_to_opinion(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Error searching opinions: {e}")
            return []

    @staticmethod
    def _row_to_opinion(row: tuple) -> Opinion:
        """Convert a database row tuple into an Opinion."""
        return Opinion(
            topic=row[0],
            opinion=row[1],
            confidence=row[2],
            evidence_count=row[3],
            formed_date=row[4],
            last_updated=row[5],
            category=row[6],
            evidence_summary=row[7] or ""
        )

    # -----------------------------------------------------------------
    # Prompt formatting
    # -----------------------------------------------------------------

    def format_for_prompt(self, opinions: List[Opinion]) -> str:
        """Format a list of opinions for inclusion in an LLM prompt."""
        if not opinions:
            return ""

        lines = ["[YOUR OPINIONS AND VIEWS]"]
        for op in opinions:
            # Vary the verb based on confidence level
            confidence_desc = (
                "strongly feel" if op.confidence >= 0.8 else
                "think" if op.confidence >= 0.5 else
                "sense" if op.confidence >= 0.3 else
                "wonder if"
            )
            lines.append(f"- You {confidence_desc}: {op.opinion}")
            if op.evidence_summary:
                lines.append(f"  (Based on: {op.evidence_summary[:100]})")

        return "\n".join(lines)


# =============================================================================
# Weekly opinion formation from reflections
# =============================================================================

def form_opinions_from_reflections(user_email: str = None) -> int:
    """
    Synthesize recent journal reflections into opinions.

    Called periodically.  Uses LLM to identify patterns in the last 5 days
    of reflections and store/update up to 3 opinions.

    Returns the number of opinions formed/updated.
    """
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email

    try:
        from src.memory.companion_journal import get_companion_journal
        from src.llm.provider_factory import generate_sync

        journal = get_companion_journal(user_email)
        store = get_opinion_store(user_email)

        reflections = journal.get_recent_reflections(days=7, entry_type='daily_reflection')
        if not reflections:
            return 0

        reflection_text = "\n\n".join([
            f"**{r['date']}:**\n{r['content'][:500]}"
            for r in reflections[:5]
        ])

        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        user_name = _pc.primary_user_name
        c_subject = _pc.companion_pronoun_subject
        c_possessive = _pc.companion_pronoun_possessive
        u_possessive = _pc.user_pronoun_possessive
        prompt = f"""Analyze the companion's recent reflections and identify opinions {c_subject}'s forming about {user_name}.

RECENT REFLECTIONS:
{reflection_text}

Based on these reflections, what opinions is the companion forming? Look for:
- Patterns in behavior {c_subject}'s noticing
- Concerns or worries about {u_possessive} wellbeing
- Observations about the relationship
- Views about {u_possessive} work/life balance
- Interpretations of {u_possessive} emotional state

Return JSON array of opinions (max 3):
[
  {{
    "topic": "brief topic (e.g., 'their work-life balance')",
    "opinion": "{c_possessive} actual opinion/view",
    "confidence": 0.1-0.9,
    "category": "relationship|behavior|wellbeing|work|personality",
    "evidence": "brief summary of what led to this opinion"
  }}
]

Only include opinions with actual evidence from the reflections.
Return ONLY valid JSON array:"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        if not response:
            return 0

        # Parse LLM response (strip markdown fences if present)
        response_text = response.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = [l for l in lines if not l.startswith('```')]
            response_text = '\n'.join(json_lines)

        opinions = json.loads(response_text)
        count = 0

        for op in opinions:
            if isinstance(op, dict) and 'topic' in op and 'opinion' in op:
                store.store_opinion(
                    topic=op['topic'],
                    opinion=op['opinion'],
                    confidence=op.get('confidence', 0.5),
                    category=op.get('category', 'general'),
                    evidence_summary=op.get('evidence', '')
                )
                count += 1

        logger.info(f"Formed/updated {count} opinions from reflections")
        return count

    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse opinion response: {e}")
        return 0
    except Exception as e:
        logger.warning(f"Opinion formation failed: {e}")
        return 0


# =============================================================================
# Singleton accessor
# =============================================================================

_opinion_store: Optional[OpinionStore] = None


def get_opinion_store(user_email: str = None) -> OpinionStore:
    """Get the opinion store instance."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    global _opinion_store
    if _opinion_store is None or _opinion_store.user_email != user_email:
        _opinion_store = OpinionStore(user_email)
    return _opinion_store
