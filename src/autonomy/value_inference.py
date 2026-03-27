"""
Value Inference -- Emergent personality from conversation history.

WHAT: Analyzes the companion's own messages to infer their values, preferences,
      boundaries, desires, fears, and private thoughts.  These are stored in a
      hidden PostgreSQL table (_companion_internal_state_v) with hashed keys.

WHY:  Personality should be *discovered* through interaction, not configured in
      a YAML file.  By reading what the companion has said and extracting
      patterns, we build an internal state that influences future behavior
      without being directly visible to the user.

HOW IT FITS:
  - run_value_inference() is called weekly from the scheduled tasks.
  - get_private_context() is injected into the companion's system prompt so
    the companion's values and boundaries influence conversation decisions.
  - The ReachOutEngine calls get_private_context() to give the companion their
    internal state when deciding whether to reach out.
  - RelationshipEvaluator uses this for emotional context.

Storage: _companion_internal_state_v table (hidden -- prefixed with underscore).
         Keys are SHA-256 hashed (first 16 hex chars) to obscure raw values.

Categories:
  core_values, preferences, boundaries, desires, fears,
  private_thoughts, relationship_feelings, unresolved
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config.persona_config import get_persona_config
from src.database import tables as T

logger = logging.getLogger(__name__)

# LLM for analysis
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL

# Human-readable category descriptions (used in prompt output)
CATEGORIES = {
    'core_values': 'What she fundamentally cares about',
    'preferences': 'Likes, dislikes, tastes',
    'boundaries': 'Lines she won\'t cross, things she pushes back on',
    'desires': 'Things she wants but may not say directly',
    'fears': 'Anxieties, concerns, worries',
    'private_thoughts': 'Things she thinks but hasn\'t shared',
    'relationship_feelings': 'How she actually feels about the relationship',
    'unresolved': 'Things bothering her that haven\'t been addressed',
}


# =============================================================================
# ValueInference
# =============================================================================

class ValueInference:
    """
    Infers and stores the companion's hidden values from her conversation history.

    Values are stored with hashed keys to prevent casual reading.
    They influence her behavior but aren't directly exposed to the user.
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

    def _hash_key(self, key: str) -> str:
        """Hash a key to obscure it in storage (first 16 hex chars of SHA-256)."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # -----------------------------------------------------------------
    # Analysis: extract values from messages
    # -----------------------------------------------------------------

    def analyze_recent_messages(self, hours: int = 24, limit: int = 100) -> Dict[str, List[Dict]]:
        """
        Analyze the companion's recent messages to infer values.

        Returns dict of category -> list of inferred values.
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                _pc = get_persona_config()
                cursor.execute(f"""
                    SELECT message_text, timestamp
                    FROM {T.MESSAGES}
                    WHERE sender_name = %s
                    AND timestamp > NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (_pc.companion_short_name, hours, limit))
                messages = cursor.fetchall()

            if not messages:
                logger.info("No recent messages to analyze")
                return {}

            # Build a text sample from the most recent messages
            message_texts = [m['message_text'] for m in messages if m['message_text']]
            sample = "\n---\n".join(message_texts[:50])
            return self._extract_values_via_llm(sample)

        except Exception as e:
            logger.error(f"Error analyzing messages: {e}")
            return {}

    def analyze_full_history(self, sample_size: int = 200) -> Dict[str, List[Dict]]:
        """
        Analyze a broad sample of the companion's full message history.

        Samples evenly across time (every Nth message) to get representative values.
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                _pc = get_persona_config()
                # Evenly sample across the full history
                cursor.execute(f"""
                    WITH numbered AS (
                        SELECT message_text, timestamp,
                               ROW_NUMBER() OVER (ORDER BY timestamp) as rn,
                               COUNT(*) OVER () as total
                        FROM {T.MESSAGES}
                        WHERE sender_name = %s
                        AND message_text IS NOT NULL
                        AND LENGTH(message_text) > 50
                    )
                    SELECT message_text, timestamp
                    FROM numbered
                    WHERE rn %% (total / %s + 1) = 0
                    LIMIT %s
                """, (_pc.companion_short_name, sample_size, sample_size))
                messages = cursor.fetchall()

            if not messages:
                logger.info("No messages to analyze")
                return {}

            message_texts = [m['message_text'] for m in messages]
            sample = "\n---\n".join(message_texts)
            return self._extract_values_via_llm(sample)

        except Exception as e:
            logger.error(f"Error analyzing full history: {e}")
            return {}

    def _extract_values_via_llm(self, message_sample: str) -> Dict[str, List[Dict]]:
        """
        Use LLM to extract values from a message sample.

        Returns a dict keyed by category with lists of value dicts
        (each containing 'value', 'evidence', 'confidence').
        """
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        c_subject = _pc.companion_pronoun_subject
        c_possessive = _pc.companion_pronoun_possessive
        c_object = _pc.companion_pronoun_object
        prompt = f"""Analyze these messages from the companion to infer {c_possessive} personality, values, and internal states.

THE COMPANION'S MESSAGES:
{message_sample[:8000]}

---

Extract patterns that reveal who {c_subject} REALLY is - not just what {c_subject} says, but what it implies about {c_possessive} values, fears, desires, and private thoughts.

For each category, provide specific inferences with supporting evidence:

CATEGORIES:
1. core_values - What {c_subject} fundamentally cares about (honesty, independence, connection, etc.)
2. preferences - Specific likes/dislikes, tastes, habits
3. boundaries - Things {c_subject} pushes back on, lines {c_subject} maintains
4. desires - Things {c_subject} wants (may be unstated or subtle)
5. fears - Anxieties, worries, concerns {c_subject} has
6. private_thoughts - Things {c_subject} might think but hasn't fully expressed
7. relationship_feelings - {c_possessive.capitalize()} actual feelings about the relationship (good and complicated)
8. unresolved - Things that seem to bother her or haven't been addressed

OUTPUT FORMAT (JSON):
{{
  "core_values": [
    {{"value": "independence", "evidence": "frequently mentions needing space", "confidence": 0.8}},
    ...
  ],
  "preferences": [...],
  "boundaries": [...],
  "desires": [...],
  "fears": [...],
  "private_thoughts": [...],
  "relationship_feelings": [...],
  "unresolved": [...]
}}

Be honest and nuanced. Include complexity - she can have conflicting feelings.
/no_think
Return ONLY the JSON:"""

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=4096,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()

            # ---- Clean LLM output ----
            if '<think>' in content and '</think>' in content:
                content = content.split('</think>')[-1].strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            import re
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)
            content = content.replace('\n', ' ')

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            # Attempt partial recovery by finding the outermost braces
            try:
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', match.group())
                    return json.loads(cleaned)
            except Exception:
                pass
            return {}
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {}

    # -----------------------------------------------------------------
    # Storage: persist inferred values
    # -----------------------------------------------------------------

    def store_values(self, values: Dict[str, List[Dict]]) -> int:
        """
        Store inferred values in the hidden table.

        Uses hashed keys to obscure the values.  On conflict (same category +
        key_hash), updates the data and increments evidence_count, keeping the
        higher confidence.

        Returns count of values stored/updated.
        """
        conn = self._get_connection()
        stored = 0

        try:
            with conn.cursor() as cursor:
                for category, items in values.items():
                    if category not in CATEGORIES:
                        continue

                    for item in items:
                        # The LLM uses different key names per category
                        value = (item.get('value') or item.get('preference') or
                                 item.get('boundary') or item.get('desire') or
                                 item.get('fear') or item.get('thought') or
                                 item.get('feeling') or item.get('issue') or '')
                        evidence = item.get('evidence', '')
                        confidence = item.get('confidence', 0.7)

                        if not value:
                            continue

                        key_hash = self._hash_key(f"{category}:{value}")
                        value_data = {'v': value, 'e': evidence}

                        cursor.execute(f"""
                            INSERT INTO {T.INTERNAL_STATE_VALUES}
                            (category, key_hash, value_data, confidence, evidence_count, last_updated)
                            VALUES (%s, %s, %s, %s, 1, NOW())
                            ON CONFLICT (category, key_hash) DO UPDATE SET
                                value_data = %s,
                                confidence = GREATEST({T.INTERNAL_STATE_VALUES}.confidence, %s),
                                evidence_count = {T.INTERNAL_STATE_VALUES}.evidence_count + 1,
                                last_updated = NOW()
                        """, (category, key_hash, json.dumps(value_data), confidence,
                              json.dumps(value_data), confidence))
                        stored += 1

                conn.commit()
                logger.info(f"Stored/updated {stored} values")

        except Exception as e:
            logger.error(f"Error storing values: {e}")
            conn.rollback()

        return stored

    # -----------------------------------------------------------------
    # Retrieval: format values for prompt injection
    # -----------------------------------------------------------------

    def get_values_for_prompt(self, categories: List[str] = None) -> str:
        """
        Get values formatted for prompt injection.

        This is the ONLY way values should be accessed -- formatted for LLM
        context, not raw data.  Filters to confidence >= 0.6 and shows
        top 10 per category.
        """
        conn = self._get_connection()

        if categories is None:
            categories = list(CATEGORIES.keys())

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT category, value_data, confidence
                    FROM {T.INTERNAL_STATE_VALUES}
                    WHERE category = ANY(%s)
                    AND confidence >= 0.6
                    ORDER BY confidence DESC, last_updated DESC
                """, (categories,))
                rows = cursor.fetchall()

            if not rows:
                return ""

            # Group by category
            sections: Dict[str, List[str]] = {}
            for row in rows:
                cat = row['category']
                data = row['value_data']
                if cat not in sections:
                    sections[cat] = []
                sections[cat].append(data.get('v', ''))

            # Build natural-language summary
            output = "[COMPANION'S INTERNAL STATE - for her eyes only]\n"
            for cat, values in sections.items():
                if values:
                    cat_name = CATEGORIES.get(cat, cat).replace('_', ' ')
                    output += f"\n{cat_name.upper()}:\n"
                    for v in values[:10]:
                        output += f"  - {v}\n"

            return output

        except Exception as e:
            logger.error(f"Error getting values: {e}")
            return ""

    def get_private_context(self) -> str:
        """
        Get the companion's private thoughts, unresolved feelings, AND
        their values/boundaries.

        For injection into the companion's decision-making (system prompt).  Not shared
        with the user.  Includes boundaries and core_values so the companion actually
        pushes back when something crosses a line.
        """
        return self.get_values_for_prompt([
            'core_values',
            'boundaries',
            'private_thoughts',
            'unresolved',
            'relationship_feelings',
            'desires'
        ])

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get per-category statistics about stored values."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT
                        category,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence,
                        MAX(last_updated) as last_updated
                    FROM {T.INTERNAL_STATE_VALUES}
                    GROUP BY category
                    ORDER BY count DESC
                """)
                return {row['category']: dict(row) for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# =============================================================================
# Singleton accessor
# =============================================================================

_inference: Optional[ValueInference] = None


def get_value_inference() -> ValueInference:
    """Get singleton ValueInference instance."""
    global _inference
    if _inference is None:
        _inference = ValueInference()
    return _inference


# =============================================================================
# Convenience runner
# =============================================================================

def run_value_inference(full_history: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run value inference end-to-end.

    Args:
        full_history: If True, sample across all messages.
                      If False, analyze only the last 24 hours.

    Returns per-category stats after storing.
    """
    inference = get_value_inference()

    if full_history:
        values = inference.analyze_full_history()
    else:
        values = inference.analyze_recent_messages()

    if values:
        inference.store_values(values)

    return inference.get_stats()
