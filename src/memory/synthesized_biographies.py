"""
Synthesized Biographies - Theme-grouped facts as coherent paragraphs.

WHAT: Groups extracted facts by theme (work, family, preferences, crisis, etc.),
synthesizes each group into a natural-language paragraph via LLM, and stores
the paragraphs with temporal-decay-aware importance scores. Paragraphs are
retrieved by context_builder for prompt injection.

WHY: A paragraph about "James's work situation" is far more useful to the
LLM than a scattered list of SPO triples like "works at Cavallo", "job
stress", "remote work." Synthesized paragraphs give the companion a
narrative understanding of each aspect of a person's life.

HOW it fits:
  - The biography refresh task (daily cron) calls refresh_all_biographies()
    to regenerate paragraphs from the latest facts.
  - context_builder.py calls get_biography_context() to inject the most
    important paragraphs into the LLM prompt.
  - Temporal decay ensures recent crises dominate over stale preferences.

Temporal Decay Formula:
    effective_importance = base_importance * 0.5^(age_days / half_life_days)

Half-lives by category:
    - crisis:    7 days   (recent crises stay top-of-mind)
    - event:    14 days   (interviews, appointments)
    - plan:     21 days   (future plans)
    - relationship / work: 30 days
    - detail:   60 days   (personal details)
    - preference: 90 days (likes, dislikes, habits)
    - identity: 365 days  (core biographical facts)
"""

import os
import json
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL
from src.database import tables as T

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# Category half-lives in days (how long until importance halves)
CATEGORY_HALF_LIVES = {
    'crisis': 7,        # Police, hospital, running away
    'event': 14,        # Interviews, appointments
    'plan': 21,         # Future plans (interview, move, etc.)
    'relationship': 30, # Relationship changes, emotions
    'work': 30,         # Job changes, work situations
    'detail': 60,       # Personal details
    'preference': 90,   # Likes, dislikes, habits
    'identity': 365,    # Core identity facts
    'default': 45,      # Default half-life
}

# Keywords that indicate crisis/urgent categories
CRISIS_KEYWORDS = [
    'police', 'hospital', 'emergency', 'ran away', 'missing',
    'arrested', 'crisis', 'accident', 'died', 'death', 'funeral'
]

EVENT_KEYWORDS = [
    'interview', 'appointment', 'meeting', 'scheduled', 'tomorrow',
    'today', 'job offer', 'hired', 'fired', 'quit'
]


@dataclass
class SynthesizedParagraph:
    """A synthesized paragraph from grouped facts."""
    id: int
    theme: str  # e.g., "work", "family", "crisis"
    subject: str  # Who it's about (James, Jesse, etc.)
    content: str  # The synthesized paragraph
    fact_ids: List[int]  # Source fact IDs
    base_importance: float  # Average importance of source facts
    created_at: datetime
    updated_at: datetime
    embedding: Optional[List[float]] = None


def classify_fact_category(fact: Dict[str, Any]) -> str:
    """
    Classify a fact into a temporal decay category.

    Uses predicate (category) and content analysis to determine
    which half-life to apply.
    """
    predicate = fact.get('predicate', '').lower()
    content = fact.get('object', '').lower()

    # Check for crisis indicators
    for keyword in CRISIS_KEYWORDS:
        if keyword in content:
            return 'crisis'

    # Check for event indicators
    for keyword in EVENT_KEYWORDS:
        if keyword in content:
            return 'event'

    # Map predicate to category
    predicate_mapping = {
        'plan': 'plan',
        'relationship': 'relationship',
        'work': 'work',
        'job': 'work',
        'preference': 'preference',
        'detail': 'detail',
        'identity': 'identity',
    }

    for key, category in predicate_mapping.items():
        if key in predicate:
            return category

    return 'default'


def calculate_decay_factor(age_days: float, category: str) -> float:
    """
    Calculate temporal decay factor based on age and category.

    Uses exponential decay: factor = 0.5 ^ (age / half_life)

    Returns value between 0 and 1 (1 = no decay, 0.5 = half-life reached)
    """
    half_life = CATEGORY_HALF_LIVES.get(category, CATEGORY_HALF_LIVES['default'])

    # Minimum decay factor of 0.1 (facts never completely disappear)
    decay = math.pow(0.5, age_days / half_life)
    return max(0.1, decay)


def calculate_effective_importance(
    fact: Dict[str, Any],
    reference_time: datetime = None
) -> float:
    """
    Calculate effective importance with temporal decay applied.

    Args:
        fact: Fact dict with 'importance', 'created_at', 'predicate', 'object'
        reference_time: Time to calculate decay from (default: now)

    Returns:
        Effective importance (0-10 scale, decayed)
    """
    if reference_time is None:
        reference_time = datetime.now(PST)

    base_importance = fact.get('importance', 5)
    created_at = fact.get('created_at')

    if created_at is None:
        return float(base_importance)

    # Ensure timezone awareness
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=PST)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=PST)

    # Calculate age in days
    age = reference_time - created_at
    age_days = age.total_seconds() / 86400

    # Get category and decay factor
    category = classify_fact_category(fact)
    decay_factor = calculate_decay_factor(age_days, category)

    effective = base_importance * decay_factor

    logger.debug(
        f"Fact importance: {base_importance} -> {effective:.1f} "
        f"(age={age_days:.1f}d, category={category}, decay={decay_factor:.2f})"
    )

    return effective


class SynthesizedBiographyStore:
    """
    Stores and retrieves synthesized biography paragraphs.

    Uses PostgreSQL for storage with optional embedding-based retrieval.
    """

    def __init__(self):
        self._conn = None

    def _get_connection(self):
        """Get database connection."""
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

    def store_paragraph(
        self,
        theme: str,
        subject: str,
        content: str,
        fact_ids: List[int],
        base_importance: float,
        embedding: List[float] = None,
        user_email: str = None
    ) -> Optional[int]:
        """Store or update a synthesized paragraph."""
        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                # Upsert: update if exists, insert if not
                cursor.execute(f"""
                    INSERT INTO {T.SYNTHESIZED_PARAGRAPHS}
                    (theme, subject, content, fact_ids, base_importance, embedding_vec, user_email, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (theme, subject, user_email)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        fact_ids = EXCLUDED.fact_ids,
                        base_importance = EXCLUDED.base_importance,
                        embedding_vec = EXCLUDED.embedding_vec,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (theme, subject, content, fact_ids, base_importance, embedding, user_email))

                result = cursor.fetchone()
                conn.commit()

                paragraph_id = result[0] if result else None
                logger.info(f"Stored paragraph: {theme}/{subject} (id={paragraph_id})")
                return paragraph_id

        except Exception as e:
            logger.error(f"Failed to store paragraph: {e}")
            conn.rollback()
            return None

    def get_paragraphs_for_subject(
        self,
        subject: str,
        min_importance: float = 3.0,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """Get all paragraphs about a subject, with temporal decay applied."""
        conn = self._get_connection()

        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT * FROM {T.SYNTHESIZED_PARAGRAPHS}
                    WHERE subject = %s
                    AND (user_email = %s OR user_email IS NULL)
                    ORDER BY base_importance DESC
                """, (subject, user_email))

                paragraphs = []
                now = datetime.now(PST)

                for row in cursor.fetchall():
                    # Apply temporal decay to base_importance
                    updated_at = row['updated_at']
                    if updated_at and updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=PST)

                    if updated_at:
                        age_days = (now - updated_at).total_seconds() / 86400
                        # Use theme to determine decay rate
                        category = row['theme'] if row['theme'] in CATEGORY_HALF_LIVES else 'default'
                        decay = calculate_decay_factor(age_days, category)
                        effective_importance = row['base_importance'] * decay
                    else:
                        effective_importance = row['base_importance']

                    if effective_importance >= min_importance:
                        para = dict(row)
                        para['effective_importance'] = effective_importance
                        paragraphs.append(para)

                # Sort by effective importance
                paragraphs.sort(key=lambda x: x['effective_importance'], reverse=True)
                return paragraphs

        except Exception as e:
            logger.error(f"Failed to get paragraphs for {subject}: {e}")
            return []

    def get_all_paragraphs(
        self,
        min_importance: float = 3.0,
        user_email: str = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all paragraphs with temporal decay applied."""
        conn = self._get_connection()

        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT * FROM {T.SYNTHESIZED_PARAGRAPHS}
                    WHERE (user_email = %s OR user_email IS NULL)
                    ORDER BY base_importance DESC
                    LIMIT %s
                """, (user_email, limit * 2))  # Get extra to filter after decay

                paragraphs = []
                now = datetime.now(PST)

                for row in cursor.fetchall():
                    updated_at = row['updated_at']
                    if updated_at and updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=PST)

                    if updated_at:
                        age_days = (now - updated_at).total_seconds() / 86400
                        category = row['theme'] if row['theme'] in CATEGORY_HALF_LIVES else 'default'
                        decay = calculate_decay_factor(age_days, category)
                        effective_importance = row['base_importance'] * decay
                    else:
                        effective_importance = row['base_importance']

                    if effective_importance >= min_importance:
                        para = dict(row)
                        para['effective_importance'] = effective_importance
                        paragraphs.append(para)

                # Sort and limit
                paragraphs.sort(key=lambda x: x['effective_importance'], reverse=True)
                return paragraphs[:limit]

        except Exception as e:
            logger.error(f"Failed to get all paragraphs: {e}")
            return []


# =============================================================================
# Biography synthesis pipeline
# =============================================================================

class BiographySynthesizer:
    """
    Synthesizes grouped facts into coherent paragraphs.

    Pipeline:
      1. Fetch all facts for a subject from fact_store.
      2. Group by theme using keyword matching on predicate/content.
      3. For each theme group with enough facts, call the LLM to generate
         a 2-4 sentence paragraph.
      4. Store the paragraph in synthesized_paragraphs with base_importance
         (average of source fact importances).

    The LLM prompt includes importance scores so it can prioritize the most
    significant facts when space is limited.
    """

    THEME_MAPPING = {
        'work': ['work', 'job', 'career', 'employment'],
        'family': ['family', 'children', 'kids', 'son', 'daughter', 'parent'],
        'relationship': ['relationship', 'partner', 'girlfriend', 'boyfriend', 'spouse'],
        'preferences': ['preference', 'likes', 'dislikes', 'hobby', 'interest'],
        'crisis': ['crisis', 'emergency', 'police', 'hospital'],
        'plans': ['plan', 'future', 'scheduled', 'appointment'],
        'personal': ['detail', 'personal', 'background'],
    }

    def __init__(self):
        self.store = SynthesizedBiographyStore()
        self._llm_client = None

    def _get_llm_client(self):
        """Get Fireworks LLM client."""
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ.get('FIREWORKS_API_KEY')
            )
        return self._llm_client

    def _classify_fact_theme(self, fact: Dict[str, Any]) -> str:
        """Classify a fact into a theme based on predicate and content."""
        predicate = fact.get('predicate', '').lower()
        content = fact.get('object', '').lower()

        for theme, keywords in self.THEME_MAPPING.items():
            for keyword in keywords:
                if keyword in predicate or keyword in content:
                    return theme

        return 'personal'  # Default theme

    def _group_facts_by_theme(
        self,
        facts: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group facts by theme."""
        groups = {}

        for fact in facts:
            theme = self._classify_fact_theme(fact)
            if theme not in groups:
                groups[theme] = []
            groups[theme].append(fact)

        return groups

    def _synthesize_paragraph(
        self,
        subject: str,
        theme: str,
        facts: List[Dict[str, Any]]
    ) -> str:
        """Use LLM to synthesize facts into a coherent paragraph."""
        client = self._get_llm_client()

        # Sort facts by effective importance (with decay)
        facts_with_importance = [
            (f, calculate_effective_importance(f))
            for f in facts
        ]
        facts_with_importance.sort(key=lambda x: x[1], reverse=True)

        # Format facts for LLM
        facts_text = "\n".join([
            f"- [{f[1]:.1f}] {f[0].get('object', '')}"
            for f in facts_with_importance[:15]  # Limit to top 15
        ])

        prompt = f"""/no_think
Synthesize these facts about {subject}'s {theme} into a coherent paragraph.

FACTS (sorted by importance, [score]):
{facts_text}

REQUIREMENTS:
1. Write a natural, conversational paragraph (2-4 sentences)
2. Prioritize higher-importance facts
3. Maintain factual accuracy - don't add information not in the facts
4. Use present tense for current states, past tense for past events
5. If facts seem contradictory, prefer the higher-scored one
6. Do NOT include any thinking, reasoning, or explanation - ONLY the paragraph

Write the paragraph now (just the paragraph, nothing else):"""

        try:
            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()

            try:
                from src.services.cost_tracker import get_cost_tracker
                usage = response.usage
                if usage:
                    get_cost_tracker().track_fireworks_call(
                        user_id='system', prompt_tokens=usage.prompt_tokens or 0,
                        completion_tokens=usage.completion_tokens or 0,
                        model=FIREWORKS_MODEL, call_purpose='biography_synthesis')
            except Exception:
                pass

            # Remove any thinking tags if present (handle both complete and incomplete)
            import re
            if '<think>' in content:
                # Try to remove complete thinking blocks first
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                # Also remove any incomplete thinking at the start (no closing tag)
                content = re.sub(r'^<think>.*', '', content, flags=re.DOTALL)
                content = content.strip()

            # Remove any remaining thinking content that starts with <think>
            if content.startswith('<think>'):
                # Find where actual content starts (after thinking or at end)
                close_idx = content.find('</think>')
                if close_idx != -1:
                    content = content[close_idx + 8:].strip()
                else:
                    # No closing tag, try to find where actual content starts
                    # Look for content after empty lines following thinking
                    lines = content.split('\n')
                    actual_start = 0
                    for i, line in enumerate(lines):
                        if not line.strip().startswith('<') and line.strip():
                            actual_start = i
                            break
                    content = '\n'.join(lines[actual_start:]).strip()

            return content if content else "No synthesis generated."

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            # Fallback: just join the top facts
            return " ".join([f[0].get('object', '') for f in facts_with_importance[:3]])

    def synthesize_for_subject(
        self,
        subject: str,
        user_email: str = None,
        min_facts_per_theme: int = 2
    ) -> List[SynthesizedParagraph]:
        """
        Synthesize all biography paragraphs for a subject.

        Args:
            subject: Who to synthesize for (e.g., "Alex", "Jesse")
            user_email: User context for facts
            min_facts_per_theme: Minimum facts needed to create a paragraph

        Returns:
            List of synthesized paragraphs
        """
        from src.memory.fact_store import get_fact_store

        fact_store = get_fact_store()

        # Get all facts for subject
        facts = fact_store.get_facts_for_subject(subject, limit=100)

        if not facts:
            logger.info(f"No facts found for {subject}")
            return []

        # Group by theme
        groups = self._group_facts_by_theme(facts)

        paragraphs = []

        for theme, theme_facts in groups.items():
            if len(theme_facts) < min_facts_per_theme:
                continue

            # Calculate average base importance
            base_importance = sum(f.get('importance', 5) for f in theme_facts) / len(theme_facts)

            # Synthesize paragraph
            content = self._synthesize_paragraph(subject, theme, theme_facts)

            if not content:
                continue

            # Store in database
            fact_ids = [f.get('id') for f in theme_facts if f.get('id')]

            paragraph_id = self.store.store_paragraph(
                theme=theme,
                subject=subject,
                content=content,
                fact_ids=fact_ids,
                base_importance=base_importance,
                user_email=user_email
            )

            if paragraph_id:
                paragraphs.append(SynthesizedParagraph(
                    id=paragraph_id,
                    theme=theme,
                    subject=subject,
                    content=content,
                    fact_ids=fact_ids,
                    base_importance=base_importance,
                    created_at=datetime.now(PST),
                    updated_at=datetime.now(PST)
                ))

        logger.info(f"Synthesized {len(paragraphs)} paragraphs for {subject}")
        return paragraphs

    def get_biography_context(
        self,
        user_email: str,
        subjects: List[str] = None,
        min_importance: float = 3.0,
        max_paragraphs: int = 10
    ) -> str:
        """
        Get formatted biography context for inclusion in prompt.

        This is the main interface for context_builder.

        Args:
            user_email: User context
            subjects: Specific subjects to include (default: all)
            min_importance: Minimum effective importance threshold
            max_paragraphs: Maximum paragraphs to include

        Returns:
            Formatted string for prompt inclusion
        """
        if subjects:
            paragraphs = []
            for subject in subjects:
                paragraphs.extend(
                    self.store.get_paragraphs_for_subject(
                        subject,
                        min_importance=min_importance,
                        user_email=user_email
                    )
                )
        else:
            paragraphs = self.store.get_all_paragraphs(
                min_importance=min_importance,
                user_email=user_email,
                limit=max_paragraphs * 2
            )

        if not paragraphs:
            return ""

        # Sort by effective importance and limit
        paragraphs.sort(key=lambda x: x.get('effective_importance', 0), reverse=True)
        paragraphs = paragraphs[:max_paragraphs]

        # Group by subject for formatting
        by_subject = {}
        for p in paragraphs:
            subj = p.get('subject', 'Unknown')
            if subj not in by_subject:
                by_subject[subj] = []
            by_subject[subj].append(p)

        # Format output — clearly label who each biography is ABOUT
        lines = ["[SYNTHESIZED BIOGRAPHICAL CONTEXT]"]
        lines.append("(Each section is about a SPECIFIC person. Do NOT mix up whose biography is whose.)")
        lines.append("")

        from src.config.persona_config import get_persona_config
        _companion_name = get_persona_config().companion_short_name.lower()
        for subject, subj_paragraphs in by_subject.items():
            if subject.lower() == _companion_name:
                lines.append(f"**About {subject} (you):**")
            elif subject.lower() == 'james':
                lines.append(f"**About {subject} (your partner):**")
            else:
                lines.append(f"**About {subject}:**")
            for p in subj_paragraphs:
                eff_imp = p.get('effective_importance', 0)
                theme = p.get('theme', 'unknown')
                lines.append(f"[{theme}, imp={eff_imp:.1f}] {p.get('content', '')}")
            lines.append("")

        return "\n".join(lines)


# Singleton instances
_store: Optional[SynthesizedBiographyStore] = None
_synthesizer: Optional[BiographySynthesizer] = None


def get_biography_store() -> SynthesizedBiographyStore:
    """Get singleton store instance."""
    global _store
    if _store is None:
        _store = SynthesizedBiographyStore()
    return _store


def get_biography_synthesizer() -> BiographySynthesizer:
    """Get singleton synthesizer instance."""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = BiographySynthesizer()
    return _synthesizer


def get_biography_context(user_email: str, max_paragraphs: int = 8) -> Optional[str]:
    """
    Main entry point - get biography context for prompt.

    Called by context_builder.py
    """
    try:
        synthesizer = get_biography_synthesizer()
        context = synthesizer.get_biography_context(
            user_email=user_email,
            max_paragraphs=max_paragraphs
        )
        return context if context else None
    except Exception as e:
        logger.error(f"Error getting biography context: {e}")
        return None


# Batch job for synthesis
def refresh_all_biographies(user_email: str = None):
    """
    Refresh all synthesized biographies.

    Call this periodically (daily) or when facts change significantly.
    """
    synthesizer = get_biography_synthesizer()

    # Get all unique subjects from facts
    from src.memory.fact_store import get_fact_store
    fact_store = get_fact_store()

    try:
        conn = fact_store._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT DISTINCT subject FROM {T.FACTS}
                WHERE archived_at IS NULL
                AND subject IS NOT NULL
            """)
            subjects = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get subjects: {e}")
        subjects = ['James', 'Jesse', 'Kyler']  # Fallback

    results = {}
    for subject in subjects:
        try:
            paragraphs = synthesizer.synthesize_for_subject(subject, user_email)
            results[subject] = {
                'success': True,
                'paragraphs': len(paragraphs)
            }
        except Exception as e:
            results[subject] = {
                'success': False,
                'error': str(e)
            }

    return results
