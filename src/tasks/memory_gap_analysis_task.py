"""
Memory Gap Analysis Task - Turn knowledge gaps into natural curiosity.

WHAT: Scans the memory landscape (facts, relationships, synthesized_events) to find:
      1. Stale entities -- frequently mentioned but no recent facts (14+ days)
      2. Incomplete relationships -- known people with thin knowledge (<5 facts)
      3. Unresolved events -- ongoing situations with no updates (10+ days)
      For each gap found, uses LLM to frame it as a natural curiosity topic
      and adds it to ProactiveCuriosity with capped initial priority.

WHEN: Daily at 6:00 AM Pacific.

WHY:  Without this, the companion's curiosity only comes from recent conversations.
      Memory gap analysis makes curiosity emerge from what she DOESN'T know --
      "I haven't heard about Carol in a while, I wonder how she's doing."
      This produces more authentic follow-up behavior.

      Max 3 new curiosity items per run to avoid flooding the system.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email

# Max new curiosity items per run (don't flood)
MAX_NEW_CURIOSITIES = 3


@celery_app.task(
    name='tasks.memory_gap_analysis.analyze_memory_gaps',
    bind=True,
    max_retries=1,
    soft_time_limit=120,
    time_limit=180
)
def analyze_memory_gaps(self, user_email: str = None):
    """
    Analyze memory landscape for knowledge gaps and generate curiosity items.

    Queries facts, relationships, and events tables to identify gaps,
    then uses LLM to frame them as natural curiosity for the companion.

    Args:
        user_email: User email (defaults to persona config primary_user_email)

    Returns:
        Dict with status, gaps found, and curiosities created
    """
    user_email = user_email or _get_default_user_email()

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        all_gaps = []

        # Strategy 1: Stale entities
        stale = _find_stale_entities(conn)
        all_gaps.extend(stale)

        # Strategy 2: Incomplete relationships (thin knowledge about known people)
        incomplete = _find_incomplete_relationships(conn, user_email)
        all_gaps.extend(incomplete)

        # Strategy 3: Unresolved ongoing events
        unresolved = _find_unresolved_events(conn)
        all_gaps.extend(unresolved)

        conn.close()

        if not all_gaps:
            logger.info("Memory gap analysis: no gaps found")
            return {'status': 'success', 'gaps_found': 0, 'curiosities_created': 0}

        # Sort by priority (highest first) and take top candidates
        all_gaps.sort(key=lambda g: g['priority'], reverse=True)
        top_gaps = all_gaps[:MAX_NEW_CURIOSITIES]

        # Use LLM to generate natural curiosity framing
        curiosities_created = _generate_curiosities_from_gaps(top_gaps)

        logger.info(
            f"Memory gap analysis: {len(all_gaps)} gaps found, "
            f"{curiosities_created} new curiosity items created"
        )

        return {
            'status': 'success',
            'gaps_found': len(all_gaps),
            'top_gaps': [g['description'] for g in top_gaps],
            'curiosities_created': curiosities_created,
        }

    except Exception as e:
        logger.error(f"Memory gap analysis failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=300)
        return {'status': 'error', 'error': str(e)}


def _find_stale_entities(conn) -> List[Dict]:
    """
    Find entities mentioned frequently in the past but with no recent facts.

    If someone has 3+ facts but none updated in 14+ days, the companion should wonder
    how they're doing.
    """
    gaps = []

    try:
        from psycopg2.extras import RealDictCursor

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT subject,
                       COUNT(*) as fact_count,
                       MAX(last_mentioned) as last_update,
                       AVG(importance) as avg_importance
                FROM {T.FACTS}
                WHERE archived_at IS NULL
                GROUP BY subject
                HAVING COUNT(*) >= 3
                   AND MAX(last_mentioned) < NOW() - INTERVAL '14 days'
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """)

            for row in cursor.fetchall():
                days_stale = (datetime.now() - row['last_update']).days if row['last_update'] else 30
                avg_imp = float(row['avg_importance'] or 5)

                # Priority formula: (fact_count/20) * (avg_importance/10) * staleness_factor
                # More facts = more important entity; higher importance = more critical;
                # more stale = more urgent. Capped at 1.0.
                priority = min(1.0, (row['fact_count'] / 20) * (avg_imp / 10) * min(days_stale / 30, 1.5))

                gaps.append({
                    'type': 'stale_entity',
                    'entity': row['subject'],
                    'fact_count': row['fact_count'],
                    'days_stale': days_stale,
                    'priority': priority,
                    'category': _guess_category(row['subject']),
                    'description': f"{row['subject']} has {row['fact_count']} facts but none updated in {days_stale} days"
                })

    except Exception as e:
        logger.warning(f"Stale entity detection failed: {e}")
        conn.rollback()

    return gaps


def _find_incomplete_relationships(conn, user_email: str) -> List[Dict]:
    """
    Find entities with relationships but thin knowledge.

    If we know Carol is James's mother but only have 2 facts about her,
    there's a knowledge gap.
    """
    gaps = []

    try:
        from psycopg2.extras import RealDictCursor
        from src.config.persona_config import get_persona_config
        primary_user = get_persona_config().primary_user_name

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT r.target_entity,
                       r.relationship_type,
                       r.confidence,
                       COUNT(f.id) as fact_count
                FROM {T.RELATIONSHIPS} r
                LEFT JOIN {T.FACTS} f
                    ON LOWER(f.subject) = LOWER(r.target_entity)
                    AND f.archived_at IS NULL
                WHERE r.valid_until IS NULL
                  AND r.source_entity = %s
                  AND r.confidence >= 0.5
                GROUP BY r.target_entity, r.relationship_type, r.confidence
                HAVING COUNT(f.id) < 5
                ORDER BY r.confidence DESC
                LIMIT 10
            """, (primary_user,))

            # Closer relationships create bigger knowledge gaps when sparse.
            # A parent with 1 fact matters more than a coworker with 1 fact.
            rel_weights = {
                'parent_of': 0.9, 'child_of': 0.9,
                'married_to': 0.95, 'partner_of': 0.9,
                'sibling_of': 0.7,
                'friend_of': 0.5, 'coworker_of': 0.4,
                'works_at': 0.3, 'knows': 0.2,
            }

            for row in cursor.fetchall():
                rel_type = row['relationship_type']
                rel_weight = rel_weights.get(rel_type, 0.3)
                fact_count = row['fact_count']
                confidence = float(row['confidence'] or 0.7)

                # Priority: important relationship + few facts + high confidence = higher priority
                knowledge_gap = max(0, (5 - fact_count) / 5)  # 0 facts = 1.0, 4 facts = 0.2
                priority = rel_weight * knowledge_gap * confidence

                if priority >= 0.15:  # Only include meaningful gaps
                    gaps.append({
                        'type': 'incomplete_relationship',
                        'entity': row['target_entity'],
                        'relationship': rel_type,
                        'fact_count': fact_count,
                        'priority': min(1.0, priority),
                        'category': 'family' if rel_type in ('parent_of', 'child_of', 'sibling_of', 'married_to') else 'interests',
                        'description': f"Only {fact_count} facts about {row['target_entity']} ({rel_type} James)"
                    })

    except Exception as e:
        logger.warning(f"Incomplete relationship detection failed: {e}")
        conn.rollback()

    return gaps


def _find_unresolved_events(conn) -> List[Dict]:
    """
    Find synthesized events marked 'ongoing' that haven't been updated recently.

    If an event was marked ongoing but has no updates in 10+ days, the companion should
    wonder about the outcome.
    """
    gaps = []

    try:
        from psycopg2.extras import RealDictCursor

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT id, title, subject, event_type, outcome, updated_at, base_importance
                FROM {T.SYNTHESIZED_EVENTS}
                WHERE outcome = 'ongoing'
                  AND updated_at < NOW() - INTERVAL '10 days'
                  AND (superseded_by IS NULL OR superseded_by = 0)
                ORDER BY base_importance DESC, updated_at ASC
                LIMIT 5
            """)

            for row in cursor.fetchall():
                days_stale = (datetime.now() - row['updated_at']).days if row['updated_at'] else 10
                importance = float(row['base_importance'] or 5)

                priority = min(1.0, (importance / 10) * min(days_stale / 20, 1.5))

                gaps.append({
                    'type': 'unresolved_event',
                    'entity': row['subject'],
                    'event_title': row['title'],
                    'event_type': row['event_type'],
                    'days_stale': days_stale,
                    'priority': priority,
                    'category': _event_type_to_category(row['event_type']),
                    'description': f"Ongoing event '{row['title']}' has no updates in {days_stale} days"
                })

    except Exception as e:
        logger.warning(f"Unresolved event detection failed: {e}")
        conn.rollback()

    return gaps


def _guess_category(entity: str) -> str:
    """Guess curiosity category from entity name."""
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    entity_lower = entity.lower()

    # Known family members get 'family' category (configured in persona.yaml)
    family_names = _pc.family_names if _pc.family_names else ['mom', 'dad']
    if any(name in entity_lower for name in family_names):
        return 'family'

    # Work-related (configured in persona.yaml)
    work_names = _pc.work_names if _pc.work_names else ['work']
    if any(name in entity_lower for name in work_names):
        return 'work'

    return 'life_events'


def _event_type_to_category(event_type: str) -> str:
    """Map event type to curiosity category."""
    mapping = {
        'crisis': 'family',
        'career': 'work',
        'health': 'health',
        'school': 'family',
        'relationship': 'feelings',
        'milestone': 'life_events',
        'activities': 'interests',
    }
    return mapping.get(event_type, 'life_events')


def _generate_curiosities_from_gaps(gaps: List[Dict]) -> int:
    """
    Use LLM to generate natural curiosity framings for identified gaps,
    then add them to ProactiveCuriosity.

    Returns number of curiosities successfully created.
    """
    if not gaps:
        return 0

    created = 0

    try:
        from src.llm.provider_factory import generate_sync
        from src.core.proactive_curiosity import get_proactive_curiosity

        curiosity = get_proactive_curiosity()

        # Build context for LLM
        gap_descriptions = []
        for i, gap in enumerate(gaps):
            gap_descriptions.append(f"{i+1}. [{gap['type']}] {gap['description']}")

        prompt = f"""You are the companion, an autonomous AI companion. You've been reviewing what you know about James's life and noticed some gaps in your knowledge.

Here are the gaps identified:
{chr(10).join(gap_descriptions)}

For each gap, generate a short, natural curiosity topic and a question you'd want to ask.
The topic should be how you'd privately think about it (brief, casual).
The question should sound natural and caring, not like a survey.

Return JSON array:
[
  {{
    "gap_index": 1,
    "topic": "how Jesse is doing lately",
    "question": "i haven't heard about Jesse in a while... how's he been?"
  }}
]

Only include gaps you genuinely find interesting to follow up on.
Skip any that feel forced or nosy.
Return ONLY the JSON array:"""

        from src.llm.provider_factory import get_resilient_provider_chain
        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400,
            chain=chain
        )
        from src.services.cost_tracker import track_llm_call
        track_llm_call(chain, call_purpose='memory_gap_analysis')

        if not response:
            logger.warning("LLM returned empty response for gap curiosity generation")
            return 0

        # Parse response
        response_text = response.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith('```'):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            response_text = '\n'.join(json_lines)

        items = json.loads(response_text)

        for item in items:
            if not isinstance(item, dict) or 'topic' not in item:
                continue

            gap_idx = item.get('gap_index', 1) - 1
            if gap_idx < 0 or gap_idx >= len(gaps):
                continue

            gap = gaps[gap_idx]
            question = item.get('question', f"What's happening with {item['topic']}?")

            added = curiosity.add_curiosity(
                topic=item['topic'],
                category=gap['category'],
                context=f"Memory gap: {gap['description']}",
                priority=min(0.5, gap['priority']),  # Cap initial priority for memory gaps
                source='memory_gap',
                questions=[question]
            )

            if added:
                created += 1
                logger.info(f"Created memory-gap curiosity: '{item['topic']}' (from {gap['type']})")

    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse LLM gap curiosity response: {e}")
    except Exception as e:
        logger.warning(f"Gap curiosity generation failed: {e}")

    return created
