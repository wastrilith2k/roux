"""
Episodic-to-Semantic Consolidation Task — Bridge between episodic and semantic memory.

Cognitive science identifies the episodic→semantic consolidation pathway as
critical for human-like memory. Multiple experiences of the same recurring topic
should consolidate into a semantic fact (e.g. "user's job is a significant source
of stress").

This task:
1. Scans episodes for recurring topics (3+ occurrences in 30 days)
2. Clusters related episodes and extracts the recurring pattern
3. Uses LLM to generate a high-confidence semantic fact
4. Stores the fact with provenance linking back to source episodes
5. Updates episode patterns table for conversation guidance

Different from episode_learning_task (which extracts per-episode lessons),
this task synthesizes ACROSS episodes to find higher-order patterns.

Schedule: Daily at 2:00 AM Pacific (after reflections, before pruning)
Feature flag: COMPANION_EPISODIC_CONSOLIDATION_ENABLED (default: true)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo
import uuid

from src.celery_app import celery_app
from src.database import tables as T
from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

COMPANION_EPISODIC_CONSOLIDATION_ENABLED = os.environ.get(
    'COMPANION_EPISODIC_CONSOLIDATION_ENABLED', 'true'
).lower() == 'true'

# Minimum episode recurrence to trigger consolidation
MIN_RECURRENCE = 3
# Lookback window for episode scanning
LOOKBACK_DAYS = 30
# Max patterns to process per run
MAX_PATTERNS_PER_RUN = 10


def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


def _get_connection(user_email: str):
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        dbname=os.environ.get('POSTGRES_DB', 'companion'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', '')
    )
    if user_email:
        from src.database.schema_manager import set_search_path
        set_search_path(conn, user_email)
    return conn


def find_recurring_episode_topics(
    conn,
    lookback_days: int = LOOKBACK_DAYS,
    min_recurrence: int = MIN_RECURRENCE,
) -> List[Dict[str, Any]]:
    """
    Find episode topics that recur 3+ times in the lookback window.

    Returns grouped topics with episode details for consolidation.
    """
    from psycopg2.extras import RealDictCursor

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT
                    topic,
                    COUNT(*) as episode_count,
                    array_agg(DISTINCT emotional_state) FILTER (WHERE emotional_state IS NOT NULL) as emotions,
                    AVG(user_satisfaction) as avg_satisfaction,
                    MIN(started_at) as first_seen,
                    MAX(started_at) as last_seen,
                    array_agg(episode_id ORDER BY started_at) as episode_ids,
                    array_agg(DISTINCT resolution) FILTER (WHERE resolution IS NOT NULL) as resolutions,
                    array_agg(DISTINCT companion_approach) FILTER (WHERE companion_approach IS NOT NULL) as approaches
                FROM {T.EPISODES}
                WHERE topic IS NOT NULL
                AND topic != ''
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '{lookback_days} days'
                GROUP BY topic
                HAVING COUNT(*) >= {min_recurrence}
                ORDER BY COUNT(*) DESC
                LIMIT {MAX_PATTERNS_PER_RUN}
            """)
            return [dict(row) for row in cursor.fetchall()]

    except Exception as e:
        logger.error(f"Failed to find recurring episode topics: {e}")
        return []


def get_episode_messages_for_topic(conn, episode_ids: List[str], limit_per_episode: int = 10) -> List[Dict]:
    """Get sample messages from episodes for richer consolidation context."""
    from psycopg2.extras import RealDictCursor

    messages = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for ep_id in episode_ids[:5]:  # Sample from up to 5 episodes
                cursor.execute(f"""
                    SELECT m.sender_name, m.message_text, m.created_at
                    FROM {T.EPISODE_MESSAGES} em
                    JOIN {T.MESSAGES} m ON em.message_id = m.id
                    WHERE em.episode_id = %s
                    ORDER BY em.turn_number
                    LIMIT %s
                """, (str(ep_id), limit_per_episode))
                ep_msgs = cursor.fetchall()
                if ep_msgs:
                    messages.append({
                        'episode_id': str(ep_id),
                        'messages': [dict(m) for m in ep_msgs],
                    })
    except Exception as e:
        logger.debug(f"Could not get episode messages: {e}")

    return messages


def is_already_consolidated(conn, topic: str) -> bool:
    """Check if we've already created a consolidation fact for this topic."""
    from psycopg2.extras import RealDictCursor

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Check facts table for episodic_consolidation source
            cursor.execute(f"""
                SELECT id FROM {T.FACTS}
                WHERE source = 'episodic_consolidation'
                AND context ILIKE %s
                AND archived_at IS NULL
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '14 days'
            """, (f'%{topic[:40]}%',))
            return cursor.fetchone() is not None
    except Exception:
        return False


def consolidate_topic_to_fact(
    conn,
    topic_data: Dict[str, Any],
    episode_messages: List[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to synthesize a recurring episode topic into a semantic fact.

    Args:
        topic_data: Recurring topic with episode details
        episode_messages: Sample messages for richer context

    Returns:
        Dict with generated fact (subject, predicate, object) or None
    """
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    user_name = _pc.primary_user_name

    topic = topic_data['topic']
    count = topic_data['episode_count']
    emotions = topic_data.get('emotions') or []
    avg_sat = topic_data.get('avg_satisfaction')
    resolutions = topic_data.get('resolutions') or []
    approaches = topic_data.get('approaches') or []
    first_seen = topic_data.get('first_seen')
    last_seen = topic_data.get('last_seen')

    # Build message samples for context
    message_context = ""
    if episode_messages:
        samples = []
        for ep in episode_messages[:3]:
            msgs = ep['messages'][:5]
            sample = '\n'.join(
                f"  {m['sender_name']}: {(m['message_text'] or '')[:100]}"
                for m in msgs
            )
            samples.append(sample)
        if samples:
            message_context = f"\nSAMPLE CONVERSATIONS:\n{'---'.join(samples)}\n"

    prompt = f"""Analyze this recurring conversation pattern and generate a semantic fact.

RECURRING TOPIC: "{topic}"
Appeared in {count} separate conversations over the past {LOOKBACK_DAYS} days.
First seen: {first_seen}
Last seen: {last_seen}
Emotional states: {', '.join(str(e) for e in emotions[:5]) if emotions else 'varied'}
Average satisfaction: {f'{avg_sat:.1f}/1.0' if avg_sat is not None else 'unknown'}
Resolutions: {', '.join(str(r) for r in resolutions[:3]) if resolutions else 'varied'}
{message_context}
Based on this recurring pattern, generate:

1. A factual statement (subject | predicate | object format) that captures
   the underlying truth this pattern reveals. This should be a general
   insight, not a description of a single event.

2. A brief explanation of what this pattern means for the relationship.

3. An importance score (1-10, where 10 is critical identity-level knowledge).

Respond in JSON:
{{
  "subject": "entity name",
  "predicate": "relationship or attribute",
  "object": "value or description",
  "explanation": "what this pattern reveals",
  "importance": 7
}}

/no_think
Respond with JSON only, no markdown."""

    try:
        from src.config.models import FIREWORKS_DEFAULT_MODEL
        from src.llm.fireworks_client import fireworks_chat

        response = fireworks_chat(
            model=FIREWORKS_DEFAULT_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=300,
        )

        # Parse response
        content = response.strip()
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        if all(k in result for k in ['subject', 'predicate', 'object']):
            return {
                'subject': result['subject'],
                'predicate': result['predicate'],
                'object': result['object'],
                'explanation': result.get('explanation', ''),
                'importance': min(10, max(1, result.get('importance', 7))),
            }

    except Exception as e:
        logger.warning(f"LLM consolidation failed for topic '{topic}': {e}")

    return None


def store_consolidated_fact(
    conn,
    fact_data: Dict[str, Any],
    topic_data: Dict[str, Any],
) -> Optional[int]:
    """Store the consolidated fact with provenance metadata."""
    try:
        with conn.cursor() as cursor:
            context = (
                f"Consolidated from {topic_data['episode_count']} episodes about: "
                f"{topic_data['topic']}. {fact_data.get('explanation', '')}"
            )

            cursor.execute(f"""
                INSERT INTO {T.FACTS} (
                    subject, predicate, object, confidence, importance,
                    temporal, context, source, mention_count,
                    last_mentioned, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, 'recurring', %s,
                    'episodic_consolidation', %s, CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                RETURNING id
            """, (
                fact_data['subject'],
                fact_data['predicate'],
                fact_data['object'],
                0.85,  # High confidence for consolidated facts
                fact_data.get('importance', 7),
                context,
                topic_data['episode_count'],
            ))
            fact_id = cursor.fetchone()[0]
            conn.commit()
            return fact_id

    except Exception as e:
        logger.error(f"Failed to store consolidated fact: {e}")
        conn.rollback()
        return None


def update_episode_pattern(
    conn,
    topic_data: Dict[str, Any],
    fact_data: Dict[str, Any],
) -> None:
    """Create or update an episode pattern for this recurring topic."""
    try:
        with conn.cursor() as cursor:
            pattern_id = str(uuid.uuid4())
            cursor.execute(f"""
                INSERT INTO {T.EPISODE_PATTERNS} (
                    pattern_id, pattern_name, description,
                    topic_category, emotional_context,
                    episode_count, avg_satisfaction,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (pattern_id) DO UPDATE SET
                    episode_count = EXCLUDED.episode_count,
                    avg_satisfaction = EXCLUDED.avg_satisfaction,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                pattern_id,
                topic_data['topic'][:100],
                fact_data.get('explanation', ''),
                topic_data['topic'],
                ', '.join(str(e) for e in (topic_data.get('emotions') or [])[:3]),
                topic_data['episode_count'],
                topic_data.get('avg_satisfaction'),
            ))
            conn.commit()
    except Exception as e:
        logger.debug(f"Could not update episode pattern: {e}")
        conn.rollback()


@celery_app.task(
    name='tasks.episodic_consolidation.consolidate_episodes',
    bind=True,
    max_retries=1,
    soft_time_limit=240,
    time_limit=300
)
def consolidate_episodes(self, user_email: str = None):
    """
    Main task: scan episodes for recurring patterns and promote to semantic facts.

    Returns:
        Dict with consolidation results
    """
    if not COMPANION_EPISODIC_CONSOLIDATION_ENABLED:
        return {'status': 'disabled'}

    if user_email is None:
        user_email = _get_default_user_email()

    logger.info(f"Starting episodic consolidation for {user_email}")

    conn = _get_connection(user_email)
    results = {
        'status': 'completed',
        'topics_scanned': 0,
        'facts_created': 0,
        'already_consolidated': 0,
        'details': [],
    }

    try:
        # Find recurring topics
        recurring = find_recurring_episode_topics(conn)
        results['topics_scanned'] = len(recurring)

        if not recurring:
            logger.info("No recurring episode topics found")
            conn.close()
            return results

        for topic_data in recurring:
            topic = topic_data['topic']

            # Skip if already consolidated recently
            if is_already_consolidated(conn, topic):
                results['already_consolidated'] += 1
                continue

            # Get sample messages for richer context
            episode_ids = topic_data.get('episode_ids', [])
            episode_messages = get_episode_messages_for_topic(conn, episode_ids)

            # LLM consolidation
            fact_data = consolidate_topic_to_fact(conn, topic_data, episode_messages)

            if fact_data:
                # Store the fact
                fact_id = store_consolidated_fact(conn, fact_data, topic_data)

                if fact_id:
                    results['facts_created'] += 1
                    results['details'].append({
                        'topic': topic,
                        'episodes': topic_data['episode_count'],
                        'fact': f"{fact_data['subject']} {fact_data['predicate']} {fact_data['object']}",
                        'fact_id': fact_id,
                    })

                    # Update episode patterns table
                    update_episode_pattern(conn, topic_data, fact_data)

                    logger.info(
                        f"Consolidated '{topic}' ({topic_data['episode_count']} episodes) "
                        f"-> fact #{fact_id}: {fact_data['subject']} {fact_data['predicate']} {fact_data['object']}"
                    )

        conn.close()
        logger.info(f"Episodic consolidation complete: {results['facts_created']} facts from {results['topics_scanned']} topics")
        return results

    except Exception as e:
        logger.error(f"Episodic consolidation failed: {e}")
        try:
            conn.close()
        except Exception:
            pass
        raise
