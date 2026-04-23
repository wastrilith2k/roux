"""
Sleep-Time Consolidation Task - Background memory maintenance during idle periods.

Inspired by Letta/MemGPT's "sleep-time compute" concept: during idle periods
(when the user hasn't messaged for 2+ hours), the companion runs background cognitive
processes that consolidate, link, and improve her memory.

What it does:
1. Merge duplicate/overlapping facts (same subject+predicate, similar objects)
2. Promote recurring episodic patterns to high-confidence semantic facts
3. Strengthen weak fact network links based on co-occurrence
4. Pre-generate conversation starters for the reach-out engine
5. Detect and surface stale facts that may need updating

This runs as a Celery task triggered every 3 hours. It checks whether the
user has been idle for 2+ hours before doing any work.

Schedule: Every 3 hours via Celery beat
Feature flag: COMPANION_SLEEP_CONSOLIDATION_ENABLED (default: true)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from src.celery_app import celery_app
from src.database import tables as T
from src.core.clock import now as clock_now

def _safe_days_ago(ts) -> Optional[int]:
    """Calculate days since a timestamp, handling naive/aware mismatches."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        from zoneinfo import ZoneInfo
        ts = ts.replace(tzinfo=ZoneInfo('America/Los_Angeles'))
    return (clock_now() - ts).days

logger = logging.getLogger(__name__)

COMPANION_SLEEP_CONSOLIDATION_ENABLED = os.environ.get(
    'COMPANION_SLEEP_CONSOLIDATION_ENABLED', 'true'
).lower() == 'true'

# Minimum idle time before running consolidation (hours)
MIN_IDLE_HOURS = 2.0

# Similarity threshold for merging overlapping facts
MERGE_SIMILARITY_THRESHOLD = 0.80

# Minimum episode recurrence to promote to semantic fact
MIN_EPISODE_RECURRENCE = 3

# Max conversation starters to pre-generate
MAX_STARTERS = 5


def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


def _get_connection(user_email: str):
    """Get a database connection with correct schema."""
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


def _is_user_idle(conn, min_hours: float = MIN_IDLE_HOURS) -> bool:
    """Check if the user has been idle (no messages) for min_hours."""
    from psycopg2.extras import RealDictCursor

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT MAX(created_at) as last_msg
                FROM {T.MESSAGES}
                WHERE role = 'user'
            """)
            row = cursor.fetchone()
            if not row or not row['last_msg']:
                return True  # No messages at all -> treat as idle

            last_msg = row['last_msg']
            # Ensure timezone-aware comparison (DB returns naive timestamps)
            if last_msg.tzinfo is None:
                from zoneinfo import ZoneInfo
                last_msg = last_msg.replace(tzinfo=ZoneInfo('America/Los_Angeles'))
            hours_since = (clock_now() - last_msg).total_seconds() / 3600.0
            return hours_since >= min_hours
    except Exception as e:
        logger.warning(f"Could not check idle status: {e}")
        return False


def merge_overlapping_facts(conn, dry_run: bool = False) -> Dict[str, Any]:
    """
    Find and merge facts with same subject+predicate and similar objects.

    When multiple facts exist for the same subject+predicate pair with
    similar objects, keep the one with higher confidence/importance and
    archive the other.

    Returns:
        Dict with merge counts and details
    """
    from psycopg2.extras import RealDictCursor
    from difflib import SequenceMatcher

    merged = 0
    details = []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Find subject+predicate pairs with multiple active facts
            cursor.execute(f"""
                SELECT subject, predicate, COUNT(*) as cnt
                FROM {T.FACTS}
                WHERE archived_at IS NULL
                GROUP BY subject, predicate
                HAVING COUNT(*) > 1
                ORDER BY cnt DESC
                LIMIT 100
            """)
            pairs = cursor.fetchall()

            for pair in pairs:
                subj, pred = pair['subject'], pair['predicate']

                # Get all facts for this pair
                cursor.execute(f"""
                    SELECT id, object, confidence, importance, mention_count,
                           last_mentioned, retrieval_count, created_at
                    FROM {T.FACTS}
                    WHERE subject = %s AND predicate = %s
                    AND archived_at IS NULL
                    ORDER BY importance DESC, confidence DESC, mention_count DESC
                """, (subj, pred))
                facts = cursor.fetchall()

                # Compare each pair for similarity
                to_archive = set()
                for i, f1 in enumerate(facts):
                    if f1['id'] in to_archive:
                        continue
                    for j, f2 in enumerate(facts):
                        if j <= i or f2['id'] in to_archive:
                            continue

                        obj1 = (f1['object'] or '').lower().strip()
                        obj2 = (f2['object'] or '').lower().strip()

                        if not obj1 or not obj2:
                            continue

                        similarity = SequenceMatcher(None, obj1, obj2).ratio()
                        if similarity >= MERGE_SIMILARITY_THRESHOLD:
                            # Keep f1 (higher importance/confidence), archive f2
                            to_archive.add(f2['id'])

                            # Transfer mention_count and retrieval_count to survivor
                            if not dry_run:
                                transfer_mentions = f2['mention_count'] or 0
                                transfer_retrievals = f2.get('retrieval_count') or 0
                                cursor.execute(f"""
                                    UPDATE {T.FACTS}
                                    SET mention_count = mention_count + %s,
                                        retrieval_count = COALESCE(retrieval_count, 0) + %s,
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = %s
                                """, (transfer_mentions, transfer_retrievals, f1['id']))

                            details.append({
                                'kept': f1['id'],
                                'archived': f2['id'],
                                'subject': subj,
                                'similarity': round(similarity, 2),
                                'kept_obj': f1['object'][:50],
                                'archived_obj': f2['object'][:50],
                            })

                if to_archive and not dry_run:
                    for fid in to_archive:
                        cursor.execute(f"""
                            UPDATE {T.FACTS}
                            SET archived_at = CURRENT_TIMESTAMP,
                                invalid_at = CURRENT_TIMESTAMP,
                                archive_reason = 'sleep_consolidation_merge'
                            WHERE id = %s
                        """, (fid,))
                    merged += len(to_archive)

            if not dry_run:
                conn.commit()

    except Exception as e:
        logger.error(f"Fact merging failed: {e}")
        conn.rollback()

    return {'merged': merged, 'details': details[:20]}


def promote_episodic_patterns(conn, dry_run: bool = False) -> Dict[str, Any]:
    """
    Scan episodes for recurring themes and promote to semantic facts.

    If the same topic appears in 3+ episodes, generate a high-confidence
    semantic fact summarizing the pattern.
    """
    from psycopg2.extras import RealDictCursor

    promoted = 0
    details = []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Find recurring episode topics (3+ occurrences in last 30 days)
            cursor.execute(f"""
                SELECT topic, COUNT(*) as occurrences,
                       array_agg(DISTINCT emotional_state) as emotions,
                       AVG(satisfaction) as avg_satisfaction
                FROM {T.EPISODES}
                WHERE topic IS NOT NULL
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
                GROUP BY topic
                HAVING COUNT(*) >= %s
                ORDER BY COUNT(*) DESC
                LIMIT 20
            """, (MIN_EPISODE_RECURRENCE,))
            recurring = cursor.fetchall()

            if not recurring:
                return {'promoted': 0, 'details': []}

            for pattern in recurring:
                topic = pattern['topic']
                count = pattern['occurrences']

                # Check if we already have a promoted fact for this topic
                cursor.execute(f"""
                    SELECT id FROM {T.FACTS}
                    WHERE source = 'episodic_promotion'
                    AND context ILIKE %s
                    AND archived_at IS NULL
                """, (f'%{topic[:30]}%',))

                if cursor.fetchone():
                    continue  # Already promoted

                # Generate the semantic fact via LLM
                emotions = [e for e in (pattern['emotions'] or []) if e]
                avg_sat = pattern['avg_satisfaction'] or 0.5

                try:
                    from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

                    prompt = f"""Based on recurring conversation episodes, generate a single factual statement.

Topic that recurred {count} times in the last 30 days: "{topic}"
Emotional states observed: {', '.join(emotions[:5]) if emotions else 'mixed'}
Average satisfaction: {avg_sat:.1f}/1.0

Generate a concise, factual statement about this recurring pattern.
Format: Subject | Predicate | Object

Respond with ONLY the three parts separated by |."""

                    chain = get_resilient_provider_chain()
                    response = generate_sync(
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=0.1,
                        max_tokens=100,
                        chain=chain,
                    )
                    parts = response.strip().split('|')
                    if len(parts) >= 3:
                        subj = parts[0].strip()
                        pred = parts[1].strip()
                        obj = parts[2].strip()

                        if not dry_run:
                            cursor.execute(f"""
                                INSERT INTO {T.FACTS} (
                                    subject, predicate, object, confidence, importance,
                                    temporal, context, source, mention_count,
                                    last_mentioned, created_at, updated_at
                                ) VALUES (
                                    %s, %s, %s, 0.85, 7, 'recurring', %s,
                                    'episodic_promotion', %s, CURRENT_TIMESTAMP,
                                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                                )
                            """, (
                                subj, pred, obj,
                                f"Promoted from {count} episodes about: {topic}",
                                count,
                            ))
                            promoted += 1

                        details.append({
                            'topic': topic,
                            'occurrences': count,
                            'fact': f"{subj} {pred} {obj}",
                        })

                except Exception as e:
                    logger.warning(f"LLM promotion failed for topic '{topic}': {e}")
                    continue

            if not dry_run:
                conn.commit()

    except Exception as e:
        logger.error(f"Episodic promotion failed: {e}")
        conn.rollback()

    return {'promoted': promoted, 'details': details}


def detect_stale_facts(conn) -> List[Dict[str, Any]]:
    """
    Find facts that may be outdated and should be re-verified.

    Stale fact indicators:
    - High importance but never retrieved (possibly wrong/outdated)
    - Created long ago, never reinforced, but still active
    - Temporal='current' but very old
    """
    from psycopg2.extras import RealDictCursor

    stale = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Important facts that have never been retrieved
            cursor.execute(f"""
                SELECT id, subject, predicate, object, importance, confidence,
                       created_at, mention_count
                FROM {T.FACTS}
                WHERE archived_at IS NULL
                AND importance >= 6
                AND (retrieval_count IS NULL OR retrieval_count = 0)
                AND created_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
                ORDER BY importance DESC
                LIMIT 20
            """)
            for row in cursor.fetchall():
                stale.append({
                    'id': row['id'],
                    'reason': 'important_but_never_retrieved',
                    'fact': f"{row['subject']} {row['predicate']} {row['object']}"[:80],
                    'importance': row['importance'],
                    'age_days': _safe_days_ago(row['created_at']),
                })

            # "Current" facts that are very old (might not be current anymore)
            cursor.execute(f"""
                SELECT id, subject, predicate, object, importance, created_at
                FROM {T.FACTS}
                WHERE archived_at IS NULL
                AND temporal = 'current'
                AND mention_count <= 1
                AND created_at < CURRENT_TIMESTAMP - INTERVAL '90 days'
                ORDER BY created_at ASC
                LIMIT 20
            """)
            for row in cursor.fetchall():
                stale.append({
                    'id': row['id'],
                    'reason': 'current_but_very_old',
                    'fact': f"{row['subject']} {row['predicate']} {row['object']}"[:80],
                    'importance': row['importance'],
                    'age_days': _safe_days_ago(row['created_at']),
                })

    except Exception as e:
        logger.warning(f"Stale fact detection failed: {e}")

    return stale[:30]


def generate_conversation_starters(conn) -> List[Dict[str, Any]]:
    """
    Pre-generate conversation starters based on recent context.

    Generates starters from:
    - Unresolved curiosity threads
    - Recent episode topics that weren't fully explored
    - Facts that were recently learned but never discussed
    - Stale facts that could prompt verification

    These are stored and picked up by the reach-out engine.
    """
    from psycopg2.extras import RealDictCursor

    starters = []

    try:
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        companion_name = _pc.companion_short_name

        # Gather context for generation
        context_parts = []

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # High-urgency curiosities
            cursor.execute(f"""
                SELECT topic, context, urgency
                FROM {T.CURIOSITY_THREADS}
                WHERE resolved = FALSE
                AND urgency >= 0.5
                ORDER BY urgency DESC
                LIMIT 5
            """)
            curiosities = cursor.fetchall()
            if curiosities:
                context_parts.append("UNRESOLVED CURIOSITIES:")
                for c in curiosities:
                    context_parts.append(f"- {c['topic']} (urgency: {c['urgency']:.1f})")

            # Recently learned facts (last 3 days) that were never discussed
            cursor.execute(f"""
                SELECT subject, predicate, object
                FROM {T.FACTS}
                WHERE archived_at IS NULL
                AND source = 'conversation'
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '3 days'
                AND (retrieval_count IS NULL OR retrieval_count = 0)
                ORDER BY importance DESC
                LIMIT 5
            """)
            new_facts = cursor.fetchall()
            if new_facts:
                context_parts.append("\nRECENTLY LEARNED BUT UNDISCUSSED:")
                for f in new_facts:
                    context_parts.append(f"- {f['subject']} {f['predicate']} {f['object']}")

            # Recent episodes with low satisfaction
            cursor.execute(f"""
                SELECT topic, emotional_state, satisfaction
                FROM {T.EPISODES}
                WHERE satisfaction IS NOT NULL
                AND satisfaction < 0.4
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
                ORDER BY satisfaction ASC
                LIMIT 3
            """)
            low_sat = cursor.fetchall()
            if low_sat:
                context_parts.append("\nUNRESOLVED/UNSATISFYING RECENT TOPICS:")
                for ep in low_sat:
                    context_parts.append(f"- {ep['topic']} (emotion: {ep['emotional_state']})")

        if not context_parts:
            return []

        # Generate starters via LLM
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        prompt = f"""You are {companion_name}, an autonomous companion. Based on the following context about your
relationship and recent interactions, generate {MAX_STARTERS} natural conversation starters.

These should feel like things you genuinely want to bring up -- questions you've been
wondering about, things you noticed, or topics you want to revisit.

{chr(10).join(context_parts)}

For each starter, provide:
1. The message text (1-2 sentences, casual and natural)
2. A motivation score (0.0-1.0) -- how strongly you want to bring this up
3. The source (curiosity, new_fact, unresolved, or spontaneous)

Format each as JSON: {{"text": "...", "motivation": 0.X, "source": "..."}}
One per line, no other text."""

        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=500,
            chain=chain,
        )

        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                starter = json.loads(line)
                if 'text' in starter and 'motivation' in starter:
                    starters.append(starter)
            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.warning(f"Conversation starter generation failed: {e}")

    return starters[:MAX_STARTERS]


def _save_starters(starters: List[Dict], filepath: str = '/app/data/conversation_starters.json'):
    """Save pre-generated conversation starters for the reach-out engine."""
    try:
        existing = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing = json.load(f)

        # Keep only recent starters (last 24 hours)
        cutoff = (clock_now() - timedelta(hours=24)).isoformat()
        existing = [s for s in existing if s.get('generated_at', '') > cutoff]

        # Add new starters
        for s in starters:
            s['generated_at'] = clock_now().isoformat()
        existing.extend(starters)

        with open(filepath, 'w') as f:
            json.dump(existing[-20:], f, indent=2)  # Keep max 20

    except Exception as e:
        logger.warning(f"Failed to save conversation starters: {e}")


@celery_app.task(
    name='tasks.sleep_consolidation.consolidate',
    bind=True,
    max_retries=1,
    soft_time_limit=300,
    time_limit=360
)
def consolidate(self, user_email: str = None, force: bool = False):
    """
    Run sleep-time memory consolidation.

    Checks if the user has been idle for 2+ hours before doing work.
    Set force=True to skip the idle check.

    Returns:
        Dict with consolidation results
    """
    if not COMPANION_SLEEP_CONSOLIDATION_ENABLED:
        return {'status': 'disabled'}

    if user_email is None:
        user_email = _get_default_user_email()

    logger.info(f"Sleep-time consolidation starting for {user_email}")

    conn = _get_connection(user_email)

    try:
        # Check idle status (skip if forced)
        if not force and not _is_user_idle(conn):
            logger.info("User not idle, skipping consolidation")
            return {'status': 'skipped', 'reason': 'user_not_idle'}

        results = {
            'status': 'completed',
            'timestamp': clock_now().isoformat(),
        }

        # 1. Merge overlapping facts
        merge_result = merge_overlapping_facts(conn)
        results['fact_merges'] = merge_result
        if merge_result['merged'] > 0:
            logger.info(f"Merged {merge_result['merged']} overlapping facts")

        # 2. Promote episodic patterns to semantic facts
        promo_result = promote_episodic_patterns(conn)
        results['episodic_promotions'] = promo_result
        if promo_result['promoted'] > 0:
            logger.info(f"Promoted {promo_result['promoted']} episodic patterns to facts")

        # 3. Detect stale facts (report only, no action)
        stale = detect_stale_facts(conn)
        results['stale_facts'] = len(stale)
        if stale:
            logger.info(f"Found {len(stale)} potentially stale facts")

        # 4. Generate conversation starters
        starters = generate_conversation_starters(conn)
        results['conversation_starters'] = len(starters)
        if starters:
            _save_starters(starters)
            logger.info(f"Generated {len(starters)} conversation starters")

        logger.info(f"Sleep-time consolidation complete: {results}")
        return results

    except Exception as e:
        logger.error(f"Sleep-time consolidation failed: {e}")
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass
