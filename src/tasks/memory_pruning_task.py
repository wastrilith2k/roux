"""
Memory Pruning Task - Archive low-value facts to keep memory focused.

WHAT: Scores all facts older than 30 days using a retention formula:
        retention = importance * recency_factor * mention_boost * confidence
      Facts scoring below the threshold (default 1.5) are archived (NOT deleted).
      Also cleans up orphaned fact_links where both endpoints are archived.

WHEN: Weekly on Sunday at 5:30 AM Pacific.

WHY:  The facts table grows unbounded. Without pruning, low-value facts
      ("James had pasta for dinner" from 3 months ago) crowd out important
      ones in retrieval. Archiving preserves data for audit while keeping
      active memory clean and relevant.

Safety rails (hard-coded, never bypassed):
- Never archive importance >= 8 (critical memories)
- Never archive facts mentioned in last 30 days (still relevant)
- Never archive facts with 5+ mentions (well-established knowledge)
- Minimum age: 30 days (give facts time to prove their value)
- Always archive, never delete (sets archived_at + archive_reason)

Supports dry_run mode for safe testing.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

# --- Category-specific decay rates ---
# Different fact types age at different rates. Crisis facts become stale faster
# (situations resolve) while identity facts persist almost indefinitely.
CATEGORY_DECAY_DAYS = {
    'crisis': 90,
    'work': 120,
    'family': 180,
    'preferences': 270,
    'identity': 365,
    'default': 180,
}

# Keywords for category detection
CATEGORY_KEYWORDS = {
    'crisis': ['crisis', 'emergency', 'hospital', 'er visit', 'ran away', 'police', 'suicidal', 'overdose'],
    'work': ['work', 'job', 'career', 'interview', 'hired', 'fired', 'meeting', 'project', 'salary', 'promotion', 'cavallo'],
    'family': ['parent', 'child', 'son', 'daughter', 'mother', 'father', 'wife', 'husband', 'sibling', 'brother', 'sister', 'family'],
    'preferences': ['likes', 'dislikes', 'prefers', 'favorite', 'hates', 'loves', 'enjoys', 'hobby', 'tea', 'coffee'],
    'identity': ['born', 'name', 'age', 'identity', 'adhd', 'diagnosis', 'religion', 'ethnicity'],
}


@celery_app.task(
    name='tasks.memory_pruning.prune_stale_facts',
    bind=True,
    max_retries=1,
    soft_time_limit=120,
    time_limit=180
)
def prune_stale_facts(self, dry_run: bool = True):
    """
    Score all facts with a retention formula and archive low-scoring ones.

    Args:
        dry_run: If True, only report what would be pruned (no changes).
                 Set to False for actual pruning.

    Returns:
        Dict with status, counts, and pruning details
    """
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

        threshold = float(os.environ.get('MEMORY_PRUNE_THRESHOLD', '1.5'))

        # Step 1: Get all active facts older than 30 days
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, subject, predicate, object, importance,
                       mention_count, last_mentioned, created_at
                FROM facts
                WHERE archived_at IS NULL
                  AND created_at < NOW() - INTERVAL '30 days'
                ORDER BY created_at ASC
            """)
            candidates = [dict(row) for row in cursor.fetchall()]

        if not candidates:
            conn.close()
            logger.info("Memory pruning: no candidates (all facts younger than 30 days)")
            return {'status': 'success', 'candidates': 0, 'pruned': 0}

        # Step 2: Score each fact
        to_archive = []
        safety_skipped = {'high_importance': 0, 'recently_mentioned': 0, 'well_established': 0}
        importance_buckets = {
            '1-3': {'candidates': 0, 'pruned': 0},
            '4-6': {'candidates': 0, 'pruned': 0},
            '7-9': {'candidates': 0, 'pruned': 0},
            '10': {'candidates': 0, 'pruned': 0},
        }

        now = datetime.now()

        for fact in candidates:
            importance = fact['importance'] or 5
            mention_count = fact['mention_count'] or 1
            last_mentioned = fact['last_mentioned'] or fact['created_at']
            created_at = fact['created_at']

            # Determine importance bucket
            if importance >= 10:
                bucket = '10'
            elif importance >= 7:
                bucket = '7-9'
            elif importance >= 4:
                bucket = '4-6'
            else:
                bucket = '1-3'
            importance_buckets[bucket]['candidates'] += 1

            # Safety rail 1: Never archive importance >= 8
            if importance >= 8:
                safety_skipped['high_importance'] += 1
                continue

            # Safety rail 2: Never archive facts mentioned in last 30 days
            days_since_mentioned = (now - last_mentioned).days if last_mentioned else 999
            if days_since_mentioned < 30:
                safety_skipped['recently_mentioned'] += 1
                continue

            # Safety rail 3: Never archive facts with 5+ mentions
            if mention_count >= 5:
                safety_skipped['well_established'] += 1
                continue

            # Calculate retention score (using confidence decay for recency/mention factors)
            category = _detect_category(fact)
            decay_days = CATEGORY_DECAY_DAYS.get(category, CATEGORY_DECAY_DAYS['default'])
            days_old = (now - created_at).days if created_at else 30

            # Use confidence decay module for consistent calculations
            try:
                from src.memory.confidence_decay import calculate_effective_confidence
                eff_conf = calculate_effective_confidence(
                    base_confidence=fact.get('confidence', 0.7),
                    mention_count=mention_count,
                    last_mentioned=last_mentioned,
                    created_at=created_at
                )
                # Retention uses both importance and effective confidence
                recency_factor = max(0.1, 1.0 - (days_old / decay_days))
                mention_boost = min(2.0, 1.0 + (mention_count - 1) * 0.1)
                retention = importance * recency_factor * mention_boost * (eff_conf / 0.7)
            except Exception:
                # Fallback to original formula if confidence decay module fails
                recency_factor = max(0.1, 1.0 - (days_old / decay_days))
                mention_boost = min(2.0, 1.0 + (mention_count - 1) * 0.1)
                retention = importance * recency_factor * mention_boost

            if retention < threshold:
                to_archive.append({
                    'id': fact['id'],
                    'subject': fact['subject'],
                    'predicate': fact['predicate'],
                    'object': (fact['object'] or '')[:80],
                    'importance': importance,
                    'retention': round(retention, 3),
                    'category': category,
                    'days_old': days_old,
                    'mention_count': mention_count,
                })
                importance_buckets[bucket]['pruned'] += 1

        # Step 3: Archive (or just report in dry_run)
        pruned_count = 0
        orphaned_links = 0

        if to_archive and not dry_run:
            archive_ids = [f['id'] for f in to_archive]

            with conn.cursor() as cursor:
                # Archive the facts
                cursor.execute("""
                    UPDATE facts
                    SET archived_at = CURRENT_TIMESTAMP,
                        archive_reason = 'pruned_low_retention'
                    WHERE id = ANY(%s)
                """, (archive_ids,))
                pruned_count = cursor.rowcount

                # Clean up orphaned fact_links (where both facts are archived)
                cursor.execute("""
                    DELETE FROM fact_links
                    WHERE source_fact_id IN (
                        SELECT id FROM facts WHERE archived_at IS NOT NULL
                    )
                    AND target_fact_id IN (
                        SELECT id FROM facts WHERE archived_at IS NOT NULL
                    )
                """)
                orphaned_links = cursor.rowcount

                conn.commit()

        # Step 4: Get total active count
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM facts WHERE archived_at IS NULL")
            active_remaining = cursor.fetchone()[0]

        conn.close()

        mode = "DRY RUN" if dry_run else "EXECUTED"
        logger.info(
            f"Memory pruning [{mode}]: "
            f"{len(candidates)} candidates, "
            f"{len(to_archive)} would be pruned (threshold={threshold}), "
            f"{pruned_count} actually archived, "
            f"{orphaned_links} orphaned links removed, "
            f"{active_remaining} active facts remaining"
        )
        logger.info(f"Safety skips: {safety_skipped}")
        logger.info(f"Importance buckets: {importance_buckets}")

        return {
            'status': 'success',
            'dry_run': dry_run,
            'candidates': len(candidates),
            'would_prune': len(to_archive),
            'actually_pruned': pruned_count,
            'orphaned_links_removed': orphaned_links,
            'active_remaining': active_remaining,
            'threshold': threshold,
            'safety_skipped': safety_skipped,
            'importance_buckets': importance_buckets,
            'sample_pruned': [
                f"{f['subject']}: {f['object']} (retention={f['retention']}, imp={f['importance']})"
                for f in to_archive[:5]
            ],
        }

    except Exception as e:
        logger.error(f"Memory pruning failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=300)
        return {'status': 'error', 'error': str(e)}


def _detect_category(fact: Dict) -> str:
    """
    Detect fact category from subject/predicate/object keywords.

    Returns category string for decay_days lookup.
    """
    text = ' '.join([
        fact.get('subject', ''),
        fact.get('predicate', ''),
        (fact.get('object', '') or '')[:200]
    ]).lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category

    return 'default'
