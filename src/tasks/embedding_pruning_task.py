"""
Embedding Pruning Task - Archive stale pgvector embeddings to control index size.

WHAT: Nullifies embedding_vec on old messages that are no longer useful for
      semantic search. Does NOT delete the message row — only the vector column
      is set to NULL, preserving message history for display and analytics.

WHEN: Weekly on Sunday at 6:00 AM Pacific (after fact pruning at 5:30 AM).

WHY:  Each 1536-dim embedding is ~6KB. At 100 messages/day that's ~220MB/year
      of vectors. pgvector search time scales with table size, so pruning old,
      rarely-retrieved embeddings keeps search fast and the index small.

Safety rails (hard-coded, never bypassed):
- Never archive embeddings with importance >= 8 (messages linked to critical facts)
- Never archive embeddings retrieved in the last 60 days (still actively useful)
- Never archive embeddings younger than 180 days (configurable)
- Never archive embeddings referenced by active (non-archived) facts
- Archive = SET embedding_vec = NULL (preserves message row)

Supports dry_run mode for safe testing.
"""

import os
import logging
from datetime import datetime
from typing import Dict

from src.celery_app import celery_app
from src.database.db import get_db
from src.database import tables as T

logger = logging.getLogger(__name__)

# Default retention: 180 days before embeddings become candidates
EMBEDDING_MAX_AGE_DAYS = int(os.environ.get('EMBEDDING_MAX_AGE_DAYS', '180'))
# Last-retrieved threshold: skip if retrieved within this many days
EMBEDDING_RETRIEVAL_RECENCY_DAYS = int(os.environ.get('EMBEDDING_RETRIEVAL_RECENCY_DAYS', '60'))
# Importance threshold: never archive messages linked to facts with importance >= this
EMBEDDING_IMPORTANCE_THRESHOLD = int(os.environ.get('EMBEDDING_IMPORTANCE_THRESHOLD', '8'))


@celery_app.task(
    name='tasks.embedding_pruning.prune_stale_embeddings',
    bind=True,
    max_retries=1,
    soft_time_limit=120,
    time_limit=180
)
def prune_stale_embeddings(self, dry_run: bool = True):
    """
    Nullify embedding_vec on old, rarely-retrieved messages.

    Args:
        dry_run: If True, only report what would be pruned (no changes).
                 Set to False for actual pruning.

    Returns:
        Dict with status, counts, and pruning details
    """
    try:
        db = get_db()
        max_age_days = EMBEDDING_MAX_AGE_DAYS
        retrieval_recency_days = EMBEDDING_RETRIEVAL_RECENCY_DAYS

        # Step 1: Count total embeddings
        result = db.execute(f"""
            SELECT COUNT(*) AS cnt FROM {T.MESSAGES}
            WHERE embedding_vec IS NOT NULL
        """)
        total_embeddings = result.fetchone()['cnt']

        # Step 2: Find candidate embeddings (old enough, not recently retrieved)
        result = db.execute(f"""
            SELECT m.id, m.timestamp, m.last_retrieved_at, m.sender_name
            FROM {T.MESSAGES} m
            WHERE m.embedding_vec IS NOT NULL
              AND m.timestamp < NOW() - INTERVAL '1 day' * %s
              AND (
                  m.last_retrieved_at IS NULL
                  OR m.last_retrieved_at < NOW() - INTERVAL '1 day' * %s
              )
        """, (max_age_days, retrieval_recency_days))
        candidates = [dict(row) for row in result.fetchall()]

        if not candidates:
            logger.info("Embedding pruning: no candidates (all embeddings younger than "
                        f"{max_age_days} days or recently retrieved)")
            return {
                'status': 'success',
                'total_embeddings': total_embeddings,
                'candidates': 0,
                'pruned': 0
            }

        candidate_ids = [c['id'] for c in candidates]

        # Step 3: Exclude messages referenced by active (non-archived) facts
        result = db.execute(f"""
            SELECT DISTINCT message_id
            FROM {T.FACTS}
            WHERE message_id = ANY(%s)
              AND archived_at IS NULL
        """, (candidate_ids,))
        fact_referenced_ids = {row['message_id'] for row in result.fetchall()}

        # Step 4: Exclude messages linked to high-importance facts
        result = db.execute(f"""
            SELECT DISTINCT message_id
            FROM {T.FACTS}
            WHERE message_id = ANY(%s)
              AND importance >= %s
        """, (candidate_ids, EMBEDDING_IMPORTANCE_THRESHOLD))
        high_importance_ids = {row['message_id'] for row in result.fetchall()}

        # Apply safety filters
        safe_to_prune = []
        safety_skipped = {
            'active_fact_reference': 0,
            'high_importance': 0,
        }

        for candidate in candidates:
            msg_id = candidate['id']

            if msg_id in fact_referenced_ids:
                safety_skipped['active_fact_reference'] += 1
                continue

            if msg_id in high_importance_ids:
                safety_skipped['high_importance'] += 1
                continue

            safe_to_prune.append(msg_id)

        # Step 5: Archive (or just report in dry_run)
        pruned_count = 0

        if safe_to_prune and not dry_run:
            result = db.execute(f"""
                UPDATE {T.MESSAGES}
                SET embedding_vec = NULL
                WHERE id = ANY(%s)
            """, (safe_to_prune,))
            pruned_count = len(safe_to_prune)

        # Step 6: Get remaining embedding count
        result = db.execute(f"""
            SELECT COUNT(*) AS cnt FROM {T.MESSAGES}
            WHERE embedding_vec IS NOT NULL
        """)
        remaining_embeddings = result.fetchone()['cnt']

        mode = "DRY RUN" if dry_run else "EXECUTED"
        logger.info(
            f"Embedding pruning [{mode}]: "
            f"{total_embeddings} total embeddings, "
            f"{len(candidates)} candidates (>{max_age_days}d old), "
            f"{len(safe_to_prune)} safe to prune, "
            f"{pruned_count} actually pruned, "
            f"{remaining_embeddings} remaining"
        )
        logger.info(f"Safety skips: {safety_skipped}")

        return {
            'status': 'success',
            'dry_run': dry_run,
            'total_embeddings': total_embeddings,
            'candidates': len(candidates),
            'would_prune': len(safe_to_prune),
            'actually_pruned': pruned_count,
            'remaining_embeddings': remaining_embeddings,
            'safety_skipped': safety_skipped,
            'max_age_days': max_age_days,
            'retrieval_recency_days': retrieval_recency_days,
        }

    except Exception as e:
        logger.error(f"Embedding pruning failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=300)
        return {'status': 'error', 'error': str(e)}
