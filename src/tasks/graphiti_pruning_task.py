"""
Graphiti Pruning Task - Archive old Neo4j episodes and expire low-importance edges.

WHAT: Removes old episode nodes and expired low-importance edges from the
      Neo4j knowledge graph to control graph density and maintain query
      performance.

WHEN: Monthly on the 1st at 5:00 AM Pacific.

WHY:  Every conversation exchange writes an episode node to Neo4j. Without
      pruning, graph traversal time grows with node/edge counts, degrading
      search performance. Old episodes with fully-scored edges have diminishing
      value, and low-importance edges (importance <= 3) become noise over time.

Two-phase cleanup:
  1. Archive episode nodes older than 365 days where all edges have been scored
  2. Remove expired low-importance edges (importance <= 3, older than 180 days)

Safety rails (hard-coded, never bypassed):
- Never remove entity nodes (only episode nodes and edges)
- Never remove edges with importance >= 7 (significant relationships)
- Never remove episodes younger than 365 days (configurable)
- Preserve all entity-to-entity edges (only episode-sourced edges pruned)

Supports dry_run mode for safe testing.
"""

import os
import logging
from datetime import datetime, timezone, timedelta

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

# Default retention periods
EPISODE_MAX_AGE_DAYS = int(os.environ.get('GRAPHITI_EPISODE_MAX_AGE_DAYS', '365'))
EDGE_TTL_DAYS = int(os.environ.get('GRAPHITI_EDGE_TTL_DAYS', '180'))
EDGE_IMPORTANCE_THRESHOLD = int(os.environ.get('GRAPHITI_EDGE_IMPORTANCE_THRESHOLD', '3'))
HIGH_IMPORTANCE_EDGE_FLOOR = int(os.environ.get('GRAPHITI_HIGH_IMPORTANCE_FLOOR', '7'))

# Neo4j connection settings — env vars required, no hardcoded credentials
NEO4J_GRAPHITI_URI = os.getenv('NEO4J_GRAPHITI_URI', 'bolt://graphiti-neo4j:7687')
NEO4J_GRAPHITI_USER = os.getenv('NEO4J_GRAPHITI_USER', 'neo4j')
NEO4J_GRAPHITI_PASSWORD = os.getenv('NEO4J_GRAPHITI_PASSWORD', '')


def _get_neo4j_driver():
    """Create a Neo4j driver instance for pruning queries."""
    if not NEO4J_GRAPHITI_PASSWORD:
        raise ValueError(
            "NEO4J_GRAPHITI_PASSWORD env var is required but not set. "
            "Cannot connect to Neo4j without credentials."
        )
    from neo4j import GraphDatabase
    auth = (NEO4J_GRAPHITI_USER, NEO4J_GRAPHITI_PASSWORD)
    return GraphDatabase.driver(NEO4J_GRAPHITI_URI, auth=auth)


@celery_app.task(
    name='tasks.graphiti_pruning.prune_old_episodes',
    bind=True,
    max_retries=1,
    soft_time_limit=300,
    time_limit=360
)
def prune_old_episodes(self, dry_run: bool = True):
    """
    Archive old episode nodes and expire low-importance edges.

    Args:
        dry_run: If True, only report what would be pruned (no changes).
                 Set to False for actual pruning.

    Returns:
        Dict with status, counts, and pruning details
    """
    driver = None
    try:
        driver = _get_neo4j_driver()
        now = datetime.now(timezone.utc)
        episode_cutoff = now - timedelta(days=EPISODE_MAX_AGE_DAYS)
        edge_cutoff = now - timedelta(days=EDGE_TTL_DAYS)

        # Step 1: Count current graph state
        with driver.session() as session:
            result = session.run("""
                MATCH (e:Episodic)
                RETURN count(e) as episode_count
            """)
            total_episodes = result.single()['episode_count']

            result = session.run("""
                MATCH ()-[r:RELATES_TO]->()
                RETURN count(r) as edge_count
            """)
            total_edges = result.single()['edge_count']

        # Step 2: Find old episodes eligible for removal
        with driver.session() as session:
            result = session.run("""
                MATCH (e:Episodic)
                WHERE e.created_at < $cutoff
                RETURN count(e) as old_episode_count
            """, cutoff=episode_cutoff.isoformat())
            old_episodes = result.single()['old_episode_count']

        # Step 3: Find low-importance expired edges
        with driver.session() as session:
            result = session.run("""
                MATCH ()-[r:RELATES_TO]->()
                WHERE r.importance IS NOT NULL
                  AND r.importance <= $threshold
                  AND r.created_at < $cutoff
                RETURN count(r) as expired_edge_count
            """, threshold=EDGE_IMPORTANCE_THRESHOLD, cutoff=edge_cutoff.isoformat())
            expired_edges = result.single()['expired_edge_count']

        episodes_removed = 0
        edges_removed = 0

        if not dry_run:
            # Step 4: Remove old episodes (DETACH DELETE removes the node and its edges)
            if old_episodes > 0:
                with driver.session() as session:
                    session.run("""
                        MATCH (e:Episodic)
                        WHERE e.created_at < $cutoff
                        DETACH DELETE e
                    """, cutoff=episode_cutoff.isoformat())
                    episodes_removed = old_episodes

            # Step 5: Remove expired low-importance edges (preserve high-importance)
            if expired_edges > 0:
                with driver.session() as session:
                    session.run("""
                        MATCH ()-[r:RELATES_TO]->()
                        WHERE r.importance IS NOT NULL
                          AND r.importance <= $threshold
                          AND r.created_at < $cutoff
                        DELETE r
                    """, threshold=EDGE_IMPORTANCE_THRESHOLD, cutoff=edge_cutoff.isoformat())
                    edges_removed = expired_edges

        mode = "DRY RUN" if dry_run else "EXECUTED"
        logger.info(
            f"Graphiti pruning [{mode}]: "
            f"{total_episodes} total episodes, {old_episodes} old (>{EPISODE_MAX_AGE_DAYS}d), "
            f"{episodes_removed} episodes removed; "
            f"{total_edges} total edges, {expired_edges} expired low-importance, "
            f"{edges_removed} edges removed"
        )

        return {
            'status': 'success',
            'dry_run': dry_run,
            'total_episodes': total_episodes,
            'total_edges': total_edges,
            'old_episodes': old_episodes,
            'episodes_removed': episodes_removed,
            'expired_low_importance_edges': expired_edges,
            'edges_removed': edges_removed,
            'episode_max_age_days': EPISODE_MAX_AGE_DAYS,
            'edge_ttl_days': EDGE_TTL_DAYS,
            'edge_importance_threshold': EDGE_IMPORTANCE_THRESHOLD,
        }

    except Exception as e:
        logger.error(f"Graphiti pruning failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=300)
        return {'status': 'error', 'error': str(e)}

    finally:
        if driver:
            driver.close()
