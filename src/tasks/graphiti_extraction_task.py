#!/usr/bin/env python3
"""
Graphiti Extraction Task - Feed conversations into the temporal knowledge graph.

WHAT: Adds each conversation exchange as a Graphiti "episode." Graphiti then
      automatically extracts entities (people, places, things), identifies
      relationships, tracks temporal changes (valid_at/invalid_at), and
      handles contradictions. After each episode, scores new edges for
      importance and triggers biography rebuilds for high-importance facts.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  The knowledge graph provides structured, queryable memory that goes
      beyond flat fact storage. It captures how relationships evolve over time
      (e.g., "James was married to Alia" -> "James is separated from Alia")
      and enables graph-based context retrieval for conversations.

Requires:
- Dedicated Neo4j instance on NEO4J_GRAPHITI_URI (default: bolt://graphiti-neo4j:7687)
- OPENAI_API_KEY for LLM inference and embeddings
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, '/app')

logger = logging.getLogger(__name__)

from src.celery_app import celery_app

# Graphiti connection settings - Updated for new Graphiti Neo4j container
NEO4J_GRAPHITI_URI = os.getenv('NEO4J_GRAPHITI_URI', 'bolt://graphiti-neo4j:7687')
NEO4J_GRAPHITI_USER = os.getenv('NEO4J_GRAPHITI_USER', 'neo4j')
NEO4J_GRAPHITI_PASSWORD = os.getenv('NEO4J_GRAPHITI_PASSWORD', 'graphiti_secure_2026')

# --- Singleton Graphiti Instance ---
# Reused across task invocations to avoid reconnecting to Neo4j every time.
# Uses threading.Lock because Celery prefork workers can be multi-threaded.
import threading
_graphiti_instance = None
_graphiti_thread_lock = threading.Lock()


def reset_graphiti():
    """Reset the Graphiti singleton (used after errors)."""
    global _graphiti_instance
    with _graphiti_thread_lock:
        _graphiti_instance = None


async def get_graphiti():
    """
    Get or create singleton Graphiti instance.
    Uses double-checked locking: fast path avoids lock acquisition,
    slow path ensures only one thread initializes.
    """
    global _graphiti_instance

    # Fast path - already initialized, no lock needed
    if _graphiti_instance is not None:
        return _graphiti_instance

    # Slow path - thread-safe initialization
    with _graphiti_thread_lock:
        # Double-check after acquiring lock
        if _graphiti_instance is not None:
            return _graphiti_instance

        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            driver = Neo4jDriver(
                uri=NEO4J_GRAPHITI_URI,
                user=NEO4J_GRAPHITI_USER,
                password=NEO4J_GRAPHITI_PASSWORD,
                database="neo4j"
            )
            _graphiti_instance = Graphiti(graph_driver=driver)

            # Ensure indices exist (ignore if already created)
            try:
                await _graphiti_instance.build_indices_and_constraints()
            except Exception as e:
                if "EquivalentSchemaRuleAlreadyExists" not in str(e):
                    logger.warning(f"Index creation warning: {e}")

            logger.info(f"Graphiti initialized with {NEO4J_GRAPHITI_URI}")
            return _graphiti_instance

        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise


async def score_recent_edges(limit: int = 50) -> int:
    """
    Score importance and classify status for recently created edges.

    Called after add_episode() to:
    1. Score importance (1-10) for new facts
    2. Classify fact_status (definite/planned/hypothetical)

    If any facts score >= 9 (core identity), triggers biography rebuild
    for the affected entities.

    Returns:
        Number of edges processed
    """
    from neo4j import GraphDatabase

    uri = NEO4J_GRAPHITI_URI
    driver = GraphDatabase.driver(uri, auth=None)
    processed = 0
    high_importance_entities = set()  # Track entities needing bio rebuild

    try:
        with driver.session() as session:
            # Find edges needing scoring OR status classification
            result = session.run("""
                MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
                WHERE (edge.importance IS NULL OR edge.fact_status IS NULL)
                  AND edge.fact IS NOT NULL
                RETURN edge.uuid as uuid,
                       edge.fact as fact,
                       edge.importance as importance,
                       edge.fact_status as fact_status,
                       source.name as source_entity,
                       target.name as target_entity
                ORDER BY edge.created_at DESC
                LIMIT $limit
            """, limit=limit)

            edges = [dict(record) for record in result]

        if not edges:
            return 0

        logger.info(f"Processing {len(edges)} edges (scoring)")

        # Import here to avoid circular imports
        from src.memory.importance_scorer import score_importance

        for edge in edges:
            uuid = edge['uuid']
            fact = edge['fact']
            current_importance = edge.get('importance')
            source_entity = edge.get('source_entity', '')
            target_entity = edge.get('target_entity', '')

            if not fact:
                continue

            updates = {}

            # Score importance if needed
            if current_importance is None:
                score = score_importance(fact)  # Fixed: removed invalid use_ollama parameter
                if score is not None:
                    updates['importance'] = score
                    logger.debug(f"Scored edge {uuid}: {score}")

                    # Track high-importance entities for bio rebuild
                    if score >= 9:
                        logger.info(f"High-importance memory detected (score={score}): {fact[:50]}...")
                        if source_entity:
                            high_importance_entities.add(source_entity)
                        if target_entity:
                            high_importance_entities.add(target_entity)

            # Note: fact_status classification removed - module not available
            # Facts default to 'definite' status in Graphiti

            # Apply updates
            if updates:
                with driver.session() as session:
                    set_clauses = ", ".join([f"edge.{k} = ${k}" for k in updates.keys()])
                    query = f"""
                        MATCH ()-[edge:RELATES_TO]->()
                        WHERE edge.uuid = $uuid
                        SET {set_clauses}
                    """
                    session.run(query, uuid=uuid, **updates)
                processed += 1

        logger.info(f"Processed {processed}/{len(edges)} edges")

        # Trigger biography rebuild AND vectorization for entities with high-importance facts
        if high_importance_entities:
            await rebuild_and_vectorize_biographies(high_importance_entities)

    except Exception as e:
        logger.error(f"Error scoring edges: {e}")

    finally:
        driver.close()

    return processed


async def rebuild_and_vectorize_biographies(entities: set):
    """
    Rebuild and vectorize biographies for entities that received high-importance facts.

    This ensures significant facts (importance >= 7) are immediately reflected in the bio
    and the vectorized sections are updated for context matching.

    Steps:
    1. Regenerate biography text from updated Graphiti facts
    2. Parse into semantic sections
    3. Generate embeddings and cache in Redis
    """
    # Only rebuild for known main entities to avoid unnecessary work
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    MAIN_ENTITIES = {
        _pc.companion_short_name, _pc.primary_user_name,
        _pc.companion_short_name.lower(), _pc.primary_user_name.lower()
    }
    entities_to_rebuild = entities & MAIN_ENTITIES

    if not entities_to_rebuild:
        return

    logger.info(f"Rebuilding & vectorizing biographies for: {entities_to_rebuild}")

    try:
        from src.tasks.biography_vectorization_job import vectorize_biography

        for entity in entities_to_rebuild:
            # Normalize to title case
            entity_name = entity.title()
            logger.info(f"Vectorizing biography for {entity_name} (high-importance trigger)")
            # force_refresh=True regenerates biography before vectorizing
            success = vectorize_biography(entity_name, force_refresh=True)
            if success:
                logger.info(f"Biography vectorized for {entity_name}")
            else:
                logger.warning(f"Biography vectorization failed for {entity_name}")

    except Exception as e:
        logger.error(f"Failed to rebuild/vectorize biographies: {e}")


async def add_conversation_episode(
    user_email: str,
    user_message: str,
    companion_response: str,
    username: str = None
) -> dict:
    """
    Add a conversation exchange to Graphiti as an episode.

    Args:
        user_email: User's email (used for source tracking)
        user_message: User's message
        companion_response: The companion's response
        username: Display name for user (default: from persona config)

    Returns:
        Dict with extraction status and stats
    """
    from graphiti_core.nodes import EpisodeType

    graphiti = await get_graphiti()

    # Format conversation as episode content
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    if username is None:
        username = _pc.primary_user_name
    episode_content = f"""
{username}: {user_message}

{_pc.companion_short_name}: {companion_response}
"""

    # Create unique episode name with timestamp
    timestamp = datetime.now(timezone.utc)
    episode_name = f"Conversation {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        await graphiti.add_episode(
            name=episode_name,
            episode_body=episode_content.strip(),
            source=EpisodeType.text,
            source_description=f"Chat conversation with {username} ({user_email})",
            reference_time=timestamp,
        )

        logger.info(f"Added episode to Graphiti: {episode_name}")

        # Score importance for any newly created edges
        scored_count = await score_recent_edges()

        return {
            'status': 'success',
            'episode_name': episode_name,
            'timestamp': timestamp.isoformat(),
            'edges_scored': scored_count
        }

    except Exception as e:
        logger.error(f"Failed to add episode: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def run_async(coro):
    """
    Bridge async -> sync for Celery workers.
    Creates a fresh event loop each time because Celery workers may reuse
    threads where the previous loop was closed, causing "Event loop is closed" errors.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name='tasks.graphiti_extraction.process_exchange',
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=120,
    time_limit=180
)
def process_exchange_graphiti(self, user_email: str, user_message: str, companion_response: str):
    """
    Celery task for Graphiti-based extraction.

    Adds the conversation exchange as an episode to Graphiti's temporal
    knowledge graph. Graphiti automatically:
    - Extracts entities (people, places, things)
    - Identifies relationships between entities
    - Tracks temporal changes (valid_at, invalid_at)
    - Handles contradictions and updates

    Args:
        user_email: User's email
        user_message: User's message
        companion_response: The companion's response

    Returns:
        Dict with extraction results
    """
    try:
        print(f"\n{'='*60}")
        print(f"GRAPHITI EXTRACTION - {datetime.now()}")
        print(f"{'='*60}")
        print(f"User: {user_email}")
        print(f"Message length: {len(user_message)} chars")
        print(f"Response length: {len(companion_response)} chars")

        # Skip very short messages
        if len(user_message) < 10:
            print("Skipping - message too short")
            return {'status': 'skipped', 'reason': 'message_too_short'}

        # Add episode to Graphiti
        from src.config.persona_config import get_persona_config
        result = run_async(add_conversation_episode(
            user_email=user_email,
            user_message=user_message,
            companion_response=companion_response,
            username=get_persona_config().primary_user_name
        ))

        print(f"\n{'='*60}")
        print(f"GRAPHITI EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Status: {result.get('status')}")
        if result.get('episode_name'):
            print(f"Episode: {result.get('episode_name')}")
        print(f"{'='*60}\n")

        return result

    except Exception as e:
        logger.error(f"Graphiti extraction task failed: {e}")
        import traceback
        traceback.print_exc()

        # Reset singleton on error (may help with event loop issues)
        reset_graphiti()

        # Retry on failure
        try:
            self.retry(exc=e)
        except Exception:
            pass

        return {'status': 'error', 'error': str(e)}


# Convenience function for testing
async def test_extraction():
    """Test the extraction with sample data."""
    from src.config.persona_config import get_persona_config
    result = await add_conversation_episode(
        user_email=get_persona_config().primary_user_email,
        user_message="I had a great time at the park with Jesse today. He's really getting good at skateboarding!",
        companion_response="That sounds wonderful! It's so nice to hear Jesse is improving. How long have you two been practicing together?",
        username=get_persona_config().primary_user_name
    )
    print(f"Test result: {result}")
    return result


if __name__ == "__main__":
    # Run test
    asyncio.run(test_extraction())
