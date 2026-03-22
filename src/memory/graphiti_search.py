"""
Graphiti Search - Query the temporal knowledge graph for relevant facts.

WHAT: Provides search and retrieval over Graphiti's Neo4j-backed temporal knowledge
graph. Converts graph edges (facts stored as relationships between Entity nodes) into
simple dicts that can be injected into the companion's LLM prompt.

WHY: The companion needs to recall facts from past conversations. Graphiti stores
these as a temporal knowledge graph where facts are edges between entity nodes.
This module bridges that graph representation into the flat context strings the
LLM consumes.

HOW it fits: context_builder.py calls get_graphiti_context() or
get_graphiti_context_with_importance() to pull relevant facts for the current
message. biography_synthesizer uses get_entity_facts() /
get_weighted_facts_for_biography() for periodic bio regeneration. Both paths
ultimately query the same Neo4j database via either the Graphiti SDK (vector
search) or direct Cypher queries (entity-scoped lookups).

Two query paths:
  1. Graphiti SDK vector search (search_graphiti_async) - semantic similarity
  2. Direct Neo4j Cypher queries (get_entity_facts, etc.) - entity/time scoped
"""

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Timestamp formatting helpers
# =============================================================================

def _format_time_ago(timestamp_str: str) -> str:
    """
    Format a timestamp as a human-readable "time ago" string.

    Args:
        timestamp_str: ISO format timestamp (e.g., "2025-12-18T12:16:11.531940Z")

    Returns:
        Human-readable string like "2 days ago", "just now", etc.
    """
    try:
        if not timestamp_str:
            return ""

        # Normalize the many ISO-8601 variants Graphiti may produce into a
        # consistent format that datetime.fromisoformat() can parse.
        ts_str = timestamp_str.replace('Z[UTC]', '+00:00').replace('Z', '+00:00')
        if '+' not in ts_str and 'T' in ts_str:
            ts_str += '+00:00'

        created = datetime.fromisoformat(ts_str.replace('[UTC]', ''))
        now = datetime.now(timezone.utc)
        diff = now - created

        if diff.days == 0:
            if diff.seconds < 3600:
                return "just now"
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"

    except Exception as e:
        logger.debug(f"Failed to parse timestamp {timestamp_str}: {e}")
        return ""


# =============================================================================
# Graphiti SDK connection (vector search path)
# =============================================================================

# Connection settings for the dedicated Graphiti Neo4j container.
NEO4J_GRAPHITI_URI = os.getenv('NEO4J_GRAPHITI_URI', 'bolt://graphiti-neo4j:7687')
NEO4J_GRAPHITI_USER = os.getenv('NEO4J_GRAPHITI_USER', 'neo4j')
NEO4J_GRAPHITI_PASSWORD = os.getenv('NEO4J_GRAPHITI_PASSWORD', 'graphiti_secure_2026')

# No singleton -- graphiti_core internally binds asyncio futures to the event
# loop that creates them, so we must create fresh instances when running in
# ThreadPoolExecutor with new event loops.


def _create_graphiti_instance():
    """
    Create a fresh Graphiti instance.

    We don't cache instances because graphiti_core internally uses asyncio
    futures that get bound to specific event loops. When using ThreadPoolExecutor
    with new event loops, we need fresh instances.
    """
    try:
        from graphiti_core import Graphiti
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        driver = Neo4jDriver(
            uri=NEO4J_GRAPHITI_URI,
            user=NEO4J_GRAPHITI_USER,
            password=NEO4J_GRAPHITI_PASSWORD,
            database="neo4j"
        )
        graphiti = Graphiti(graph_driver=driver)
        logger.debug(f"Graphiti instance created: {NEO4J_GRAPHITI_URI}")
        return graphiti

    except Exception as e:
        logger.warning(f"Failed to create Graphiti instance: {e}")
        return None


async def _get_graphiti():
    """Create a fresh Graphiti instance (async context)."""
    return _create_graphiti_instance()


async def search_graphiti_async(
    query: str,
    limit: int = 10,
    center_node_uuid: Optional[str] = None,
    group_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search Graphiti knowledge graph for relevant facts (async).

    Graphiti's search returns Edge objects representing facts stored as
    relationships between Entity nodes. We flatten these into simple dicts
    that downstream code can format for prompt injection.

    Args:
        query: Natural language search query
        limit: Maximum number of results
        center_node_uuid: Optional node to center search around
        group_id: Optional group ID for multi-agent scoping (companion_id)

    Returns:
        List of dicts with fact information
    """
    graphiti = await _get_graphiti()
    if not graphiti:
        return []

    try:
        search_kwargs = dict(
            query=query,
            num_results=limit,
            center_node_uuid=center_node_uuid
        )
        if group_id:
            search_kwargs['group_ids'] = [group_id]
        results = await graphiti.search(**search_kwargs)

        # Convert Edge objects to simple dicts for context building
        facts = []
        for edge in results:
            fact_info = {
                'fact': getattr(edge, 'fact', str(edge)),
                'name': getattr(edge, 'name', ''),
                'created_at': str(getattr(edge, 'created_at', '')),
                'valid_at': str(getattr(edge, 'valid_at', '')),
                'invalid_at': str(getattr(edge, 'invalid_at', '')),
            }

            if hasattr(edge, 'source_node_uuid') and hasattr(edge, 'target_node_uuid'):
                fact_info['source'] = getattr(edge, 'source_node_uuid', '')
                fact_info['target'] = getattr(edge, 'target_node_uuid', '')

            facts.append(fact_info)

        return facts

    except Exception as e:
        logger.warning(f"Graphiti search error: {e}")
        return []


def search_graphiti(query: str, limit: int = 10, group_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for Graphiti vector search.

    The web server runs under eventlet which monkey-patches asyncio. To avoid
    conflicts, we spin up a fresh event loop in a dedicated thread via
    ThreadPoolExecutor. The 10s timeout prevents a slow Neo4j query from
    blocking the entire context build.

    Args:
        query: Natural language search query
        limit: Maximum number of results
        group_id: Optional group ID for multi-agent scoping (companion_id)

    Returns:
        List of dicts with fact information
    """
    import concurrent.futures

    def _run_in_thread():
        """Run async search in a new event loop in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(search_graphiti_async(query, limit, group_id=group_id))
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_thread)
            return future.result(timeout=10)
    except concurrent.futures.TimeoutError:
        logger.warning("Graphiti search timed out after 10s")
        return []
    except Exception as e:
        logger.warning(f"Graphiti search error: {e}")
        return []


# =============================================================================
# Prompt formatting - convert raw facts into LLM-readable context blocks
# =============================================================================

def format_graphiti_facts_for_prompt(facts: List[Dict[str, Any]], max_facts: int = 15) -> str:
    """
    Format Graphiti search results for inclusion in the companion's prompt.

    - Invalidated facts (invalid_at is set) are filtered out entirely.
    - Corrections are boosted to the top.
    - All facts get temporal tags so the LLM knows how old they are.
    - Duplicates are deduplicated by exact text match.
    """
    if not facts:
        return ""

    corrections = []
    regular_facts = []

    for fact_info in facts:
        fact_text = fact_info.get('fact', '')
        if not fact_text:
            continue

        # Skip invalidated facts — they've been superseded
        invalid_at = fact_info.get('invalid_at', '')
        if invalid_at and invalid_at not in ('', 'None', 'null'):
            continue

        if 'CORRECTION' in fact_text or 'LEARNED CORRECTION' in fact_text:
            corrections.append(fact_info)
        else:
            regular_facts.append(fact_info)

    ordered_facts = corrections + regular_facts

    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    lines = ["[KNOWLEDGE GRAPH - Temporal facts from conversations]"]
    lines.append(
        f"(Facts reference specific people by name. "
        f"Do NOT confuse {_pc.primary_user_name}'s facts with your own facts.)"
    )
    lines.append(
        "(Facts have timestamps showing when they were learned. "
        "Older facts may be outdated — trust entity profiles and recent "
        "corrections over old memories. If a fact seems wrong based on "
        "what you know now, ignore it.)"
    )

    if corrections:
        lines.append("")
        lines.append("[LEARNED CORRECTIONS - These override older information]")

    seen_facts: set = set()
    count = 0

    for fact_info in ordered_facts:
        fact_text = fact_info.get('fact', '')

        if not fact_text or fact_text in seen_facts:
            continue
        seen_facts.add(fact_text)

        time_tag = _format_time_ago(fact_info.get('created_at', ''))

        if 'CORRECTION' in fact_text:
            lines.append(f"  \u26a0\ufe0f {fact_text}")
        elif time_tag:
            lines.append(f"  - [{time_tag}] {fact_text}")
        else:
            lines.append(f"  - {fact_text}")

        count += 1
        if count >= max_facts:
            break

    if count == 0:
        return ""

    return "\n".join(lines)


def get_graphiti_context(query: str, limit: int = 15, group_id: Optional[str] = None) -> str:
    """
    Get formatted Graphiti context for a query.

    This is the main function used by context_builder.

    Args:
        query: User's message or search query
        limit: Maximum facts to return
        group_id: Optional group ID for multi-agent scoping (companion_id)

    Returns:
        Formatted context string for prompt
    """
    try:
        facts = search_graphiti(query, limit=limit, group_id=group_id)

        if not facts:
            logger.debug("No Graphiti facts found for query")
            return ""

        formatted = format_graphiti_facts_for_prompt(facts, max_facts=limit)
        logger.info(f"Graphiti returned {len(facts)} facts for context")
        return formatted

    except Exception as e:
        logger.warning(f"Graphiti context error: {e}")
        return ""


def get_graphiti_context_with_importance(query: str, limit: int = 10, group_id: Optional[str] = None) -> str:
    """
    Get Graphiti context with importance-boosted ranking.

    This is the preferred context function when importance scores are available.
    It fetches 3x the requested limit via vector search, then re-ranks by a
    weighted combination of:
      - Relevance (60%): position in the vector search results
      - Importance (40%): 0-10 score stored on the edge in Neo4j

    High-importance facts (>= 8) are labelled "[important]" in output.
    Each fact also gets a temporal tag ("2 days ago", "3 weeks ago") so the
    companion has a sense of how fresh each memory is.

    Falls back to get_graphiti_context() if importance lookup fails.

    Args:
        query: User's message
        limit: Maximum facts to return (default 10, reduced from 15 to minimize noise)
        group_id: Optional group ID for multi-agent scoping (companion_id)

    Returns:
        Formatted context string for prompt
    """
    try:
        # Fetch 3x candidates so we have room to re-rank and still fill `limit`.
        raw_facts = search_graphiti(query, limit=limit * 3, group_id=group_id)

        if not raw_facts:
            logger.debug("No Graphiti facts found for query")
            return ""

        # --- Fetch importance scores from Neo4j ---
        driver = _get_graphiti_neo4j_driver()
        fact_texts = [f.get('fact', '') for f in raw_facts if f.get('fact')]

        importance_map: Dict[str, int] = {}
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH ()-[edge:RELATES_TO]->()
                    WHERE edge.fact IN $facts
                    RETURN edge.fact as fact, edge.importance as importance
                """, facts=fact_texts)

                for record in result:
                    importance_map[record['fact']] = record['importance']
        except Exception as e:
            logger.warning(f"Importance lookup failed: {e}")

        # --- Re-rank: combine relevance position with importance score ---
        scored_facts = []
        for i, fact_info in enumerate(raw_facts):
            fact_text = fact_info.get('fact', '')
            if not fact_text:
                continue

            # Skip invalidated facts — they've been superseded by newer info
            invalid_at = fact_info.get('invalid_at', '')
            if invalid_at and invalid_at not in ('', 'None', 'null'):
                continue

            importance = importance_map.get(fact_text, 5) or 5  # Default 5 if unscored / None

            # Relevance: linear decay from 1.0 (first result) to ~0.0 (last)
            relevance = 1.0 - (i / len(raw_facts))

            # Normalize importance to 0-1 scale
            importance_normalized = importance / 10.0

            # Weighted combination: 60% relevance, 40% importance
            combined_score = (relevance * 0.6) + (importance_normalized * 0.4)

            scored_facts.append({
                'fact': fact_text,
                'score': combined_score,
                'importance': importance,
                'relevance_rank': i + 1,
                'created_at': fact_info.get('created_at', ''),
            })

        scored_facts.sort(key=lambda x: x['score'], reverse=True)

        # --- Format top results with temporal + importance annotations ---
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        lines = ["[CONTEXTUAL MEMORIES - Related to what you mentioned]"]
        lines.append(
            f"(Facts reference specific people by name. "
            f"Do NOT confuse {_pc.primary_user_name}'s facts with your own facts.)"
        )
        lines.append(
            "(Facts have timestamps. Older facts may be outdated — trust entity "
            "profiles and recent corrections over old memories.)"
        )
        seen: set = set()
        count = 0

        for sf in scored_facts:
            fact_text = sf['fact']
            if fact_text in seen:
                continue
            seen.add(fact_text)

            time_prefix = _format_time_ago(sf.get('created_at', ''))

            # Build annotation tags
            tags = []
            if sf['importance'] >= 8:
                tags.append("important")
            if time_prefix:
                tags.append(time_prefix)

            if tags:
                lines.append(f"  - [{', '.join(tags)}] {fact_text}")
            else:
                lines.append(f"  - {fact_text}")

            count += 1
            if count >= limit:
                break

        if count == 0:
            return ""

        logger.info(f"Graphiti context: {count} facts (importance-boosted)")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Graphiti context with importance error: {e}")
        # Fall back to regular (non-importance-boosted) context
        return get_graphiti_context(query, limit, group_id=group_id)


# =============================================================================
# Direct Neo4j Cypher queries (for biography synthesis and entity lookups)
#
# These bypass the Graphiti SDK entirely and issue Cypher directly. Used when
# we need ALL facts about an entity rather than the top-K vector matches.
# =============================================================================

_neo4j_driver = None


def _get_graphiti_neo4j_driver():
    """Get or create a cached Neo4j driver for direct Cypher queries."""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(
            NEO4J_GRAPHITI_URI,
            auth=(NEO4J_GRAPHITI_USER, NEO4J_GRAPHITI_PASSWORD) if NEO4J_GRAPHITI_PASSWORD else None
        )
    return _neo4j_driver


def get_entity_facts(entity_name: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get all facts (edges) involving an entity from Graphiti.

    Used by biography_synthesizer to get comprehensive facts about a person,
    not filtered by vector similarity.

    Args:
        entity_name: Name of the entity (e.g., "Companion", "User")
        limit: Maximum facts to return

    Returns:
        List of fact dicts with 'fact', 'fact_type', 'created_at', etc.
    """
    try:
        driver = _get_graphiti_neo4j_driver()

        with driver.session() as session:
            # Query edges where entity is source or target
            # Graphiti stores facts on edges between Entity nodes
            # Include importance score for weighted sampling
            result = session.run("""
                MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
                WHERE toLower(source.name) CONTAINS toLower($name)
                   OR toLower(target.name) CONTAINS toLower($name)
                RETURN edge.fact as fact,
                       edge.name as name,
                       edge.uuid as uuid,
                       edge.importance as importance,
                       source.name as source_entity,
                       target.name as target_entity,
                       edge.created_at as created_at,
                       edge.valid_at as valid_at,
                       edge.invalid_at as invalid_at
                ORDER BY edge.created_at DESC
                LIMIT $limit
            """, name=entity_name, limit=limit)

            facts = []
            for record in result:
                fact_dict = dict(record)
                # Map to format expected by biography_synthesizer
                facts.append({
                    'summary': fact_dict.get('fact', ''),
                    'fact': fact_dict.get('fact', ''),
                    'fact_type': 'graphiti_edge',
                    'uuid': fact_dict.get('uuid', ''),
                    'importance': fact_dict.get('importance'),  # None if not scored
                    'source_entity': fact_dict.get('source_entity', ''),
                    'target_entity': fact_dict.get('target_entity', ''),
                    'created_at': str(fact_dict.get('created_at', '')),
                    'valid_at': str(fact_dict.get('valid_at', '')),
                    'invalid_at': str(fact_dict.get('invalid_at', '')),
                })

            logger.info(f"Retrieved {len(facts)} facts for entity '{entity_name}' from Graphiti")
            return facts

    except Exception as e:
        logger.error(f"Failed to get entity facts from Graphiti: {e}")
        return []


def get_weighted_facts_for_biography(entity_name: str, medium_sample_size: int = 50) -> List[Dict[str, Any]]:
    """
    Get facts for biography using importance-weighted sampling.

    Called by the biography synthesis pipeline to gather material for
    generating a person's bio. Uses a tiered strategy to balance
    comprehensiveness with signal quality:

    Tiers:
      - Core (importance >= 7): Always included -- these are defining facts.
      - Medium (importance 5-6): Recency-sampled up to medium_sample_size.
      - Low (importance < 5): Excluded -- too mundane for a biography.
      - Unscored (importance is None): Treated as medium; backfill included
        if total selected facts is below the 100-fact minimum threshold.

    Planned/hypothetical facts are annotated with [PLANNED]/[HYPOTHETICAL]
    prefixes so the synthesizer can distinguish them from confirmed facts.

    Args:
        entity_name: Name of the entity (e.g., "Companion")
        medium_sample_size: How many medium-importance facts to sample

    Returns:
        List of facts suitable for biography synthesis
    """
    try:
        driver = _get_graphiti_neo4j_driver()

        with driver.session() as session:
            # Get ALL facts for this entity (we'll filter in Python)
            # Include fact_status for plan/action differentiation
            result = session.run("""
                MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
                WHERE toLower(source.name) CONTAINS toLower($name)
                   OR toLower(target.name) CONTAINS toLower($name)
                RETURN edge.fact as fact,
                       edge.name as name,
                       edge.uuid as uuid,
                       edge.importance as importance,
                       edge.fact_status as fact_status,
                       source.name as source_entity,
                       target.name as target_entity,
                       edge.created_at as created_at
                ORDER BY edge.created_at DESC
            """, name=entity_name)

            all_facts = []
            for record in result:
                fact_dict = dict(record)
                fact_status = fact_dict.get('fact_status', 'definite')  # Default to definite
                fact_text = fact_dict.get('fact', '')

                # Annotate fact text if planned/hypothetical
                if fact_status == 'planned':
                    annotated_text = f"[PLANNED] {fact_text}"
                elif fact_status == 'hypothetical':
                    annotated_text = f"[HYPOTHETICAL] {fact_text}"
                else:
                    annotated_text = fact_text

                all_facts.append({
                    'summary': annotated_text,
                    'fact': fact_text,
                    'fact_type': 'graphiti_edge',
                    'uuid': fact_dict.get('uuid', ''),
                    'importance': fact_dict.get('importance'),
                    'fact_status': fact_status,
                    'source_entity': fact_dict.get('source_entity', ''),
                    'target_entity': fact_dict.get('target_entity', ''),
                    'created_at': str(fact_dict.get('created_at', '')),
                })

        if not all_facts:
            return []

        # --- Bucket facts by importance tier ---
        core_facts = []      # importance >= 7: always include
        medium_facts = []    # importance 5-6: sample from these
        unscored_facts = []  # importance is None: treat as medium

        # Track fact_status distribution for diagnostics
        status_counts: Dict[str, int] = {'definite': 0, 'planned': 0, 'hypothetical': 0, 'unknown': 0}

        for fact in all_facts:
            importance = fact.get('importance')
            status = fact.get('fact_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

            if importance is None:
                unscored_facts.append(fact)
            elif importance >= 7:
                core_facts.append(fact)
            elif importance >= 5:
                medium_facts.append(fact)
            # importance < 5: intentionally excluded from biography

        logger.info(
            f"Weighted sampling for {entity_name}: "
            f"{len(core_facts)} core, {len(medium_facts)} medium, "
            f"{len(unscored_facts)} unscored, "
            f"{len(all_facts) - len(core_facts) - len(medium_facts) - len(unscored_facts)} excluded"
        )
        logger.info(
            f"Fact status distribution: "
            f"{status_counts.get('definite', 0)} definite, "
            f"{status_counts.get('planned', 0)} planned, "
            f"{status_counts.get('hypothetical', 0)} hypothetical"
        )

        # --- Assemble final selection ---
        selected_facts = core_facts.copy()

        # Medium facts are already sorted by recency (ORDER BY created_at DESC
        # in the Cypher query), so slicing gives us the most recent ones.
        if medium_facts:
            selected_facts.extend(medium_facts[:medium_sample_size])

        # Backfill with unscored facts if we have not yet reached the
        # minimum 100-fact threshold needed for a meaningful biography.
        if len(selected_facts) < 100 and unscored_facts:
            needed = min(100 - len(selected_facts), len(unscored_facts))
            selected_facts.extend(unscored_facts[:needed])
            logger.info(f"Added {needed} unscored facts to reach minimum threshold")

        logger.info(f"Selected {len(selected_facts)} facts for biography")
        return selected_facts

    except Exception as e:
        logger.error(f"Failed to get weighted facts: {e}")
        return []


# =============================================================================
# Temporal queries - for when the user references a specific time window
# =============================================================================

def search_facts_in_time_range(
    query: str,
    time_start: datetime,
    time_end: datetime,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Search Graphiti facts within a specific time range.

    Used when the user references a time period like "yesterday" or "last
    night". Issues a direct Cypher query with both a time window filter and
    an optional case-insensitive text match on the fact body.

    Args:
        query: Search query (e.g., "lunch", "dinner") -- pass '' for all
        time_start: Start of time range (inclusive)
        time_end: End of time range (inclusive)
        limit: Maximum facts to return

    Returns:
        List of facts created within the time range, ordered by recency
    """
    try:
        driver = _get_graphiti_neo4j_driver()

        with driver.session() as session:
            # Query facts within time range, optionally filtered by query terms
            # We do a case-insensitive search on the fact text
            result = session.run("""
                MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
                WHERE edge.created_at >= $time_start
                  AND edge.created_at <= $time_end
                  AND ($query = '' OR toLower(edge.fact) CONTAINS toLower($query))
                RETURN edge.fact as fact,
                       edge.importance as importance,
                       edge.created_at as created_at,
                       source.name as source_entity,
                       target.name as target_entity
                ORDER BY edge.created_at DESC
                LIMIT $limit
            """, time_start=time_start.isoformat(), time_end=time_end.isoformat(),
                query=query, limit=limit)

            facts = []
            for record in result:
                facts.append({
                    'fact': record['fact'],
                    'importance': record['importance'],
                    'created_at': str(record['created_at']),
                    'source_entity': record['source_entity'],
                    'target_entity': record['target_entity'],
                })

            logger.info(f"Temporal search: {len(facts)} facts for '{query}' between {time_start} and {time_end}")
            return facts

    except Exception as e:
        logger.error(f"Temporal fact search failed: {e}")
        return []


def format_temporal_facts_for_prompt(facts: List[Dict[str, Any]], period_description: str) -> str:
    """
    Format temporally-filtered facts for prompt inclusion.

    Args:
        facts: Facts from search_facts_in_time_range
        period_description: Human-readable time description (e.g., "yesterday")

    Returns:
        Formatted string for prompt
    """
    if not facts:
        return ""

    lines = [f"[MEMORIES FROM {period_description.upper()}]"]

    for fact in facts:
        importance = fact.get('importance')
        fact_text = fact.get('fact', '')

        if importance and importance >= 7:
            lines.append(f"  - [important] {fact_text}")
        else:
            lines.append(f"  - {fact_text}")

    return "\n".join(lines)


def get_entity_relationships(entity_name: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get relationships for an entity from Graphiti.

    Args:
        entity_name: Name of the entity
        limit: Maximum relationships to return

    Returns:
        List of relationship dicts
    """
    try:
        driver = _get_graphiti_neo4j_driver()

        with driver.session() as session:
            result = session.run("""
                MATCH (source:Entity)-[edge:RELATES_TO]->(target:Entity)
                WHERE toLower(source.name) = toLower($name)
                RETURN edge.name as relationship,
                       target.name as target,
                       edge.fact as description,
                       edge.created_at as created_at
                ORDER BY edge.created_at DESC
                LIMIT $limit
            """, name=entity_name, limit=limit)

            relationships = []
            for record in result:
                rel_dict = dict(record)
                relationships.append({
                    'relationship': rel_dict.get('relationship', 'RELATES_TO'),
                    'target': rel_dict.get('target', ''),
                    'description': rel_dict.get('description', ''),
                    'created_at': str(rel_dict.get('created_at', '')),
                })

            logger.info(f"Retrieved {len(relationships)} relationships for '{entity_name}' from Graphiti")
            return relationships

    except Exception as e:
        logger.error(f"Failed to get entity relationships from Graphiti: {e}")
        return []


# =============================================================================
# Convenience wrapper class + singleton accessor
# =============================================================================

class GraphitiSearch:
    """Stateless convenience class that delegates to the module-level functions."""

    def search(self, query: str, limit: int = 10, group_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for facts matching query."""
        return search_graphiti(query, limit, group_id=group_id)

    def get_context(self, query: str, limit: int = 15, group_id: Optional[str] = None) -> str:
        """Get formatted context for prompt."""
        return get_graphiti_context(query, limit, group_id=group_id)

    def get_entity_facts(self, entity_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all facts about an entity (for biography synthesis)."""
        return get_entity_facts(entity_name, limit)

    def get_entity_relationships(self, entity_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get relationships for an entity."""
        return get_entity_relationships(entity_name, limit)


_graphiti_search: Optional[GraphitiSearch] = None


def get_graphiti_search() -> GraphitiSearch:
    """Get GraphitiSearch singleton."""
    global _graphiti_search
    if _graphiti_search is None:
        _graphiti_search = GraphitiSearch()
    return _graphiti_search
