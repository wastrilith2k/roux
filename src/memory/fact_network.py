"""
Fact Network - Brain-like associative memory linking facts to facts.

WHAT: Maintains a directed graph of typed links between facts in PostgreSQL.
Supports spreading activation retrieval where querying one fact automatically
pulls in causally, temporally, or semantically connected facts -- mimicking
how human memory works.

WHY: Individual facts in isolation lack context. "Jesse went to the ER"
is more useful when linked to "Jesse has ADHD" (explains) and "James is
stressed at work" (causes). The fact network captures these relationships
so the companion can retrieve richer, more connected context without the
user having to ask for it explicitly.

HOW it fits:
  - fact_store.store_fact() calls detect_links_for_new_fact() in a
    background thread after each new fact is persisted.
  - fact_store.search_with_spreading_activation() delegates to
    spreading_activation() here to expand seed facts through the network.
  - relationship_store provides entity-level bridges that augment the
    fact-level links (e.g., parent_of relationship lets Jesse's facts
    activate James's facts).

Link types:
  - explains: A provides context for B (ADHD explains ER visits)
  - causes: A leads to B (crisis causes stress)
  - contradicts: A conflicts with B (employed vs unemployed)
  - temporal_before: A happened before B (interview then job offer)
  - same_event: Both part of one event (related ER visits)
  - similar: Semantically related (both about work)
  - supports: A reinforces B (multiple evidence)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import psycopg2
from psycopg2.extras import RealDictCursor

from src.database import tables as T

logger = logging.getLogger(__name__)


class LinkType(str, Enum):
    """Types of relationships between facts."""
    EXPLAINS = "explains"        # A provides context for B
    CAUSES = "causes"            # A leads to B
    CONTRADICTS = "contradicts"  # A conflicts with B
    TEMPORAL_BEFORE = "temporal_before"  # A happened before B
    SAME_EVENT = "same_event"    # Both part of same event
    SIMILAR = "similar"          # Semantically related
    SUPPORTS = "supports"        # A reinforces/supports B


@dataclass
class FactLink:
    """A link between two facts."""
    source_fact_id: int
    target_fact_id: int
    link_type: str
    strength: float  # 0.0-1.0, how strong the association
    context: str = ""  # Why they're linked
    created_at: Optional[datetime] = None


class FactNetwork:
    """
    Manages fact-to-fact links and spreading activation retrieval.

    Like a brain's associative memory - facts prime related facts.
    """

    def __init__(self):
        self._conn = None

    def _get_connection(self):
        """Get database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def create_link(
        self,
        source_fact_id: int,
        target_fact_id: int,
        link_type: str,
        strength: float = 0.5,
        context: str = ""
    ) -> bool:
        """
        Create a link between two facts.

        Returns True if created, False if already exists or error.
        """
        if source_fact_id == target_fact_id:
            return False

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.FACT_LINKS} (source_fact_id, target_fact_id, link_type, strength, context)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_fact_id, target_fact_id, link_type)
                    DO UPDATE SET strength = GREATEST({T.FACT_LINKS}.strength, EXCLUDED.strength),
                                  context = EXCLUDED.context
                    RETURNING id
                """, (source_fact_id, target_fact_id, link_type, strength, context))

                conn.commit()
                logger.debug(f"Created link: {source_fact_id} --{link_type}--> {target_fact_id}")
                return True

        except Exception as e:
            logger.warning(f"Failed to create fact link: {e}")
            conn.rollback()
            return False

    def get_linked_facts(
        self,
        fact_id: int,
        link_types: List[str] = None,
        min_strength: float = 0.3,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Dict[str, Any]]:
        """
        Get facts linked to a given fact.

        Args:
            fact_id: Source fact ID
            link_types: Filter by link types (None = all)
            min_strength: Minimum link strength
            direction: Which direction to follow links

        Returns:
            List of linked facts with link metadata
        """
        conn = self._get_connection()
        results = []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Outgoing links (this fact -> other facts)
                if direction in ("outgoing", "both"):
                    query = f"""
                        SELECT f.*, fl.link_type, fl.strength, fl.context as link_context,
                               'outgoing' as link_direction
                        FROM {T.FACT_LINKS} fl
                        JOIN {T.FACTS} f ON fl.target_fact_id = f.id
                        WHERE fl.source_fact_id = %s
                          AND fl.strength >= %s
                          AND f.archived_at IS NULL
                    """
                    params = [fact_id, min_strength]

                    if link_types:
                        query += " AND fl.link_type = ANY(%s)"
                        params.append(link_types)

                    cursor.execute(query, params)
                    results.extend([dict(row) for row in cursor.fetchall()])

                # Incoming links (other facts -> this fact)
                if direction in ("incoming", "both"):
                    query = f"""
                        SELECT f.*, fl.link_type, fl.strength, fl.context as link_context,
                               'incoming' as link_direction
                        FROM {T.FACT_LINKS} fl
                        JOIN {T.FACTS} f ON fl.source_fact_id = f.id
                        WHERE fl.target_fact_id = %s
                          AND fl.strength >= %s
                          AND f.archived_at IS NULL
                    """
                    params = [fact_id, min_strength]

                    if link_types:
                        query += " AND fl.link_type = ANY(%s)"
                        params.append(link_types)

                    cursor.execute(query, params)
                    results.extend([dict(row) for row in cursor.fetchall()])

            return results

        except Exception as e:
            logger.warning(f"Failed to get linked facts: {e}")
            return []

    # =========================================================================
    # Spreading activation retrieval
    # =========================================================================

    def spreading_activation(
        self,
        seed_fact_ids: List[int],
        max_depth: int = 2,
        min_activation: float = 0.2,
        decay_factor: float = 0.6,
        max_results: int = 20,
        relationship_bridge: bool = True,
        bridge_decay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Spreading activation retrieval -- brain-like associative recall.

        Algorithm:
          1. Seed facts start with activation = 1.0.
          2. At each depth level, activation spreads to linked facts:
             spread = source_activation * link_strength * decay_factor^depth
          3. If relationship_bridge is True, entity relationships (parent_of,
             works_at, etc.) act as additional bridges: facts about entity B
             get activated when entity A's facts are active, weighted by the
             relationship confidence * bridge_decay.
          4. Facts accumulate activation from multiple sources (max, not sum,
             to avoid runaway amplification).
          5. All facts with activation >= min_activation are returned.

        Each result dict includes:
          - activation: float score
          - is_seed: whether it was a direct search hit
          - activation_source: 'direct' | 'fact_link' | 'relationship_bridge'

        Args:
            seed_fact_ids: Initial facts (e.g., from semantic search)
            max_depth: How many hops to follow (1-3 recommended)
            min_activation: Minimum activation to include in results
            decay_factor: Activation decay per hop (0-1)
            max_results: Maximum facts to return
            relationship_bridge: Use entity relationships as additional bridges
            bridge_decay: Extra decay multiplier for relationship bridges

        Returns:
            List of fact dicts with activation scores, sorted descending
        """
        if not seed_fact_ids:
            return []

        # Optionally load relationship store for bridge activation
        relationship_store = None
        if relationship_bridge:
            try:
                from src.memory.relationship_store import get_relationship_store
                relationship_store = get_relationship_store()
            except Exception as e:
                logger.debug(f"Could not load relationship store for bridge activation: {e}")

        # Track activation levels: fact_id -> activation
        activations: Dict[int, float] = {}

        # Initialize seeds with full activation
        for fact_id in seed_fact_ids:
            activations[fact_id] = 1.0

        # Spread activation through network
        current_frontier = set(seed_fact_ids)
        visited = set(seed_fact_ids)

        # Cache subjects for activated facts (for relationship bridging)
        # fact_id -> subject
        fact_subjects: Dict[int, str] = {}
        # Track which facts came from relationship bridging
        bridge_activated: Set[int] = set()

        # Pre-load subjects for seed facts
        if relationship_store:
            try:
                conn = self._get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(f"""
                        SELECT id, subject FROM {T.FACTS}
                        WHERE id = ANY(%s) AND archived_at IS NULL
                    """, (list(seed_fact_ids),))
                    for row in cursor.fetchall():
                        fact_subjects[row['id']] = row['subject']
            except Exception as e:
                logger.debug(f"Could not pre-load fact subjects: {e}")

        for depth in range(max_depth):
            next_frontier = set()
            current_decay = decay_factor ** (depth + 1)

            for fact_id in current_frontier:
                source_activation = activations.get(fact_id, 0)

                # Get linked facts via fact_links
                linked = self.get_linked_facts(fact_id, min_strength=0.3)

                for linked_fact in linked:
                    linked_id = linked_fact['id']
                    link_strength = linked_fact.get('strength', 0.5)

                    # Calculate activation spreading to this fact
                    spread_activation = source_activation * link_strength * current_decay

                    if spread_activation >= min_activation:
                        # Accumulate activation (facts can receive from multiple sources)
                        current = activations.get(linked_id, 0)
                        activations[linked_id] = max(current, spread_activation)

                        if linked_id not in visited:
                            next_frontier.add(linked_id)
                            visited.add(linked_id)
                            # Cache subject for bridge use
                            if relationship_store and 'subject' in linked_fact:
                                fact_subjects[linked_id] = linked_fact['subject']

            # Relationship bridge: use entity relationships to cross-activate facts
            if relationship_store:
                bridged = self._apply_relationship_bridge(
                    current_frontier=current_frontier,
                    fact_subjects=fact_subjects,
                    activations=activations,
                    visited=visited,
                    relationship_store=relationship_store,
                    bridge_decay=bridge_decay,
                    current_decay=current_decay,
                    min_activation=min_activation,
                )
                bridge_activated.update(bridged)
                next_frontier.update(bridged)

            current_frontier = next_frontier

            if not current_frontier:
                break

        # Fetch full fact details for activated facts
        conn = self._get_connection()
        results = []

        try:
            # Filter and sort by activation
            activated_ids = [
                (fid, act) for fid, act in activations.items()
                if act >= min_activation
            ]
            activated_ids.sort(key=lambda x: x[1], reverse=True)
            activated_ids = activated_ids[:max_results]

            if not activated_ids:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                fact_ids = [fid for fid, _ in activated_ids]
                cursor.execute(f"""
                    SELECT * FROM {T.FACTS}
                    WHERE id = ANY(%s)
                    AND archived_at IS NULL
                """, (fact_ids,))

                facts_by_id = {row['id']: dict(row) for row in cursor.fetchall()}

                for fact_id, activation in activated_ids:
                    if fact_id in facts_by_id:
                        fact = facts_by_id[fact_id]
                        fact['activation'] = activation
                        fact['is_seed'] = fact_id in seed_fact_ids
                        if fact_id in bridge_activated:
                            fact['activation_source'] = 'relationship_bridge'
                        elif fact_id in seed_fact_ids:
                            fact['activation_source'] = 'direct'
                        else:
                            fact['activation_source'] = 'fact_link'
                        results.append(fact)

            bridge_count = sum(1 for f in results if not f.get('is_seed') and f['id'] not in seed_fact_ids)
            logger.debug(
                f"Spreading activation: {len(seed_fact_ids)} seeds -> {len(results)} activated facts"
                f" ({bridge_count} via links/bridges)"
            )
            return results

        except Exception as e:
            logger.warning(f"Spreading activation failed: {e}")
            return []

    def _apply_relationship_bridge(
        self,
        current_frontier: Set[int],
        fact_subjects: Dict[int, str],
        activations: Dict[int, float],
        visited: Set[int],
        relationship_store,
        bridge_decay: float,
        current_decay: float,
        min_activation: float,
    ) -> Set[int]:
        """
        Apply relationship bridging: facts about entity A activate facts about
        entity B when A and B have a known relationship.

        Returns set of newly activated fact IDs to add to next frontier.
        """
        bridged_frontier: Set[int] = set()

        try:
            # Collect unique subjects from current frontier
            activated_subjects: Dict[str, float] = {}  # subject -> max activation
            for fact_id in current_frontier:
                subject = fact_subjects.get(fact_id)
                if subject:
                    act = activations.get(fact_id, 0)
                    activated_subjects[subject] = max(activated_subjects.get(subject, 0), act)

            if not activated_subjects:
                return bridged_frontier

            # For each subject, find related entities via relationships
            for subject, source_activation in activated_subjects.items():
                try:
                    related = relationship_store.get_relationships_for_entity(
                        entity=subject,
                        include_as_target=True,
                        only_current=True,
                        min_confidence=0.5
                    )
                except Exception:
                    continue

                for rel in related:
                    # Determine the "other" entity in the relationship
                    src = rel.get('source_entity', '')
                    tgt = rel.get('target_entity', '')
                    other_entity = tgt if src.lower() == subject.lower() else src
                    rel_confidence = rel.get('confidence', 0.7)

                    # Get top facts about the related entity
                    try:
                        from src.memory.fact_store import get_fact_store
                        store = get_fact_store()
                        related_facts = store.get_facts_for_entities(
                            entities=[other_entity], limit=5
                        )
                    except Exception:
                        continue

                    for fact in related_facts:
                        fact_id = fact['id']
                        bridge_activation = source_activation * rel_confidence * bridge_decay * current_decay

                        if bridge_activation >= min_activation and fact_id not in visited:
                            current = activations.get(fact_id, 0)
                            activations[fact_id] = max(current, bridge_activation)
                            bridged_frontier.add(fact_id)
                            visited.add(fact_id)
                            # Cache subject for potential deeper bridging
                            fact_subjects[fact_id] = fact.get('subject', other_entity)

            if bridged_frontier:
                logger.debug(f"Relationship bridge activated {len(bridged_frontier)} additional facts")

        except Exception as e:
            logger.debug(f"Relationship bridge failed (non-critical): {e}")

        return bridged_frontier

    def detect_links_for_new_fact(
        self,
        new_fact_id: int,
        new_fact_subject: str,
        new_fact_object: str,
        recent_fact_ids: List[int] = None,
        use_llm: bool = True
    ) -> List[FactLink]:
        """
        Detect links between a new fact and existing facts.

        Called after storing a new fact to wire it into the network.

        Args:
            new_fact_id: ID of newly stored fact
            new_fact_subject: Subject of new fact
            new_fact_object: Object/content of new fact
            recent_fact_ids: Specific facts to check (None = auto-select)
            use_llm: Whether to use LLM for link detection

        Returns:
            List of created links
        """
        conn = self._get_connection()
        created_links = []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get candidate facts to link with
                if recent_fact_ids:
                    cursor.execute(f"""
                        SELECT id, subject, predicate, object, importance
                        FROM {T.FACTS}
                        WHERE id = ANY(%s)
                        AND id != %s
                        AND archived_at IS NULL
                    """, (recent_fact_ids, new_fact_id))
                else:
                    # Get recent facts about same subject or high importance
                    cursor.execute(f"""
                        SELECT id, subject, predicate, object, importance
                        FROM {T.FACTS}
                        WHERE id != %s
                        AND archived_at IS NULL
                        AND (
                            subject = %s
                            OR importance >= 7
                            OR created_at > NOW() - INTERVAL '7 days'
                        )
                        ORDER BY
                            CASE WHEN subject = %s THEN 0 ELSE 1 END,
                            importance DESC,
                            created_at DESC
                        LIMIT 30
                    """, (new_fact_id, new_fact_subject, new_fact_subject))

                candidates = cursor.fetchall()

            if not candidates:
                return []

            # Detect links using LLM or heuristics
            if use_llm:
                links = self._detect_links_llm(
                    new_fact_id, new_fact_subject, new_fact_object,
                    [dict(c) for c in candidates]
                )
            else:
                links = self._detect_links_heuristic(
                    new_fact_id, new_fact_subject, new_fact_object,
                    [dict(c) for c in candidates]
                )

            # Create the links
            for link in links:
                if self.create_link(
                    source_fact_id=link.source_fact_id,
                    target_fact_id=link.target_fact_id,
                    link_type=link.link_type,
                    strength=link.strength,
                    context=link.context
                ):
                    created_links.append(link)

            if created_links:
                logger.info(f"Created {len(created_links)} links for fact {new_fact_id}")

            return created_links

        except Exception as e:
            logger.warning(f"Link detection failed: {e}")
            return []

    def _detect_links_heuristic(
        self,
        new_fact_id: int,
        new_subject: str,
        new_object: str,
        candidates: List[Dict]
    ) -> List[FactLink]:
        """
        Detect links using simple heuristics (fast, no LLM).

        Rules:
        - Same subject -> SIMILAR link
        - Temporal keywords -> TEMPORAL_BEFORE link
        - Crisis keywords shared -> SAME_EVENT link
        """
        links = []
        new_object_lower = new_object.lower()

        # Keywords that indicate crisis/event grouping
        crisis_keywords = ['er', 'hospital', 'ran away', 'police', 'crisis', 'emergency', 'suicidal']
        work_keywords = ['interview', 'job', 'hired', 'fired', 'layoff', 'offer']

        def has_keywords(text, keywords):
            text_lower = text.lower()
            return any(kw in text_lower for kw in keywords)

        new_is_crisis = has_keywords(new_object, crisis_keywords)
        new_is_work = has_keywords(new_object, work_keywords)

        for candidate in candidates:
            cand_id = candidate['id']
            cand_subject = candidate['subject']
            cand_object = candidate['object'] or ''
            cand_object_lower = cand_object.lower()

            # Same subject -> SIMILAR
            if cand_subject == new_subject:
                # Higher strength for same predicate
                strength = 0.6

                # Crisis facts about same person -> SAME_EVENT
                cand_is_crisis = has_keywords(cand_object, crisis_keywords)
                if new_is_crisis and cand_is_crisis:
                    links.append(FactLink(
                        source_fact_id=new_fact_id,
                        target_fact_id=cand_id,
                        link_type=LinkType.SAME_EVENT.value,
                        strength=0.8,
                        context="Both crisis-related facts about same person"
                    ))
                    continue

                # Work facts about same person -> SIMILAR
                cand_is_work = has_keywords(cand_object, work_keywords)
                if new_is_work and cand_is_work:
                    links.append(FactLink(
                        source_fact_id=new_fact_id,
                        target_fact_id=cand_id,
                        link_type=LinkType.SIMILAR.value,
                        strength=0.7,
                        context="Both work-related facts about same person"
                    ))
                    continue

                # Generic same-subject link
                links.append(FactLink(
                    source_fact_id=new_fact_id,
                    target_fact_id=cand_id,
                    link_type=LinkType.SIMILAR.value,
                    strength=strength,
                    context=f"Both facts about {new_subject}"
                ))

        return links

    def _detect_links_llm(
        self,
        new_fact_id: int,
        new_subject: str,
        new_object: str,
        candidates: List[Dict]
    ) -> List[FactLink]:
        """
        Detect links using LLM for deeper understanding.

        More accurate but slower than heuristics.
        """
        try:
            from src.llm.provider_factory import generate_sync

            # Format candidates for prompt
            candidate_text = "\n".join([
                f"[{c['id']}] {c['subject']}: {c['object'][:100]}"
                for c in candidates[:15]  # Limit to avoid token overflow
            ])

            prompt = f"""Analyze relationships between a new fact and existing facts.

NEW FACT (ID: {new_fact_id}):
{new_subject}: {new_object}

EXISTING FACTS:
{candidate_text}

For each existing fact that has a meaningful relationship to the new fact, output a line:
LINK: [existing_fact_id] [link_type] [strength] [reason]

Link types (use exactly these):
- explains: The existing fact explains/provides context for the new fact
- causes: The existing fact caused or led to the new fact
- contradicts: The facts conflict with each other
- temporal_before: The existing fact happened before the new fact
- same_event: Both facts are about the same event/situation
- similar: Facts are semantically related (same topic)
- supports: Facts reinforce each other

Strength: 0.5 (weak) to 0.9 (strong)

Only output links for facts with clear relationships. Skip weak/tenuous connections.
Output nothing if no meaningful links exist.

Example output:
LINK: 42 explains 0.8 ADHD explains behavioral issues
LINK: 37 same_event 0.9 Both about January crisis"""

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2
            )

            if not response:
                return self._detect_links_heuristic(new_fact_id, new_subject, new_object, candidates)

            # Parse response
            links = []
            for line in response.strip().split('\n'):
                if line.startswith('LINK:'):
                    parts = line.replace('LINK:', '').strip().split(' ', 3)
                    if len(parts) >= 3:
                        try:
                            target_id = int(parts[0])
                            link_type = parts[1]
                            strength = float(parts[2])
                            context = parts[3] if len(parts) > 3 else ""

                            # Validate link type
                            valid_types = [lt.value for lt in LinkType]
                            if link_type in valid_types:
                                links.append(FactLink(
                                    source_fact_id=new_fact_id,
                                    target_fact_id=target_id,
                                    link_type=link_type,
                                    strength=min(0.95, max(0.3, strength)),
                                    context=context
                                ))
                        except (ValueError, IndexError):
                            continue

            return links if links else self._detect_links_heuristic(new_fact_id, new_subject, new_object, candidates)

        except Exception as e:
            logger.warning(f"LLM link detection failed: {e}")
            return self._detect_links_heuristic(new_fact_id, new_subject, new_object, candidates)

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the fact network."""
        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT
                        COUNT(*) as total_links,
                        COUNT(DISTINCT source_fact_id) as facts_with_outgoing,
                        COUNT(DISTINCT target_fact_id) as facts_with_incoming,
                        AVG(strength) as avg_strength,
                        COUNT(*) FILTER (WHERE link_type = 'explains') as explains_count,
                        COUNT(*) FILTER (WHERE link_type = 'causes') as causes_count,
                        COUNT(*) FILTER (WHERE link_type = 'similar') as similar_count,
                        COUNT(*) FILTER (WHERE link_type = 'same_event') as same_event_count
                    FROM {T.FACT_LINKS}
                """)
                return dict(cursor.fetchone())

        except Exception as e:
            logger.warning(f"Could not get network stats: {e}")
            return {}


# Singleton instance
_fact_network: Optional[FactNetwork] = None


def get_fact_network() -> FactNetwork:
    """Get singleton FactNetwork instance."""
    global _fact_network
    if _fact_network is None:
        _fact_network = FactNetwork()
    return _fact_network
