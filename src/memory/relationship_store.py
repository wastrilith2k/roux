"""
Relationship Store - Structured relationships between known entities.

WHAT: PostgreSQL-backed storage for typed, confidence-scored relationships
between entities (people, organizations, pets). Supports bidirectional
queries, contradiction detection, temporal validity (ended relationships),
and prompt-ready formatting.

WHY: The generic fact store and Graphiti both represent relationships as
untyped text blobs. This store solves the "trash relationship" problem by
enforcing explicit RelationshipType enums, tracking how many times each
relationship has been mentioned (verification count), detecting logical
contradictions (e.g., two simultaneous marriages), and supporting
temporal validity so ended relationships are preserved but filtered out of
active queries.

HOW it fits:
  - relationship_extractor.py calls store_relationship() after extracting
    typed relationships from conversation.
  - context_builder.py calls format_relationships_for_prompt() to inject
    an entity's relationship map into the LLM prompt.
  - Spreading activation in fact_network.py uses get_relationships_for_entity()
    to bridge between entities (e.g., facts about Jesse activate when James
    is the seed entity, via the parent_of relationship).

Quality model:
  - confidence (0-1): certainty level, grows with repeated mentions
  - mention_count: how many times referenced in conversation
  - last_verified: timestamp of most recent confirmation
  - contradiction_notes: flagged conflicts with existing relationships
  - valid_until: NULL while active; set when a relationship ends
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

from src.database import tables as T

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


class RelationshipType(str, Enum):
    """Explicit relationship types - no generic RELATES_TO."""

    # Family relationships
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    SIBLING_OF = "sibling_of"
    MARRIED_TO = "married_to"
    DIVORCED_FROM = "divorced_from"
    SEPARATED_FROM = "separated_from"

    # Romantic relationships
    PARTNER_OF = "partner_of"
    DATING = "dating"
    EX_PARTNER_OF = "ex_partner_of"

    # Professional relationships
    WORKS_AT = "works_at"
    COWORKER_OF = "coworker_of"
    MANAGER_OF = "manager_of"
    REPORTS_TO = "reports_to"

    # Social relationships
    FRIEND_OF = "friend_of"
    ACQUAINTANCE_OF = "acquaintance_of"
    NEIGHBOR_OF = "neighbor_of"

    # Pet relationships
    OWNS_PET = "owns_pet"
    PET_OF = "pet_of"

    # Living arrangements
    LIVES_WITH = "lives_with"
    ROOMMATE_OF = "roommate_of"

    # Other
    KNOWS = "knows"  # Fallback for unclear relationships


# Inverse relationships for bidirectional queries
INVERSE_RELATIONSHIPS = {
    RelationshipType.PARENT_OF: RelationshipType.CHILD_OF,
    RelationshipType.CHILD_OF: RelationshipType.PARENT_OF,
    RelationshipType.SIBLING_OF: RelationshipType.SIBLING_OF,
    RelationshipType.MARRIED_TO: RelationshipType.MARRIED_TO,
    RelationshipType.DIVORCED_FROM: RelationshipType.DIVORCED_FROM,
    RelationshipType.SEPARATED_FROM: RelationshipType.SEPARATED_FROM,
    RelationshipType.PARTNER_OF: RelationshipType.PARTNER_OF,
    RelationshipType.DATING: RelationshipType.DATING,
    RelationshipType.EX_PARTNER_OF: RelationshipType.EX_PARTNER_OF,
    RelationshipType.WORKS_AT: RelationshipType.WORKS_AT,
    RelationshipType.COWORKER_OF: RelationshipType.COWORKER_OF,
    RelationshipType.MANAGER_OF: RelationshipType.REPORTS_TO,
    RelationshipType.REPORTS_TO: RelationshipType.MANAGER_OF,
    RelationshipType.FRIEND_OF: RelationshipType.FRIEND_OF,
    RelationshipType.ACQUAINTANCE_OF: RelationshipType.ACQUAINTANCE_OF,
    RelationshipType.NEIGHBOR_OF: RelationshipType.NEIGHBOR_OF,
    RelationshipType.OWNS_PET: RelationshipType.PET_OF,
    RelationshipType.PET_OF: RelationshipType.OWNS_PET,
    RelationshipType.LIVES_WITH: RelationshipType.LIVES_WITH,
    RelationshipType.ROOMMATE_OF: RelationshipType.ROOMMATE_OF,
    RelationshipType.KNOWS: RelationshipType.KNOWS,
}


@dataclass
class Relationship:
    """A structured relationship between two entities."""
    id: int
    source_entity: str  # e.g., "James"
    relationship_type: RelationshipType
    target_entity: str  # e.g., "Jesse"

    # Quality metrics
    confidence: float  # 0-1, how certain is this?
    mention_count: int  # How many times referenced
    last_verified: datetime  # Last time confirmed

    # Temporal validity
    valid_from: Optional[datetime]  # When relationship started
    valid_until: Optional[datetime]  # When relationship ended (None if current)

    # Context
    context: Optional[str]  # Additional context about the relationship
    source_message_id: Optional[int]  # Message that established this

    # Flags
    is_primary: bool  # Is this the primary relationship between these entities?
    contradiction_notes: Optional[str]  # Notes about conflicts


class RelationshipStore:
    """PostgreSQL-backed storage for structured relationships."""

    def __init__(self):
        self._conn = None
        self._current_email = None

    def _get_connection(self, user_email: str = None):
        """Get database connection with correct schema search path."""
        needs_new = (self._conn is None or self._conn.closed or
                     (user_email and user_email != self._current_email))
        if needs_new:
            if self._conn and not self._conn.closed:
                self._conn.close()
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
            if user_email:
                from src.database.schema_manager import set_search_path
                set_search_path(self._conn, user_email)
                self._current_email = user_email
        return self._conn

    def store_relationship(
        self,
        source_entity: str,
        relationship_type: RelationshipType,
        target_entity: str,
        confidence: float = 0.7,
        context: str = None,
        source_message_id: int = None,
        user_email: str = None,
        valid_from: datetime = None
    ) -> Optional[int]:
        """
        Store a new relationship, or reinforce an existing one.

        Deduplication: if the same (source, type, target) triple already
        exists and is still active (valid_until IS NULL), we reinforce it
        by incrementing mention_count and computing a running weighted
        average of confidence scores. This means a relationship mentioned
        10 times converges toward high confidence.

        Before inserting a genuinely new relationship, we check for logical
        contradictions (e.g., two simultaneous marriages) and attach
        contradiction notes if found.

        Returns relationship ID or None if failed.
        """
        conn = self._get_connection(user_email)

        # Normalize entity names
        source_entity = source_entity.strip().title()
        target_entity = target_entity.strip().title()

        try:
            with conn.cursor() as cursor:
                # Check for existing relationship
                cursor.execute(f"""
                    SELECT id, confidence, mention_count
                    FROM {T.RELATIONSHIPS}
                    WHERE source_entity = %s
                    AND relationship_type = %s
                    AND target_entity = %s
                    AND (user_email = %s OR user_email IS NULL)
                    AND valid_until IS NULL
                """, (source_entity, relationship_type.value, target_entity, user_email))

                existing = cursor.fetchone()

                if existing:
                    # Reinforce existing relationship
                    rel_id, old_confidence, old_count = existing
                    new_count = old_count + 1
                    # Weighted average confidence (new observation has weight 1)
                    new_confidence = (old_confidence * old_count + confidence) / new_count
                    new_confidence = min(0.99, new_confidence)  # Cap at 0.99

                    cursor.execute(f"""
                        UPDATE {T.RELATIONSHIPS}
                        SET confidence = %s,
                            mention_count = %s,
                            last_verified = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (new_confidence, new_count, rel_id))

                    conn.commit()
                    logger.debug(f"Reinforced relationship {rel_id}: {source_entity} -{relationship_type.value}-> {target_entity} (count={new_count})")
                    return rel_id

                # Check for contradictions before inserting
                contradictions = self._check_contradictions(
                    cursor, source_entity, relationship_type, target_entity, user_email
                )

                # Insert new relationship
                cursor.execute(f"""
                    INSERT INTO {T.RELATIONSHIPS} (
                        source_entity, relationship_type, target_entity,
                        confidence, context, source_message_id, user_email,
                        valid_from, contradiction_notes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    source_entity, relationship_type.value, target_entity,
                    confidence, context, source_message_id, user_email,
                    valid_from, contradictions
                ))

                rel_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Stored relationship {rel_id}: {source_entity} -{relationship_type.value}-> {target_entity}")
                return rel_id

        except Exception as e:
            logger.error(f"Failed to store relationship: {e}")
            conn.rollback()
            return None

    def _check_contradictions(
        self,
        cursor,
        source_entity: str,
        relationship_type: RelationshipType,
        target_entity: str,
        user_email: str
    ) -> Optional[str]:
        """
        Check for contradicting relationships.

        Examples of contradictions:
        - James MARRIED_TO Alia, but trying to add James MARRIED_TO Carol
        - James PARENT_OF Jesse (age 15), but trying to add Jesse PARENT_OF James

        Returns contradiction notes or None.
        """
        contradictions = []

        # Check for exclusive relationships (can only have one)
        exclusive_types = {
            RelationshipType.MARRIED_TO,
            RelationshipType.PARTNER_OF,
            RelationshipType.DATING,
        }

        if relationship_type in exclusive_types:
            cursor.execute(f"""
                SELECT target_entity FROM {T.RELATIONSHIPS}
                WHERE source_entity = %s
                AND relationship_type = %s
                AND target_entity != %s
                AND valid_until IS NULL
                AND (user_email = %s OR user_email IS NULL)
            """, (source_entity, relationship_type.value, target_entity, user_email))

            existing = cursor.fetchone()
            if existing:
                contradictions.append(
                    f"Existing {relationship_type.value}: {source_entity} -> {existing[0]}"
                )

        # Check for inverse contradictions (A parent of B, but B parent of A)
        inverse = INVERSE_RELATIONSHIPS.get(relationship_type)
        if inverse and inverse != relationship_type:
            cursor.execute(f"""
                SELECT id FROM {T.RELATIONSHIPS}
                WHERE source_entity = %s
                AND relationship_type = %s
                AND target_entity = %s
                AND valid_until IS NULL
                AND (user_email = %s OR user_email IS NULL)
            """, (target_entity, relationship_type.value, source_entity, user_email))

            if cursor.fetchone():
                contradictions.append(
                    f"Inverse exists: {target_entity} -{relationship_type.value}-> {source_entity}"
                )

        return "; ".join(contradictions) if contradictions else None

    def get_relationships_for_entity(
        self,
        entity: str,
        include_as_target: bool = True,
        only_current: bool = True,
        min_confidence: float = 0.5,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """Get all relationships involving an entity."""
        conn = self._get_connection(user_email)
        entity = entity.strip().title()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if include_as_target:
                    # Get relationships where entity is source OR target
                    query = f"""
                        SELECT * FROM {T.RELATIONSHIPS}
                        WHERE (source_entity = %s OR target_entity = %s)
                        AND confidence >= %s
                        AND (user_email = %s OR user_email IS NULL)
                    """
                    params = [entity, entity, min_confidence, user_email]
                else:
                    # Only where entity is source
                    query = f"""
                        SELECT * FROM {T.RELATIONSHIPS}
                        WHERE source_entity = %s
                        AND confidence >= %s
                        AND (user_email = %s OR user_email IS NULL)
                    """
                    params = [entity, min_confidence, user_email]

                if only_current:
                    query += " AND valid_until IS NULL"

                query += " ORDER BY confidence DESC, mention_count DESC"

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get relationships for {entity}: {e}")
            return []

    def get_relationship_between(
        self,
        entity1: str,
        entity2: str,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """Get all relationships between two specific entities."""
        conn = self._get_connection(user_email)
        entity1 = entity1.strip().title()
        entity2 = entity2.strip().title()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT * FROM {T.RELATIONSHIPS}
                    WHERE ((source_entity = %s AND target_entity = %s)
                           OR (source_entity = %s AND target_entity = %s))
                    AND valid_until IS NULL
                    AND (user_email = %s OR user_email IS NULL)
                    ORDER BY confidence DESC
                """, (entity1, entity2, entity2, entity1, user_email))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get relationship between {entity1} and {entity2}: {e}")
            return []

    def end_relationship(
        self,
        relationship_id: int,
        reason: str = None,
        user_email: str = None
    ) -> bool:
        """Mark a relationship as ended (set valid_until)."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.RELATIONSHIPS}
                    SET valid_until = CURRENT_TIMESTAMP,
                        context = COALESCE(context || '; ', '') || %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (f"Ended: {reason}" if reason else "Ended", relationship_id))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to end relationship {relationship_id}: {e}")
            conn.rollback()
            return False

    def format_relationships_for_prompt(
        self,
        entity: str,
        user_email: str = None,
        max_relationships: int = 10
    ) -> str:
        """Format an entity's relationships for inclusion in prompt."""
        relationships = self.get_relationships_for_entity(
            entity,
            include_as_target=True,
            only_current=True,
            min_confidence=0.6,
            user_email=user_email
        )

        if not relationships:
            return ""

        lines = [f"[RELATIONSHIPS - {entity}]"]

        for rel in relationships[:max_relationships]:
            source = rel['source_entity']
            target = rel['target_entity']
            rel_type = rel['relationship_type'].replace('_', ' ')
            conf = rel['confidence']
            count = rel['mention_count']

            # Format direction
            if source.lower() == entity.lower():
                line = f"- {rel_type} {target}"
            else:
                # Inverse direction
                inverse_type = INVERSE_RELATIONSHIPS.get(
                    RelationshipType(rel['relationship_type']),
                    RelationshipType.KNOWS
                ).value.replace('_', ' ')
                line = f"- {inverse_type} {source}"

            # Add confidence indicator for less certain relationships
            if conf < 0.8:
                line += f" (uncertain)"
            elif count > 3:
                line += f" (confirmed)"

            lines.append(line)

        return "\n".join(lines)

    def get_stats(self, user_email: str = None) -> Dict[str, Any]:
        """Get relationship store statistics."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT
                        COUNT(*) as total_relationships,
                        COUNT(*) FILTER (WHERE valid_until IS NULL) as active_relationships,
                        COUNT(DISTINCT source_entity) as unique_sources,
                        COUNT(DISTINCT target_entity) as unique_targets,
                        AVG(confidence) as avg_confidence,
                        AVG(mention_count) as avg_mentions
                    FROM {T.RELATIONSHIPS}
                """)
                stats = dict(cursor.fetchone())

                # Get relationship type distribution
                cursor.execute(f"""
                    SELECT relationship_type, COUNT(*) as count
                    FROM {T.RELATIONSHIPS}
                    WHERE valid_until IS NULL
                    GROUP BY relationship_type
                    ORDER BY count DESC
                """)
                stats['by_type'] = {row['relationship_type']: row['count'] for row in cursor.fetchall()}

                return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Singleton instance
_store: Optional[RelationshipStore] = None


def get_relationship_store() -> RelationshipStore:
    """Get singleton RelationshipStore instance."""
    global _store
    if _store is None:
        _store = RelationshipStore()
    return _store


def format_entity_relationships(entity: str, user_email: str = None) -> str:
    """Main entry point for context builder."""
    store = get_relationship_store()
    return store.format_relationships_for_prompt(entity, user_email)
