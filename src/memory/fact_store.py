"""
Fact Store - PostgreSQL storage for extracted facts.

WHAT: CRUD and search operations for subject-predicate-object facts learned
from conversations. Provides semantic deduplication (string similarity +
known pattern matching), contradiction detection (string + embedding
similarity), hybrid search (BM25 + pgvector), and spreading activation
retrieval through the fact network.

WHY: The companion needs to recall what it knows about people and things.
Raw conversation messages are too noisy and verbose. The fact store holds
distilled, structured knowledge -- each fact is a (subject, predicate,
object) triple with confidence, importance, mention count, and temporal
metadata.

HOW it fits:
  - The fact extraction pipeline (run after each message) calls store_fact()
    to persist new facts, which handles dedup, contradiction archival, and
    wiring into the fact network.
  - context_builder.py uses get_facts_for_subject() and search_facts_hybrid()
    to pull relevant facts into the LLM prompt.
  - confidence_decay.py enriches facts with effective_confidence at read time.
  - fact_network.py creates associative links between facts for spreading
    activation retrieval.
  - Replaces the old Neo4j Entity/Attribute storage from v1 of the system.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

import psycopg2
from psycopg2.extras import RealDictCursor

from src.database import tables as T

logger = logging.getLogger(__name__)


class FactStore:
    """PostgreSQL-backed storage for extracted facts."""

    def __init__(self):
        self._conn = None
        self._current_email = None

    def _get_connection(self, user_email: str = None):
        """
        Get database connection with correct schema search path.

        Recovers from poisoned transaction state (TRANSACTION_STATUS_INERROR)
        and reconnects when the user changes (different schema needed).
        """
        if self._conn is not None and not self._conn.closed:
            try:
                if self._conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
                    logger.warning("FactStore connection in INERROR state - rolling back")
                    self._conn.rollback()
            except Exception:
                self._conn = None
        # Reconnect if closed or if user changed (need different schema)
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

    def store_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.7,
        importance: int = 5,
        temporal: str = 'current',
        context: str = None,
        source: str = 'conversation',
        user_email: str = None,
        message_id: int = None,
        embedding: List[float] = None
    ) -> Optional[int]:
        """
        Store a new fact, handling deduplication.

        Returns:
            Fact ID if stored, None if deduplicated or error
        """
        conn = self._get_connection(user_email)

        try:
            # Check for semantic duplicates
            if self.is_duplicate(subject, predicate, obj, user_email=user_email):
                # Reinforce existing fact instead
                self.reinforce_fact(subject, predicate, obj, user_email=user_email)
                return None

            # Check for contradictions and archive old facts
            archived_count, detected_contradictions = self.handle_contradictions(subject, predicate, obj, embedding, user_email=user_email)
            if detected_contradictions:
                logger.info(f"Detected {len(detected_contradictions)} contradiction(s) for {subject}")

            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.FACTS} (
                        subject, predicate, object, confidence, importance,
                        temporal, context, source, user_email, message_id,
                        embedding, created_at, updated_at, mention_count, last_mentioned
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP
                    )
                    RETURNING id
                """, (
                    subject, predicate, obj, confidence, importance,
                    temporal, context, source, user_email, message_id,
                    embedding
                ))
                fact_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Stored fact {fact_id}: {subject} - {predicate} - {obj[:50]}...")

                # Wire into fact network (create links to related facts)
                self._create_fact_links(fact_id, subject, obj)

                return fact_id

        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            conn.rollback()
            return None

    # =========================================================================
    # Deduplication
    # =========================================================================

    def is_duplicate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        threshold: float = 0.85,
        user_email: str = None
    ) -> bool:
        """
        Check if a semantically similar fact already exists.

        Three-tier duplicate detection:
          1. Exact text match on the object field (fast).
          2. Known semantic pattern match -- hand-curated equivalence classes
             for frequently repeated facts (e.g., "works at Cavallo" matches
             "employed at Cavallo").
          3. SequenceMatcher string similarity >= threshold.

        If a duplicate is found, the caller should call reinforce_fact()
        instead of inserting.
        """
        # Hand-curated semantic equivalence classes for common repeated facts.
        # Each group maps variant phrasings to a canonical pattern name so
        # that "loves tea" and "prefers tea" are recognized as duplicates.
        # Instance-specific patterns are loaded from persona.yaml; generic
        # patterns that apply to any instance are defined here.
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        SEMANTIC_PATTERNS = dict(_pc.semantic_patterns) if _pc.semantic_patterns else {}

        def get_pattern(text):
            text_lower = text.lower()
            for pattern_name, phrases in SEMANTIC_PATTERNS.items():
                for phrase in phrases:
                    if phrase in text_lower:
                        return pattern_name
            return None

        conn = self._get_connection(user_email)
        new_pattern = get_pattern(obj)
        obj_lower = obj.lower().strip()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT id, object FROM {T.FACTS}
                    WHERE subject = %s
                    AND predicate = %s
                    AND archived_at IS NULL
                """, (subject, predicate))

                for row in cursor.fetchall():
                    existing = row['object']
                    if not existing:
                        continue

                    existing_lower = existing.lower().strip()

                    # Exact match
                    if obj_lower == existing_lower:
                        return True

                    # Semantic pattern match
                    existing_pattern = get_pattern(existing)
                    if new_pattern and existing_pattern and new_pattern == existing_pattern:
                        logger.debug(f"Semantic duplicate: '{obj[:40]}' matches pattern '{new_pattern}'")
                        return True

                    # String similarity
                    similarity = SequenceMatcher(None, obj_lower, existing_lower).ratio()
                    if similarity >= threshold:
                        logger.debug(f"Similar ({similarity:.0%}): '{obj[:40]}' ~ '{existing[:40]}'")
                        return True

            return False

        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            conn.rollback()
            return False

    def reinforce_fact(self, subject: str, predicate: str, obj: str, user_email: str = None) -> None:
        """Reinforce an existing fact by incrementing mention count."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor() as cursor:
                # Find the most similar existing fact
                cursor.execute(f"""
                    SELECT id, object FROM {T.FACTS}
                    WHERE subject = %s
                    AND predicate = %s
                    AND archived_at IS NULL
                    ORDER BY last_mentioned DESC
                    LIMIT 1
                """, (subject, predicate))

                row = cursor.fetchone()
                if row:
                    cursor.execute(f"""
                        UPDATE {T.FACTS}
                        SET mention_count = mention_count + 1,
                            last_mentioned = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP,
                            confidence = LEAST(confidence + 0.02, 0.99)
                        WHERE id = %s
                    """, (row[0],))
                    conn.commit()
                    logger.debug(f"Reinforced fact {row[0]}: {subject} - {predicate}")

        except Exception as e:
            logger.error(f"Failed to reinforce fact: {e}")
            conn.rollback()

    # =========================================================================
    # Contradiction detection and archival
    # =========================================================================

    def handle_contradictions(
        self,
        subject: str,
        predicate: str,
        new_obj: str,
        embedding: List[float] = None,
        user_email: str = None
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Detect and archive facts that the new fact contradicts.

        Two-stage detection:
          1. String similarity < 0.4 = very different text for same
             subject+predicate => likely a correction (e.g., "works at
             Cavallo" replacing "works at LeanTaaS").
          2. Embedding cosine similarity < 0.7 = semantically different
             meaning => catches subtler contradictions like "likes coffee"
             vs. "prefers tea."

        Contradicted facts are soft-deleted (archived_at set) rather than
        hard-deleted, preserving history. Detected contradictions are also
        recorded in a dedicated table so they can optionally be surfaced to
        the companion for acknowledgment.

        Returns:
            Tuple of (archived_count, detected_contradictions)
        """
        conn = self._get_connection(user_email)
        archived = 0
        contradictions = []

        try:
            # Get embedding for new fact if not provided
            if embedding is None:
                embedding = self._get_embedding(new_obj)

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT id, object, embedding FROM {T.FACTS}
                    WHERE subject = %s
                    AND predicate = %s
                    AND archived_at IS NULL
                """, (subject, predicate))

                for row in cursor.fetchall():
                    existing = row['object']
                    existing_embedding = row.get('embedding')

                    # Stage 1: String similarity check
                    string_similarity = SequenceMatcher(None, new_obj.lower(), existing.lower()).ratio()

                    # Stage 2: Semantic similarity check (if embeddings available)
                    semantic_similarity = None
                    if embedding and existing_embedding:
                        semantic_similarity = self._cosine_similarity(embedding, existing_embedding)

                    # Determine if contradicted:
                    # - String similarity < 0.4 = very different text
                    # - Semantic similarity < 0.7 = semantically different meaning
                    is_string_contradiction = string_similarity < 0.4
                    is_semantic_contradiction = semantic_similarity is not None and semantic_similarity < 0.7

                    if is_string_contradiction or is_semantic_contradiction:
                        # Record the contradiction before archiving
                        contradiction_info = {
                            'fact_id': row['id'],
                            'subject': subject,
                            'predicate': predicate,
                            'old_value': existing,
                            'new_value': new_obj,
                            'string_similarity': string_similarity,
                            'semantic_similarity': semantic_similarity,
                            'detection_method': 'semantic' if is_semantic_contradiction and not is_string_contradiction else 'string'
                        }
                        contradictions.append(contradiction_info)

                        # Archive the old fact
                        archive_reason = 'contradicted_by_newer_fact'
                        if is_semantic_contradiction and not is_string_contradiction:
                            archive_reason = 'semantically_contradicted'

                        cursor.execute(f"""
                            UPDATE {T.FACTS}
                            SET archived_at = CURRENT_TIMESTAMP,
                                archive_reason = %s
                            WHERE id = %s
                        """, (archive_reason, row['id']))
                        archived += 1

                        logger.info(
                            f"Archived contradicted fact {row['id']}: '{existing[:40]}...' "
                            f"(method: {contradiction_info['detection_method']}, "
                            f"str_sim: {string_similarity:.2f}, "
                            f"sem_sim: {semantic_similarity:.2f if semantic_similarity else 'N/A'})"
                        )

                conn.commit()

            # Store detected contradictions for later surfacing in context
            if contradictions:
                self._store_detected_contradictions(subject, contradictions, user_email=user_email)

        except Exception as e:
            logger.error(f"Contradiction handling failed: {e}")
            conn.rollback()

        return archived, contradictions

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _store_detected_contradictions(self, subject: str, contradictions: List[Dict], user_email: str = None) -> None:
        """Store detected contradictions for later surfacing in context."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor() as cursor:
                for c in contradictions:
                    cursor.execute(f"""
                        INSERT INTO {T.DETECTED_CONTRADICTIONS} (
                            subject, old_value, new_value, detection_method,
                            string_similarity, semantic_similarity, created_at, surfaced
                        ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, FALSE)
                        ON CONFLICT DO NOTHING
                    """, (
                        c['subject'], c['old_value'], c['new_value'],
                        c['detection_method'], c['string_similarity'],
                        c.get('semantic_similarity')
                    ))
                conn.commit()
        except Exception as e:
            # Table might not exist yet - that's OK
            logger.debug(f"Could not store contradiction (table may not exist): {e}")
            conn.rollback()

    def get_unsurfaced_contradictions(self, subject: str = None, limit: int = 5, user_email: str = None) -> List[Dict]:
        """Get contradictions that haven't been surfaced to the LLM yet."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if subject:
                    cursor.execute(f"""
                        SELECT * FROM {T.DETECTED_CONTRADICTIONS}
                        WHERE subject = %s AND surfaced = FALSE
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (subject, limit))
                else:
                    cursor.execute(f"""
                        SELECT * FROM {T.DETECTED_CONTRADICTIONS}
                        WHERE surfaced = FALSE
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))

                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"Could not get contradictions (table may not exist): {e}")
            conn.rollback()
            return []

    def mark_contradiction_surfaced(self, contradiction_id: int, user_email: str = None) -> None:
        """Mark a contradiction as surfaced (shown to LLM)."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.DETECTED_CONTRADICTIONS}
                    SET surfaced = TRUE, surfaced_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (contradiction_id,))
                conn.commit()
        except Exception as e:
            logger.debug(f"Could not mark contradiction surfaced: {e}")
            conn.rollback()

    def get_facts_for_subject(
        self,
        subject: str,
        include_archived: bool = False,
        limit: int = 50,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """Get all facts about a subject, enriched with effective confidence."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if include_archived:
                    cursor.execute(f"""
                        SELECT * FROM {T.FACTS}
                        WHERE subject = %s
                        ORDER BY importance DESC, last_mentioned DESC
                        LIMIT %s
                    """, (subject, limit))
                else:
                    cursor.execute(f"""
                        SELECT * FROM {T.FACTS}
                        WHERE subject = %s
                        AND archived_at IS NULL
                        ORDER BY importance DESC, last_mentioned DESC
                        LIMIT %s
                    """, (subject, limit))

                results = [dict(row) for row in cursor.fetchall()]

                # Enrich with effective confidence
                try:
                    from src.memory.confidence_decay import enrich_facts_with_effective_confidence
                    enrich_facts_with_effective_confidence(results)
                except Exception as e:
                    logger.debug(f"Confidence enrichment skipped: {e}")

                return results

        except Exception as e:
            logger.error(f"Failed to get facts for {subject}: {e}")
            conn.rollback()
            return []

    def get_important_facts(
        self,
        min_importance: int = 7,
        limit: int = 100,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """Get high-importance facts across all subjects."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT * FROM {T.FACTS}
                    WHERE importance >= %s
                    AND archived_at IS NULL
                    ORDER BY importance DESC, last_mentioned DESC
                    LIMIT %s
                """, (min_importance, limit))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get important facts: {e}")
            conn.rollback()
            return []

    def get_facts_for_entities(
        self,
        entities: List[str],
        limit: int = 10,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get top facts about a list of entities.

        Used by relationship bridge in spreading activation to fetch facts
        about related entities (e.g., facts about Jesse when James is activated
        via parent_of relationship).

        Args:
            entities: List of entity names to look up
            limit: Max facts per entity

        Returns:
            List of fact dicts, sorted by importance
        """
        if not entities:
            return []

        conn = self._get_connection(user_email)
        results = []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Case-insensitive match on subject
                cursor.execute(f"""
                    SELECT * FROM {T.FACTS}
                    WHERE archived_at IS NULL
                    AND LOWER(subject) = ANY(%s)
                    ORDER BY importance DESC, last_mentioned DESC
                    LIMIT %s
                """, ([e.lower() for e in entities], limit))

                results = [dict(row) for row in cursor.fetchall()]

            return results

        except Exception as e:
            logger.error(f"Failed to get facts for entities {entities}: {e}")
            conn.rollback()
            return []

    def search_facts(
        self,
        query: str,
        limit: int = 20,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """Search facts by text match, enriched with effective confidence."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT * FROM {T.FACTS}
                    WHERE archived_at IS NULL
                    AND (
                        subject ILIKE %s
                        OR predicate ILIKE %s
                        OR object ILIKE %s
                        OR context ILIKE %s
                    )
                    ORDER BY importance DESC, last_mentioned DESC
                    LIMIT %s
                """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', limit))

                results = [dict(row) for row in cursor.fetchall()]

                try:
                    from src.memory.confidence_decay import enrich_facts_with_effective_confidence
                    enrich_facts_with_effective_confidence(results)
                except Exception as e:
                    logger.debug(f"Confidence enrichment skipped: {e}")

                return results

        except Exception as e:
            logger.error(f"Failed to search facts: {e}")
            conn.rollback()
            return []

    # =========================================================================
    # Hybrid search (BM25 + pgvector)
    # =========================================================================

    def search_facts_hybrid(
        self,
        query: str,
        limit: int = 20,
        vector_weight: float = 0.6,
        text_weight: float = 0.4,
        min_score: float = 0.1,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 full-text and pgvector cosine similarity.

        Inspired by Clawdbot's approach. The two signals are complementary:
          - BM25 (text_weight): excels at exact token matches -- names, error
            codes, specific phrases like "Cavallo" or "peppermint tea."
          - pgvector (vector_weight): excels at semantic matches -- paraphrases,
            conceptually similar facts.

        The query runs both searches as CTEs and combines scores with the
        provided weights. Falls back to simple ILIKE search if the hybrid
        query fails (e.g., search_vector column not yet backfilled).

        Args:
            query: Search query
            limit: Max results
            vector_weight: Weight for vector similarity (0-1)
            text_weight: Weight for BM25 text match (0-1)
            min_score: Minimum combined score to include

        Returns:
            List of facts with hybrid_score field
        """
        conn = self._get_connection(user_email)

        try:
            # Get embedding for query (if embeddings are available)
            embedding = self._get_embedding(query)

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if embedding:
                    # Full hybrid search with both vector and BM25
                    cursor.execute(f"""
                        WITH vector_scores AS (
                            SELECT id,
                                   1 - (embedding <=> %s::vector) as vector_score
                            FROM {T.FACTS}
                            WHERE archived_at IS NULL
                            AND embedding IS NOT NULL
                        ),
                        text_scores AS (
                            SELECT id,
                                   ts_rank_cd(search_vector, plainto_tsquery('english', %s)) as text_score
                            FROM {T.FACTS}
                            WHERE archived_at IS NULL
                            AND search_vector IS NOT NULL
                            AND search_vector @@ plainto_tsquery('english', %s)
                        )
                        SELECT f.*,
                               COALESCE(v.vector_score, 0) as vector_score,
                               COALESCE(t.text_score, 0) as text_score,
                               COALESCE(v.vector_score, 0) * %s +
                               COALESCE(t.text_score, 0) * %s as hybrid_score
                        FROM {T.FACTS} f
                        LEFT JOIN vector_scores v ON f.id = v.id
                        LEFT JOIN text_scores t ON f.id = t.id
                        WHERE f.archived_at IS NULL
                        AND (v.vector_score IS NOT NULL OR t.text_score IS NOT NULL)
                        AND (COALESCE(v.vector_score, 0) * %s +
                             COALESCE(t.text_score, 0) * %s) >= %s
                        ORDER BY hybrid_score DESC, f.importance DESC
                        LIMIT %s
                    """, (
                        embedding, query, query,
                        vector_weight, text_weight,
                        vector_weight, text_weight, min_score,
                        limit
                    ))
                else:
                    # BM25 only (no embedding available)
                    cursor.execute(f"""
                        SELECT f.*,
                               0 as vector_score,
                               ts_rank_cd(search_vector, plainto_tsquery('english', %s)) as text_score,
                               ts_rank_cd(search_vector, plainto_tsquery('english', %s)) as hybrid_score
                        FROM {T.FACTS} f
                        WHERE archived_at IS NULL
                        AND search_vector IS NOT NULL
                        AND search_vector @@ plainto_tsquery('english', %s)
                        ORDER BY hybrid_score DESC, importance DESC
                        LIMIT %s
                    """, (query, query, query, limit))

                results = [dict(row) for row in cursor.fetchall()]

                if results:
                    logger.debug(
                        f"Hybrid search '{query[:30]}...': {len(results)} results "
                        f"(top score: {results[0].get('hybrid_score', 0):.3f})"
                    )

                return results

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to simple search: {e}")
            conn.rollback()
            # Fall back to simple ILIKE search
            return self.search_facts(query, limit, user_email=user_email)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (if embedding service available)."""
        try:
            from src.memory.embeddings import get_embedding
            return get_embedding(text)
        except Exception as e:
            logger.debug(f"Could not get embedding: {e}")
            return None

    def _create_fact_links(self, fact_id: int, subject: str, obj: str) -> None:
        """
        Wire a new fact into the fact network.

        Called after storing a fact to create links to related facts.
        Runs in background to not slow down fact storage.
        """
        try:
            import threading

            def create_links():
                try:
                    from src.memory.fact_network import get_fact_network

                    network = get_fact_network()
                    # Use heuristics by default (fast), LLM for important facts
                    use_llm = False  # Can enable with env var COMPANION_FACT_LINKS_LLM=true
                    if os.environ.get('COMPANION_FACT_LINKS_LLM', 'false').lower() == 'true':
                        use_llm = True

                    network.detect_links_for_new_fact(
                        new_fact_id=fact_id,
                        new_fact_subject=subject,
                        new_fact_object=obj,
                        use_llm=use_llm
                    )
                except Exception as e:
                    logger.debug(f"Fact link creation failed (non-critical): {e}")

            # Run in background thread
            thread = threading.Thread(target=create_links, daemon=True)
            thread.start()

        except Exception as e:
            logger.debug(f"Could not start fact link thread: {e}")

    def ensure_search_vectors(self, user_email: str = None) -> int:
        """
        Ensure all facts have search_vector populated.

        Run this after migration to backfill any missing vectors.

        Returns:
            Number of facts updated
        """
        conn = self._get_connection(user_email)

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.FACTS} SET search_vector =
                        setweight(to_tsvector('english', COALESCE(subject, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(predicate, '')), 'B') ||
                        setweight(to_tsvector('english', COALESCE(object, '')), 'B') ||
                        setweight(to_tsvector('english', COALESCE(context, '')), 'C')
                    WHERE search_vector IS NULL
                    RETURNING id
                """)
                updated = cursor.rowcount
                conn.commit()

                if updated > 0:
                    logger.info(f"Backfilled search_vector for {updated} facts")

                return updated

        except Exception as e:
            logger.error(f"Failed to backfill search vectors: {e}")
            conn.rollback()
            return 0

    def update_importance(self, fact_id: int, importance: int, user_email: str = None) -> bool:
        """Update the importance score for a fact."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.FACTS}
                    SET importance = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (importance, fact_id))
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to update importance: {e}")
            conn.rollback()
            return False

    def get_stats(self, user_email: str = None) -> Dict[str, Any]:
        """Get fact store statistics."""
        conn = self._get_connection(user_email)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT
                        COUNT(*) as total_facts,
                        COUNT(*) FILTER (WHERE archived_at IS NULL) as active_facts,
                        COUNT(*) FILTER (WHERE archived_at IS NOT NULL) as archived_facts,
                        COUNT(DISTINCT subject) as unique_subjects,
                        AVG(importance) as avg_importance,
                        AVG(mention_count) as avg_mentions
                    FROM {T.FACTS}
                """)
                return dict(cursor.fetchone())

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            conn.rollback()
            return {}

    # =========================================================================
    # Spreading activation search (brain-like associative retrieval)
    # =========================================================================

    def search_with_spreading_activation(
        self,
        query: str,
        limit: int = 20,
        activation_depth: int = 2,
        min_activation: float = 0.25,
        relationship_bridge: bool = True,
        user_email: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search facts with spreading activation through the fact network.

        This is the most sophisticated retrieval mode, mimicking how human
        memory works -- activating one concept primes related concepts:

          1. Hybrid search finds seed facts (direct matches).
          2. Activation spreads through fact_links (explains, causes, similar).
          3. If relationship_bridge is True, activation also crosses entity
             boundaries via known relationships (e.g., asking about "Jesse
             crisis" also activates James's stress facts via parent_of).

        Falls back to hybrid search if the fact network is unavailable.

        Args:
            query: Search query
            limit: Max results
            activation_depth: How many link hops to follow (1-3)
            min_activation: Minimum activation to include
            relationship_bridge: Whether to use entity relationships as bridges

        Returns:
            List of facts with activation scores, sorted by activation
        """
        try:
            # Step 1: Get seed facts via hybrid search
            seed_facts = self.search_facts_hybrid(query, limit=10, min_score=0.15, user_email=user_email)

            if not seed_facts:
                # Fall back to simple search
                return self.search_facts(query, limit, user_email=user_email)

            seed_ids = [f['id'] for f in seed_facts]

            # Step 2: Spread activation through network (with relationship bridges)
            from src.memory.fact_network import get_fact_network

            network = get_fact_network()
            activated_facts = network.spreading_activation(
                seed_fact_ids=seed_ids,
                max_depth=activation_depth,
                min_activation=min_activation,
                max_results=limit,
                relationship_bridge=relationship_bridge,
                user_email=user_email,
            )

            if activated_facts:
                logger.info(
                    f"Spreading activation: '{query[:30]}...' -> "
                    f"{len(seed_ids)} seeds -> {len(activated_facts)} activated facts"
                )
                return activated_facts

            # Fall back to seed facts if no activation spread
            return seed_facts

        except Exception as e:
            logger.warning(f"Spreading activation search failed: {e}")
            return self.search_facts_hybrid(query, limit, user_email=user_email)


# Singleton instance
_fact_store: Optional[FactStore] = None


def get_fact_store() -> FactStore:
    """Get singleton FactStore instance."""
    global _fact_store
    if _fact_store is None:
        _fact_store = FactStore()
    return _fact_store
