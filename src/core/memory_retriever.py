"""
Memory Retriever -- multi-source search for verified memories.

WHAT: Searches PostgreSQL conversation history, Graphiti knowledge graph
      (deprecated), and entity profiles for memories matching search terms.
      Returns a list of VerifiedMemory dataclasses with content, source,
      timestamp, and relevance score.

WHY:  When the MemoryValidationAgent detects a memory query ("remember when
      we...?"), it needs actual verified records to inject into the prompt.
      This module provides those records from every available storage backend.

HOW:  Search strategy varies by query type:
      - specific_event: Full-text search on messages table + Graphiti edges
      - emotional_vague: Broader search with lower relevance threshold
      - factual: Entity profile lookup + facts table
      Results are deduplicated and sorted by relevance. Graphiti calls are
      wrapped in try/except since that backend is being phased out.

Pipeline position: MemoryQueryClassifier -> [this] -> MemoryValidationAgent
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VerifiedMemory:
    """A verified memory record from storage."""
    content: str  # The memory content
    source: str  # 'postgres' | 'graphiti' | 'entity_profile'
    timestamp: Optional[str] = None  # When this was recorded
    relevance: float = 0.5  # How relevant to the query (0-1)


class MemoryRetriever:
    """
    Retrieves verified memory records from multiple storage sources.

    Searches:
    1. PostgreSQL messages (full-text search on conversation history)
    2. Graphiti facts (semantic search on knowledge graph)
    3. Entity profiles (for factual queries about people)
    """

    def __init__(self):
        self._db = None
        self._graphiti = None

    def _get_db(self):
        """Get database instance."""
        if self._db is None:
            from src.database.db import get_db
            self._db = get_db()
        return self._db

    def search(
        self,
        search_terms: List[str],
        user_email: str,
        query_type: str,
        limit: int = 10
    ) -> List[VerifiedMemory]:
        """
        Search for verified memories matching the query.

        Args:
            search_terms: Terms to search for
            user_email: User's email for filtering
            query_type: 'specific_event' | 'emotional_vague' | 'factual'
            limit: Maximum results to return

        Returns:
            List of VerifiedMemory objects
        """
        memories = []

        # Search strategy depends on query type
        if query_type == 'specific_event':
            # Search both Postgres and Graphiti for specific events
            memories.extend(self._search_postgres(search_terms, user_email, limit // 2))
            memories.extend(self._search_graphiti(search_terms, limit // 2))

        elif query_type == 'emotional_vague':
            # For emotional queries, look for high-importance facts and significant messages
            memories.extend(self._search_graphiti_important(limit))
            memories.extend(self._search_postgres_significant(user_email, limit // 2))

        elif query_type == 'factual':
            # For factual queries, prioritize entity profiles and Graphiti
            memories.extend(self._search_entity_profiles(search_terms))
            memories.extend(self._search_graphiti(search_terms, limit))

        # Sort by relevance and deduplicate
        memories = self._deduplicate_and_rank(memories)

        return memories[:limit]

    def _search_postgres(
        self,
        search_terms: List[str],
        user_email: str,
        limit: int
    ) -> List[VerifiedMemory]:
        """Search PostgreSQL messages for matching content."""
        try:
            db = self._get_db()

            # Get recent messages and search through them
            # This is a simple implementation - could be optimized with full-text search
            messages = db.get_recent_messages(user_email, limit=500)

            matching = []
            search_lower = [term.lower() for term in search_terms if len(term) > 2]

            for msg in messages:
                text = msg.get('message_text', '').lower()
                sender = msg.get('sender_name', '')
                timestamp = msg.get('created_at', '')

                # Count how many search terms match
                matches = sum(1 for term in search_lower if term in text)

                if matches > 0:
                    relevance = min(1.0, matches / max(1, len(search_lower)))

                    # Format the memory
                    if sender == 'User':
                        content = f"James said: \"{msg.get('message_text', '')[:200]}\""
                    else:
                        content = f"The companion said: \"{msg.get('message_text', '')[:200]}\""

                    matching.append(VerifiedMemory(
                        content=content,
                        source='postgres',
                        timestamp=str(timestamp) if timestamp else None,
                        relevance=relevance
                    ))

            # Sort by relevance and return top results
            matching.sort(key=lambda m: m.relevance, reverse=True)
            return matching[:limit]

        except Exception as e:
            logger.warning(f"PostgreSQL search failed: {e}")
            return []

    def _search_postgres_significant(
        self,
        user_email: str,
        limit: int
    ) -> List[VerifiedMemory]:
        """Search for significant/emotional messages in PostgreSQL."""
        try:
            db = self._get_db()

            # Get recent messages
            messages = db.get_recent_messages(user_email, limit=200)

            # Look for messages with emotional content
            emotional_keywords = [
                'love', 'happy', 'sad', 'excited', 'worried', 'beautiful',
                'amazing', 'wonderful', 'remember', 'favorite', 'special',
                'important', 'miss', 'grateful', 'proud', 'scared'
            ]

            matching = []
            for msg in messages:
                text = msg.get('message_text', '').lower()
                sender = msg.get('sender_name', '')
                timestamp = msg.get('created_at', '')

                # Count emotional keywords
                emotion_count = sum(1 for kw in emotional_keywords if kw in text)

                if emotion_count > 0:
                    relevance = min(1.0, emotion_count / 3)  # Max out at 3 matches

                    if sender == 'User':
                        content = f"James said: \"{msg.get('message_text', '')[:200]}\""
                    else:
                        content = f"The companion said: \"{msg.get('message_text', '')[:200]}\""

                    matching.append(VerifiedMemory(
                        content=content,
                        source='postgres',
                        timestamp=str(timestamp) if timestamp else None,
                        relevance=relevance
                    ))

            matching.sort(key=lambda m: m.relevance, reverse=True)
            return matching[:limit]

        except Exception as e:
            logger.warning(f"PostgreSQL significant search failed: {e}")
            return []

    def _search_graphiti(
        self,
        search_terms: List[str],
        limit: int
    ) -> List[VerifiedMemory]:
        """
        DEPRECATED: Graphiti has been removed.
        Memory search now uses pgvector semantic search.
        """
        # Graphiti removed - return empty
        return []

    def _search_graphiti_important(self, limit: int) -> List[VerifiedMemory]:
        """
        DEPRECATED: Graphiti has been removed.
        Memory search now uses pgvector semantic search.
        """
        # Graphiti removed - return empty
        return []

    def _search_entity_profiles(
        self,
        search_terms: List[str]
    ) -> List[VerifiedMemory]:
        """
        Search YAML entity profiles for factual information.

        YAML profiles are the SOURCE OF TRUTH and should override
        any conflicting information from Graphiti.
        """
        try:
            from src.core.entity_profile_loader import get_entity_profile_loader

            memories = []
            loader = get_entity_profile_loader()

            # Use all loaded entity profile names
            entity_names = list(loader.profiles.keys())

            for term in search_terms:
                term_lower = term.lower()

                for entity_name in entity_names:
                    if entity_name in term_lower or term_lower in entity_name:
                        # Get profile from YAML
                        profile = loader.get_profile(entity_name)

                        if profile:
                            # Extract key facts from profile
                            facts = self._extract_profile_facts(profile, entity_name)

                            for fact in facts:
                                memories.append(VerifiedMemory(
                                    content=f"[GROUND TRUTH] {fact}",
                                    source='entity_profile',
                                    timestamp=None,
                                    relevance=1.0  # YAML profiles are highest priority
                                ))

            return memories[:15]  # Limit profile facts

        except Exception as e:
            logger.warning(f"Entity profile search failed: {e}")
            return []

    def _extract_profile_facts(self, profile: Dict[str, Any], entity_name: str = None) -> List[str]:
        """Extract key facts from an entity profile (YAML structure)."""
        facts = []
        name = profile.get('name', entity_name or 'Unknown')

        # Basic identity
        if profile.get('role'):
            facts.append(f"{name}'s role: {profile['role']}")

        # Employment (for James)
        if profile.get('employment'):
            emp = profile['employment']
            if emp.get('work_status'):
                facts.append(f"{name}: {emp['work_status']}")
            if emp.get('previous_employer'):
                facts.append(f"{name} previously worked at: {emp['previous_employer']}")

        # Family relationships
        if profile.get('family'):
            fam = profile['family']
            if fam.get('spouse'):
                facts.append(f"{name}'s spouse: {fam['spouse']}")
            if fam.get('children'):
                for child in fam['children']:
                    facts.append(f"{name}'s child: {child.get('name')} ({child.get('relationship', '')})")

        # Romantic relationship (critical)
        if profile.get('romantic_relationship'):
            rom = profile['romantic_relationship']
            if rom.get('partner'):
                facts.append(f"{name}'s romantic partner: {rom['partner']}")
            if rom.get('status'):
                facts.append(f"{name}'s relationship status: {rom['status']}")

        # Preferences (especially important for James's tea/coffee)
        if profile.get('preferences'):
            prefs = profile['preferences']
            if prefs.get('drinks'):
                drinks = prefs['drinks']
                if drinks.get('coffee'):
                    facts.append(f"{name} and coffee: {drinks['coffee']}")
                if drinks.get('tea'):
                    facts.append(f"{name} and tea: {drinks['tea']}")
                if drinks.get('default'):
                    facts.append(f"Drink rule for {name}: {drinks['default']}")

        # Family situation (for sibling relationships etc.)
        if profile.get('sister') or profile.get('siblings'):
            sibs = profile.get('siblings', {})
            if isinstance(sibs, dict):
                for sib_name, sib_info in sibs.items():
                    if isinstance(sib_info, dict):
                        status = sib_info.get('relationship_status', '')
                        if status:
                            facts.append(f"{name}'s relationship with {sib_name}: {status}")

        # Important notes
        if profile.get('notes'):
            notes = profile['notes']
            if isinstance(notes, list):
                for note in notes[:5]:  # Limit to 5 notes
                    facts.append(f"About {name}: {note}")

        return facts

    def _deduplicate_and_rank(
        self,
        memories: List[VerifiedMemory]
    ) -> List[VerifiedMemory]:
        """Remove duplicates and rank by relevance."""
        seen_content = set()
        unique = []

        for memory in memories:
            # Normalize content for deduplication
            normalized = memory.content.lower().strip()[:100]

            if normalized not in seen_content:
                seen_content.add(normalized)
                unique.append(memory)

        # Sort by relevance
        unique.sort(key=lambda m: m.relevance, reverse=True)

        return unique


# Singleton instance
_retriever: Optional[MemoryRetriever] = None


def get_memory_retriever() -> MemoryRetriever:
    """Get singleton MemoryRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = MemoryRetriever()
    return _retriever


def search_memories(
    search_terms: List[str],
    user_email: str,
    query_type: str,
    limit: int = 10
) -> List[VerifiedMemory]:
    """Convenience function to search for memories."""
    retriever = get_memory_retriever()
    return retriever.search(search_terms, user_email, query_type, limit)
