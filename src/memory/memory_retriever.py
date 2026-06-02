"""
MemoryRetriever -- multi-source search engine for verified memories.

WHAT: Searches PostgreSQL conversation history and entity profiles for memories
      matching search terms. Returns a list of VerifiedMemory dataclasses with
      content, source, timestamp, and relevance score.

WHY:  Internal implementation used by RetrievalAgent.search_verified(). Not
      intended to be called directly by application code -- use
      get_retrieval_agent().search_verified() instead.

HOW:  Search strategy varies by query type:
      - specific_event: Full-text search on messages table
      - emotional_vague: Broader search for emotionally significant messages
      - factual: Entity profile lookup (YAML ground truth)
      Results are deduplicated and sorted by relevance.

Pipeline position: RetrievalAgent.search_verified() -> [this] -> MemoryValidationAgent
"""

import logging
from typing import List, Optional, Dict, Any

from src.core.verified_memory import VerifiedMemory

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Retrieves verified memory records from multiple storage sources.

    Searches:
    1. PostgreSQL messages (full-text search on conversation history)
    2. Entity profiles (for factual queries about people)

    Note: Graphiti backend has been removed; those methods are no-ops.
    """

    def __init__(self):
        self._db = None

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

        if query_type == 'specific_event':
            memories.extend(self._search_postgres(search_terms, user_email, limit // 2))

        elif query_type == 'emotional_vague':
            memories.extend(self._search_postgres_significant(user_email, limit))

        elif query_type == 'factual':
            memories.extend(self._search_entity_profiles(search_terms))
            memories.extend(self._search_postgres(search_terms, user_email, limit // 2))

        else:
            # Default: plain postgres full-text search
            memories.extend(self._search_postgres(search_terms, user_email, limit))

        return self._deduplicate_and_rank(memories)[:limit]

    def _search_postgres(
        self,
        search_terms: List[str],
        user_email: str,
        limit: int
    ) -> List[VerifiedMemory]:
        """Search PostgreSQL messages for matching content."""
        try:
            db = self._get_db()
            messages = db.get_recent_messages(user_email, limit=500)

            matching = []
            search_lower = [term.lower() for term in search_terms if len(term) > 2]

            for msg in messages:
                text = msg.get('message_text', '').lower()
                sender = msg.get('sender_name', '')
                timestamp = msg.get('created_at', '')

                matches = sum(1 for term in search_lower if term in text)

                if matches > 0:
                    relevance = min(1.0, matches / max(1, len(search_lower)))
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
            messages = db.get_recent_messages(user_email, limit=200)

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

                emotion_count = sum(1 for kw in emotional_keywords if kw in text)

                if emotion_count > 0:
                    relevance = min(1.0, emotion_count / 3)
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

    def _search_entity_profiles(
        self,
        search_terms: List[str]
    ) -> List[VerifiedMemory]:
        """
        Search YAML entity profiles for factual information.

        YAML profiles are the SOURCE OF TRUTH and should override
        any conflicting information from other backends.
        """
        try:
            from src.core.entity_profile_loader import get_entity_profile_loader

            memories = []
            loader = get_entity_profile_loader()
            entity_names = list(loader.profiles.keys())

            for term in search_terms:
                term_lower = term.lower()
                for entity_name in entity_names:
                    if entity_name in term_lower or term_lower in entity_name:
                        profile = loader.get_profile(entity_name)
                        if profile:
                            facts = self._extract_profile_facts(profile, entity_name)
                            for fact in facts:
                                memories.append(VerifiedMemory(
                                    content=f"[GROUND TRUTH] {fact}",
                                    source='entity_profile',
                                    timestamp=None,
                                    relevance=1.0
                                ))

            return memories[:15]

        except Exception as e:
            logger.warning(f"Entity profile search failed: {e}")
            return []

    def _extract_profile_facts(self, profile: Dict[str, Any], entity_name: str = None) -> List[str]:
        """Extract key facts from an entity profile (YAML structure)."""
        facts = []
        name = profile.get('name', entity_name or 'Unknown')

        if profile.get('role'):
            facts.append(f"{name}'s role: {profile['role']}")

        if profile.get('employment'):
            emp = profile['employment']
            if emp.get('work_status'):
                facts.append(f"{name}: {emp['work_status']}")
            if emp.get('previous_employer'):
                facts.append(f"{name} previously worked at: {emp['previous_employer']}")

        if profile.get('family'):
            fam = profile['family']
            if fam.get('spouse'):
                facts.append(f"{name}'s spouse: {fam['spouse']}")
            if fam.get('children'):
                for child in fam['children']:
                    facts.append(f"{name}'s child: {child.get('name')} ({child.get('relationship', '')})")

        if profile.get('romantic_relationship'):
            rom = profile['romantic_relationship']
            if rom.get('partner'):
                facts.append(f"{name}'s romantic partner: {rom['partner']}")
            if rom.get('status'):
                facts.append(f"{name}'s relationship status: {rom['status']}")

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

        if profile.get('sister') or profile.get('siblings'):
            sibs = profile.get('siblings', {})
            if isinstance(sibs, dict):
                for sib_name, sib_info in sibs.items():
                    if isinstance(sib_info, dict):
                        status = sib_info.get('relationship_status', '')
                        if status:
                            facts.append(f"{name}'s relationship with {sib_name}: {status}")

        if profile.get('notes'):
            notes = profile['notes']
            if isinstance(notes, list):
                for note in notes[:5]:
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
            normalized = memory.content.lower().strip()[:100]
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique.append(memory)
        unique.sort(key=lambda m: m.relevance, reverse=True)
        return unique


# Singleton
_retriever: Optional[MemoryRetriever] = None


def get_memory_retriever() -> MemoryRetriever:
    """Get singleton MemoryRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = MemoryRetriever()
    return _retriever
