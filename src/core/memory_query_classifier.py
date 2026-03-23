"""
Memory Query Classifier - Detects memory-related queries

Uses Claude Haiku for fast classification of user messages to determine
if they're asking about past memories, facts, or events.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryQueryResult:
    """Result of memory query classification."""
    is_memory_query: bool
    query_type: str  # 'specific_event' | 'emotional_vague' | 'factual' | 'none'
    search_terms: List[str]
    confidence: float
    reasoning: str = ""


class MemoryQueryClassifier:
    """
    Classifies user messages for memory-related queries.

    Uses Claude Haiku for fast (~200-300ms) classification to determine
    if a message is asking about memories, and what type of query it is.
    """

    # Keywords that suggest memory queries (for pre-filtering)
    MEMORY_KEYWORDS = [
        'remember', 'recall', 'memory', 'memories', 'when did', 'when we',
        'favorite', 'favourite', 'best', 'worst', 'first time', 'last time',
        'that time', 'the time', 'do you know', 'tell me about', 'what about',
        'how did', 'where did', 'who was', 'what was', 'what happened',
        'did we', 'have we', 'did i', 'have i', 'used to', 'back when',
        'ago', 'before', 'earlier', 'yesterday', 'last week', 'last month'
    ]

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Get or create Fireworks client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
        return self._client

    def _has_memory_keywords(self, message: str) -> bool:
        """Quick check for memory-related keywords."""
        message_lower = message.lower()
        return any(kw in message_lower for kw in self.MEMORY_KEYWORDS)

    def classify(self, message: str) -> MemoryQueryResult:
        """
        Classify a user message for memory-related queries.

        Args:
            message: The user's message

        Returns:
            MemoryQueryResult with classification details
        """
        # Quick filter - skip obvious non-memory messages
        if len(message) < 10 or not self._has_memory_keywords(message):
            # Check for factual queries that might not have keywords
            factual_patterns = ['where does', 'where is', 'what is', 'who is',
                               'what\'s his', 'what\'s her', 'what\'s my',
                               'who\'s', 'does she', 'does he',
                               'is she', 'is he', 'her name', 'his name',
                               'how old is']
            if not any(p in message.lower() for p in factual_patterns):
                return MemoryQueryResult(
                    is_memory_query=False,
                    query_type='none',
                    search_terms=[],
                    confidence=0.9,
                    reasoning="No memory keywords detected"
                )

        try:
            return self._classify_with_llm(message)
        except Exception as e:
            logger.error(f"Memory classification failed: {e}")
            # Fall back to keyword-based classification
            return self._fallback_classify(message)

    def _classify_with_llm(self, message: str) -> MemoryQueryResult:
        """Use Haiku to classify the message."""
        client = self._get_client()

        prompt = f"""Analyze this message and determine if it's asking about a past memory, event, or fact.

Message: "{message}"

Respond with JSON only:
{{
  "is_memory_query": true/false,
  "query_type": "specific_event" | "emotional_vague" | "factual" | "none",
  "search_terms": ["term1", "term2"],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Query types:
- specific_event: Asking about a particular past event ("when did we go to...", "remember the time...")
- emotional_vague: Asking for emotional/significant memories ("favorite memory", "best moment", "happiest time")
- factual: Asking for facts about people/places/things ("where does X live", "what's my job", "who is X")
- none: Not a memory query (greetings, requests, roleplay, present-tense conversation)

Extract search_terms that would help find relevant memories (names, places, events, dates).
For emotional_vague queries, use terms like ["significant", "memorable", "emotional", "important"].

IMPORTANT: Roleplay and present-tense conversation is NOT a memory query.
"What should we do tonight?" → none
"What did we do last night?" → specific_event"""

        from src.llm.fireworks_models import call_fireworks

        text = call_fireworks(
            client,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        if not text:
            return self._fallback_classify("")

        # Handle markdown code blocks
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]

        data = json.loads(text)

        result = MemoryQueryResult(
            is_memory_query=data.get('is_memory_query', False),
            query_type=data.get('query_type', 'none'),
            search_terms=data.get('search_terms', []),
            confidence=data.get('confidence', 0.5),
            reasoning=data.get('reasoning', '')
        )

        logger.info(f"Memory classification: {result.query_type} ({result.confidence:.0%}) - {result.reasoning}")

        return result

    def _fallback_classify(self, message: str) -> MemoryQueryResult:
        """Fallback keyword-based classification."""
        message_lower = message.lower()

        # Check for emotional/vague queries
        emotional_patterns = ['favorite', 'favourite', 'best', 'happiest',
                            'saddest', 'most', 'memorable', 'special']
        if any(p in message_lower for p in emotional_patterns):
            return MemoryQueryResult(
                is_memory_query=True,
                query_type='emotional_vague',
                search_terms=['significant', 'memorable', 'emotional'],
                confidence=0.6,
                reasoning="Fallback: emotional keywords detected"
            )

        # Check for specific event queries
        event_patterns = ['remember when', 'that time', 'when did', 'when we',
                         'did we', 'have we', 'last time']
        if any(p in message_lower for p in event_patterns):
            # Extract potential search terms (simple approach)
            words = message.split()
            search_terms = [w for w in words if len(w) > 4 and w.isalpha()][:5]

            return MemoryQueryResult(
                is_memory_query=True,
                query_type='specific_event',
                search_terms=search_terms,
                confidence=0.6,
                reasoning="Fallback: event keywords detected"
            )

        # Check for factual queries
        factual_patterns = ['where does', 'where is', 'what is', 'who is',
                           'what\'s his', 'what\'s her', 'what\'s my',
                           'who\'s', 'how old is']
        if any(p in message_lower for p in factual_patterns):
            words = message.split()
            search_terms = [w for w in words if len(w) > 3 and w.isalpha()][:5]

            return MemoryQueryResult(
                is_memory_query=True,
                query_type='factual',
                search_terms=search_terms,
                confidence=0.6,
                reasoning="Fallback: factual keywords detected"
            )

        return MemoryQueryResult(
            is_memory_query=False,
            query_type='none',
            search_terms=[],
            confidence=0.5,
            reasoning="Fallback: no clear memory pattern"
        )


# Singleton instance
_classifier: Optional[MemoryQueryClassifier] = None


def get_memory_query_classifier() -> MemoryQueryClassifier:
    """Get singleton MemoryQueryClassifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = MemoryQueryClassifier()
    return _classifier


def classify_memory_query(message: str) -> MemoryQueryResult:
    """Convenience function to classify a message."""
    classifier = get_memory_query_classifier()
    return classifier.classify(message)
