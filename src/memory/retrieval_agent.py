"""
Retrieval Agent - LLM-driven smart memory retrieval orchestration.

WHAT: Uses GPT-4o-mini to analyze each user message and produce a structured
RetrievalPlan: which entities are mentioned, what time period matters, which
memory sources to prioritize, and what specific search queries to run. Falls
back to keyword-based heuristics if the LLM call fails.

WHY: Naive memory retrieval just embeds the user's message and does vector
search. But "how are you?" doesn't semantically match "Jesse ran away" even
though that's exactly what the companion should remember. The retrieval agent
understands intent: "how is Jesse?" triggers entity-focused searches across
events and biographies, not just raw message history.

HOW it fits:
  - context_builder calls analyze_query_for_retrieval() at the start of each
    turn, getting back a RetrievalPlan.
  - The plan's priority_sources and specific_queries guide which memory modules
    get queried and with what search terms.
  - Runs in ~100ms (GPT-4o-mini with json_object response format).
  - enhance_query() expands the plan into multiple search queries for
    semantic search, adding entity-focused and intent-specific variations.

Example:
    User: "What happened with Jesse last week?"
    Agent produces: entities=["jesse"], time_references=["last week"],
        priority_sources=["events", "graphiti"],
        specific_queries=["Jesse health", "Jesse crisis", "Jesse school"]
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from zoneinfo import ZoneInfo
from src.config.persona_config import get_persona_config
from src.memory.fact_store import get_fact_store

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# =============================================================================
# Configuration
# =============================================================================

# GPT-4o-mini: best structured output quality at lowest cost ($0.15/1M input)
OPENAI_MODEL = "gpt-4o-mini"


# =============================================================================
# Retrieval plan data model
# =============================================================================

@dataclass
class RetrievalPlan:
    """Structured plan for what memories to retrieve."""
    # Entities mentioned or implied in the query
    entities: List[str] = field(default_factory=list)

    # Time references detected (e.g., "yesterday", "last week", "January")
    time_references: List[str] = field(default_factory=list)

    # Start and end dates for temporal filtering
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None

    # Memory sources to prioritize (events, graphiti, facts, messages)
    priority_sources: List[str] = field(default_factory=list)

    # Specific queries to run against memory sources
    specific_queries: List[str] = field(default_factory=list)

    # Query intent (informational, emotional, contextual)
    intent: str = "informational"

    # Whether this query needs historical context
    needs_history: bool = False

    # Confidence in the plan (0-1)
    confidence: float = 0.8


# =============================================================================
# Retrieval agent
# =============================================================================

class RetrievalAgent:
    """
    Lightweight agent that analyzes user messages to plan memory retrieval.

    Two paths:
      1. LLM analysis (primary): GPT-4o-mini produces structured JSON plan
      2. Keyword fallback: simple pattern matching when LLM fails or is too slow
    """

    # Memory source descriptions -- included in the LLM prompt so the model
    # understands what each source contains and when to prefer it.
    MEMORY_SOURCES = {
        'events': 'Synthesized event narratives (crises, career changes, milestones)',
        'graphiti': 'Knowledge graph facts and relationships',
        'facts': 'Extracted facts from conversations (preferences, details)',
        'messages': 'Raw conversation history (semantic search)',
        'biographies': 'Person summaries (synthesized from facts)'
    }

    def __init__(self):
        self._client = None
        self._memory_retriever = None
        _pc = get_persona_config()
        # Known entities -- provided to the LLM so it can resolve references
        # like "the kids" -> specific names. Loaded from persona.yaml
        # known_entities.entries so instances can configure their own.
        self.KNOWN_ENTITIES = _pc.get_known_entities_dict()

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        return self._client

    def _get_memory_retriever(self):
        """Lazy-load MemoryRetriever singleton (avoids circular imports)."""
        if self._memory_retriever is None:
            from src.memory.memory_retriever import get_memory_retriever
            self._memory_retriever = get_memory_retriever()
        return self._memory_retriever

    def search_verified(
        self,
        user_email: str,
        query: str,
        limit: int = 10,
        search_terms: Optional[List[str]] = None,
        query_type: str = 'specific_event',
    ) -> list:
        """Unified replacement for legacy MemoryRetriever.search().

        Delegates to MemoryRetriever which searches PostgreSQL conversation
        history, entity profiles, and any remaining knowledge-graph backends.
        Results are deduplicated by content and trimmed to *limit*.

        Args:
            user_email: User identifier for database filtering.
            query: Primary search query string.
            limit: Maximum number of results to return.
            search_terms: Optional list of explicit terms; defaults to [query].
            query_type: Passed through to MemoryRetriever strategy selector.
                        One of: specific_event | emotional_vague | factual.

        Returns:
            List of VerifiedMemory dataclass instances (or dicts for
            forward-compatibility), deduplicated and ranked by relevance.
        """
        terms = search_terms or [query]
        retriever = self._get_memory_retriever()
        raw = retriever.search(
            search_terms=terms,
            user_email=user_email,
            query_type=query_type,
            limit=limit * 2,  # over-fetch so dedup still meets the limit
        )

        # Deduplicate by normalised content prefix (first 100 chars, lowercased)
        seen: set = set()
        unique = []
        for item in raw:
            # VerifiedMemory dataclass — use .content attribute
            key = getattr(item, 'content', '') or ''
            key = key.lower().strip()[:100]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        # Sort by importance_score descending (None treated as 5 — neutral importance)
        unique.sort(
            key=lambda x: getattr(x, "importance_score", None)
                          or (x.get("importance_score") if isinstance(x, dict) else None)
                          or 5,
            reverse=True,
        )
        return unique[:limit]


    def _search_facts(self, user_email: str, query: str, limit: int = 10) -> list:
        """Search fact store using hybrid BM25+pgvector."""
        store = get_fact_store()
        return store.search_facts_hybrid(
            query=query,
            limit=limit,
            text_weight=0.4,
            vector_weight=0.6,
            user_email=user_email,
        )

    def analyze_query(
        self,
        user_message: str,
        conversation_context: str = "",
        current_time: datetime = None
    ) -> RetrievalPlan:
        """
        Analyze a user message and return a structured retrieval plan.

        Args:
            user_message: The user's message to analyze
            conversation_context: Recent conversation for context (optional)
            current_time: Current time for temporal calculations (default: now)

        Returns:
            RetrievalPlan with entities, time ranges, queries, etc.
        """
        if not user_message or len(user_message.strip()) < 5:
            return RetrievalPlan()

        current_time = current_time or datetime.now(PST)

        try:
            client = self._get_client()

            # Build context about known entities
            entities_context = "\n".join([
                f"- {name}: {desc}" for name, desc in self.KNOWN_ENTITIES.items()
            ])

            sources_context = "\n".join([
                f"- {name}: {desc}" for name, desc in self.MEMORY_SOURCES.items()
            ])

            prompt = f"""You are a retrieval planner for an AI companion. Analyze the user's message and determine what memories to retrieve.

CURRENT TIME: {current_time.strftime('%Y-%m-%d %H:%M')} Pacific

KNOWN ENTITIES:
{entities_context}

MEMORY SOURCES:
{sources_context}

USER MESSAGE: {user_message}

{f"RECENT CONTEXT: {conversation_context[:500]}" if conversation_context else ""}

TASK: Analyze the message and return a JSON retrieval plan:

1. entities: List of entity names mentioned or implied (use lowercase: james, jesse, etc.)
2. time_references: List of time references detected (e.g., "yesterday", "last week")
3. time_range_start: ISO date string for earliest relevant time (null if no time reference)
4. time_range_end: ISO date string for latest relevant time (null if no time reference)
5. priority_sources: Which memory sources to prioritize (events, graphiti, facts, messages, biographies)
6. specific_queries: 1-3 specific search queries to run
7. intent: "informational" (asking about facts), "emotional" (checking on feelings), or "contextual" (needs background)
8. needs_history: true if the query needs past conversation context
9. confidence: How confident you are in this plan (0.0-1.0)

EXAMPLES:

"How is Jesse?" →
{{"entities": ["jesse"], "time_references": [], "time_range_start": null, "time_range_end": null, "priority_sources": ["events", "biographies"], "specific_queries": ["Jesse health", "Jesse current status"], "intent": "emotional", "needs_history": false, "confidence": 0.9}}

"What happened last week?" →
{{"entities": [], "time_references": ["last week"], "time_range_start": "{(current_time - timedelta(days=7)).strftime('%Y-%m-%d')}", "time_range_end": "{current_time.strftime('%Y-%m-%d')}", "priority_sources": ["events", "messages"], "specific_queries": ["important events", "significant conversations"], "intent": "informational", "needs_history": true, "confidence": 0.8}}

"Did Jesse go to school today?" →
{{"entities": ["jesse"], "time_references": ["today"], "time_range_start": "{current_time.strftime('%Y-%m-%d')}", "time_range_end": "{current_time.strftime('%Y-%m-%d')}", "priority_sources": ["messages", "events"], "specific_queries": ["Jesse school", "Jesse today"], "intent": "informational", "needs_history": false, "confidence": 0.85}}

Return ONLY valid JSON, no explanation:"""

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()

            try:
                from src.services.cost_tracker import get_cost_tracker
                usage = response.usage
                if usage:
                    get_cost_tracker().track_openai_call(
                        user_id='system', prompt_tokens=usage.prompt_tokens or 0,
                        completion_tokens=usage.completion_tokens or 0,
                        model=OPENAI_MODEL, service_type='retrieval_planning', call_purpose='retrieval_planning')
            except Exception:
                pass

            plan_dict = json.loads(content)

            # Convert to RetrievalPlan
            plan = RetrievalPlan(
                entities=plan_dict.get('entities', []),
                time_references=plan_dict.get('time_references', []),
                priority_sources=plan_dict.get('priority_sources', ['messages']),
                specific_queries=plan_dict.get('specific_queries', [user_message[:100]]),
                intent=plan_dict.get('intent', 'informational'),
                needs_history=plan_dict.get('needs_history', False),
                confidence=plan_dict.get('confidence', 0.8)
            )

            # Parse time range if provided
            if plan_dict.get('time_range_start'):
                try:
                    plan.time_range_start = datetime.fromisoformat(plan_dict['time_range_start']).replace(tzinfo=PST)
                except:
                    pass

            if plan_dict.get('time_range_end'):
                try:
                    plan.time_range_end = datetime.fromisoformat(plan_dict['time_range_end']).replace(tzinfo=PST)
                except:
                    pass

            logger.info(f"RetrievalAgent plan: entities={plan.entities}, sources={plan.priority_sources}, queries={plan.specific_queries}")
            return plan

        except json.JSONDecodeError as e:
            logger.warning(f"RetrievalAgent JSON parse error: {e}")
            return self._fallback_plan(user_message)
        except Exception as e:
            logger.warning(f"RetrievalAgent error: {e}")
            return self._fallback_plan(user_message)

    def _fallback_plan(self, user_message: str) -> RetrievalPlan:
        """
        Generate a fallback plan using simple keyword matching.

        Used when LLM analysis fails. Lower confidence (0.5) so downstream
        consumers know to cast a wider net.
        """
        message_lower = user_message.lower()

        # Detect entities
        entities = []
        for entity in self.KNOWN_ENTITIES.keys():
            if entity in message_lower:
                entities.append(entity)

        # Default to James if no entity detected and it's a question
        if not entities and '?' in user_message:
            entities = ['james']

        # Detect time references
        time_refs = []
        time_keywords = {
            'yesterday': 1,
            'today': 0,
            'last week': 7,
            'last night': 1,
            'this morning': 0,
            'earlier': 1,
            'recently': 7
        }

        for keyword, days in time_keywords.items():
            if keyword in message_lower:
                time_refs.append(keyword)

        # Choose sources based on query type
        sources = ['messages']  # Default
        if any(word in message_lower for word in ['how is', 'how are', 'what happened', 'crisis', 'er ', 'hospital']):
            sources = ['events', 'biographies', 'messages']
        elif any(word in message_lower for word in ['like', 'prefer', 'favorite', 'drink', 'eat']):
            sources = ['facts', 'graphiti', 'messages']

        return RetrievalPlan(
            entities=entities,
            time_references=time_refs,
            priority_sources=sources,
            specific_queries=[user_message[:100]],
            confidence=0.5
        )

    def enhance_query(self, user_message: str, plan: RetrievalPlan) -> List[str]:
        """
        Expand the retrieval plan into multiple semantic search queries.

        Adds entity-focused and intent-specific query variations beyond the
        original message. For example, an emotional intent about Jesse also
        generates "How is Jesse feeling" alongside the plan's specific queries.

        Returns up to 5 deduplicated queries.
        """
        queries = []

        # Add specific queries from the plan
        queries.extend(plan.specific_queries)

        # Add entity-focused queries
        for entity in plan.entities:
            entity_name = entity.title()
            queries.append(f"{entity_name} current status")

            if plan.intent == 'emotional':
                queries.append(f"How is {entity_name} feeling")
            elif 'events' in plan.priority_sources:
                queries.append(f"{entity_name} recent events")

        # Add original message if not already included
        if user_message not in queries:
            queries.insert(0, user_message)

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries[:5]  # Limit to 5 queries


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_retrieval_agent: Optional[RetrievalAgent] = None


def get_retrieval_agent() -> RetrievalAgent:
    """Get or create RetrievalAgent singleton."""
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = RetrievalAgent()
    return _retrieval_agent


def analyze_query_for_retrieval(
    user_message: str,
    conversation_context: str = ""
) -> RetrievalPlan:
    """
    Convenience function to analyze a query and get a retrieval plan.

    This is the main entry point for the context builder.
    """
    agent = get_retrieval_agent()
    return agent.analyze_query(user_message, conversation_context)


# Test function
def test_retrieval_agent():
    """Test the retrieval agent with sample queries."""
    test_queries = [
        "How is Jesse?",
        "What happened with Jesse last week?",
        "Does James prefer tea or coffee?",
        "Tell me about the ER visit",
        "What did we talk about yesterday?",
        "Is Kyler doing okay at school?",
    ]

    agent = get_retrieval_agent()

    for query in test_queries:
        print(f"\nQuery: {query}")
        plan = agent.analyze_query(query)
        print(f"  Entities: {plan.entities}")
        print(f"  Time refs: {plan.time_references}")
        print(f"  Sources: {plan.priority_sources}")
        print(f"  Queries: {plan.specific_queries}")
        print(f"  Intent: {plan.intent}")
        print(f"  Confidence: {plan.confidence}")


if __name__ == "__main__":
    test_retrieval_agent()
