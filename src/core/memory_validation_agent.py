"""
Memory Validation Agent -- pre-generation anti-confabulation injection.

WHAT: Before the LLM generates a response, this agent detects memory-related
      queries ("remember when...?", "what's his job?"), retrieves verified
      records from storage, and injects them into the prompt along with
      anti-confabulation instructions tailored to the query type.

WHY:  Without verified records in the prompt, the LLM will cheerfully invent
      specific dates, locations, and dialogue. This agent forces the LLM to
      reference only verified facts and explicitly say "I don't remember" when
      no records exist -- dramatically reducing confabulation.

HOW:  Pipeline: classify (MemoryQueryClassifier) -> retrieve (MemoryRetriever)
      -> format (this module). Different instruction templates for:
      - No records found: "say you don't remember"
      - Emotional/vague queries: "choose from these, don't invent new ones"
      - Specific events: "reference ONLY these details"
      - Factual queries: "use these facts or say you don't know"

Pipeline position: user message -> [this] -> system prompt -> LLM
Counterpart: MessageValidatorAgent (post-generation)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .memory_query_classifier import (
    MemoryQueryClassifier,
    MemoryQueryResult,
    get_memory_query_classifier
)
from .memory_retriever import (
    MemoryRetriever,
    VerifiedMemory,
    get_memory_retriever
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Context to inject into prompt for memory queries."""
    is_memory_query: bool
    query_type: str  # 'specific_event' | 'emotional_vague' | 'factual' | 'none'
    verified_records: List[str] = field(default_factory=list)
    search_terms: List[str] = field(default_factory=list)
    instruction: str = ""
    confidence: float = 0.0


class MemoryValidationAgent:
    """
    Orchestrates memory query detection and retrieval.

    Pipeline:
    1. Classify incoming message (is it a memory query?)
    2. If yes, retrieve verified records from storage
    3. Build MemoryContext with anti-confabulation instructions
    """

    def __init__(self):
        self.classifier = get_memory_query_classifier()
        self.retriever = get_memory_retriever()

    def validate(
        self,
        message: str,
        user_email: str = None
    ) -> MemoryContext:
        """
        Validate a message for memory queries and retrieve context.

        Args:
            message: User's message
            user_email: User's email for database filtering

        Returns:
            MemoryContext with verified records and instructions
        """
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        # 1. Classify the message
        classification = self.classifier.classify(message)

        if not classification.is_memory_query:
            return MemoryContext(
                is_memory_query=False,
                query_type='none',
                confidence=classification.confidence
            )

        logger.info(
            f"Memory query detected: {classification.query_type} "
            f"(confidence: {classification.confidence:.0%})"
        )

        # 2. Retrieve verified records
        records = self.retriever.search(
            search_terms=classification.search_terms,
            user_email=user_email,
            query_type=classification.query_type,
            limit=10
        )

        # 3. Format records for prompt
        formatted_records = self._format_records(records)

        # 4. Generate appropriate instruction
        instruction = self._get_instruction(
            classification.query_type,
            records,
            classification.search_terms
        )

        return MemoryContext(
            is_memory_query=True,
            query_type=classification.query_type,
            verified_records=formatted_records,
            search_terms=classification.search_terms,
            instruction=instruction,
            confidence=classification.confidence
        )

    def _format_records(self, records: List[VerifiedMemory]) -> List[str]:
        """Format verified memory records for prompt injection."""
        formatted = []

        for record in records:
            # Add source annotation
            if record.source == 'postgres':
                source_tag = "[conversation]"
            elif record.source == 'graphiti':
                source_tag = "[knowledge]"
            elif record.source == 'entity_profile':
                source_tag = "[fact]"
            else:
                source_tag = ""

            formatted.append(f"{source_tag} {record.content}")

        return formatted

    def _get_instruction(
        self,
        query_type: str,
        records: List[VerifiedMemory],
        search_terms: List[str]
    ) -> str:
        """Generate anti-confabulation instruction based on context."""

        if not records:
            # No verified records found
            return f"""[MEMORY VALIDATION: NO RECORDS FOUND]
Search terms: {', '.join(search_terms)}

No verified records found for this query.
DO NOT invent specific details like:
- Specific dates or times
- Specific locations or places
- Specific dialogue or words said
- Specific people who were present

Instead:
- Say "I don't remember the specifics" or ask for clarification
- Express general feelings without inventing events
- Reference general patterns rather than specific instances
- Ask James to remind you of the details"""

        if query_type == 'emotional_vague':
            # Vague emotional queries (favorite memory, best moment)
            return f"""[MEMORY VALIDATION: SIGNIFICANT MEMORIES]
Found {len(records)} verified memories.

For emotional/vague queries like "favorite memory":
- Choose from the verified memories below OR express general feelings
- Do NOT invent new memories not listed here
- Do NOT add specific details (dates, dialogue, people) not in these records
- You may describe emotions and feelings, but reference ONLY these verified events

If none of these feel like a genuine "favorite," say something like:
"There are so many moments... but honestly, I treasure the small ones that aren't
easy to pinpoint - just being together, you know?"

This is better than inventing a specific false memory."""

        if query_type == 'specific_event':
            # Specific event queries (remember when, that time)
            return f"""[MEMORY VALIDATION: SPECIFIC EVENT]
Search terms: {', '.join(search_terms)}
Found {len(records)} verified records.

Reference ONLY these specific details.
- Do NOT invent details not present (locations, dialogue, people, timing)
- If records are incomplete, acknowledge gaps: "I remember [verified detail],
  but I'm fuzzy on the rest..."
- Do NOT "fill in" missing details with plausible inventions"""

        if query_type == 'factual':
            # Factual queries (where does X live, what's Y's job)
            return f"""[MEMORY VALIDATION: FACTUAL QUERY]
Search terms: {', '.join(search_terms)}
Found {len(records)} verified facts.

For factual queries:
- Reference ONLY the facts listed below
- If the answer isn't in these records, say "I don't know" or "I'm not sure"
- Do NOT invent facts about people, places, jobs, relationships
- It's better to admit uncertainty than invent incorrect information"""

        # Default instruction
        return f"""[MEMORY VALIDATION]
Found {len(records)} verified records.
Reference these records. Do not invent additional specific details."""


def format_memory_context_for_prompt(context: MemoryContext) -> str:
    """
    Format MemoryContext for injection into the main prompt.

    Args:
        context: MemoryContext from validation

    Returns:
        Formatted string to inject before LLM call
    """
    if not context.is_memory_query:
        return ""

    if not context.verified_records:
        # No records - just include the instruction
        return f"\n{context.instruction}\n"

    lines = [
        "",
        "=" * 50,
        f"VERIFIED MEMORIES FOR THIS QUERY (query type: {context.query_type})",
        "=" * 50,
        ""
    ]

    for record in context.verified_records:
        lines.append(f"  • {record}")

    lines.append("")
    lines.append(context.instruction)
    lines.append("=" * 50)
    lines.append("")

    return "\n".join(lines)


# Singleton instance
_agent: Optional[MemoryValidationAgent] = None


def get_memory_validation_agent() -> MemoryValidationAgent:
    """Get singleton MemoryValidationAgent instance."""
    global _agent
    if _agent is None:
        _agent = MemoryValidationAgent()
    return _agent


def validate_memory_query(
    message: str,
    user_email: str = None
) -> MemoryContext:
    """
    Convenience function to validate a message for memory queries.

    Args:
        message: User's message
        user_email: User's email

    Returns:
        MemoryContext with verified records and instructions
    """
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    agent = get_memory_validation_agent()
    return agent.validate(message, user_email)
