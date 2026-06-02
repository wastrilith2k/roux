"""
VerifiedMemory -- shared dataclass for verified memory records.

This module holds the VerifiedMemory dataclass that is passed between
the retrieval layer (MemoryRetriever, RetrievalAgent) and the prompt
injection layer (MemoryValidationAgent).

Separated here so the dataclass can be imported without pulling in the
full retrieval machinery.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VerifiedMemory:
    """A verified memory record from storage."""
    content: str           # The memory content
    source: str            # 'postgres' | 'graphiti' | 'entity_profile' | 'search_tool'
    timestamp: Optional[str] = None   # When this was recorded
    relevance: float = 0.5            # How relevant to the query (0-1)
