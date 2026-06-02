"""
Tests for RetrievalAgent.search_verified() -- the unified replacement for
the legacy MemoryRetriever.search() interface.
"""
from unittest.mock import patch, MagicMock
from src.memory.retrieval_agent import get_retrieval_agent


def test_retrieval_agent_has_search_verified():
    agent = get_retrieval_agent()
    assert hasattr(agent, 'search_verified'), "RetrievalAgent must have search_verified()"
    assert callable(agent.search_verified)


def test_search_verified_returns_list():
    agent = get_retrieval_agent()
    # Mock the internal MemoryRetriever to avoid needing a real DB
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    with patch.object(agent, '_get_memory_retriever', return_value=mock_retriever):
        results = agent.search_verified(user_email="test@example.com", query="test", limit=5)
    assert isinstance(results, list)


def test_search_verified_deduplicates_results():
    """Duplicate content entries must be collapsed to one."""
    from src.core.verified_memory import VerifiedMemory

    agent = get_retrieval_agent()
    duplicate = VerifiedMemory(content="same content here", source="postgres", relevance=0.8)
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [duplicate, duplicate]
    with patch.object(agent, '_get_memory_retriever', return_value=mock_retriever):
        results = agent.search_verified(user_email="test@example.com", query="test", limit=10)
    assert len(results) == 1


def test_search_verified_respects_limit():
    """Result list must not exceed the requested limit."""
    from src.core.verified_memory import VerifiedMemory

    agent = get_retrieval_agent()
    many = [
        VerifiedMemory(content=f"memory {i}", source="postgres", relevance=0.5)
        for i in range(20)
    ]
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = many
    with patch.object(agent, '_get_memory_retriever', return_value=mock_retriever):
        results = agent.search_verified(user_email="test@example.com", query="test", limit=5)
    assert len(results) <= 5
