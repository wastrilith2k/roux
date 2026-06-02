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


def test_search_verified_orders_by_importance():
    """Higher importance facts should appear earlier in results."""
    from src.core.verified_memory import VerifiedMemory

    agent = get_retrieval_agent()
    # Low importance fact first in the raw list — without sorting it would stay first
    low = VerifiedMemory(content="low importance fact", source="postgres", relevance=0.5,
                         importance_score=2)
    high = VerifiedMemory(content="high importance fact", source="postgres", relevance=0.5,
                          importance_score=9)
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [low, high]
    with patch.object(agent, '_get_memory_retriever', return_value=mock_retriever):
        results = agent.search_verified(user_email="t@t.com", query="test", limit=10)
    assert len(results) >= 2
    # High importance fact should appear before low importance fact
    contents = [getattr(r, 'content', r.get('content', '') if isinstance(r, dict) else '')
                for r in results]
    high_idx = next(i for i, c in enumerate(contents) if 'high' in c)
    low_idx = next(i for i, c in enumerate(contents) if 'low' in c)
    assert high_idx < low_idx, (
        f"High importance (idx {high_idx}) should rank before low (idx {low_idx})"
    )


def test_retrieval_agent_calls_hybrid_search():
    """_search_facts must call search_facts_hybrid, not plain search_facts."""
    agent = get_retrieval_agent()
    mock_store = MagicMock()
    mock_store.search_facts_hybrid.return_value = []

    with patch('src.memory.retrieval_agent.get_fact_store', return_value=mock_store):
        try:
            agent._search_facts(user_email='t@t.com', query='test query', limit=10)
        except Exception:
            pass  # May fail due to DB, but we just need to check what was called

    mock_store.search_facts_hybrid.assert_called()
    mock_store.search_facts.assert_not_called()
