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
    with patch.object(agent, '_search_facts', return_value=[]), \
         patch.object(agent, '_search_graphiti', return_value=[]):
        results = agent.search_verified(user_email="test@example.com", query="test", limit=5)
    assert isinstance(results, list)


def test_search_verified_deduplicates_results():
    """Duplicate content entries must be collapsed to one."""
    agent = get_retrieval_agent()
    duplicate = {'content': 'same content here', 'fact': 'same content here'}
    with patch.object(agent, '_search_facts', return_value=[duplicate, duplicate]), \
         patch.object(agent, '_search_graphiti', return_value=[]):
        results = agent.search_verified(user_email="test@example.com", query="test", limit=10)
    assert len(results) == 1


def test_search_verified_respects_limit():
    """Result list must not exceed the requested limit."""
    agent = get_retrieval_agent()
    many = [{'content': f'memory {i}', 'fact': f'memory {i}'} for i in range(20)]
    with patch.object(agent, '_search_facts', return_value=many), \
         patch.object(agent, '_search_graphiti', return_value=[]):
        results = agent.search_verified(user_email="test@example.com", query="test", limit=5)
    assert len(results) <= 5


def test_search_verified_orders_by_importance():
    """Higher importance facts should appear earlier in results."""
    agent = get_retrieval_agent()
    low  = {'content': 'low importance fact',  'fact': 'low importance fact',  'importance_score': 2}
    high = {'content': 'high importance fact', 'fact': 'high importance fact', 'importance_score': 9}
    with patch.object(agent, '_search_facts', return_value=[low, high]), \
         patch.object(agent, '_search_graphiti', return_value=[]):
        results = agent.search_verified(user_email="t@t.com", query="test", limit=10)
    assert len(results) >= 2
    contents = [r.get('content', '') for r in results]
    high_idx = next(i for i, c in enumerate(contents) if 'high' in c)
    low_idx  = next(i for i, c in enumerate(contents) if 'low' in c)
    assert high_idx < low_idx, (
        f"High importance (idx {high_idx}) should rank before low (idx {low_idx})"
    )


def test_retrieval_agent_calls_hybrid_search():
    """_search_facts must call search_facts_hybrid, not plain search_facts."""
    agent = get_retrieval_agent()
    mock_store = MagicMock()
    mock_store.search_facts_hybrid.return_value = []

    with patch('src.memory.retrieval_agent.get_fact_store', return_value=mock_store), \
         patch.object(agent, '_search_graphiti', return_value=[]):
        agent._search_facts(user_email='t@t.com', query='test query', limit=10)

    mock_store.search_facts_hybrid.assert_called()
    mock_store.search_facts.assert_not_called()


def test_search_verified_fuses_both_sources():
    """search_verified must call both _search_facts and _search_graphiti."""
    agent = get_retrieval_agent()
    with patch.object(agent, '_search_facts',
                      return_value=[{'content': 'fact1', 'fact': 'fact1'}]) as mock_facts, \
         patch.object(agent, '_search_graphiti',
                      return_value=[{'content': 'graph1', 'fact': 'graph1'}]) as mock_graph:
        results = agent.search_verified(user_email='t@t.com', query='test', limit=10)
    mock_facts.assert_called_once()
    mock_graph.assert_called_once()
    contents = [r.get('content', r.get('fact', '')) for r in results]
    assert 'fact1' in contents
    assert 'graph1' in contents


# =============================================================================
# RRF fusion tests
# =============================================================================

from src.memory.retrieval_agent import RetrievalAgent


def test_rrf_fuse_promotes_items_appearing_in_both_lists():
    """An item in both fact and graphiti results should have higher RRF score than one in only one list."""
    fact_results = [
        {'content': 'alpha', 'fact': 'alpha', 'hybrid_score': 0.9},
        {'content': 'beta',  'fact': 'beta',  'hybrid_score': 0.8},
        {'content': 'gamma', 'fact': 'gamma', 'hybrid_score': 0.7},
    ]
    graph_results = [
        {'content': 'gamma', 'fact': 'gamma'},  # appears in both — should rank up
        {'content': 'delta', 'fact': 'delta'},
    ]
    fused = RetrievalAgent._rrf_fuse(fact_results, graph_results, k=60)
    contents = [r.get('content', r.get('fact', '')) for r in fused]
    # gamma appears in both lists — must rank above beta (only in facts at position 2)
    assert 'gamma' in contents
    assert 'beta' in contents
    gamma_idx = contents.index('gamma')
    beta_idx = contents.index('beta')
    assert gamma_idx < beta_idx, f"gamma (idx {gamma_idx}) should beat beta (idx {beta_idx})"


def test_rrf_fuse_deduplicates():
    fact_results = [{'content': 'same', 'fact': 'same thing'}]
    graph_results = [{'content': 'same', 'fact': 'same thing'}]
    fused = RetrievalAgent._rrf_fuse(fact_results, graph_results, k=60)
    assert len(fused) == 1, f"Expected 1 result, got {len(fused)}"


def test_rrf_fuse_includes_all_items():
    fact_results = [{'content': 'only_in_facts', 'fact': 'only_in_facts'}]
    graph_results = [{'content': 'only_in_graph', 'fact': 'only_in_graph'}]
    fused = RetrievalAgent._rrf_fuse(fact_results, graph_results, k=60)
    contents = [r.get('content', '') for r in fused]
    assert 'only_in_facts' in contents
    assert 'only_in_graph' in contents
