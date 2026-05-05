"""
Tests for per-user group_id isolation in the Graphiti knowledge graph.

Regression tests for: episodes/edges written without group_id stamp (all data
shared in a flat namespace) and reads returning cross-user facts.
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call


# ---------------------------------------------------------------------------
# _group_id_from_email
# ---------------------------------------------------------------------------

def test_group_id_from_email_basic():
    from src.memory.graphiti_search import _group_id_from_email
    assert _group_id_from_email("wastrilith@gmail.com") == "user_wastrilith"


def test_group_id_from_email_none_returns_none():
    from src.memory.graphiti_search import _group_id_from_email
    assert _group_id_from_email(None) is None


def test_group_id_from_email_empty_returns_none():
    from src.memory.graphiti_search import _group_id_from_email
    assert _group_id_from_email("") is None


def test_group_id_from_email_different_users_are_distinct():
    from src.memory.graphiti_search import _group_id_from_email
    gid_a = _group_id_from_email("alice@example.com")
    gid_b = _group_id_from_email("bob@example.com")
    assert gid_a != gid_b
    assert gid_a == "user_alice"
    assert gid_b == "user_bob"


# ---------------------------------------------------------------------------
# _stamp_group_id — verifies Cypher calls are made with correct group_id
# ---------------------------------------------------------------------------

def _make_neo4j_mock(mock_session):
    """Inject a fake neo4j module so tests don't need neo4j installed."""
    import sys
    mock_neo4j = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = lambda s, *a: mock_session
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_neo4j.GraphDatabase.driver.return_value = mock_driver
    sys.modules.setdefault('neo4j', mock_neo4j)
    return mock_neo4j


def test_stamp_group_id_makes_cypher_calls():
    """After add_episode, group_id must be stamped on all returned UUIDs."""
    import sys
    mock_session = MagicMock()
    mock_neo4j = _make_neo4j_mock(mock_session)
    sys.modules['neo4j'] = mock_neo4j

    from src.tasks.graphiti_extraction_task import _stamp_group_id

    episode_mock = MagicMock()
    episode_mock.uuid = "ep-uuid-1"
    node_mock = MagicMock(); node_mock.uuid = "node-uuid-1"
    edge_mock = MagicMock(); edge_mock.uuid = "edge-uuid-1"
    ep_edge_mock = MagicMock(); ep_edge_mock.uuid = "epedge-uuid-1"

    add_result = MagicMock()
    add_result.episode = episode_mock
    add_result.nodes = [node_mock]
    add_result.edges = [edge_mock]
    add_result.episodic_edges = [ep_edge_mock]

    asyncio.run(_stamp_group_id(add_result, "wastrilith@gmail.com"))

    assert mock_session.run.call_count == 4
    calls = [c.args[0] for c in mock_session.run.call_args_list]
    assert any("Episodic" in q for q in calls)
    assert any("Entity" in q for q in calls)
    assert any("RELATES_TO" in q for q in calls)
    assert any("MENTIONS" in q for q in calls)
    for c in mock_session.run.call_args_list:
        assert c.kwargs.get("gid") == "user_wastrilith"


def test_stamp_group_id_handles_empty_lists_gracefully():
    """Empty nodes/edges should not cause errors (just skip those Cypher calls)."""
    import sys
    mock_session = MagicMock()
    mock_neo4j = _make_neo4j_mock(mock_session)
    sys.modules['neo4j'] = mock_neo4j

    from src.tasks.graphiti_extraction_task import _stamp_group_id

    add_result = MagicMock()
    add_result.episode = MagicMock(); add_result.episode.uuid = "ep-uuid-2"
    add_result.nodes = []
    add_result.edges = []
    add_result.episodic_edges = []

    asyncio.run(_stamp_group_id(add_result, "wastrilith@gmail.com"))
    assert mock_session.run.call_count == 1


# ---------------------------------------------------------------------------
# context_builder passes group_id to graphiti search
# ---------------------------------------------------------------------------

def test_get_graphiti_context_passes_group_id():
    """_get_graphiti_context must derive group_id from user_email and pass it."""
    with patch("src.memory.graphiti_search.search_graphiti") as mock_search:
        mock_search.return_value = []

        from src.core.conversation.context_builder import ContextBuilder
        cb = ContextBuilder.__new__(ContextBuilder)

        with patch("src.memory.graphiti_search.get_graphiti_context_with_importance") as mock_ctx:
            mock_ctx.return_value = ""
            cb._get_graphiti_context("wastrilith@gmail.com", "hello there")
            if mock_ctx.called:
                _, kwargs = mock_ctx.call_args
                assert kwargs.get("group_id") == "user_wastrilith"
