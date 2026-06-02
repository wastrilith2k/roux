from unittest.mock import patch, MagicMock, call


def test_expand_entity_hops_returns_connected_entities():
    """expand_entity_hops should return entities reachable within N hops."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.__enter__ = MagicMock(return_value=mock_driver)
    mock_driver.__exit__ = MagicMock(return_value=False)
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = mock_session

    mock_session.run.return_value = [
        {'name': 'Jesse', 'distance': 1},
        {'name': 'Dr. Smith', 'distance': 2},
    ]

    with patch('src.memory.graphiti_search._get_graphiti_neo4j_driver', return_value=mock_driver):
        from src.memory.graphiti_search import expand_entity_hops
        result = expand_entity_hops(entity_name='James', hops=2, group_id='test-group')

    assert 'Jesse' in result
    assert 'Dr. Smith' in result


def test_expand_entity_hops_returns_empty_on_no_driver():
    with patch('src.memory.graphiti_search._get_graphiti_neo4j_driver', return_value=None):
        from src.memory.graphiti_search import expand_entity_hops
        result = expand_entity_hops(entity_name='James', hops=2, group_id='test')
    assert result == []


def test_expand_entity_hops_returns_empty_on_exception():
    mock_driver = MagicMock()
    mock_driver.session.side_effect = Exception("Connection refused")

    with patch('src.memory.graphiti_search._get_graphiti_neo4j_driver', return_value=mock_driver):
        from src.memory.graphiti_search import expand_entity_hops
        result = expand_entity_hops(entity_name='James', hops=2, group_id='test')
    assert result == []
