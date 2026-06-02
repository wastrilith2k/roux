from unittest.mock import patch, MagicMock


def test_lift_facts_calls_store_fact_with_inferred_preference():
    """lift_facts() should call store_fact with fact_type='inferred_preference'."""
    mock_store = MagicMock()
    mock_store.get_important_facts.return_value = [
        {'id': 1, 'subject': 'James', 'predicate': 'likes', 'object': 'ramen', 'importance': 8},
        {'id': 2, 'subject': 'James', 'predicate': 'enjoyed', 'object': 'Thai food', 'importance': 8},
        {'id': 3, 'subject': 'James', 'predicate': 'likes', 'object': 'sushi', 'importance': 7},
    ]

    with patch('src.tasks.semantic_lifting_task.get_fact_store', return_value=mock_store), \
         patch('src.tasks.semantic_lifting_task._cluster_and_lift', return_value=[
             {'subject': 'James', 'predicate': 'prefers', 'obj': 'Asian cuisine',
              'source': 'inferred_preference', 'confidence': 0.7}
         ]):
        from src.tasks.semantic_lifting_task import lift_facts
        result = lift_facts(user_email='test@example.com')

    mock_store.store_fact.assert_called_once()
    call_kwargs = mock_store.store_fact.call_args
    # Verify source='inferred_preference' was passed
    assert call_kwargs.kwargs.get('source') == 'inferred_preference' or \
           (call_kwargs.args and 'inferred_preference' in str(call_kwargs))
    assert result['lifted'] == 1


def test_lift_facts_skips_when_no_high_importance_facts():
    mock_store = MagicMock()
    mock_store.get_important_facts.return_value = []
    with patch('src.tasks.semantic_lifting_task.get_fact_store', return_value=mock_store):
        from src.tasks.semantic_lifting_task import lift_facts
        result = lift_facts(user_email='test@example.com')
    mock_store.store_fact.assert_not_called()
    assert result['lifted'] == 0
