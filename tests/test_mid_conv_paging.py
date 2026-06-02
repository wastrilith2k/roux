from unittest.mock import patch, MagicMock


def test_detect_topic_shift_returns_false_on_empty_prior():
    """Empty prior context should never trigger a shift."""
    from src.core.conversation.pipeline import ConversationPipeline
    pipeline = ConversationPipeline()
    assert pipeline._detect_topic_shift('', 'any message') is False
    assert pipeline._detect_topic_shift(None, 'any message') is False


def test_detect_topic_shift_handles_embedding_failure_gracefully():
    """If embedding call fails, should return False (never block the pipeline)."""
    from src.core.conversation.pipeline import ConversationPipeline
    pipeline = ConversationPipeline()
    with patch('src.core.conversation.pipeline.get_embedding', side_effect=Exception("API down")):
        result = pipeline._detect_topic_shift("prior topic", "new message")
    assert result is False


def test_page_in_memory_handles_failure_gracefully():
    """_page_in_memory must not raise on any error."""
    from src.core.conversation.pipeline import ConversationPipeline
    from src.core.conversation.context_builder import ConversationContext
    pipeline = ConversationPipeline()
    ctx = ConversationContext(user_email='t@t.com')
    with patch('src.core.conversation.pipeline.get_retrieval_agent', side_effect=Exception("DB down")):
        # Must not raise
        pipeline._page_in_memory(ctx, user_email='t@t.com', new_message='test')
