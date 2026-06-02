from src.core.conversation.pipeline import ConversationPipeline


def test_compress_tool_result_returns_original_when_short():
    pipeline = ConversationPipeline()
    short_result = "No events today."
    result = pipeline._compress_tool_result(tool_name="calendar", raw=short_result)
    assert result == short_result


def test_compress_tool_result_truncates_when_llm_unavailable():
    """When LLM compression fails, must truncate and not raise."""
    from unittest.mock import patch
    pipeline = ConversationPipeline()
    long_result = "Meeting at 3pm. " * 100  # ~1600 chars
    with patch.object(pipeline, '_llm_compress', side_effect=Exception("API down")):
        result = pipeline._compress_tool_result(tool_name="calendar", raw=long_result)
    assert len(result) <= 900, f"Expected <= 900 chars after fallback truncation, got {len(result)}"
    assert '[truncated]' in result


def test_compress_tool_result_short_enough_means_no_llm_call():
    """Should not call _llm_compress for short inputs."""
    from unittest.mock import patch
    pipeline = ConversationPipeline()
    with patch.object(pipeline, '_llm_compress') as mock_compress:
        pipeline._compress_tool_result(tool_name="weather", raw="Sunny, 72°F")
    mock_compress.assert_not_called()
