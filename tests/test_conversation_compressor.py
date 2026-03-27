"""Regression tests for issue #23: conversation compression.

The bug: Messages beyond the 25-message history limit simply vanish — they're
not summarized or compressed into any persistent memory object. In long
conversations (50+ messages), the companion forgets the first half entirely,
creating an "amnesia cliff."

The fix: When conversation history exceeds COMPRESSION_THRESHOLD, older
messages are compressed into a structured session summary via LLM. The
summary is cached in Redis and included in the prompt as session_summary.
Recent messages remain as raw turns.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePersonaConfig:
    companion_short_name: str = "Kai"
    companion_name: str = "Kai Tanaka"
    primary_user_name: str = "James"
    c_possessive: str = "her"
    c_pronoun_subject: str = "she"
    c_pronoun_object: str = "her"
    u_possessive: str = "his"
    u_pronoun_subject: str = "he"
    u_pronoun_object: str = "him"
    primary_user_email: str = "test@test.com"


def _fake_persona():
    return FakePersonaConfig()


def _make_messages(count, start_id=1):
    """Generate a list of fake messages in chronological order (oldest first)."""
    messages = []
    for i in range(count):
        msg_id = start_id + i
        sender = "James" if i % 2 == 0 else "Kai"
        messages.append({
            'id': msg_id,
            'sender_name': sender,
            'message_text': f"Message #{msg_id} from {sender}",
        })
    return messages


# ---------------------------------------------------------------------------
# Tests for get_or_create_session_summary
# ---------------------------------------------------------------------------

class TestGetOrCreateSessionSummary:
    """Test the core split+compress logic."""

    def test_no_compression_below_threshold(self):
        """Messages below COMPRESSION_THRESHOLD should not be compressed."""
        from src.memory.conversation_compressor import get_or_create_session_summary

        messages = _make_messages(20)

        with patch('src.memory.conversation_compressor.COMPRESSION_THRESHOLD', 25):
            summary, recent = get_or_create_session_summary("test@test.com", messages)

        assert summary == ""
        assert recent == messages  # All messages returned as-is

    def test_compression_at_threshold(self):
        """Messages at exactly COMPRESSION_THRESHOLD should not trigger compression."""
        from src.memory.conversation_compressor import get_or_create_session_summary

        messages = _make_messages(25)

        with patch('src.memory.conversation_compressor.COMPRESSION_THRESHOLD', 25):
            summary, recent = get_or_create_session_summary("test@test.com", messages)

        assert summary == ""
        assert len(recent) == 25

    def test_compression_above_threshold(self):
        """Messages above threshold should split into summary + recent."""
        from src.memory.conversation_compressor import get_or_create_session_summary

        messages = _make_messages(40)
        fake_summary = "[Session summary — 25 earlier messages]\n- Topics discussed: tests"

        with patch('src.memory.conversation_compressor.COMPRESSION_THRESHOLD', 25), \
             patch('src.memory.conversation_compressor.RECENT_MESSAGES_KEEP', 15), \
             patch('src.memory.conversation_compressor._get_redis') as mock_redis, \
             patch('src.memory.conversation_compressor.compress_messages', return_value=fake_summary) as mock_compress:
            mock_r = MagicMock()
            mock_r.get.return_value = None  # Cache miss
            mock_redis.return_value = mock_r

            summary, recent = get_or_create_session_summary("test@test.com", messages)

        # Recent should be the last 15 messages
        assert len(recent) == 15
        assert recent[0]['id'] == 26  # message IDs 26-40

        # Summary should come from compress_messages
        assert summary == fake_summary

        # Older messages (1-25) should have been passed to compress_messages
        mock_compress.assert_called_once()
        compressed_msgs = mock_compress.call_args[0][0]
        assert len(compressed_msgs) == 25
        assert compressed_msgs[0]['id'] == 1
        assert compressed_msgs[-1]['id'] == 25

    def test_cached_summary_returned_on_hit(self):
        """Redis cache hit should skip LLM compression."""
        from src.memory.conversation_compressor import get_or_create_session_summary

        messages = _make_messages(40)
        cached_summary = "[Session summary — cached]"

        with patch('src.memory.conversation_compressor.COMPRESSION_THRESHOLD', 25), \
             patch('src.memory.conversation_compressor.RECENT_MESSAGES_KEEP', 15), \
             patch('src.memory.conversation_compressor._get_redis') as mock_redis, \
             patch('src.memory.conversation_compressor.compress_messages') as mock_compress:
            mock_r = MagicMock()
            mock_r.get.return_value = cached_summary  # Cache hit
            mock_redis.return_value = mock_r

            summary, recent = get_or_create_session_summary("test@test.com", messages)

        assert summary == cached_summary
        assert len(recent) == 15
        mock_compress.assert_not_called()  # Should NOT call LLM

    def test_empty_messages_returns_empty(self):
        """Empty message list should return empty summary."""
        from src.memory.conversation_compressor import get_or_create_session_summary

        summary, recent = get_or_create_session_summary("test@test.com", [])
        assert summary == ""
        assert recent == []

    def test_redis_failure_falls_through_to_compression(self):
        """Redis failure should not prevent compression — just skip caching."""
        from src.memory.conversation_compressor import get_or_create_session_summary

        messages = _make_messages(40)
        fake_summary = "[Session summary — 25 earlier messages]"

        with patch('src.memory.conversation_compressor.COMPRESSION_THRESHOLD', 25), \
             patch('src.memory.conversation_compressor.RECENT_MESSAGES_KEEP', 15), \
             patch('src.memory.conversation_compressor._get_redis') as mock_redis, \
             patch('src.memory.conversation_compressor.compress_messages', return_value=fake_summary):
            mock_r = MagicMock()
            mock_r.get.side_effect = Exception("Redis down")
            mock_r.set.side_effect = Exception("Redis down")
            mock_redis.return_value = mock_r

            summary, recent = get_or_create_session_summary("test@test.com", messages)

        # Should still produce a summary despite Redis failure
        assert summary == fake_summary
        assert len(recent) == 15


# ---------------------------------------------------------------------------
# Tests for compress_messages
# ---------------------------------------------------------------------------

class TestCompressMessages:
    """Test the LLM compression call."""

    def test_empty_messages_returns_existing(self):
        """compress_messages with empty list returns existing summary."""
        from src.memory.conversation_compressor import compress_messages

        result = compress_messages([], existing_summary="previous summary")
        assert result == "previous summary"

    def test_llm_called_with_formatted_messages(self):
        """compress_messages should format messages and call the LLM."""
        from src.memory.conversation_compressor import compress_messages

        messages = _make_messages(5)
        expected_summary = "[Session summary — 5 earlier messages]\n- Topics: test"

        with patch('src.config.persona_config.get_persona_config', _fake_persona), \
             patch('src.llm.provider_factory.generate_sync', return_value=expected_summary), \
             patch('src.llm.provider_factory.get_resilient_provider_chain'):

            result = compress_messages(messages)

        assert result == expected_summary

    def test_llm_failure_returns_existing_summary(self):
        """LLM failure should return existing summary gracefully."""
        from src.memory.conversation_compressor import compress_messages

        messages = _make_messages(5)

        with patch('src.config.persona_config.get_persona_config', _fake_persona), \
             patch('src.llm.provider_factory.generate_sync', side_effect=Exception("LLM down")), \
             patch('src.llm.provider_factory.get_resilient_provider_chain'):

            result = compress_messages(messages, existing_summary="old summary")

        assert result == "old summary"

    def test_rolling_compression_includes_existing_summary(self):
        """When existing_summary is provided, it should be included in the prompt."""
        from src.memory.conversation_compressor import compress_messages

        messages = _make_messages(5)
        existing = "[Session summary — previous batch]"

        with patch('src.config.persona_config.get_persona_config', _fake_persona), \
             patch('src.llm.provider_factory.generate_sync', return_value="extended summary") as mock_llm, \
             patch('src.llm.provider_factory.get_resilient_provider_chain'):

            result = compress_messages(messages, existing_summary=existing)

        # The LLM should have received the existing summary in its prompt
        call_args = mock_llm.call_args
        user_content = call_args[1]['messages'][1]['content']
        assert "EXISTING SUMMARY" in user_content
        assert existing in user_content


# ---------------------------------------------------------------------------
# Tests for cache key generation
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic_key(self):
        """Same message IDs should produce the same cache key."""
        from src.memory.conversation_compressor import _build_cache_key

        key1 = _build_cache_key("user@test.com", [1, 2, 3])
        key2 = _build_cache_key("user@test.com", [1, 2, 3])
        assert key1 == key2

    def test_different_ids_different_key(self):
        """Different message IDs should produce different cache keys."""
        from src.memory.conversation_compressor import _build_cache_key

        key1 = _build_cache_key("user@test.com", [1, 2, 3])
        key2 = _build_cache_key("user@test.com", [4, 5, 6])
        assert key1 != key2

    def test_order_independent(self):
        """Message ID order should not affect the cache key (sorted internally)."""
        from src.memory.conversation_compressor import _build_cache_key

        key1 = _build_cache_key("user@test.com", [3, 1, 2])
        key2 = _build_cache_key("user@test.com", [1, 2, 3])
        assert key1 == key2

    def test_different_users_different_key(self):
        """Different users should produce different cache keys."""
        from src.memory.conversation_compressor import _build_cache_key

        key1 = _build_cache_key("user1@test.com", [1, 2, 3])
        key2 = _build_cache_key("user2@test.com", [1, 2, 3])
        assert key1 != key2


# ---------------------------------------------------------------------------
# Tests for ConversationContext integration
# ---------------------------------------------------------------------------

class TestConversationContextSessionSummary:
    """Verify session_summary is properly wired into ConversationContext."""

    def test_session_summary_field_exists(self):
        """ConversationContext should have a session_summary field."""
        from src.core.conversation.context_builder import ConversationContext

        ctx = ConversationContext()
        assert hasattr(ctx, 'session_summary')
        assert ctx.session_summary == ""

    def test_session_summary_in_prompt_sections(self):
        """Non-empty session_summary should appear in to_prompt_sections()."""
        from src.core.conversation.context_builder import ConversationContext

        ctx = ConversationContext(
            session_summary="[Session summary — 25 earlier messages]\n- Topics: weather"
        )

        with patch('src.core.conversation.token_budget.apply_source_budgets',
                   side_effect=lambda x: x):
            sections = ctx.to_prompt_sections()

        assert 'session_summary' in sections

    def test_empty_session_summary_excluded_from_sections(self):
        """Empty session_summary should not appear in to_prompt_sections()."""
        from src.core.conversation.context_builder import ConversationContext

        ctx = ConversationContext()

        with patch('src.core.conversation.token_budget.apply_source_budgets',
                   side_effect=lambda x: x):
            sections = ctx.to_prompt_sections()

        assert 'session_summary' not in sections


# ---------------------------------------------------------------------------
# Tests for token budget integration
# ---------------------------------------------------------------------------

class TestTokenBudgetIntegration:
    """Verify session_summary has proper token budget and tier assignment."""

    def test_session_summary_has_budget(self):
        """session_summary should have a token budget defined."""
        from src.core.conversation.token_budget import SOURCE_TOKEN_BUDGETS

        assert 'session_summary' in SOURCE_TOKEN_BUDGETS
        assert SOURCE_TOKEN_BUDGETS['session_summary'] > 0

    def test_session_summary_in_tier_2(self):
        """session_summary should be in TIER_2 (proportional trim, not first to go)."""
        from src.core.conversation.token_budget import TIER_2, get_tier

        assert 'session_summary' in TIER_2
        assert get_tier('session_summary') == 2


# ---------------------------------------------------------------------------
# Tests for pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify session_summary is included in the pipeline's prompt assembly."""

    def test_session_summary_in_section_priority(self):
        """session_summary should have a priority in ConversationPipeline.SECTION_PRIORITY."""
        with patch('src.config.persona_config.get_persona_config', _fake_persona):
            from src.core.conversation.pipeline import ConversationPipeline
            assert 'session_summary' in ConversationPipeline.SECTION_PRIORITY

    def test_session_summary_priority_is_high(self):
        """session_summary should have high priority (2) since it's core context."""
        with patch('src.config.persona_config.get_persona_config', _fake_persona):
            from src.core.conversation.pipeline import ConversationPipeline
            assert ConversationPipeline.SECTION_PRIORITY['session_summary'] == 2


# ---------------------------------------------------------------------------
# Integration test: the amnesia cliff scenario
# ---------------------------------------------------------------------------

class TestAmnesiaCliffRegression:
    """
    Regression test for the core bug: in a 50-message conversation,
    messages beyond the 25-message window were simply lost. Now they
    should be compressed into a session summary.
    """

    def test_long_conversation_produces_summary(self):
        """
        50 messages should produce a session summary + 15 recent turns,
        NOT 25 turns with the first 25 lost.
        """
        from src.memory.conversation_compressor import get_or_create_session_summary

        messages = _make_messages(50)
        fake_summary = "[Session summary — 35 earlier messages]\n- Topics discussed: various"

        with patch('src.memory.conversation_compressor.COMPRESSION_THRESHOLD', 25), \
             patch('src.memory.conversation_compressor.RECENT_MESSAGES_KEEP', 15), \
             patch('src.memory.conversation_compressor._get_redis') as mock_redis, \
             patch('src.memory.conversation_compressor.compress_messages', return_value=fake_summary):
            mock_r = MagicMock()
            mock_r.get.return_value = None
            mock_redis.return_value = mock_r

            summary, recent = get_or_create_session_summary("test@test.com", messages)

        # The old behavior: 25 messages, first 25 lost, no summary
        # The new behavior: 15 recent messages + summary of 35 older messages
        assert summary != ""
        assert len(recent) == 15
        # Verify recent messages are the NEWEST ones
        assert recent[0]['id'] == 36
        assert recent[-1]['id'] == 50

    def test_structured_history_returns_three_tuple(self):
        """
        _get_conversation_history_structured should return a 3-tuple
        including session_summary (previously was 2-tuple).
        """
        from src.core.conversation.context_builder import ContextBuilder

        mock_db = MagicMock()
        mock_db.get_recent_messages.return_value = _make_messages(10)
        mock_db.get_minutes_since_last_message.return_value = 5

        with patch('src.core.conversation.context_builder.ContextBuilder.__init__', lambda self: None):
            builder = ContextBuilder()
            builder.db = mock_db

        with patch.object(builder, '_get_conversation_continuity_context', return_value="active"):
            result = builder._get_conversation_history_structured("test@test.com")

        # Should be a 3-tuple: (turns, continuity, session_summary)
        assert isinstance(result, tuple)
        assert len(result) == 3
        turns, continuity, session_summary = result
        assert isinstance(turns, list)
        assert isinstance(session_summary, str)


# ---------------------------------------------------------------------------
# Regression test for issue #54: dead code stub removed
# ---------------------------------------------------------------------------

class TestNoDeadCodeStub:
    """
    Regression test for issue #54: conversation_compressor.py contained a dead
    try/except block that fetched Redis but did nothing, leaving
    existing_summary as "" unconditionally. The stub has been removed.
    """

    def test_no_dead_try_except_in_get_or_create(self):
        """
        get_or_create_session_summary should NOT contain a try/except block
        whose body only assigns to Redis without using the result.
        Verify by checking that compress_messages is called with only
        older_messages (no existing_summary arg) on a cache miss.
        """
        import inspect
        from src.memory.conversation_compressor import get_or_create_session_summary

        source = inspect.getsource(get_or_create_session_summary)
        # The dead code had 'existing_summary = ""' followed by a try block
        # that did nothing with it. Ensure that pattern is gone.
        assert 'existing_summary = ""' not in source

    def test_compress_called_without_existing_summary_on_cache_miss(self):
        """
        On a cache miss, compress_messages should be called with only
        older_messages — no existing_summary argument.
        """
        from src.memory.conversation_compressor import get_or_create_session_summary

        messages = _make_messages(40)
        fake_summary = "[Session summary]"

        with patch('src.memory.conversation_compressor.COMPRESSION_THRESHOLD', 25), \
             patch('src.memory.conversation_compressor.RECENT_MESSAGES_KEEP', 15), \
             patch('src.memory.conversation_compressor._get_redis') as mock_redis, \
             patch('src.memory.conversation_compressor.compress_messages', return_value=fake_summary) as mock_compress:
            mock_r = MagicMock()
            mock_r.get.return_value = None  # Cache miss
            mock_redis.return_value = mock_r

            get_or_create_session_summary("test@test.com", messages)

        # Should be called with just older_messages, no existing_summary kwarg
        mock_compress.assert_called_once()
        args, kwargs = mock_compress.call_args
        assert len(args) == 1  # Only older_messages
        assert 'existing_summary' not in kwargs


# ---------------------------------------------------------------------------
# Regression test for issue #62: invalid provider string in compressor
# ---------------------------------------------------------------------------

class TestCompressorProviderChain:
    """
    Regression test for issue #62: compress_messages passed
    primary="openai:gpt-4o-mini" to get_resilient_provider_chain(), which
    didn't match any branch — compression silently never ran.
    """

    def test_compress_messages_passes_valid_primary(self):
        """
        compress_messages should pass primary='openai' (not 'openai:gpt-4o-mini')
        to get_resilient_provider_chain, so OpenAI is added as primary provider.
        """
        from src.memory.conversation_compressor import compress_messages

        messages = _make_messages(5)

        with patch('src.config.persona_config.get_persona_config', _fake_persona), \
             patch('src.llm.provider_factory.generate_sync', return_value="[Session summary]") as mock_gen, \
             patch('src.llm.provider_factory.get_resilient_provider_chain') as mock_chain:
            mock_chain.return_value = MagicMock()

            compress_messages(messages)

        # Verify get_resilient_provider_chain was called with primary="openai"
        mock_chain.assert_called_once_with(primary="openai")

    def test_openai_primary_creates_openai_provider_first(self):
        """
        get_resilient_provider_chain(primary='openai') should place OpenAI
        as the first provider in the chain (not just as a fallback).
        """
        from src.llm.provider_factory import get_resilient_provider_chain
        from src.llm.openai_provider import OpenAIProvider

        env = {
            'OPENAI_API_KEY': 'test-key',
            'LLM_PROVIDER': 'fireworks',
        }
        with patch.dict('os.environ', env, clear=True):
            chain = get_resilient_provider_chain(primary="openai")

        # First provider should be OpenAI
        assert isinstance(chain.providers[0], OpenAIProvider)
        assert 'gpt-4o-mini' in chain.providers[0].get_model_name()

    def test_openai_primary_not_duplicated_in_fallback(self):
        """
        When primary='openai', OpenAI should not appear twice in the chain
        (once as primary and again as fallback).
        """
        from src.llm.provider_factory import get_resilient_provider_chain
        from src.llm.openai_provider import OpenAIProvider
        from src.llm.openrouter_provider import OpenRouterProvider

        env = {
            'OPENAI_API_KEY': 'test-key',
            'LLM_PROVIDER': 'fireworks',
        }
        with patch.dict('os.environ', env, clear=True):
            chain = get_resilient_provider_chain(primary="openai")

        openai_providers = [
            p for p in chain.providers
            if isinstance(p, OpenAIProvider) and not isinstance(p, OpenRouterProvider)
        ]
        assert len(openai_providers) == 1, (
            f"Expected exactly 1 OpenAI provider, got {len(openai_providers)}"
        )

    def test_invalid_primary_string_would_produce_empty_chain(self):
        """
        Verify the original bug: passing an unrecognised primary like
        'openai:gpt-4o-mini' with no other API keys configured results
        in ValueError (empty provider list). This is the failure mode
        that caused compression to silently skip.
        """
        import pytest
        from src.llm.provider_factory import get_resilient_provider_chain

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="No LLM providers configured"):
                get_resilient_provider_chain(primary="openai:gpt-4o-mini")
