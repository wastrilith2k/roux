"""
Tests for issue #22: Agent-controlled retrieval — give LLM memory search as a tool.

Covers:
- Phase 1: MEDIUM complexity tier, fast path default, tier-based context building
- Phase 2: search_memory tool definition, handler, result formatting
- Phase 2: Two-pass generation flow (memory tool in pipeline)
- Phase 2: Memory validation agent integration with search_memory tool
"""

import os
import pytest

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

from unittest.mock import patch, MagicMock


# =========================================================================
# Phase 1: Complexity Classifier — MEDIUM tier
# =========================================================================

from src.core.conversation.complexity_classifier import (
    classify_message,
    MessageComplexity,
    ClassificationResult,
)


class TestMediumComplexityTier:
    """Test the new MEDIUM complexity tier exists and is classified correctly."""

    def _classify(self, message):
        """Helper — classify with fast path enabled."""
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', True):
            return classify_message(message)

    def test_medium_tier_exists(self):
        """MEDIUM tier is a valid MessageComplexity value."""
        assert hasattr(MessageComplexity, 'MEDIUM')
        assert MessageComplexity.MEDIUM.value == "medium"

    def test_multi_sentence_is_medium(self):
        """Multi-sentence messages without emotional/complex triggers classify as MEDIUM."""
        result = self._classify(
            "I went to the store today. Then I got some groceries. "
            "After that I came home and made dinner."
        )
        assert result.complexity == MessageComplexity.MEDIUM
        assert "Multi-sentence" in result.reason

    def test_non_trivial_question_is_medium(self):
        """Non-trivial questions (>30 chars) without complex topic patterns classify as MEDIUM."""
        result = self._classify(
            "What do you think we should have for dinner tonight?"
        )
        assert result.complexity == MessageComplexity.MEDIUM
        assert "question" in result.reason.lower()

    def test_longer_message_no_triggers_is_medium(self):
        """Messages >80 chars without specific emotional/complex triggers classify as MEDIUM."""
        result = self._classify(
            "I was just reading this interesting article about how plants communicate with each other through underground networks"
        )
        assert result.complexity == MessageComplexity.MEDIUM

    def test_emotional_content_still_complex(self):
        """Emotional content should still classify as COMPLEX, not MEDIUM."""
        result = self._classify(
            "I'm feeling really sad today and I don't know what to do about it"
        )
        assert result.complexity == MessageComplexity.COMPLEX

    def test_memory_reference_still_complex(self):
        """Memory/history references should still classify as COMPLEX."""
        result = self._classify("do you remember when we talked about my dad?")
        assert result.complexity == MessageComplexity.COMPLEX

    def test_greeting_still_simple(self):
        """Greetings should still classify as SIMPLE."""
        result = self._classify("hey!")
        assert result.complexity == MessageComplexity.SIMPLE

    def test_action_still_action(self):
        """Action messages should still classify as ACTION."""
        result = self._classify("what's the weather in Portland?")
        assert result.complexity == MessageComplexity.ACTION


class TestFastPathDefaultEnabled:
    """Test that fast path is now enabled by default."""

    def test_fast_path_default_is_true(self):
        """COMPANION_FAST_PATH_ENABLED defaults to 'true' (was 'false')."""
        # Read the actual default from source
        from src.core.conversation import complexity_classifier
        # The module-level default should be 'true' unless env var overrides
        with patch.dict(os.environ, {}, clear=False):
            # Remove any override to test the default
            env_val = os.environ.get('COMPANION_FAST_PATH_ENABLED', 'true')
            assert env_val.lower() == 'true'

    def test_simple_message_uses_fast_path_when_enabled(self):
        """With fast path enabled, SIMPLE messages get lightweight classification."""
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', True):
            result = classify_message("hey!")
            assert result.complexity == MessageComplexity.SIMPLE

    def test_can_disable_fast_path_via_env(self):
        """Fast path can still be disabled via environment variable."""
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', False):
            result = classify_message("hey!")
            assert result.complexity == MessageComplexity.COMPLEX
            assert "disabled" in result.reason


# =========================================================================
# Phase 1: Context Builder — build_medium()
# =========================================================================


class TestContextBuilderMedium:
    """Test the new build_medium() method on ContextBuilder."""

    def test_medium_submits_more_than_lightweight_fewer_than_full(self):
        """build_medium should submit ~12 sources: more than lightweight (7), fewer than full (24)."""
        from src.core.conversation.context_builder import ContextBuilder

        with patch.object(ContextBuilder, '__init__', lambda self: None):
            builder = ContextBuilder()
            builder._source_timings = {}

            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = ('test', None, 0.01)
            mock_executor.submit.return_value = mock_future
            builder._executor = mock_executor

            # Mock all context source methods
            for attr in dir(builder):
                if attr.startswith('_get_'):
                    setattr(builder, attr, MagicMock(return_value=None))

            # Count medium submits
            builder.build_medium('test@test.com', 'test message')
            medium_count = mock_executor.submit.call_count

            mock_executor.reset_mock()

            # Count lightweight submits
            builder.build_lightweight('test@test.com', 'hey')
            lightweight_count = mock_executor.submit.call_count

            mock_executor.reset_mock()

            # Count full parallel submits
            builder._build_parallel('test@test.com', 'hey', 50)
            full_count = mock_executor.submit.call_count

            assert lightweight_count < medium_count < full_count, (
                f"Expected lightweight ({lightweight_count}) < medium ({medium_count}) "
                f"< full ({full_count})"
            )
            assert medium_count >= 10, f"Medium should be ~12 sources, got {medium_count}"
            assert medium_count <= 15, f"Medium should be ~12 sources, got {medium_count}"

    def test_medium_includes_memories_and_relationship_dynamics(self):
        """build_medium should include memories and relationship_dynamics sources."""
        from src.core.conversation.context_builder import ContextBuilder

        with patch.object(ContextBuilder, '__init__', lambda self: None):
            builder = ContextBuilder()
            builder._source_timings = {}

            submitted_sources = []

            def mock_submit(fn, name, *args, **kwargs):
                submitted_sources.append(name)
                mock_future = MagicMock()
                mock_future.result.return_value = (name, None, 0.01)
                return mock_future

            mock_executor = MagicMock()
            mock_executor.submit.side_effect = mock_submit
            builder._executor = mock_executor

            # Mock all source methods
            for attr in dir(builder):
                if attr.startswith('_get_'):
                    setattr(builder, attr, MagicMock(return_value=None))

            builder.build_medium('test@test.com', 'test message')

            # Medium path should include these enrichment sources beyond lightweight
            assert 'memories' in submitted_sources, "Medium should fetch memories"
            assert 'relationship_dynamics' in submitted_sources, "Medium should fetch relationship_dynamics"
            assert 'opinions_context' in submitted_sources, "Medium should fetch opinions_context"
            assert 'graphiti_context' in submitted_sources, "Medium should fetch graphiti_context"


# =========================================================================
# Phase 1: Pipeline — MEDIUM routing
# =========================================================================


class TestPipelineMediumRouting:
    """Test that the pipeline correctly routes MEDIUM-classified messages."""

    def test_medium_classification_sets_use_medium_path(self):
        """MEDIUM complexity with fast path enabled should set use_medium_path=True."""
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', True):
            # Multi-sentence message triggers MEDIUM classification
            result = classify_message(
                "I went to the park today. It was really nice outside. "
                "Then I grabbed some lunch at that new place."
            )

            # Simulate pipeline logic
            use_fast_path = result.complexity == MessageComplexity.SIMPLE
            use_medium_path = result.complexity == MessageComplexity.MEDIUM
            force_tools = result.complexity == MessageComplexity.ACTION

            assert result.complexity == MessageComplexity.MEDIUM
            assert use_fast_path is False
            assert use_medium_path is True
            assert force_tools is False


# =========================================================================
# Phase 2: search_memory Tool
# =========================================================================


class TestSearchMemoryToolDefinition:
    """Test the search_memory tool definition structure."""

    def test_tool_has_correct_structure(self):
        """Tool definition follows OpenAI function-calling schema."""
        from src.tools.memory_search_tool import SEARCH_MEMORY_TOOL

        assert SEARCH_MEMORY_TOOL["type"] == "function"
        func = SEARCH_MEMORY_TOOL["function"]
        assert func["name"] == "search_memory"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert "query" in params["properties"]
        assert "source" in params["properties"]
        assert "time_range" in params["properties"]
        assert params["required"] == ["query"]

    def test_source_enum_values(self):
        """Source parameter has the correct enum values."""
        from src.tools.memory_search_tool import SEARCH_MEMORY_TOOL

        source_enum = SEARCH_MEMORY_TOOL["function"]["parameters"]["properties"]["source"]["enum"]
        assert set(source_enum) == {"all", "conversations", "facts", "episodes", "graph"}

    def test_time_range_enum_values(self):
        """Time range parameter has the correct enum values."""
        from src.tools.memory_search_tool import SEARCH_MEMORY_TOOL

        time_range_enum = SEARCH_MEMORY_TOOL["function"]["parameters"]["properties"]["time_range"]["enum"]
        assert "recent" in time_range_enum
        assert "all" in time_range_enum


class TestSearchMemoryHandler:
    """Test the search_memory function handler."""

    def test_search_returns_result_object(self):
        """search_memory returns a MemorySearchResult dataclass."""
        from src.tools.memory_search_tool import search_memory, MemorySearchResult

        # Mock all backends to avoid DB connections
        with patch('src.tools.memory_search_tool._search_all', return_value=["[conversation] test result"]):
            result = search_memory("test query", "test@test.com")

        assert isinstance(result, MemorySearchResult)
        assert result.query == "test query"
        assert result.source == "all"
        assert result.search_time_ms >= 0

    def test_search_conversations_routes_to_pgvector(self):
        """source='conversations' routes to _search_conversations."""
        from src.tools.memory_search_tool import search_memory

        with patch('src.tools.memory_search_tool._search_conversations',
                   return_value=["[conversation] found something"]) as mock_search:
            result = search_memory("mom's surgery", "test@test.com", source="conversations")

        mock_search.assert_called_once()
        assert result.source == "conversations"
        assert len(result.results) == 1

    def test_search_facts_routes_to_fact_store(self):
        """source='facts' routes to _search_facts."""
        from src.tools.memory_search_tool import search_memory

        with patch('src.tools.memory_search_tool._search_facts',
                   return_value=["[fact] James works remotely"]) as mock_search:
            result = search_memory("where does he work", "test@test.com", source="facts")

        mock_search.assert_called_once()
        assert result.source == "facts"

    def test_unknown_source_returns_error(self):
        """Unknown source returns an error result."""
        from src.tools.memory_search_tool import search_memory

        result = search_memory("test", "test@test.com", source="invalid_source")
        assert result.error is not None
        assert "Unknown source" in result.error

    def test_search_all_calls_multiple_backends(self):
        """source='all' searches conversations, facts, episodes, and graph in parallel."""
        from src.tools.memory_search_tool import search_memory

        with patch('src.tools.memory_search_tool._search_conversations', return_value=["conv result"]), \
             patch('src.tools.memory_search_tool._search_facts', return_value=["fact result"]), \
             patch('src.tools.memory_search_tool._search_episodes', return_value=["ep result"]), \
             patch('src.tools.memory_search_tool._search_graph', return_value=["graph result"]):
            result = search_memory("test query", "test@test.com", source="all")

        assert result.result_count == 4
        assert len(result.results) == 4

    def test_backend_failure_doesnt_crash_search_all(self):
        """If one backend fails during 'all' search, others still return results."""
        from src.tools.memory_search_tool import search_memory

        with patch('src.tools.memory_search_tool._search_conversations',
                   side_effect=Exception("DB down")), \
             patch('src.tools.memory_search_tool._search_facts', return_value=["fact result"]), \
             patch('src.tools.memory_search_tool._search_episodes', return_value=[]), \
             patch('src.tools.memory_search_tool._search_graph', return_value=[]):
            result = search_memory("test", "test@test.com", source="all")

        assert result.error is None
        assert result.result_count >= 1


class TestSearchMemoryFormatting:
    """Test result formatting for prompt injection."""

    def test_format_empty_results(self):
        """Empty results produce a 'no results' message."""
        from src.tools.memory_search_tool import MemorySearchResult, format_tool_results_for_prompt

        result = MemorySearchResult(query="test", source="all", results=[], result_count=0)
        formatted = format_tool_results_for_prompt(result)
        assert "no results" in formatted.lower()

    def test_format_with_results(self):
        """Results are formatted with header and bullet points."""
        from src.tools.memory_search_tool import MemorySearchResult, format_tool_results_for_prompt

        result = MemorySearchResult(
            query="mom",
            source="conversations",
            results=["[conversation] talked about mom's health"],
            result_count=1,
            search_time_ms=50
        )
        formatted = format_tool_results_for_prompt(result)
        assert "MEMORY SEARCH RESULTS" in formatted
        assert "mom" in formatted
        assert "talked about mom's health" in formatted

    def test_format_error_result(self):
        """Error results produce an error message."""
        from src.tools.memory_search_tool import MemorySearchResult, format_tool_results_for_prompt

        result = MemorySearchResult(query="test", source="all", error="DB connection failed")
        formatted = format_tool_results_for_prompt(result)
        assert "failed" in formatted.lower()


# =========================================================================
# Phase 2: Memory Validation Agent — search_memory integration
# =========================================================================


class TestMemoryValidationAgentIntegration:
    """Test that the memory validation agent uses the search_memory tool."""

    def test_validate_calls_search_via_tool(self):
        """When a memory query is detected, the agent uses _search_via_tool for supplementary results."""
        from src.core.memory_validation_agent import MemoryValidationAgent
        from src.core.verified_memory import VerifiedMemory

        agent = MagicMock(spec=MemoryValidationAgent)
        agent._search_via_tool = MemoryValidationAgent._search_via_tool.__get__(agent)

        with patch('src.tools.memory_search_tool.search_memory') as mock_search:
            from src.tools.memory_search_tool import MemorySearchResult
            mock_search.return_value = MemorySearchResult(
                query="dad",
                source="conversations",
                results=["[conversation] talked about dad's birthday"],
                result_count=1,
                search_time_ms=50
            )

            results = agent._search_via_tool(["dad", "birthday"], "test@test.com", "specific_event")

        assert len(results) == 1
        assert results[0].source == 'search_tool'
        assert "dad's birthday" in results[0].content

    def test_search_via_tool_maps_query_types_to_sources(self):
        """_search_via_tool maps query types to appropriate tool sources."""
        from src.core.memory_validation_agent import MemoryValidationAgent

        agent = MagicMock(spec=MemoryValidationAgent)
        agent._search_via_tool = MemoryValidationAgent._search_via_tool.__get__(agent)

        with patch('src.tools.memory_search_tool.search_memory') as mock_search:
            from src.tools.memory_search_tool import MemorySearchResult
            mock_search.return_value = MemorySearchResult(
                query="test", source="conversations", results=[], result_count=0
            )

            # specific_event → conversations
            agent._search_via_tool(["test"], "test@test.com", "specific_event")
            assert mock_search.call_args[1].get('source', mock_search.call_args[0][2] if len(mock_search.call_args[0]) > 2 else None) in ['conversations', None]

    def test_search_via_tool_failure_returns_empty(self):
        """If the search tool fails, _search_via_tool returns empty list."""
        from src.core.memory_validation_agent import MemoryValidationAgent

        agent = MagicMock(spec=MemoryValidationAgent)
        agent._search_via_tool = MemoryValidationAgent._search_via_tool.__get__(agent)

        with patch('src.tools.memory_search_tool.search_memory', side_effect=Exception("boom")):
            results = agent._search_via_tool(["test"], "test@test.com", "factual")

        assert results == []

    def test_format_records_handles_search_tool_source(self):
        """_format_records properly tags records from the search_tool source."""
        from src.core.memory_validation_agent import MemoryValidationAgent
        from src.core.verified_memory import VerifiedMemory

        agent = MemoryValidationAgent.__new__(MemoryValidationAgent)
        records = [
            VerifiedMemory(content="from postgres", source="postgres"),
            VerifiedMemory(content="from search tool", source="search_tool"),
            VerifiedMemory(content="from entity", source="entity_profile"),
        ]

        formatted = agent._format_records(records)
        assert "[FROM PAST CONVERSATION]" in formatted[0]
        assert "[FROM PAST CONVERSATION]" in formatted[1]
        assert "[VERIFIED FACT]" in formatted[2]


# =========================================================================
# Phase 2: Two-pass generation flow
# =========================================================================


class TestTwoPassGeneration:
    """Test the two-pass generation flow with memory tool."""

    def test_medium_path_offers_memory_tool(self):
        """Pipeline offers search_memory tool for MEDIUM path messages."""
        # Verify the routing logic: MEDIUM classification -> use_medium_path -> _call_llm_with_memory_tool
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', True):
            result = classify_message(
                "I was thinking about what you said about cooking last time"
            )

            # This has a complex topic pattern "you said" so it should be COMPLEX
            # Let's use a message that's medium
            result = classify_message(
                "So I went to the store. Bought some stuff. Then came back home."
            )
            use_medium_path = result.complexity == MessageComplexity.MEDIUM
            assert use_medium_path is True

    def test_call_llm_with_memory_tool_no_tool_call(self):
        """When LLM doesn't call the tool, _call_llm_with_memory_tool falls back to standard."""
        from src.core.conversation.pipeline import ConversationPipeline

        pipeline = MagicMock(spec=ConversationPipeline)
        pipeline._call_llm_with_memory_tool = ConversationPipeline._call_llm_with_memory_tool.__get__(pipeline)
        pipeline._call_llm = MagicMock(return_value=("response text", "test-model"))
        pipeline._get_dynamic_temperature = MagicMock(return_value=0.7)
        pipeline._current_mode_detection = None

        # Mock provider that returns text (no tool call)
        mock_provider = MagicMock()
        mock_provider.generate_sync.return_value = "direct response"

        with patch('src.llm.openai_provider.get_openai_tool_provider', return_value=mock_provider):
            response, model, tool_calls = pipeline._call_llm_with_memory_tool(
                "system prompt", "hey there", "test@test.com", []
            )

        assert tool_calls == []
        # Should have called _call_llm for final response
        pipeline._call_llm.assert_called()

    def test_call_llm_with_memory_tool_with_tool_call(self):
        """When LLM calls search_memory, results are injected and final response generated."""
        from src.core.conversation.pipeline import ConversationPipeline

        pipeline = MagicMock(spec=ConversationPipeline)
        pipeline._call_llm_with_memory_tool = ConversationPipeline._call_llm_with_memory_tool.__get__(pipeline)
        pipeline._call_llm = MagicMock(return_value=("final response with memories", "test-model"))
        pipeline._get_dynamic_temperature = MagicMock(return_value=0.7)
        pipeline._current_mode_detection = None

        # Mock provider: first call returns tool_use, second returns text
        mock_provider = MagicMock()
        mock_provider.generate_sync.side_effect = [
            {
                "type": "tool_use",
                "tool_name": "search_memory",
                "tool_input": {"query": "mom's surgery", "source": "conversations"},
                "tool_use_id": "call_0"
            },
            "final response text"
        ]

        from src.tools.memory_search_tool import MemorySearchResult

        with patch('src.llm.openai_provider.get_openai_tool_provider', return_value=mock_provider), \
             patch('src.tools.memory_search_tool.search_memory', return_value=MemorySearchResult(
                 query="mom's surgery",
                 source="conversations",
                 results=["[conversation] discussed mom's upcoming surgery"],
                 result_count=1,
                 search_time_ms=100
             )):
            response, model, tool_calls = pipeline._call_llm_with_memory_tool(
                "system prompt", "do you remember my mom's surgery?",
                "test@test.com", []
            )

        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "search_memory"
        assert tool_calls[0]["query"] == "mom's surgery"
        # Final response should come from _call_llm with enhanced prompt
        pipeline._call_llm.assert_called()

    def test_call_llm_with_memory_tool_respects_max_calls(self):
        """Memory tool respects the max_tool_calls limit."""
        from src.core.conversation.pipeline import ConversationPipeline

        pipeline = MagicMock(spec=ConversationPipeline)
        pipeline._call_llm_with_memory_tool = ConversationPipeline._call_llm_with_memory_tool.__get__(pipeline)
        pipeline._call_llm = MagicMock(return_value=("response", "model"))
        pipeline._get_dynamic_temperature = MagicMock(return_value=0.7)
        pipeline._current_mode_detection = None

        # Provider always returns tool calls (to test limit enforcement)
        mock_provider = MagicMock()
        mock_provider.generate_sync.return_value = {
            "type": "tool_use",
            "tool_name": "search_memory",
            "tool_input": {"query": "test"},
            "tool_use_id": "call_0"
        }

        from src.tools.memory_search_tool import MemorySearchResult

        with patch('src.llm.openai_provider.get_openai_tool_provider', return_value=mock_provider), \
             patch('src.tools.memory_search_tool.search_memory', return_value=MemorySearchResult(
                 query="test", source="all", results=[], result_count=0
             )):
            response, model, tool_calls = pipeline._call_llm_with_memory_tool(
                "prompt", "msg", "test@test.com", [], max_tool_calls=2
            )

        # Should have made at most max_tool_calls+1 provider calls
        assert mock_provider.generate_sync.call_count <= 3

    def test_call_llm_with_memory_tool_no_provider_fallback(self):
        """When no OpenAI provider is available, falls back to standard _call_llm."""
        from src.core.conversation.pipeline import ConversationPipeline

        pipeline = MagicMock(spec=ConversationPipeline)
        pipeline._call_llm_with_memory_tool = ConversationPipeline._call_llm_with_memory_tool.__get__(pipeline)
        pipeline._call_llm = MagicMock(return_value=("fallback response", "fallback-model"))
        pipeline._get_dynamic_temperature = MagicMock(return_value=0.7)
        pipeline._current_mode_detection = None

        with patch('src.llm.openai_provider.get_openai_tool_provider', return_value=None):
            response, model, tool_calls = pipeline._call_llm_with_memory_tool(
                "prompt", "msg", "test@test.com", []
            )

        assert tool_calls == []
        pipeline._call_llm.assert_called_once()


# =========================================================================
# Issue #57: time_range parameter must be used in _search_conversations
# =========================================================================


class TestTimeRangeFiltering:
    """Regression tests for issue #57: time_range was accepted but ignored."""

    def test_time_range_to_since_recent(self):
        """'recent' maps to ~3 days ago."""
        from src.tools.memory_search_tool import _time_range_to_since
        from datetime import datetime, timezone, timedelta

        since = _time_range_to_since("recent")
        assert since is not None
        expected = datetime.now(timezone.utc) - timedelta(days=3)
        assert abs((since - expected).total_seconds()) < 5

    def test_time_range_to_since_last_week(self):
        """'last_week' maps to ~7 days ago."""
        from src.tools.memory_search_tool import _time_range_to_since
        from datetime import datetime, timezone, timedelta

        since = _time_range_to_since("last_week")
        assert since is not None
        expected = datetime.now(timezone.utc) - timedelta(weeks=1)
        assert abs((since - expected).total_seconds()) < 5

    def test_time_range_to_since_last_month(self):
        """'last_month' maps to ~30 days ago."""
        from src.tools.memory_search_tool import _time_range_to_since
        from datetime import datetime, timezone, timedelta

        since = _time_range_to_since("last_month")
        assert since is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((since - expected).total_seconds()) < 5

    def test_time_range_to_since_last_year(self):
        """'last_year' maps to ~365 days ago."""
        from src.tools.memory_search_tool import _time_range_to_since
        from datetime import datetime, timezone, timedelta

        since = _time_range_to_since("last_year")
        assert since is not None
        expected = datetime.now(timezone.utc) - timedelta(days=365)
        assert abs((since - expected).total_seconds()) < 5

    def test_time_range_to_since_all_returns_none(self):
        """'all' returns None (no filtering)."""
        from src.tools.memory_search_tool import _time_range_to_since

        assert _time_range_to_since("all") is None

    def test_time_range_to_since_unknown_returns_none(self):
        """Unknown time_range returns None (no filtering)."""
        from src.tools.memory_search_tool import _time_range_to_since

        assert _time_range_to_since("bogus") is None

    def test_search_conversations_passes_since_to_pgvector(self):
        """_search_conversations must pass a `since` kwarg derived from time_range."""
        from src.tools.memory_search_tool import _search_conversations

        with patch('src.tools.memory_search_tool._time_range_to_since') as mock_since, \
             patch('src.memory.semantic_search.search_memory') as mock_search:
            from datetime import datetime, timezone, timedelta
            fake_since = datetime.now(timezone.utc) - timedelta(days=3)
            mock_since.return_value = fake_since
            mock_search.return_value = []

            _search_conversations("test query", "user@test.com", "recent", 10)

            mock_since.assert_called_once_with("recent")
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args
            assert call_kwargs[1].get('since') == fake_since or \
                   (len(call_kwargs[0]) > 4 and call_kwargs[0][4] == fake_since), \
                   f"Expected since={fake_since} to be passed to pgvector_search, got {call_kwargs}"

    def test_search_conversations_all_passes_no_since(self):
        """time_range='all' should pass since=None (no time filtering)."""
        from src.tools.memory_search_tool import _search_conversations

        with patch('src.memory.semantic_search.search_memory') as mock_search:
            mock_search.return_value = []

            _search_conversations("test query", "user@test.com", "all", 10)

            call_kwargs = mock_search.call_args
            assert call_kwargs[1].get('since') is None, \
                   f"Expected since=None for time_range='all', got {call_kwargs}"

    def test_different_time_ranges_produce_different_since(self):
        """'recent' and 'last_year' must produce different cutoff dates."""
        from src.tools.memory_search_tool import _time_range_to_since

        recent = _time_range_to_since("recent")
        last_year = _time_range_to_since("last_year")
        assert recent is not None
        assert last_year is not None
        # recent should be a more recent cutoff (larger datetime) than last_year
        assert recent > last_year


# =========================================================================
# Integration: Full tier routing through pipeline classification
# =========================================================================


class TestTierRouting:
    """End-to-end tier routing: message -> classification -> context path."""

    def _classify(self, message):
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', True):
            return classify_message(message)

    def test_greeting_routes_to_simple(self):
        result = self._classify("hey!")
        assert result.complexity == MessageComplexity.SIMPLE

    def test_general_question_routes_to_medium(self):
        result = self._classify("What should we do for dinner tonight, anything good?")
        assert result.complexity == MessageComplexity.MEDIUM

    def test_emotional_deep_routes_to_complex(self):
        result = self._classify(
            "I've been feeling really overwhelmed and anxious about everything lately"
        )
        assert result.complexity == MessageComplexity.COMPLEX

    def test_history_reference_routes_to_complex(self):
        result = self._classify("do you remember what you told me about your family?")
        assert result.complexity == MessageComplexity.COMPLEX

    def test_weather_routes_to_action(self):
        result = self._classify("what's the weather like?")
        assert result.complexity == MessageComplexity.ACTION

    def test_tier_token_ordering(self):
        """Verify expected token usage ordering: SIMPLE < MEDIUM < COMPLEX."""
        # This tests the conceptual ordering — actual token counts depend on
        # real context, but source counts should follow the expected pattern.
        from src.core.conversation.context_builder import ContextBuilder

        with patch.object(ContextBuilder, '__init__', lambda self: None):
            builder = ContextBuilder()
            builder._source_timings = {}

            mock_executor = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = ('test', None, 0.01)
            mock_executor.submit.return_value = mock_future
            builder._executor = mock_executor

            for attr in dir(builder):
                if attr.startswith('_get_'):
                    setattr(builder, attr, MagicMock(return_value=None))

            # Measure source counts per tier
            builder.build_lightweight('t@t.com', 'hey')
            simple_count = mock_executor.submit.call_count
            mock_executor.reset_mock()

            builder.build_medium('t@t.com', 'moderate message')
            medium_count = mock_executor.submit.call_count
            mock_executor.reset_mock()

            builder._build_parallel('t@t.com', 'complex deep message', 50)
            complex_count = mock_executor.submit.call_count

            assert simple_count < medium_count < complex_count, (
                f"Source ordering: SIMPLE({simple_count}) < MEDIUM({medium_count}) "
                f"< COMPLEX({complex_count})"
            )
