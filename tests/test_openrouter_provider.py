"""Regression tests for issue #36: OpenRouter as LLM provider with per-request model selection.

Covers:
1. OpenRouterProvider implements LLMProvider interface correctly
2. OpenRouterProvider uses correct defaults and env vars
3. Per-request model override works (model changes for one call, then reverts)
4. Provider factory creates OpenRouterProvider and identifies it correctly
5. get_last_provider_type returns 'openrouter' for OpenRouterProvider instances
6. model_override flows through get_resilient_provider_chain
7. CostTracker.track_openrouter_call writes rows to openrouter_usage table
8. CostTracker OpenRouter daily summary aggregation works
9. Pipeline._track_llm_cost calls track_openrouter_call for openrouter provider
10. Model-specific pricing works correctly
"""

import os
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')


# ---------------------------------------------------------------------------
# OpenRouterProvider unit tests
# ---------------------------------------------------------------------------

class TestOpenRouterProviderInterface:
    """Verify OpenRouterProvider implements LLMProvider correctly."""

    def test_is_llm_provider_subclass(self):
        from src.llm.openrouter_provider import OpenRouterProvider
        from src.llm.provider_interface import LLMProvider
        assert issubclass(OpenRouterProvider, LLMProvider)

    def test_is_openai_provider_subclass(self):
        from src.llm.openrouter_provider import OpenRouterProvider
        from src.llm.openai_provider import OpenAIProvider
        assert issubclass(OpenRouterProvider, OpenAIProvider)

    def test_default_model_from_env(self):
        """OPENROUTER_DEFAULT_MODEL env var takes priority over OPENROUTER_MODEL."""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test-key',
            'OPENROUTER_DEFAULT_MODEL': 'moonshotai/kimi-k2',
        }, clear=False):
            mock_client = MagicMock()
            with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
                from src.llm.openrouter_provider import OpenRouterProvider
                provider = OpenRouterProvider(api_key='test-key')
                assert provider.model == 'moonshotai/kimi-k2'

    def test_default_model_fallback_to_openrouter_model_env(self):
        """Falls back to OPENROUTER_MODEL env var when OPENROUTER_DEFAULT_MODEL not set."""
        env = {
            'OPENROUTER_API_KEY': 'test-key',
            'OPENROUTER_MODEL': 'nousresearch/hermes-3-llama-3.1-405b:free',
        }
        # Remove OPENROUTER_DEFAULT_MODEL if present
        with patch.dict('os.environ', env, clear=False):
            os.environ.pop('OPENROUTER_DEFAULT_MODEL', None)
            mock_client = MagicMock()
            with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
                from src.llm.openrouter_provider import OpenRouterProvider
                provider = OpenRouterProvider(api_key='test-key')
                assert provider.model == 'nousresearch/hermes-3-llama-3.1-405b:free'

    def test_default_model_hardcoded_fallback(self):
        """Falls back to deepseek/deepseek-chat when no env vars set."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}, clear=False):
            os.environ.pop('OPENROUTER_DEFAULT_MODEL', None)
            os.environ.pop('OPENROUTER_MODEL', None)
            mock_client = MagicMock()
            with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
                from src.llm.openrouter_provider import OpenRouterProvider
                provider = OpenRouterProvider(api_key='test-key')
                assert provider.model == 'deepseek/deepseek-chat'

    def test_context_limit_auto_detected(self):
        """Context limit is auto-detected from model name."""
        mock_client = MagicMock()
        with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
            from src.llm.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(
                api_key='test-key', model='anthropic/claude-sonnet-4'
            )
            assert provider.context_limit == 200000

    def test_explicit_model_override(self):
        """Explicit model parameter overrides env vars."""
        mock_client = MagicMock()
        with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
            from src.llm.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(
                api_key='test-key', model='anthropic/claude-sonnet-4'
            )
            assert provider.model == 'anthropic/claude-sonnet-4'
            assert provider.get_model_name() == 'anthropic/claude-sonnet-4'


class TestOpenRouterProviderGeneration:
    """Verify generate_sync works and captures usage."""

    def test_generate_sync_captures_usage(self):
        """generate_sync should populate _last_usage from OpenAI-format response."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10700
        mock_usage.completion_tokens = 55

        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Hello there!"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
            from src.llm.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(api_key='test-key')

        result = provider.generate_sync(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello there!"
        assert provider._last_usage == {
            'input_tokens': 10700,
            'output_tokens': 55,
        }

    def test_per_request_model_override(self):
        """model_override should change model for one call then revert."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Response"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
            from src.llm.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(
                api_key='test-key', model='deepseek/deepseek-chat'
            )

        # Call with override
        provider.generate_sync(
            messages=[{"role": "user", "content": "Hi"}],
            model_override='anthropic/claude-sonnet-4',
        )

        # Verify the override was used in the API call
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs[1]['model'] == 'anthropic/claude-sonnet-4'

        # Verify model reverts after the call
        assert provider.model == 'deepseek/deepseek-chat'
        assert provider.get_model_name() == 'deepseek/deepseek-chat'

    def test_per_request_model_override_reverts_on_error(self):
        """Model must revert even if the API call fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
            from src.llm.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(
                api_key='test-key', model='deepseek/deepseek-chat'
            )

        with pytest.raises(Exception, match="API error"):
            provider.generate_sync(
                messages=[{"role": "user", "content": "Hi"}],
                model_override='anthropic/claude-sonnet-4',
            )

        # Model must revert
        assert provider.model == 'deepseek/deepseek-chat'


# ---------------------------------------------------------------------------
# Provider factory integration
# ---------------------------------------------------------------------------

class TestProviderFactoryOpenRouter:
    """Provider factory must create and identify OpenRouterProvider correctly."""

    def test_get_last_provider_type_openrouter(self):
        """get_last_provider_type should return 'openrouter' for OpenRouterProvider."""
        from src.llm.provider_factory import ResilientProviderChain

        mock_client = MagicMock()
        with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
            from src.llm.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(api_key='key', model='test')

        chain = ResilientProviderChain(providers=[provider])
        chain._last_successful_provider = 0

        assert chain.get_last_provider_type() == 'openrouter'

    def test_openrouter_distinguished_from_openai(self):
        """OpenRouterProvider must be identified as 'openrouter', not 'openai'."""
        from src.llm.provider_factory import ResilientProviderChain

        mock_client = MagicMock()
        with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
            from src.llm.openrouter_provider import OpenRouterProvider
            or_provider = OpenRouterProvider(api_key='key', model='test')

        with patch('src.llm.openai_provider.OpenAI', return_value=mock_client):
            from src.llm.openai_provider import OpenAIProvider
            oai_provider = OpenAIProvider(api_key='key', model='gpt-4o-mini')

        # OpenRouter provider
        chain = ResilientProviderChain(providers=[or_provider])
        chain._last_successful_provider = 0
        assert chain.get_last_provider_type() == 'openrouter'

        # OpenAI provider
        chain = ResilientProviderChain(providers=[oai_provider])
        chain._last_successful_provider = 0
        assert chain.get_last_provider_type() == 'openai'

    def test_chain_uses_openrouter_provider_when_key_set(self):
        """get_resilient_provider_chain creates OpenRouterProvider when API key is set."""
        env = {
            'OPENROUTER_API_KEY': 'test-key',
            'OPENROUTER_DEFAULT_MODEL': 'deepseek/deepseek-chat',
            'LLM_PROVIDER': 'fireworks',
        }
        # Remove other provider keys to isolate
        with patch.dict('os.environ', env, clear=False):
            os.environ.pop('FIREWORKS_API_KEY', None)
            os.environ.pop('DEEPSEEK_API_KEY', None)
            os.environ.pop('OPENAI_API_KEY', None)
            os.environ.pop('ANTHROPIC_API_KEY', None)

            mock_client = MagicMock()
            with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
                from src.llm.provider_factory import get_resilient_provider_chain
                chain = get_resilient_provider_chain()

            from src.llm.openrouter_provider import OpenRouterProvider
            assert any(isinstance(p, OpenRouterProvider) for p in chain.providers)

    def test_model_override_in_chain(self):
        """model_override parameter should set the OpenRouter model in the chain."""
        env = {
            'OPENROUTER_API_KEY': 'test-key',
            'LLM_PROVIDER': 'fireworks',
        }
        with patch.dict('os.environ', env, clear=False):
            os.environ.pop('FIREWORKS_API_KEY', None)
            os.environ.pop('DEEPSEEK_API_KEY', None)
            os.environ.pop('OPENAI_API_KEY', None)
            os.environ.pop('ANTHROPIC_API_KEY', None)
            os.environ.pop('OPENROUTER_DEFAULT_MODEL', None)
            os.environ.pop('OPENROUTER_MODEL', None)

            mock_client = MagicMock()
            with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
                from src.llm.provider_factory import get_resilient_provider_chain
                chain = get_resilient_provider_chain(
                    model_override='anthropic/claude-sonnet-4'
                )

            from src.llm.openrouter_provider import OpenRouterProvider
            or_providers = [
                p for p in chain.providers if isinstance(p, OpenRouterProvider)
            ]
            assert len(or_providers) == 1
            assert or_providers[0].model == 'anthropic/claude-sonnet-4'

    def test_get_llm_provider_openrouter(self):
        """get_llm_provider('openrouter') should return an OpenRouterProvider."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}, clear=False):
            mock_client = MagicMock()
            with patch('src.llm.openrouter_provider.OpenAI', return_value=mock_client):
                from src.llm.provider_factory import get_llm_provider
                provider = get_llm_provider('openrouter', model='deepseek/deepseek-chat')

            from src.llm.openrouter_provider import OpenRouterProvider
            assert isinstance(provider, OpenRouterProvider)
            assert provider.model == 'deepseek/deepseek-chat'


# ---------------------------------------------------------------------------
# CostTracker: OpenRouter tracking
# ---------------------------------------------------------------------------

class TestCostTrackerOpenRouter:
    """CostTracker must track OpenRouter usage in its own table."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CostTracker with an isolated SQLite database."""
        db_path = str(tmp_path / "test_costs.db")
        from src.services.cost_tracker import CostTracker
        return CostTracker(db_path=db_path)

    def test_track_openrouter_call_inserts_row(self, tracker):
        cost = tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=10700,
            completion_tokens=55,
            model="deepseek/deepseek-chat",
        )

        assert cost > 0

        conn = sqlite3.connect(tracker.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM openrouter_usage").fetchone()
        conn.close()

        assert row is not None
        assert row['user_id'] == 'alice@example.com'
        assert row['prompt_tokens'] == 10700
        assert row['completion_tokens'] == 55
        assert row['model'] == 'deepseek/deepseek-chat'
        assert row['cost_usd'] == cost

    def test_openrouter_cost_uses_model_specific_pricing(self, tracker):
        """Different models should have different costs."""
        cost_deepseek = tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            model="deepseek/deepseek-chat",
        )

        cost_claude = tracker.track_openrouter_call(
            user_id="bob@example.com",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            model="anthropic/claude-sonnet-4",
        )

        # DeepSeek: $0.32/1M input, Claude: $3.00/1M input
        assert abs(cost_deepseek - 0.32) < 0.001
        assert abs(cost_claude - 3.00) < 0.001
        assert cost_claude > cost_deepseek

    def test_track_openrouter_updates_daily_summary(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=10700,
            completion_tokens=55,
            model="deepseek/deepseek-chat",
        )

        conn = sqlite3.connect(tracker.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM daily_cost_summary").fetchone()
        conn.close()

        assert row is not None
        assert row['openrouter_calls'] == 1
        assert row['openrouter_tokens_in'] == 10700
        assert row['openrouter_tokens_out'] == 55
        assert row['total_cost_usd'] > 0

    def test_multiple_openrouter_calls_accumulate(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=10000,
            completion_tokens=50,
            model="deepseek/deepseek-chat",
        )
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=20000,
            completion_tokens=100,
            model="deepseek/deepseek-chat",
        )

        conn = sqlite3.connect(tracker.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM daily_cost_summary").fetchone()
        conn.close()

        assert row['openrouter_calls'] == 2
        assert row['openrouter_tokens_in'] == 30000
        assert row['openrouter_tokens_out'] == 150


# ---------------------------------------------------------------------------
# Pipeline integration: _track_llm_cost calls CostTracker for OpenRouter
# ---------------------------------------------------------------------------

class TestPipelineOpenRouterCostIntegration:
    """Pipeline._track_llm_cost must call track_openrouter_call for openrouter."""

    def _make_pipeline(self):
        """Create a minimal ConversationPipeline for testing."""
        with patch.dict('os.environ', {
            'FIREWORKS_API_KEY': 'test',
            'ENVIRONMENT': 'test',
        }):
            with patch('src.core.conversation.context_builder.get_context_builder', return_value=MagicMock()), \
                 patch('src.core.memory_validation_agent.get_memory_validation_agent', return_value=MagicMock()), \
                 patch('src.core.message_validator_agent.get_message_validator_agent', return_value=MagicMock()):
                from src.core.conversation.pipeline import ConversationPipeline
                return ConversationPipeline()

    def test_track_openrouter_cost(self):
        pipeline = self._make_pipeline()
        pipeline._last_chain_usage = {'input_tokens': 10700, 'output_tokens': 55}
        pipeline._last_chain_provider_type = 'openrouter'

        with patch('src.services.cost_tracker.get_cost_tracker') as mock_get:
            mock_tracker = MagicMock()
            mock_get.return_value = mock_tracker

            pipeline._track_llm_cost('alice@example.com', 'deepseek/deepseek-chat')

            mock_tracker.track_openrouter_call.assert_called_once_with(
                user_id='alice@example.com',
                prompt_tokens=10700,
                completion_tokens=55,
                model='deepseek/deepseek-chat',
                call_purpose='conversation',
                conversation_id=None,
                message_id=None,
                companion_id=None,
            )

    def test_track_openrouter_clears_usage(self):
        pipeline = self._make_pipeline()
        pipeline._last_chain_usage = {'input_tokens': 100, 'output_tokens': 50}
        pipeline._last_chain_provider_type = 'openrouter'

        with patch('src.services.cost_tracker.get_cost_tracker') as mock_get:
            mock_get.return_value = MagicMock()
            pipeline._track_llm_cost('alice@example.com', 'model')

        assert pipeline._last_chain_usage is None


# ---------------------------------------------------------------------------
# Model override flows through extra_context
# ---------------------------------------------------------------------------

class TestModelOverrideFlow:
    """model_override in extra_context reaches the provider chain."""

    def _make_pipeline(self):
        with patch.dict('os.environ', {
            'FIREWORKS_API_KEY': 'test',
            'ENVIRONMENT': 'test',
        }):
            with patch('src.core.conversation.context_builder.get_context_builder', return_value=MagicMock()), \
                 patch('src.core.memory_validation_agent.get_memory_validation_agent', return_value=MagicMock()), \
                 patch('src.core.message_validator_agent.get_message_validator_agent', return_value=MagicMock()):
                from src.core.conversation.pipeline import ConversationPipeline
                return ConversationPipeline()

    def test_model_override_stored_from_extra_context(self):
        """Pipeline.process() should extract model_override from extra_context."""
        pipeline = self._make_pipeline()

        # We can't easily call process() end-to-end, but we can verify
        # that the extraction logic works by checking _call_llm reads it
        pipeline._model_override = 'anthropic/claude-sonnet-4'

        with patch('src.llm.provider_factory.get_resilient_provider_chain') as mock_chain_fn:
            mock_chain = MagicMock()
            mock_chain.get_last_usage.return_value = {'input_tokens': 0, 'output_tokens': 0}
            mock_chain.get_last_provider_type.return_value = 'openrouter'
            mock_chain.get_model_name.return_value = 'anthropic/claude-sonnet-4'
            mock_chain_fn.return_value = mock_chain

            with patch('src.llm.provider_factory.generate_sync', return_value="test response"):
                pipeline._call_llm("system prompt", "hello")

            # Verify model_override was passed to get_resilient_provider_chain
            mock_chain_fn.assert_called_once_with(
                model_override='anthropic/claude-sonnet-4'
            )
