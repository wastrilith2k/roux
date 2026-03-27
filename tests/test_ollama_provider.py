"""
Tests for Ollama LLM provider support (Issue #11).

Covers:
- OllamaProvider implements the LLMProvider interface correctly
- OllamaProvider.generate_sync hits the correct endpoint with correct payload
- OllamaProvider uses configurable base_url, model, and context_limit
- Provider factory creates Ollama provider via get_llm_provider()
- Provider factory includes Ollama in resilient chain when LLM_PROVIDER=ollama
- Ollama does not appear in chain when not configured (no behavior change)
- Error message lists Ollama as supported provider
"""

import asyncio
import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

from src.llm.ollama_provider import OllamaProvider
from src.llm.provider_interface import LLMProvider


# =========================================================================
# OllamaProvider Unit Tests
# =========================================================================

class TestOllamaProviderInterface:
    """Verify OllamaProvider implements the LLMProvider interface."""

    def test_is_llm_provider_subclass(self):
        assert issubclass(OllamaProvider, LLMProvider)

    def test_default_values(self):
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama3.1"
        assert provider.context_limit == 8192

    def test_custom_values(self):
        provider = OllamaProvider(
            base_url="http://myhost:9999",
            model="mistral",
            context_limit=32768
        )
        assert provider.base_url == "http://myhost:9999"
        assert provider.model == "mistral"
        assert provider.context_limit == 32768

    def test_trailing_slash_stripped(self):
        provider = OllamaProvider(base_url="http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"

    def test_get_model_name(self):
        provider = OllamaProvider(model="codellama")
        assert provider.get_model_name() == "codellama"

    def test_get_context_limit(self):
        provider = OllamaProvider(context_limit=16384)
        assert provider.get_context_limit() == 16384

    def test_supports_streaming(self):
        provider = OllamaProvider()
        assert provider.supports_streaming() is True


class TestOllamaProviderGenerateSync:
    """Test synchronous generation calls."""

    @patch("src.llm.ollama_provider.requests.post")
    def test_generate_sync_calls_correct_endpoint(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Ollama"}}]
        }
        mock_post.return_value = mock_response

        provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.1")
        messages = [{"role": "user", "content": "Hi"}]
        result = provider.generate_sync(messages, temperature=0.5, max_tokens=100)

        assert result == "Hello from Ollama"

        # Verify the correct URL was called
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/v1/chat/completions"

        # Verify payload
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3.1"
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert payload["stream"] is False

    @patch("src.llm.ollama_provider.requests.post")
    def test_generate_sync_raises_on_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        with pytest.raises(Exception, match="Ollama API error: 500"):
            provider.generate_sync([{"role": "user", "content": "Hi"}])

    @patch("src.llm.ollama_provider.requests.post")
    def test_generate_sync_raises_on_connection_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        provider = OllamaProvider()
        with pytest.raises(requests.ConnectionError):
            provider.generate_sync([{"role": "user", "content": "Hi"}])

    @patch("src.llm.ollama_provider.requests.post")
    def test_generate_sync_filters_invalid_kwargs(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        provider.generate_sync(
            [{"role": "user", "content": "Hi"}],
            timeout=30,
            chain="ignored",
            tools=[{"name": "foo"}],
        )

        payload = mock_post.call_args[1]["json"]
        assert "timeout" not in payload
        assert "chain" not in payload
        assert "tools" not in payload


class TestOllamaGenerateAsync:
    """Test that async generate() does not block the event loop (Issue #51)."""

    @patch("src.llm.ollama_provider.requests.post")
    def test_generate_uses_asyncio_to_thread(self, mock_post):
        """generate() must offload to a thread, not call generate_sync directly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "threaded response"}}]
        }
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        messages = [{"role": "user", "content": "Hi"}]

        async def _run():
            with patch("src.llm.ollama_provider.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = "threaded response"
                result = await provider.generate(messages, temperature=0.5, max_tokens=100)
                mock_to_thread.assert_called_once_with(
                    provider.generate_sync, messages, 0.5, 100
                )
                assert result == "threaded response"

        asyncio.run(_run())


class TestOllamaStreamKwargsFiltering:
    """Test that generate_stream filters invalid kwargs (Issue #51)."""

    def test_generate_stream_filters_invalid_kwargs(self):
        """generate_stream must filter timeout, chain, tools from payload."""

        class MockAsyncLineIterator:
            """Mock async iterator for response.aiter_lines()."""
            def __init__(self, lines):
                self._lines = iter(lines)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._lines)
                except StopIteration:
                    raise StopAsyncIteration

        mock_response = MagicMock()
        mock_response.aiter_lines = MagicMock(
            return_value=MockAsyncLineIterator(['data: [DONE]'])
        )

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        provider = OllamaProvider()
        messages = [{"role": "user", "content": "Hi"}]

        async def _run():
            with patch("httpx.AsyncClient", return_value=mock_client_ctx):
                chunks = []
                async for chunk in provider.generate_stream(
                    messages,
                    timeout=30,
                    chain="ignored",
                    tools=[{"name": "foo"}],
                ):
                    chunks.append(chunk)

                # Verify payload passed to httpx does not contain filtered keys
                call_args = mock_client.stream.call_args
                payload = call_args[1]["json"]
                assert "timeout" not in payload
                assert "chain" not in payload
                assert "tools" not in payload

        asyncio.run(_run())


class TestOllamaStreamUsesHttpx:
    """Regression test: generate_stream must not depend on aiohttp (Issue #65)."""

    def test_generate_stream_does_not_import_aiohttp(self):
        """generate_stream uses httpx (a listed dependency), not aiohttp."""
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(OllamaProvider.generate_stream))
        tree = ast.parse(source)

        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.append(node.module)

        assert "aiohttp" not in imported_names, (
            "generate_stream must not import aiohttp — it is not in requirements"
        )
        assert "httpx" in imported_names, (
            "generate_stream should use httpx (an existing dependency)"
        )

    def test_generate_stream_yields_content_chunks(self):
        """generate_stream yields content from SSE data lines via httpx."""

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: [DONE]',
        ]

        class MockAsyncLineIterator:
            def __init__(self, lines):
                self._lines = iter(lines)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._lines)
                except StopIteration:
                    raise StopAsyncIteration

        mock_response = MagicMock()
        mock_response.aiter_lines = MagicMock(
            return_value=MockAsyncLineIterator(sse_lines)
        )

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

        provider = OllamaProvider()

        async def _run():
            with patch("httpx.AsyncClient", return_value=mock_client_ctx):
                chunks = []
                async for chunk in provider.generate_stream(
                    [{"role": "user", "content": "Hi"}],
                ):
                    chunks.append(chunk)
                assert chunks == ["Hello", " world"]

        asyncio.run(_run())


# =========================================================================
# Provider Factory Integration Tests
# =========================================================================

class TestProviderFactoryOllama:
    """Test Ollama integration in the provider factory."""

    def test_get_llm_provider_ollama_explicit(self, monkeypatch):
        """get_llm_provider(provider_name='ollama') returns OllamaProvider."""
        provider = _get_ollama_provider_via_factory(monkeypatch)
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.1"

    def test_get_llm_provider_ollama_from_env(self, monkeypatch):
        """LLM_PROVIDER=ollama env var selects Ollama."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        from src.llm.provider_factory import get_llm_provider
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider)

    def test_get_llm_provider_ollama_custom_model(self, monkeypatch):
        """OLLAMA_MODEL env var is respected."""
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        provider = _get_ollama_provider_via_factory(monkeypatch)
        assert provider.model == "mistral"

    def test_get_llm_provider_ollama_custom_base_url(self, monkeypatch):
        """OLLAMA_BASE_URL env var is respected."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
        provider = _get_ollama_provider_via_factory(monkeypatch)
        assert provider.base_url == "http://gpu-box:11434"

    def test_get_llm_provider_ollama_custom_context_limit(self, monkeypatch):
        """OLLAMA_CONTEXT_LIMIT env var is respected."""
        monkeypatch.setenv("OLLAMA_CONTEXT_LIMIT", "32768")
        provider = _get_ollama_provider_via_factory(monkeypatch)
        assert provider.context_limit == 32768

    def test_get_llm_provider_ollama_model_override(self, monkeypatch):
        """Explicit model parameter overrides OLLAMA_MODEL env var."""
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")
        from src.llm.provider_factory import get_llm_provider
        provider = get_llm_provider(provider_name="ollama", model="codellama")
        assert provider.model == "codellama"

    def test_unknown_provider_error_lists_ollama(self):
        """Error message for unknown provider mentions ollama."""
        from src.llm.provider_factory import get_llm_provider
        with pytest.raises(ValueError, match="ollama"):
            get_llm_provider(provider_name="nonexistent")

    def test_resilient_chain_includes_ollama_when_configured(self, monkeypatch):
        """When LLM_PROVIDER=ollama, the resilient chain starts with Ollama."""
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        from src.llm.provider_factory import get_resilient_provider_chain
        chain = get_resilient_provider_chain()

        assert len(chain.providers) >= 1
        assert isinstance(chain.providers[0], OllamaProvider)
        assert chain.providers[0].model == "llama3.1"

    def test_resilient_chain_excludes_ollama_when_not_configured(self, monkeypatch):
        """When LLM_PROVIDER is not ollama, no OllamaProvider in chain."""
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "fireworks")

        from src.llm.provider_factory import get_resilient_provider_chain
        chain = get_resilient_provider_chain()

        for provider in chain.providers:
            assert not isinstance(provider, OllamaProvider)

    def test_resilient_chain_ollama_no_api_key_needed(self, monkeypatch):
        """Ollama requires no API key — just LLM_PROVIDER=ollama."""
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        from src.llm.provider_factory import get_resilient_provider_chain
        chain = get_resilient_provider_chain()
        assert len(chain.providers) >= 1

    def test_no_providers_error_mentions_ollama(self, monkeypatch):
        """Error when no providers configured mentions LLM_PROVIDER=ollama."""
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "fireworks")

        from src.llm.provider_factory import get_resilient_provider_chain
        with pytest.raises(ValueError, match="ollama"):
            get_resilient_provider_chain()


# =========================================================================
# Helpers
# =========================================================================

def _get_ollama_provider_via_factory(monkeypatch):
    """Create an Ollama provider through the factory."""
    from src.llm.provider_factory import get_llm_provider
    return get_llm_provider(provider_name="ollama")


def _clear_provider_env(monkeypatch):
    """Remove all provider API keys so only explicitly set ones are used."""
    for key in [
        "FIREWORKS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
