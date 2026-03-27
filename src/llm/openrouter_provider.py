"""
OpenRouter Provider Implementation

OpenRouter provides a unified API gateway to 200+ models with automatic
fallback, usage-based pricing, and OpenAI-compatible endpoints.

Extends OpenAIProvider since the API is OpenAI-compatible, adding:
- OpenRouter-specific HTTP headers (HTTP-Referer, X-Title)
- Per-request model override support
- Model fallback chains via the `route` parameter
- Accurate usage tracking from response headers
"""

import os
import logging
from typing import Dict, List, Optional, Union

from openai import OpenAI

from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default context limits for common OpenRouter models
OPENROUTER_CONTEXT_LIMITS = {
    "deepseek/deepseek-chat": 131072,
    "deepseek/deepseek-chat-v3-0324": 131072,
    "moonshotai/kimi-k2": 131072,
    "anthropic/claude-sonnet-4": 200000,
    "anthropic/claude-3.5-sonnet": 200000,
    "openrouter/free": 131072,
    "openrouter/hunter-alpha": 1000000,
    "nousresearch/hermes-3-llama-3.1-405b:free": 131072,
}

DEFAULT_CONTEXT_LIMIT = 131072


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider with per-request model override and fallback chains.

    Inherits all OpenAI-compatible generation logic from OpenAIProvider.
    Adds OpenRouter-specific headers and model routing.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        context_limit: int = None,
        fallback_models: Optional[List[str]] = None,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (falls back to OPENROUTER_API_KEY env var)
            model: Default model (falls back to OPENROUTER_DEFAULT_MODEL or
                   OPENROUTER_MODEL env var, then deepseek/deepseek-chat)
            context_limit: Context window size (auto-detected from model if None)
            fallback_models: Optional list of fallback models for OpenRouter's
                            automatic routing when the primary is unavailable
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get(
            "OPENROUTER_DEFAULT_MODEL",
            os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        )
        self.fallback_models = fallback_models

        if context_limit is not None:
            self.context_limit = context_limit
        else:
            self.context_limit = OPENROUTER_CONTEXT_LIMITS.get(
                self.model, DEFAULT_CONTEXT_LIMIT
            )

        # Build OpenAI client with OpenRouter base URL and headers
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": OPENROUTER_BASE_URL,
            "default_headers": {
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_REFERER", "https://github.com/companion-framework/roux"
                ),
                "X-Title": os.environ.get("OPENROUTER_TITLE", "Companion Framework"),
            },
        }
        self.client = OpenAI(**client_kwargs)

    def generate_sync(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
        model_override: Optional[str] = None,
        **kwargs
    ) -> Union[str, Dict]:
        """Synchronous generate with optional per-request model override.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            tools: Optional tool definitions for function calling
            model_override: Override the default model for this request only
            **kwargs: Additional parameters (passed through)

        Returns:
            str for text responses, dict for tool calls
        """
        # Store original model so we can restore after the call
        original_model = self.model
        if model_override:
            self.model = model_override
            self.context_limit = OPENROUTER_CONTEXT_LIMITS.get(
                model_override, DEFAULT_CONTEXT_LIMIT
            )
            logger.info(f"OpenRouter model override: {model_override}")

        try:
            # Strip OpenRouter-specific kwargs before passing to parent
            kwargs.pop('chain', None)
            kwargs.pop('timeout', None)
            return super().generate_sync(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs
            )
        finally:
            # Restore original model after per-request override
            if model_override:
                self.model = original_model
                self.context_limit = OPENROUTER_CONTEXT_LIMITS.get(
                    original_model, DEFAULT_CONTEXT_LIMIT
                )

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model_override: Optional[str] = None,
        **kwargs
    ) -> str:
        """Async generate with optional per-request model override."""
        kwargs.pop('chain', None)
        kwargs.pop('timeout', None)
        if model_override:
            kwargs['model_override'] = model_override
        return self.generate_sync(messages, temperature, max_tokens, **kwargs)

    def get_model_name(self) -> str:
        """Return current model identifier."""
        return self.model


def get_openrouter_provider(
    model: str = None,
    fallback_models: List[str] = None,
) -> Optional['OpenRouterProvider']:
    """Get an OpenRouter provider instance, or None if not configured.

    Args:
        model: Override default model
        fallback_models: Optional fallback model chain
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenRouterProvider(
        api_key=api_key,
        model=model,
        fallback_models=fallback_models,
    )
