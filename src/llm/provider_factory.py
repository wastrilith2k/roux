"""
LLM Provider Factory -- creates provider instances with automatic failover.

WHAT: Factory functions for creating LLM provider instances (Fireworks, Anthropic,
      OpenAI/DeepSeek) and a ResilientProviderChain that retries with exponential
      backoff and fails over to the next provider when one goes down.

WHY:  The companion must always be able to "think." If Fireworks has an outage,
      the chain automatically falls over to DeepSeek, then OpenAI, then Anthropic.
      This makes the system resilient to any single provider failing.

HOW:  `get_resilient_provider_chain()` builds a priority-ordered list of providers
      from environment variables. `ResilientProviderChain.generate()` (async) and
      `generate_sync()` try each provider with up to 3 retries and exponential
      backoff before moving to the next. On total failure, a rate-limited Telegram
      notification is sent via the ops bot.

      Default chain: Ollama (if configured) -> Fireworks primary -> Fireworks
      fallback model -> DeepSeek direct -> OpenAI gpt-4o-mini -> Anthropic
      (last resort).
"""

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any

# Enable nested asyncio.run() calls (needed when called from eventlet context)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # Will fall back to thread pool approach

from .provider_interface import LLMProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# Rate-limit LLM failure notifications (don't spam Telegram)
_last_failure_notify = 0
_FAILURE_NOTIFY_COOLDOWN = 300  # 5 minutes between notifications


def _notify_llm_failure(error):
    """Send a one-time Telegram notification when all LLM providers fail."""
    global _last_failure_notify
    import time
    now = time.time()
    if now - _last_failure_notify < _FAILURE_NOTIFY_COOLDOWN:
        return
    _last_failure_notify = now
    try:
        from src.autonomy.telegram_bridge import send_to_telegram
        send_to_telegram(f"[SYSTEM] All LLM providers are down. Last error: {str(error)[:200]}")
    except Exception:
        logger.debug("Could not send LLM failure notification to Telegram")


class ResilientProviderChain:
    """
    LLM provider chain with automatic failover and retry logic.

    Tries primary provider first with retries, then falls back to alternatives.
    This ensures the companion can always think, even when one provider has issues.
    """

    def __init__(
        self,
        providers: List[LLMProvider],
        max_retries: int = 3,
        base_delay: float = 0.5
    ):
        """
        Initialize provider chain.

        Args:
            providers: List of providers in priority order (first = primary)
            max_retries: Number of retries per provider before failover
            base_delay: Base delay for exponential backoff (seconds)
        """
        if not providers:
            raise ValueError("At least one provider required")

        self.providers = providers
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._last_successful_provider = 0

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate completion with failover support.

        Tries each provider with exponential backoff retries.
        On failure, moves to next provider in chain.

        Returns:
            Generated text response

        Raises:
            RuntimeError: If all providers fail
        """
        last_error = None

        for provider_index, provider in enumerate(self.providers):
            provider_name = provider.get_model_name()

            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Trying {provider_name} (attempt {attempt + 1}/{self.max_retries})")

                    result = await provider.generate(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )

                    # Success - track which provider worked
                    self._last_successful_provider = provider_index

                    if provider_index > 0:
                        logger.info(f"Failover successful: using {provider_name}")

                    return result

                except Exception as e:
                    last_error = e
                    delay = self.base_delay * (2 ** attempt)

                    logger.warning(
                        f"Provider {provider_name} failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(delay)

            # Exhausted retries for this provider
            if provider_index < len(self.providers) - 1:
                next_provider = self.providers[provider_index + 1].get_model_name()
                logger.warning(f"Failing over from {provider_name} to {next_provider}")

        # All providers failed - notify via Telegram
        logger.error(f"All LLM providers failed. Last error: {last_error}")
        _notify_llm_failure(last_error)
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def generate_sync(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Synchronous generate with failover (works with eventlet).

        Uses synchronous HTTP clients instead of async.
        """
        import time
        last_error = None

        for provider_index, provider in enumerate(self.providers):
            provider_name = provider.get_model_name()

            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Trying {provider_name} sync (attempt {attempt + 1}/{self.max_retries})")

                    # Use synchronous generate method
                    result = provider.generate_sync(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )

                    self._last_successful_provider = provider_index

                    if provider_index > 0:
                        logger.info(f"Failover successful: using {provider_name}")

                    return result

                except Exception as e:
                    last_error = e
                    delay = self.base_delay * (2 ** attempt)

                    logger.warning(
                        f"Provider {provider_name} failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if attempt < self.max_retries - 1:
                        time.sleep(delay)

            if provider_index < len(self.providers) - 1:
                next_provider = self.providers[provider_index + 1].get_model_name()
                logger.warning(f"Failing over from {provider_name} to {next_provider}")

        logger.error(f"All LLM providers failed. Last error: {last_error}")
        _notify_llm_failure(last_error)
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def get_model_name(self) -> str:
        """Return name of currently preferred provider."""
        return self.providers[self._last_successful_provider].get_model_name()

    def get_context_limit(self) -> int:
        """Return context limit of currently preferred provider."""
        return self.providers[self._last_successful_provider].get_context_limit()


def get_resilient_provider_chain(
    primary: str = None,
    fallback: str = None
) -> ResilientProviderChain:
    """
    Create a resilient provider chain with failover capability.

    Default order: Ollama (local) → OpenRouter (free) → Fireworks → DeepSeek direct → OpenAI → Anthropic

    Args:
        primary: Override primary provider ('fireworks' or 'anthropic')
        fallback: Override fallback provider ('fireworks' or 'anthropic')

    Returns:
        Configured ResilientProviderChain
    """
    providers = []

    # Determine primary provider
    if primary is None:
        primary = os.getenv("LLM_PROVIDER", "fireworks").lower()

    # Primary: Ollama (local, tried first when configured)
    if primary == "ollama":
        from .ollama_provider import OllamaProvider
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
        ollama_ctx = int(os.getenv("OLLAMA_CONTEXT_LIMIT", "8192"))
        providers.append(OllamaProvider(
            base_url=ollama_base_url,
            model=ollama_model,
            context_limit=ollama_ctx
        ))
        logger.debug(f"Added Ollama primary: {ollama_model} at {ollama_base_url}")

    # Primary: OpenRouter (free, tried first when configured)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        providers.append(OpenAIProvider(
            api_key=openrouter_api_key,
            model=openrouter_model,
            base_url="https://openrouter.ai/api/v1",
            context_limit=131072
        ))
        logger.debug(f"Added OpenRouter primary: {openrouter_model}")

    fireworks_api_key = os.getenv("FIREWORKS_API_KEY")

    if primary == "fireworks" and fireworks_api_key:
        from .fireworks_provider import FireworksProvider
        # Primary: Fireworks with configured model (e.g. DeepSeek v3)
        model = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2-instruct-0905")
        providers.append(FireworksProvider(api_key=fireworks_api_key, model=model))
        logger.debug(f"Added Fireworks primary: {model}")

        # Fallback 1: Fireworks Kimi K2 (different model, same provider)
        fallback_model = os.getenv("FIREWORKS_FALLBACK_MODEL", "accounts/fireworks/models/kimi-k2-instruct-0905")
        if fallback_model != model:
            providers.append(FireworksProvider(api_key=fireworks_api_key, model=fallback_model))
            logger.debug(f"Added Fireworks fallback: {fallback_model}")

    elif primary == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            from .anthropic_provider import AnthropicProvider
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            providers.append(AnthropicProvider(api_key=api_key, model=model))
            logger.debug(f"Added Anthropic primary: {model}")

    # Fallback 2: DeepSeek direct API (OpenAI-compatible)
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        providers.append(OpenAIProvider(
            api_key=deepseek_api_key,
            model=deepseek_model,
            base_url="https://api.deepseek.com",
            context_limit=65536
        ))
        logger.debug(f"Added DeepSeek direct fallback: {deepseek_model}")

    # Fallback 3: OpenAI (gpt-4o-mini — cheap and reliable)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
        providers.append(OpenAIProvider(api_key=openai_api_key, model=openai_model))
        logger.debug(f"Added OpenAI fallback: {openai_model}")

    # Last resort: Anthropic (if configured and not already primary)
    if primary != "anthropic":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            from .anthropic_provider import AnthropicProvider
            anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            providers.append(AnthropicProvider(api_key=anthropic_api_key, model=anthropic_model))
            logger.debug(f"Added Anthropic last-resort: {anthropic_model}")

    if not providers:
        raise ValueError(
            "No LLM providers configured. Set LLM_PROVIDER=ollama, or set FIREWORKS_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY, or ANTHROPIC_API_KEY."
        )

    logger.info(f"Created resilient provider chain: {[p.get_model_name() for p in providers]}")
    return ResilientProviderChain(providers)


def generate_sync(
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    chain: ResilientProviderChain = None,
    **kwargs
) -> str:
    """
    Synchronous LLM generation (works with eventlet).

    Uses purely synchronous HTTP clients to avoid asyncio/eventlet conflicts.

    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        chain: Optional pre-created chain (creates new if None)
        **kwargs: Additional provider parameters

    Returns:
        Generated text response
    """
    if chain is None:
        chain = get_resilient_provider_chain()

    # Use synchronous generation directly - no asyncio involved
    return chain.generate_sync(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def get_llm_provider(
    provider_name: Optional[str] = None,
    model: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to get configured LLM provider.

    Args:
        provider_name: Override default provider ('fireworks' or 'anthropic')
        model: Override default model for the provider

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider is unknown or API key is missing
    """

    # Get provider from parameter or environment
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "fireworks").lower()

    logger.info(f"Initializing LLM provider: {provider_name}")

    if provider_name == "ollama":
        from .ollama_provider import OllamaProvider
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if model is None:
            model = os.getenv("OLLAMA_MODEL", "llama3.1")
        context_limit = int(os.getenv("OLLAMA_CONTEXT_LIMIT", "8192"))
        return OllamaProvider(base_url=base_url, model=model, context_limit=context_limit)

    elif provider_name == "fireworks":
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable not set")

        # Default Fireworks model
        if model is None:
            model = os.getenv(
                "FIREWORKS_MODEL",
                "accounts/fireworks/models/kimi-k2-instruct-0905"
            )

        from .fireworks_provider import FireworksProvider
        return FireworksProvider(api_key=api_key, model=model)

    elif provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        # Default Anthropic model
        if model is None:
            model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-20250514")

        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model=model)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: 'ollama', 'fireworks', 'anthropic'"
        )


def get_default_provider() -> LLMProvider:
    """
    Get the default LLM provider based on environment configuration.

    This is a convenience wrapper around get_llm_provider().

    Returns:
        Default configured LLMProvider instance
    """
    return get_llm_provider()
