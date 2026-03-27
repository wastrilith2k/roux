"""
LLM Provider Interface - Abstract Base Class

All LLM providers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_context_limit(self) -> int:
        """
        Return context window size for this provider/model.

        Returns:
            Maximum context length in tokens
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model identifier.

        Returns:
            Model name/ID string
        """
        pass

    def get_last_usage(self) -> Dict:
        """
        Return usage stats from the last generate call.

        Returns:
            Dict with 'input_tokens' and 'output_tokens' (both int, default 0)
        """
        return getattr(self, '_last_usage', {})

    def supports_streaming(self) -> bool:
        """
        Whether this provider supports streaming responses.

        Returns:
            True if streaming is supported
        """
        return False

    async def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Generate completion with streaming (optional).

        Yields:
            Text chunks as they're generated

        Raises:
            NotImplementedError: If provider doesn't support streaming
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")
