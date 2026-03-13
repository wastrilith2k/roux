"""
Anthropic Claude Provider Implementation

Uses Anthropic API with Claude models (Opus, Sonnet, Haiku).
Supports tool calling for agentic code execution.
"""

import anthropic
import logging
from typing import Dict, List, Optional, Union

from .provider_interface import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude implementation"""

    CONTEXT_LIMITS = {
        "claude-opus-4-20250514": 200000,
        "claude-opus-4": 200000,
        "claude-opus-4.1": 200000,
        "claude-sonnet-4-5-20250929": 200000,
        "claude-sonnet-4.5": 200000,
        "claude-sonnet-4-20241022": 200000,
        "claude-3-7-sonnet-20250219": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-20250514"
    ):
        self.api_key = api_key
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.sync_client = anthropic.Anthropic(api_key=api_key)
        self.context_limit = self.CONTEXT_LIMITS.get(model, 200000)

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate completion using Anthropic API"""

        try:
            # Anthropic requires system message separate
            system_message = None
            chat_messages = []

            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    chat_messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })

            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": chat_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            if system_message:
                request_params["system"] = system_message

            response = await self.client.messages.create(**request_params)

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}", exc_info=True)
            raise

    def generate_sync(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Union[str, Dict]:
        """Synchronous generate with optional tool support.

        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool definitions for function calling

        Returns:
            Either a string (text response) or a dict with tool_use info
        """

        try:
            # Anthropic requires system message separate
            system_message = None
            chat_messages = []

            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    # Handle both simple content and structured content (for tool results)
                    chat_messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })

            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": chat_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            if system_message:
                request_params["system"] = system_message

            # Add tools if provided
            if tools:
                request_params["tools"] = tools

            response = self.sync_client.messages.create(**request_params)

            # Check if response contains tool use
            for block in response.content:
                if block.type == "tool_use":
                    return {
                        "type": "tool_use",
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "tool_use_id": block.id,
                        "stop_reason": response.stop_reason
                    }

            # Regular text response
            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic sync generation error: {e}", exc_info=True)
            raise

    def get_context_limit(self) -> int:
        """Return context window size for this model"""
        return self.context_limit

    def get_model_name(self) -> str:
        """Return model identifier"""
        return self.model

    def supports_streaming(self) -> bool:
        """Anthropic supports streaming"""
        return True

    async def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """Generate with streaming"""

        try:
            # Separate system message
            system_message = None
            chat_messages = []

            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    chat_messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })

            # Build request
            request_params = {
                "model": self.model,
                "messages": chat_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            if system_message:
                request_params["system"] = system_message

            # Stream response
            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}", exc_info=True)
            raise
