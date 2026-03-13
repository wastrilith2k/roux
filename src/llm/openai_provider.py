"""
OpenAI Provider Implementation

Uses OpenAI API with gpt-4o-mini for tool calling (very cost-effective).
"""

import os
import logging
from typing import Dict, List, Optional, Union

from openai import OpenAI

from .provider_interface import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI implementation with tool/function calling support."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        base_url: str = None,
        context_limit: int = 128000
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.context_limit = context_limit

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Async generate - wraps sync for now."""
        return self.generate_sync(messages, temperature, max_tokens, **kwargs)

    def generate_sync(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Union[str, Dict]:
        """
        Synchronous generate with optional tool support.

        Returns:
            str if regular text response
            dict if tool call: {"type": "tool_use", "tool_name": ..., "tool_input": ..., "tool_use_id": ...}
        """
        try:
            # Build request params
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Convert tools to OpenAI function format
            if tools:
                functions = []
                for tool in tools:
                    functions.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {})
                        }
                    })
                request_params["tools"] = functions
                request_params["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**request_params)

            message = response.choices[0].message
            usage = response.usage

            # Track usage for cost tracking
            self._last_usage = {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }

            # Check for tool calls
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                import json
                return {
                    "type": "tool_use",
                    "tool_name": tool_call.function.name,
                    "tool_input": json.loads(tool_call.function.arguments),
                    "tool_use_id": tool_call.id,
                    "usage": self._last_usage
                }

            # Regular text response
            return message.content or ""

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}", exc_info=True)
            raise

    def get_context_limit(self) -> int:
        """Return model context window size."""
        return self.context_limit

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model

    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True

    async def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """Generate with streaming."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI stream error: {e}", exc_info=True)
            raise


def get_openai_tool_provider() -> Optional[OpenAIProvider]:
    """Get an OpenAI provider for tool calling, or None if not configured."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
