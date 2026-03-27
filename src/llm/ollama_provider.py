"""
Ollama Provider Implementation

Uses Ollama's OpenAI-compatible API for fully local LLM inference.
Supports all LLM call sites via the standard provider interface.
"""

import asyncio
import requests
import logging
from typing import Dict, List, Optional

from .provider_interface import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama implementation using its OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        context_limit: int = 8192
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.context_limit = context_limit

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Async generate - offloads blocking sync call to a thread."""
        return await asyncio.to_thread(self.generate_sync, messages, temperature, max_tokens, **kwargs)

    def generate_sync(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Synchronous generate using Ollama's OpenAI-compatible endpoint."""

        url = f"{self.base_url}/v1/chat/completions"

        # Filter out kwargs that aren't valid API params
        filtered = {k: v for k, v in kwargs.items() if k not in ('timeout', 'chain', 'tools')}

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **filtered
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.ConnectionError as e:
            logger.error(f"Ollama connection error (is Ollama running at {self.base_url}?): {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama generation error: {e}", exc_info=True)
            raise

    def get_context_limit(self) -> int:
        """Return context window size."""
        return self.context_limit

    def get_model_name(self) -> str:
        """Return model identifier."""
        return self.model

    def supports_streaming(self) -> bool:
        """Ollama supports streaming."""
        return True

    async def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """Generate with streaming via Ollama's OpenAI-compatible endpoint."""

        url = f"{self.base_url}/v1/chat/completions"

        # Filter out kwargs that aren't valid API params
        filtered = {k: v for k, v in kwargs.items() if k not in ('timeout', 'chain', 'tools')}

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **filtered
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str == '[DONE]':
                                    break
                                try:
                                    import json
                                    data = json.loads(data_str)
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            yield delta['content']
                                except json.JSONDecodeError:
                                    continue

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}", exc_info=True)
            raise
