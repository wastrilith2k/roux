"""
Fireworks AI Provider Implementation

Uses Fireworks AI API with Kimi K2 model (via Fireworks).
"""

import requests
import logging
from typing import Dict, List, Optional

from .provider_interface import LLMProvider

logger = logging.getLogger(__name__)


class FireworksProvider(LLMProvider):
    """Fireworks AI implementation"""

    def __init__(
        self,
        api_key: str,
        model: str = "accounts/fireworks/models/kimi-k2-instruct-0905",
        base_url: str = "https://api.fireworks.ai/inference/v1/chat/completions"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.context_limit = 262144  # 262K tokens

    async def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate completion using Fireworks AI API"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Filter out kwargs that aren't valid Fireworks API params
        filtered = {k: v for k, v in kwargs.items() if k not in ('timeout', 'chain')}

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **filtered
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Fireworks API error: {response.status} - {error_text}")
                        raise Exception(f"Fireworks API error: {response.status}")

                    result = await response.json()
                    return result['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Fireworks generation error: {e}", exc_info=True)
            raise

    def generate_sync(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Synchronous generate using requests (works with eventlet)"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Filter out kwargs that aren't valid Fireworks API params
        filtered = {k: v for k, v in kwargs.items() if k not in ('timeout', 'chain')}

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **filtered
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                logger.error(f"Fireworks API error: {response.status_code} - {response.text}")
                raise Exception(f"Fireworks API error: {response.status_code}")

            result = response.json()
            return result['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Fireworks sync generation error: {e}", exc_info=True)
            raise

    def get_context_limit(self) -> int:
        """Return context window size"""
        return self.context_limit

    def get_model_name(self) -> str:
        """Return model identifier"""
        return self.model

    def supports_streaming(self) -> bool:
        """Fireworks supports streaming"""
        return True

    async def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """Generate with streaming"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180)
            ) as response:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
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
