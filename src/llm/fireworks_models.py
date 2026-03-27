"""
Fireworks Model Configuration - Centralized model definitions with fallbacks.

All Fireworks model references should use these constants so we can
swap models in one place when Fireworks retires them.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Fast/cheap model for extraction, classification, and simple tasks
# Primary: Qwen3 8B (fast, good at JSON)
# Fallback: Mixtral 8x22B (larger but reliable)
FAST_MODELS = [
    "accounts/fireworks/models/qwen3-8b",
    "accounts/fireworks/models/mixtral-8x22b-instruct",
]

# Medium model for autonomy, reasoning, interjection decisions
MEDIUM_MODELS = [
    "accounts/fireworks/models/kimi-k2p5-instruct",
]

# Large model for main conversation (configured via env vars, not here)


def call_fireworks(
    client,
    messages: list,
    models: list = None,
    max_tokens: int = 500,
    temperature: float = 0.1,
) -> Optional[str]:
    """
    Call Fireworks with automatic model fallback.

    Tries each model in the list until one succeeds. Returns the
    response text, or None if all models fail.

    Args:
        client: OpenAI-compatible client pointed at Fireworks
        messages: Chat messages
        models: List of model IDs to try in order (defaults to FAST_MODELS)
        max_tokens: Max response tokens
        temperature: Sampling temperature

    Returns:
        Response text or None
    """
    if models is None:
        models = FAST_MODELS

    last_error = None
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            # Strip Qwen <think> tags
            if '<think>' in text and '</think>' in text:
                text = text.split('</think>')[-1].strip()
            return text
        except Exception as e:
            last_error = e
            error_msg = str(e)[:80]
            logger.warning(f"Fireworks model {model} failed: {error_msg}")
            continue

    logger.error(f"All Fireworks models failed. Last error: {last_error}")
    return None
