"""
Shared Vision Utility — Image description via Fireworks AI vision model.

Uses Qwen3 VL 8B on Fireworks (OpenAI-compatible API) by default.
Falls back to GPT-4o-mini if FIREWORKS_API_KEY is not available.

Extracted from telegram_bridge._describe_image() so both Telegram and
WebSocket channels can describe user-sent photos consistently.
"""
import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Fireworks vision models — tried in order
FIREWORKS_VISION_MODELS = [
    "accounts/fireworks/models/qwen3-vl-8b-instruct",
    "accounts/fireworks/models/qwen2p5-vl-7b-instruct",
]

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

VISION_PROMPT = (
    "Describe this photo in 2-3 natural sentences, as if telling "
    "a close friend what you see. Focus on the subject, mood, and "
    "any notable details. If it's a selfie or person, describe their "
    "expression and what they seem to be doing. Keep it casual."
)


def _call_fireworks_vision(b64: str, mime_type: str) -> Optional[str]:
    """Call a Fireworks vision model with automatic fallback."""
    import openai

    api_key = os.environ.get('FIREWORKS_API_KEY')
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64}"
                }
            },
            {
                "type": "text",
                "text": VISION_PROMPT
            }
        ]
    }]

    last_error = None
    for model in FIREWORKS_VISION_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=300,
                messages=messages,
            )
            text = response.choices[0].message.content.strip()
            logger.info(f"Vision description via Fireworks ({model})")
            return text
        except Exception as e:
            last_error = e
            logger.warning(f"Fireworks vision model {model} failed: {str(e)[:80]}")
            continue

    logger.error(f"All Fireworks vision models failed. Last error: {last_error}")
    return None


def _call_openai_vision(b64: str, mime_type: str) -> Optional[str]:
    """Fallback: call GPT-4o-mini vision."""
    import openai

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64}"
                    }
                },
                {
                    "type": "text",
                    "text": VISION_PROMPT
                }
            ]
        }]
    )
    logger.info("Vision description via OpenAI GPT-4o-mini (fallback)")
    return response.choices[0].message.content.strip()


def describe_image(image_data: bytes, mime_type: str = 'image/jpeg') -> Optional[str]:
    """Describe an image using a vision model.

    Tries Fireworks (Qwen3 VL) first, falls back to OpenAI GPT-4o-mini.

    Args:
        image_data: Raw image bytes.
        mime_type: MIME type of the image (e.g. 'image/jpeg', 'image/png').

    Returns:
        A short text description or None on failure.
    """
    try:
        b64 = base64.b64encode(image_data).decode('utf-8')

        # Try Fireworks first
        result = _call_fireworks_vision(b64, mime_type)
        if result:
            return result

        # Fall back to OpenAI
        result = _call_openai_vision(b64, mime_type)
        if result:
            return result

        logger.error("No vision API available (need FIREWORKS_API_KEY or OPENAI_API_KEY)")
        return None

    except Exception as e:
        logger.error(f"Image description failed: {e}", exc_info=True)
        return None


def describe_image_from_path(image_path: str) -> Optional[str]:
    """Convenience wrapper that reads a file and calls describe_image().

    Detects MIME type from the file extension.
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(ext, 'image/jpeg')

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
    except Exception as e:
        logger.error(f"Could not read image file {image_path}: {e}")
        return None

    return describe_image(image_data, mime_type=mime_type)


def describe_image_from_base64(b64_data: str, mime_type: str = 'image/jpeg') -> Optional[str]:
    """Convenience wrapper for base64-encoded data (e.g. from WebSocket).

    Args:
        b64_data: Base64-encoded image string (no data URI prefix).
        mime_type: MIME type of the image.

    Returns:
        A short text description or None on failure.
    """
    try:
        image_data = base64.b64decode(b64_data)
    except Exception as e:
        logger.error(f"Invalid base64 image data: {e}")
        return None

    return describe_image(image_data, mime_type=mime_type)
