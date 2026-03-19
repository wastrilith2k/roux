"""
NanoBanana Image Provider — generates images via the NanoBanana API (Gemini-based).

WHAT: Sends text prompts (and optional reference images) to NanoBanana's API
      for image generation. Returns a URL to the generated image.

WHY:  NanoBanana uses Google's Gemini image generation, which is free with a
      Google AI Studio API key. This makes it the simplest and cheapest option
      for image generation — no ComfyUI setup, no GPU rental.

HOW:  Two modes:
      1. Text-to-image: Send a text prompt, get an image back
      2. Image-to-image: Send a reference image + prompt for mood variations,
         style transfers, etc. ("the person in this image looking sad")

      The API key comes from NANOBANANA_API_KEY or GOOGLE_AI_STUDIO_KEY env var.
"""

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

NANOBANANA_API_URL = "https://api.nanobanana.com/v1/generate"


class NanoBananaProvider:
    """Image generation via NanoBanana (Gemini-based)."""

    def __init__(self):
        self.api_key = os.environ.get('NANOBANANA_API_KEY') or os.environ.get('GOOGLE_AI_STUDIO_KEY')
        if not self.api_key:
            logger.debug("NanoBanana not configured (no NANOBANANA_API_KEY or GOOGLE_AI_STUDIO_KEY)")

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        reference_image_url: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate an image from a text prompt, optionally with a reference image.

        Args:
            prompt: Text description of the image to generate
            reference_image_url: Optional URL of a reference image for img2img
            width: Output width
            height: Output height
            style: Optional style preset

        Returns:
            URL of the generated image, or None on failure
        """
        if not self.is_enabled():
            logger.error("NanoBanana not configured")
            return None

        try:
            payload = {
                'prompt': prompt,
                'width': width,
                'height': height,
                'api_key': self.api_key,
            }

            if reference_image_url:
                payload['reference_image'] = reference_image_url

            if style:
                payload['style'] = style

            logger.info(f"NanoBanana generating: {prompt[:80]}...")

            response = requests.post(
                NANOBANANA_API_URL,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                image_url = data.get('image_url') or data.get('url') or data.get('output')
                if image_url:
                    logger.info(f"NanoBanana generated: {image_url[:80]}")
                    return image_url
                else:
                    logger.error(f"NanoBanana response missing image URL: {data}")
                    return None
            else:
                logger.error(f"NanoBanana error {response.status_code}: {response.text[:200]}")
                return None

        except Exception as e:
            logger.error(f"NanoBanana generation failed: {e}")
            return None

    def generate_mood_variation(
        self,
        base_image_url: str,
        mood: str,
        character_description: str = "",
    ) -> Optional[str]:
        """
        Generate a mood variation of a base character image.

        Args:
            base_image_url: URL of the base character image
            mood: Target mood (e.g., "sad", "happy", "tired")
            character_description: Optional description of the character for consistency

        Returns:
            URL of the mood-varied image, or None on failure
        """
        prompt = f"The person in this image with a {mood} expression."
        if character_description:
            prompt = f"{character_description}, {mood} expression. Same person, same style, same framing."

        return self.generate(
            prompt=prompt,
            reference_image_url=base_image_url,
            width=512,
            height=512,
        )


# Singleton
_provider = None

def get_nanobanana_provider() -> NanoBananaProvider:
    global _provider
    if _provider is None:
        _provider = NanoBananaProvider()
    return _provider
