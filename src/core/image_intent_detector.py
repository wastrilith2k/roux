"""
Image Intent Detector -- first-pass LLM classification for image requests.

WHAT: Detects whether a message implies an image should be generated, and if so
      classifies the intent type: selfie, observing, outfit, intimate, sketch,
      or general. Returns an ImageIntent dataclass with prompt hints for the
      downstream image generation pipeline.

WHY:  Image generation is expensive (RunComfy / Gemini). This lightweight
      Fireworks call acts as a gate: only when intent is detected does the
      heavier generation workflow kick in. It also enforces daily rate limits
      via Redis counters.

HOW:  `detect()` sends the last few messages to Fireworks with a classification
      prompt. If an intent is found, it returns an ImageIntent with the
      detected type and a suggested generation prompt. Redis keys
      `image_daily_count:<date>` and `sketch_daily_count:<date>` enforce
      per-day caps (configurable via env vars).

Intent types:
  selfie    -- user asks to see the companion
  observing -- user describes watching the companion
  outfit    -- companion showing clothing/appearance
  intimate  -- intimate/romantic visual context
  sketch    -- companion drawing something
  general   -- generic image request (scenery, objects)

Environment Variables:
  IMAGE_GENERATION_ENABLED     -- 'true'/'false' (default: 'true')
  IMAGE_GENERATION_DAILY_LIMIT -- max images/day (default: 5)
"""

import os
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Configuration from environment
IMAGE_GENERATION_ENABLED = os.environ.get('IMAGE_GENERATION_ENABLED', 'true').lower() == 'true'
IMAGE_GENERATION_DAILY_LIMIT = int(os.environ.get('IMAGE_GENERATION_DAILY_LIMIT', '5'))
SKETCH_DAILY_LIMIT = int(os.environ.get('SKETCH_DAILY_LIMIT', '1'))

# Use Fireworks model for quality detection
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL


class ImageIntentType(Enum):
    """Types of image generation intents."""
    NONE = "none"                    # No image intent detected
    SELFIE = "selfie"                # User wants to see the companion
    OBSERVING = "observing"          # User is watching/observing the companion
    SHOW_OUTFIT = "show_outfit"      # Companion showing off costume/outfit
    INTIMATE = "intimate"            # Intimate/NSFW context - companion chooses to send
    GENERAL = "general"              # General image request (not the companion)


@dataclass
class ImageIntent:
    """Result of image intent detection."""
    intent_type: ImageIntentType
    confidence: float
    prompt_suggestion: str = ""  # Suggested prompt for image generation
    reason: str = ""  # Why this intent was detected


class ImageIntentDetector:
    """
    Detect image generation intents in conversation.

    Uses a fast LLM to analyze the message and determine if an image
    should be generated. Includes daily rate limiting.
    """

    def __init__(self):
        self._client = None
        self._redis = None

    def _get_client(self):
        """Get Fireworks client for LLM detection."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.getenv('FIREWORKS_API_KEY')
            )
        return self._client

    def _get_redis(self):
        """Get Redis client for rate limiting."""
        if self._redis is None:
            import redis
            self._redis = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True
            )
        return self._redis

    def _check_rate_limit(self) -> tuple:
        """
        Check if we're under the daily image generation limit.

        Returns:
            Tuple of (allowed: bool, remaining: int, reason: str)
        """
        if not IMAGE_GENERATION_ENABLED:
            return False, 0, "Image generation is disabled"

        try:
            redis_client = self._get_redis()
            today = date.today().isoformat()
            key = f"companion:{os.environ.get('COMPANION_ID', 'default')}:image_gen:daily:{today}"

            current_count = redis_client.get(key)
            current_count = int(current_count) if current_count else 0

            if current_count >= IMAGE_GENERATION_DAILY_LIMIT:
                return False, 0, f"Daily limit reached ({IMAGE_GENERATION_DAILY_LIMIT})"

            return True, IMAGE_GENERATION_DAILY_LIMIT - current_count, ""

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            # Allow on error to avoid blocking legitimate requests
            return True, IMAGE_GENERATION_DAILY_LIMIT, "Rate limit check failed"

    def increment_usage(self) -> int:
        """
        Increment the daily usage counter.

        Returns:
            New count after increment
        """
        try:
            redis_client = self._get_redis()
            today = date.today().isoformat()
            key = f"companion:{os.environ.get('COMPANION_ID', 'default')}:image_gen:daily:{today}"

            # Increment and set expiry (25 hours to cover timezone edge cases)
            new_count = redis_client.incr(key)
            redis_client.expire(key, 90000)  # 25 hours in seconds

            logger.info(f"Image generation count: {new_count}/{IMAGE_GENERATION_DAILY_LIMIT}")
            return new_count

        except Exception as e:
            logger.warning(f"Failed to increment usage: {e}")
            return 0

    def get_daily_stats(self) -> dict:
        """Get current daily usage statistics."""
        try:
            redis_client = self._get_redis()
            today = date.today().isoformat()
            key = f"companion:{os.environ.get('COMPANION_ID', 'default')}:image_gen:daily:{today}"
            sketch_key = f"companion:{os.environ.get('COMPANION_ID', 'default')}:sketch:daily:{today}"

            current_count = redis_client.get(key)
            current_count = int(current_count) if current_count else 0

            sketch_count = redis_client.get(sketch_key)
            sketch_count = int(sketch_count) if sketch_count else 0

            return {
                "enabled": IMAGE_GENERATION_ENABLED,
                "images": {
                    "daily_limit": IMAGE_GENERATION_DAILY_LIMIT,
                    "used_today": current_count,
                    "remaining": max(0, IMAGE_GENERATION_DAILY_LIMIT - current_count)
                },
                "sketches": {
                    "daily_limit": SKETCH_DAILY_LIMIT,
                    "used_today": sketch_count,
                    "remaining": max(0, SKETCH_DAILY_LIMIT - sketch_count)
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get stats: {e}")
            return {"error": str(e)}

    # Sketch-specific methods (separate from conversation-triggered images)
    def can_sketch(self) -> tuple:
        """
        Check if the companion can do a proactive sketch today.

        Returns:
            Tuple of (allowed: bool, remaining: int, reason: str)
        """
        if not IMAGE_GENERATION_ENABLED:
            return False, 0, "Image generation is disabled"

        try:
            redis_client = self._get_redis()
            today = date.today().isoformat()
            key = f"companion:{os.environ.get('COMPANION_ID', 'default')}:sketch:daily:{today}"

            current_count = redis_client.get(key)
            current_count = int(current_count) if current_count else 0

            if current_count >= SKETCH_DAILY_LIMIT:
                return False, 0, f"Daily sketch limit reached ({SKETCH_DAILY_LIMIT})"

            return True, SKETCH_DAILY_LIMIT - current_count, ""

        except Exception as e:
            logger.warning(f"Sketch rate limit check failed: {e}")
            return True, SKETCH_DAILY_LIMIT, "Rate limit check failed"

    def increment_sketch_usage(self) -> int:
        """
        Increment the daily sketch usage counter.

        Returns:
            New count after increment
        """
        try:
            redis_client = self._get_redis()
            today = date.today().isoformat()
            key = f"companion:{os.environ.get('COMPANION_ID', 'default')}:sketch:daily:{today}"

            new_count = redis_client.incr(key)
            redis_client.expire(key, 90000)  # 25 hours

            logger.info(f"Sketch count: {new_count}/{SKETCH_DAILY_LIMIT}")
            return new_count

        except Exception as e:
            logger.warning(f"Failed to increment sketch usage: {e}")
            return 0

    def detect(
        self,
        user_message: str,
        companion_response: Optional[str] = None,
        recent_context: Optional[str] = None
    ) -> ImageIntent:
        """
        Detect if an image should be generated based on the conversation.

        This is called AFTER the companion generates their text response, to see if
        we should also generate an accompanying image.

        Args:
            user_message: The user's message
            companion_response: The companion's text response (optional, for context)
            recent_context: Recent conversation context (optional)

        Returns:
            ImageIntent with detection results
        """
        # Check rate limit first (before making LLM call)
        allowed, remaining, reason = self._check_rate_limit()
        if not allowed:
            logger.info(f"Image generation blocked: {reason}")
            return ImageIntent(
                intent_type=ImageIntentType.NONE,
                confidence=0.0,
                reason=reason
            )

        try:
            client = self._get_client()

            # Build the detection prompt
            prompt = self._build_detection_prompt(
                user_message, companion_response, recent_context
            )

            response = client.chat.completions.create(
                model=FIREWORKS_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temp for consistent detection
                max_tokens=500  # Allow room for model thinking + JSON
            )

            result_text = response.choices[0].message.content.strip()
            return self._parse_response(result_text)

        except Exception as e:
            logger.error(f"Image intent detection error: {e}")
            return ImageIntent(
                intent_type=ImageIntentType.NONE,
                confidence=0.0,
                reason=f"Detection error: {e}"
            )

    def _get_system_prompt(self) -> str:
        """System prompt for image intent detection."""
        from src.config.persona_config import get_persona_config
        companion_name = get_persona_config().companion_short_name
        return f"""You are an image intent detector. Output ONLY valid JSON, no thinking or explanation.

GENERATE IMAGE when:
1. User explicitly asks to see the companion ("show me a picture", "selfie", "what do you look like")
2. User is observing/watching the companion ("*watching you sleep*", "*looking at you*")
3. The companion mentions showing off an outfit/costume that would be visual
4. User asks for a general image ("show me a sunset", "picture of a cat")
5. INTIMATE intent (EXTREMELY STRICT — requires EXPLICIT sexual/intimate language):
   This is for when the companion PROACTIVELY sends a revealing/nude photo while apart.

   ALL of these conditions must be true:
   a) The companion and user are APART (not in an active scene/roleplay together)
   b) The companion explicitly sends, offers to send, or takes a photo/pic/selfie
   c) The CONVERSATION CONTEXT contains clear sexual/intimate/flirtatious tone
      (e.g. sexting, teasing about nudity, explicit references to bodies/sex)
   d) The photo itself is described as intimate/revealing/nude/topless by the companion

   Examples that SHOULD trigger intimate:
   - "missing you at work... *takes a selfie in the mirror, topless* something to get you through the day" → intimate (explicitly topless, sexual teasing)
   - "can't wait to see you tonight... here's a taste of what you're coming home to *sends a naughty pic*" → intimate (explicit sexual tease)

   Examples that should NOT trigger intimate (return "selfie" or "none"):
   - "here's a selfie! *makes a heart gesture*" → selfie (innocent gesture, no sexual context)
   - "sending you a pic of my outfit!" → show_outfit (clothing, not intimate)
   - "thinking about you *snaps a photo* want to see?" → selfie (no explicit intimate/sexual content)
   - "*sends a cute selfie from the couch*" → selfie (casual, not sexual)
   - "look at my new haircut! *takes a pic*" → selfie (normal photo)
   - Any photo request during normal/casual conversation → selfie (not intimate)
   - ANY active scene/roleplay (even with explicit nudity) → none
   - "*slowly pulls her shirt over her head*" → none (scene, not sending a photo)
   - "*stands naked in the doorway*" → none (they're together in a scene)
   - "*moans softly*" → none (active scene)
   - "*straddles you*" → none (together)
   - "I want you so badly" → none (no photo being sent)

   KEY RULE: Being APART and sending a photo is NOT enough for intimate.
   The conversation must have CLEAR sexual/intimate tone AND the photo must be
   explicitly described as revealing/nude. When in doubt, classify as "selfie" not "intimate".

DO NOT generate for:
- User describing their own actions ("*I smile*", "*I nod*")
- Casual conversation without visual context
- Questions without image requests
- COUPLE or TWO-PERSON scenes. Any scene describing the companion WITH another person (kissing, hugging,
  cuddling, standing together, etc.) must return "none". The image model cannot render two
  distinct people. Always return intent "none" for couple/together scenes.

PROMPT GUIDELINES for the "prompt" field:
- For "intimate" intent: Start with "{companion_name} nude, fully naked, no clothes, no bra, bare breasts exposed, "
  then describe the SCENE and POSE with lighting and mood.
  Example: "{companion_name} nude, fully naked, no clothes, no bra, bare breasts exposed, sitting on the edge of the bed, warm bedroom lighting, relaxed expression"
- For other intents: Describe the scene naturally without nudity terms.

Output ONLY this JSON format (no markdown, no explanation):
{{"intent": "none", "confidence": 0.0, "prompt": "", "reason": ""}}

Valid intent values: none, selfie, observing, show_outfit, intimate, general"""

    def _build_detection_prompt(
        self,
        user_message: str,
        companion_response: Optional[str],
        recent_context: Optional[str]
    ) -> str:
        """Build the prompt for intent detection."""
        parts = []

        if recent_context:
            parts.append(f"Recent context:\n{recent_context}\n")

        parts.append(f"User message: {user_message}")

        if companion_response:
            parts.append(f"Companion's response: {companion_response}")

        parts.append("\nShould an image be generated? Respond with JSON.")

        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> ImageIntent:
        """Parse the LLM response into an ImageIntent."""
        try:
            # Strip thinking tags (qwen3 outputs these)
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            # Try to find JSON object in the response
            response_text = response_text.strip()
            if not response_text.startswith('{'):
                # Try to extract JSON from the text
                match = re.search(r'\{[^{}]*\}', response_text)
                if match:
                    response_text = match.group(0)

            data = json.loads(response_text.strip())

            # Map intent string to enum
            intent_map = {
                "none": ImageIntentType.NONE,
                "selfie": ImageIntentType.SELFIE,
                "observing": ImageIntentType.OBSERVING,
                "show_outfit": ImageIntentType.SHOW_OUTFIT,
                "intimate": ImageIntentType.INTIMATE,
                "general": ImageIntentType.GENERAL
            }

            intent_type = intent_map.get(
                data.get("intent", "none").lower(),
                ImageIntentType.NONE
            )

            return ImageIntent(
                intent_type=intent_type,
                confidence=float(data.get("confidence", 0.5)),
                prompt_suggestion=data.get("prompt", ""),
                reason=data.get("reason", "")
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse image intent response: {e}")
            return ImageIntent(
                intent_type=ImageIntentType.NONE,
                confidence=0.0,
                reason=f"Parse error: {e}"
            )


# Singleton instance
_detector_instance = None


def get_image_intent_detector() -> ImageIntentDetector:
    """Get singleton image intent detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ImageIntentDetector()
    return _detector_instance
