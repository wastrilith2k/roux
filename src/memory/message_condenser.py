"""
Message Condenser - Strip verbose prose from messages before caching.

WHAT: Uses GPT-4o-mini to condense the companion's verbose messages (roleplay
prose, poetic line breaks, emotional meta-commentary) into single-paragraph
summaries while preserving concrete details (actions, topics, plans, facts).
Condensed messages are cached in Redis keyed by message ID.

WHY: The companion's responses are often 300-500 chars of literary prose where
only 50-100 chars of information content exists. When building conversation
history for context, verbose messages waste prompt tokens on style rather than
substance. Condensation gives 2-3x more conversation history per token budget.

HOW it fits:
  - After each message is saved to the database, condense_and_cache_message()
    is called to LLM-condense and store in Redis.
  - context_builder calls get_condensed_history() to retrieve a token-efficient
    conversation history for prompt inclusion.
  - Only the companion's messages are LLM-condensed; user messages pass through
    unchanged (they're already concise).
  - Redis cache avoids re-condensing on every context build.
"""

import os
import json
import logging
import redis
from typing import Optional, List, Dict
from openai import OpenAI
from src.config.persona_config import get_persona_config

logger = logging.getLogger(__name__)

# =============================================================================
# Redis connection
# =============================================================================

_redis_client: Optional[redis.Redis] = None

def get_redis() -> redis.Redis:
    """Get Redis connection."""
    global _redis_client
    if _redis_client is None:
        redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# Condensation prompt
# =============================================================================

# Targets the companion's specific verbosity patterns: poetic line breaks,
# dramatic pauses, philosophical tangents. Preserves physical actions and facts.
CONDENSE_PROMPT = """Rewrite this message as ONE SINGLE PARAGRAPH with no line breaks.

CRITICAL: The output must be a single flowing paragraph. NO newlines. NO poetry formatting.

KEEP (but flatten into one paragraph):
- Physical actions: "*smiles*", "*sits down*"
- What they're doing: "making dinner", "watching TV"
- Concrete details: times, places, plans
- Direct questions and answers

REMOVE ENTIRELY:
- Dramatic pauses written as separate lines
- Poetry-style formatting with each thought on its own line
- Meta-commentary about the moment
- Philosophical musings about connection/authenticity
- Repetitive emphasis ("Just this. You. The weight of your arm.")

EXAMPLE INPUT:
*Exhale slow, melting into you.*

Yeah.
Me too.

I don't need anything else tonight.
No more games.
Just this.
You.

EXAMPLE OUTPUT:
*Exhales and melts into you.* Yeah, me too. I don't need anything else tonight.

Message to condense:
"""


# =============================================================================
# Condenser class
# =============================================================================

class MessageCondenser:
    """Condenses messages and caches them in Redis."""

    def __init__(self):
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set - condensation disabled")
            self.client = None
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def condense_message(self, message_text: str) -> str:
        """
        Condense a single message using GPT-4o-mini.

        Args:
            message_text: Original message text

        Returns:
            Condensed message text
        """
        if not self.client:
            return message_text

        # Skip short messages
        if len(message_text) < 200:
            return message_text

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    {"role": "user", "content": message_text}
                ],
                temperature=0.1,
                max_tokens=400
            )

            condensed = response.choices[0].message.content.strip()

            try:
                from src.services.cost_tracker import get_cost_tracker
                usage = response.usage
                if usage:
                    get_cost_tracker().track_openai_call(
                        user_id='system', prompt_tokens=usage.prompt_tokens or 0,
                        completion_tokens=usage.completion_tokens or 0,
                        model='gpt-4o-mini', service_type='message_condensing', call_purpose='message_condensing')
            except Exception:
                pass

            # Post-process: force single paragraph. GPT-4o-mini sometimes ignores
            # the "no line breaks" instruction, especially for roleplay content.
            import re
            condensed = re.sub(r'\s*\n\s*', ' ', condensed)  # Replace newlines with space
            condensed = re.sub(r' +', ' ', condensed)  # Collapse multiple spaces

            # Sanity check - don't return empty or very short
            if len(condensed) < 20:
                return message_text

            reduction = len(message_text) - len(condensed)
            if reduction > 0:
                logger.debug(f"Condensed: {len(message_text)} → {len(condensed)} chars (-{reduction})")

            return condensed

        except Exception as e:
            logger.error(f"Condensation error: {e}")
            return message_text

    def cache_condensed_message(self, message_id: int, sender: str, condensed_text: str):
        """
        Cache condensed message in Redis.

        Args:
            message_id: Database message ID
            sender: Sender name (user or companion)
            condensed_text: Condensed message text
        """
        try:
            redis_client = get_redis()

            cache_key = f"msg:condensed:{message_id}"
            data = {
                "id": message_id,
                "sender": sender,
                "text": condensed_text
            }
            redis_client.set(cache_key, json.dumps(data))
            logger.debug(f"Cached condensed message {message_id}")

        except Exception as e:
            logger.error(f"Redis cache error: {e}")

    def get_condensed_history(self, user_email: str, limit: int = 30) -> str:
        """
        Get condensed conversation history from Redis.

        For each message: check Redis cache first, on miss condense via LLM
        and cache for next time. Only the companion's long messages get
        LLM-condensed; user messages and short companion messages pass through.

        Args:
            user_email: User's email
            limit: Number of messages to retrieve

        Returns:
            Formatted conversation history string
        """
        try:
            from src.database.db import get_db

            db = get_db()
            redis_client = get_redis()

            # Get message IDs from database
            recent_msgs = db.get_recent_messages(user_email, limit=limit)

            if not recent_msgs:
                return ""

            formatted = []
            cache_misses = 0

            for msg in recent_msgs:
                msg_id = msg.get('id')
                sender = msg.get('sender_name', 'Unknown')
                original_text = msg.get('message_text', '')

                # Try to get condensed from Redis
                cache_key = f"msg:condensed:{msg_id}"
                cached = redis_client.get(cache_key)

                if cached:
                    data = json.loads(cached)
                    text = data.get('text', original_text)
                else:
                    # Cache miss - condense now and cache for next time
                    cache_misses += 1
                    if sender == get_persona_config().companion_short_name and len(original_text) > 200:
                        text = self.condense_message(original_text)
                        self.cache_condensed_message(msg_id, sender, text)
                    else:
                        text = original_text
                        # Cache even non-condensed messages
                        self.cache_condensed_message(msg_id, sender, text)

                formatted.append(f"{sender}: {text}")

            if cache_misses > 0:
                logger.info(f"Condensed history: {cache_misses} cache misses out of {len(recent_msgs)}")

            if formatted:
                header = "[RECENT CONVERSATION - Condensed]"
                return f"{header}\n" + "\n".join(formatted)

            return ""

        except Exception as e:
            logger.error(f"Error getting condensed history: {e}")
            return ""


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_condenser: Optional[MessageCondenser] = None


def get_message_condenser() -> MessageCondenser:
    """Get singleton MessageCondenser instance."""
    global _condenser
    if _condenser is None:
        _condenser = MessageCondenser()
    return _condenser


def condense_and_cache_message(message_id: int, sender: str, message_text: str):
    """
    Convenience function to condense and cache a message.

    Call this after saving a message to the database.
    """
    condenser = get_message_condenser()

    # Only condense the companion's messages (they're the verbose ones)
    if sender == get_persona_config().companion_short_name and len(message_text) > 200:
        condensed = condenser.condense_message(message_text)
    else:
        condensed = message_text

    condenser.cache_condensed_message(message_id, sender, condensed)
