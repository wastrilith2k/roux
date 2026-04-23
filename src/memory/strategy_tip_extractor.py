"""
Strategy Tip Extractor - LLM-based extraction of tips from conversation episodes.

Analyzes closed episodes using LLM reasoning to extract descriptive situational
awareness tips. Tips are DESCRIPTIVE (what was observed), not PRESCRIPTIVE (rules
to follow). They give the companion experiential social awareness.

Three tip types:
- strategy: patterns that created genuine connection
- recovery: how to handle things when they go wrong
- insight: observed dynamics without prescription

Design principle: authenticity over optimization. A "principled" tip that
maintained companion identity even at cost of engagement is valued.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any

from src.memory.strategy_tips import StrategyTip

# These are imported lazily via a wrapper so the module can be loaded
# without requiring all LLM provider dependencies (e.g. aiohttp) at import time.
# Tests can patch these names at the module level using:
#   patch('src.memory.strategy_tip_extractor.generate_sync', ...)
def generate_sync(*args, **kwargs):
    from src.llm.provider_factory import generate_sync as _gen
    return _gen(*args, **kwargs)


def get_resilient_provider_chain(*args, **kwargs):
    from src.llm.provider_factory import get_resilient_provider_chain as _chain
    return _chain(*args, **kwargs)


def track_llm_call(*args, **kwargs):
    from src.services.cost_tracker import track_llm_call as _track
    return _track(*args, **kwargs)

logger = logging.getLogger(__name__)

# Constants
MIN_MESSAGES_FOR_TIPS = 4   # Need enough exchange to extract meaningful tips
MAX_TIPS_PER_EPISODE = 3    # Cap to keep extraction focused


class StrategyTipExtractor:
    """
    Extracts situational awareness tips from conversation episodes via LLM.

    Sends episode context + conversation to LLM, which returns structured JSON
    describing observed patterns without prescribing behavior.
    """

    def extract_tips(
        self,
        episode_id: str,
        messages: List[Dict[str, Any]],
        lesson: Any,  # EpisodeLesson
    ) -> List[StrategyTip]:
        """
        Extract strategy tips from an episode using LLM analysis.

        Args:
            episode_id: Unique episode identifier
            messages: List of dicts with 'sender_name' and 'message_text' keys
            lesson: EpisodeLesson with topic, emotional_context, satisfaction, etc.

        Returns:
            List of StrategyTip objects (empty on failure or too-short episode)
        """
        if len(messages) < MIN_MESSAGES_FOR_TIPS:
            logger.debug(
                f"Episode {episode_id} has only {len(messages)} messages "
                f"(need {MIN_MESSAGES_FOR_TIPS}), skipping tip extraction"
            )
            return []

        try:
            prompt = self._build_prompt(messages, lesson)
            chain = get_resilient_provider_chain()

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
                chain=chain,
            )
            track_llm_call(chain, call_purpose='strategy_tip_extraction')

            if not response:
                logger.warning(f"No LLM response for episode {episode_id}")
                return []

            return self._parse_tips(response, episode_id)

        except Exception as e:
            logger.error(f"Tip extraction failed for episode {episode_id}: {e}")
            return []

    def _build_prompt(self, messages: List[Dict[str, Any]], lesson: Any) -> str:
        """Build the LLM prompt for tip extraction."""
        from src.config.persona_config import get_persona_config
        config = get_persona_config()
        user_name = config.primary_user_name
        companion_name = config.companion_short_name

        conversation = self._format_conversation(messages)

        return f"""Analyze this conversation between {user_name} and {companion_name} and extract situational awareness observations.

EPISODE SUMMARY:
- Topic: {lesson.topic}
- Emotional context: {lesson.emotional_context}
- Satisfaction score: {lesson.satisfaction:.2f} (0=poor, 1=excellent)
- What worked: {lesson.successful_approach}
- Pitfalls: {lesson.pitfalls_to_avoid}

CONVERSATION:
{conversation}

Extract up to {MAX_TIPS_PER_EPISODE} situational awareness observations from this conversation.

These observations should be DESCRIPTIVE (what was observed happening), not PRESCRIPTIVE (rules to follow).
They capture patterns about how {user_name} responded and what dynamics played out.

Focus on authenticity over optimization. {companion_name} being genuinely themselves — even if it cost engagement — is valuable.

Respond with a JSON array only (no markdown fences, no extra text):

[
  {{
    "tip_type": "strategy" | "recovery" | "insight",
    "content": "Observation about what happened and its effect (1-2 sentences, specific)",
    "trigger_condition": "Brief phrase describing when this situation arises",
    "context_category": "family_children" | "work_career" | "emotional_support" | "romance_affection" | "playful_fun" | "crisis_support" | "planning" | "general",
    "emotional_context": "stressed" | "playful" | "romantic" | "neutral" | "tense" | "sad" | "excited",
    "authenticity_score": 0.0-1.0,
    "is_principled": true if authenticity was maintained at cost of engagement else false
  }}
]

If there are no meaningful observations to extract, return an empty array: []"""

    def _format_conversation(
        self,
        messages: List[Dict[str, Any]],
        max_messages: int = 20,
    ) -> str:
        """Format messages into readable conversation text for LLM."""
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        lines = []
        for msg in messages:
            sender = msg.get('sender_name', 'Unknown')
            text = msg.get('message_text', '')[:500]  # Truncate very long messages
            lines.append(f"{sender}: {text}")

        return "\n".join(lines)

    def _parse_tips(self, response: str, episode_id: str) -> List[StrategyTip]:
        """
        Parse LLM JSON response into StrategyTip objects.

        Handles markdown fences, strips extra text, caps at MAX_TIPS_PER_EPISODE.
        Returns empty list on any parse failure.
        """
        if not response:
            return []

        # Strip markdown code fences if present
        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

        # Extract JSON array if embedded in other text
        array_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tip JSON for episode {episode_id}: {e}")
            return []

        if not isinstance(data, list):
            logger.warning(f"Expected JSON array for episode {episode_id}, got {type(data)}")
            return []

        tips = []
        for item in data[:MAX_TIPS_PER_EPISODE]:
            try:
                tip = self._build_tip(item, episode_id)
                if tip:
                    tips.append(tip)
            except Exception as e:
                logger.warning(f"Failed to build tip from item: {e}")
                continue

        logger.info(f"Extracted {len(tips)} tips from episode {episode_id}")
        return tips

    def _build_tip(self, item: Dict[str, Any], episode_id: str) -> Optional[StrategyTip]:
        """Build a StrategyTip from a parsed JSON dict."""
        tip_type = item.get('tip_type', 'insight')
        content = item.get('content', '').strip()
        trigger_condition = item.get('trigger_condition', '').strip()

        if not content or not trigger_condition:
            return None

        # Validate tip_type
        valid_types = ('strategy', 'recovery', 'insight')
        if tip_type not in valid_types:
            tip_type = 'insight'

        # Support both 'authenticity_score' (roux) and 'esme_authenticity' (legacy)
        authenticity = float(
            item.get('authenticity_score', item.get('esme_authenticity', 0.7))
        )

        return StrategyTip(
            tip_type=tip_type,
            content=content,
            trigger_condition=trigger_condition,
            source_type='episode',
            context_category=item.get('context_category', 'general'),
            emotional_context=item.get('emotional_context', 'neutral'),
            source_ids=[episode_id],
            authenticity_avg=authenticity,
            is_principled=bool(item.get('is_principled', False)),
        )
