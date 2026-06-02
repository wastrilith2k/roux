"""
Response Self-Critique — Companion Framework

WHAT: Post-generation quality checker that evaluates whether the companion's response
      sounds authentic, addresses the user's message, and avoids repetition.
WHY:  LLMs sometimes produce generic, off-topic, or repetitive responses. The critic
      catches these before they reach the user. If quality is below threshold, the
      pipeline regenerates with improvement hints.
HOW:  A fast LLM call (~100-300ms, tight 3s timeout) scores the response on:
      - Authenticity (does it sound like this specific companion, not a chatbot?)
      - Relevance (does it actually address what the user said?)
      - Emotional register (is the tone appropriate for the user's message?)
      - Repetition (is it reusing phrases from recent messages?)
      If the score is below COMPANION_QUALITY_THRESHOLD, the pipeline gets
      format_quality_hints() to append to the prompt and retries once.

Feature flags:
  COMPANION_RESPONSE_CRITIC_ENABLED (default: true) — master toggle
  COMPANION_QUALITY_THRESHOLD (default: 4) — scores below this trigger regeneration
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

COMPANION_RESPONSE_CRITIC_ENABLED = os.environ.get('COMPANION_RESPONSE_CRITIC_ENABLED', 'true').lower() == 'true'
COMPANION_QUALITY_THRESHOLD = int(os.environ.get('COMPANION_QUALITY_THRESHOLD', '4'))

# Module-level imports — lazy-safe with fallbacks so tests can patch these names
try:
    from src.llm.provider_factory import generate_sync, get_resilient_provider_chain
except ImportError:
    generate_sync = None  # type: ignore[assignment]
    get_resilient_provider_chain = None  # type: ignore[assignment]

try:
    from src.memory.strategy_tips import get_strategy_tip_store
except ImportError:
    get_strategy_tip_store = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Critique result
# ---------------------------------------------------------------------------

@dataclass
class QualityCritique:
    score: int                     # 1-10 overall quality
    is_generic: bool               # Sounds like a generic chatbot?
    addresses_message: bool        # Actually responds to what user said?
    emotional_register_match: bool # Tone matches user's emotional state?
    is_repetitive: bool            # Reuses phrases from recent messages?
    suggestions: List[str] = field(default_factory=list)
    should_regenerate: bool = False
    processing_time_ms: int = 0


# ---------------------------------------------------------------------------
# Strategy tips helper
# ---------------------------------------------------------------------------

def _get_strategy_tips(user_email: str, message: str, limit: int = 3) -> list:
    """Fetch relevant strategy tips for the user. Returns empty list on any failure."""
    try:
        store = get_strategy_tip_store(user_email=user_email)
        return store.retrieve_relevant_tips(query_text=message, limit=limit)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Critic — evaluates response quality via a fast LLM call
# ---------------------------------------------------------------------------

class ResponseCritic:
    """Evaluates response quality and personality authenticity."""

    def critique(
        self,
        response: str,
        user_message: str,
        recent_companion_messages: List[str] = None,
        mode: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> QualityCritique:
        """Score a generated response for quality. Returns passing defaults on failure."""
        start = time.time()

        try:
            # Resolve LLM callables — prefer module-level imports, fall back to lazy import
            _generate_sync = generate_sync
            _get_chain = get_resilient_provider_chain
            if _generate_sync is None or _get_chain is None:
                from src.llm.provider_factory import generate_sync as _gs, get_resilient_provider_chain as _gc
                _generate_sync = _gs
                _get_chain = _gc

            # Build recent messages context for repetition detection
            recent_context = ""
            if recent_companion_messages:
                for i, msg in enumerate(recent_companion_messages[:3]):
                    recent_context += f"Companion's message {i+1} ago: {msg[:150]}\n"

            # Fetch strategy tips for this user (procedural memory)
            tips_section = ''
            if user_email:
                tips = _get_strategy_tips(user_email=user_email, message=user_message)
                if tips:
                    tip_lines = '\n'.join(f"- {t.content}" for t in tips)
                    tips_section = f"\nKnown communication patterns for this user:\n{tip_lines}\n"

            prompt = f"""Evaluate this response from the companion (a person, not an AI) to the user.

User said: {user_message}

Companion's response: {response}

{f"Conversation mode: {mode}" if mode else ""}
{f"Companion's recent messages for repetition check:\\n{recent_context}" if recent_context else ""}{tips_section}
Rate 1-10 and answer yes/no for each:
1. SCORE: Overall quality (1=terrible, 10=perfect)
2. GENERIC: Does this sound like a generic chatbot? (yes/no)
3. ADDRESSES: Does she actually address what the user said? (yes/no)
4. REGISTER: Is the emotional tone right for what the user said? (yes/no)
5. REPETITIVE: Is she repeating phrases from her recent messages? (yes/no)
6. SUGGESTIONS: If score < 6, one brief suggestion to improve

Format:
SCORE: N
GENERIC: yes/no
ADDRESSES: yes/no
REGISTER: yes/no
REPETITIVE: yes/no
SUGGESTION: ..."""

            chain = _get_chain()
            result = _generate_sync(
                messages=[
                    {"role": "system", "content": "Evaluate response quality. Be brief and honest."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150,
                chain=chain,
                timeout=3
            )

            # Parse structured response
            score = 7  # Default to passing if parsing fails
            is_generic = False
            addresses = True
            register_match = True
            is_repetitive = False
            suggestions = []

            for line in result.strip().split('\n'):
                line = line.strip()
                upper = line.upper()
                if upper.startswith('SCORE:'):
                    try:
                        score = int(line.split(':')[1].strip().split()[0])
                        score = max(1, min(10, score))
                    except (ValueError, IndexError):
                        pass
                elif upper.startswith('GENERIC:'):
                    is_generic = 'yes' in line.lower()
                elif upper.startswith('ADDRESSES:'):
                    addresses = 'yes' in line.lower()
                elif upper.startswith('REGISTER:'):
                    register_match = 'yes' in line.lower()
                elif upper.startswith('REPETITIVE:'):
                    is_repetitive = 'yes' in line.lower()
                elif upper.startswith('SUGGESTION:'):
                    suggestion = line.split(':', 1)[1].strip()
                    if suggestion and suggestion.lower() not in ('none', 'n/a', ''):
                        suggestions.append(suggestion)

            processing_time = int((time.time() - start) * 1000)
            should_regenerate = score < COMPANION_QUALITY_THRESHOLD

            critique = QualityCritique(
                score=score,
                is_generic=is_generic,
                addresses_message=addresses,
                emotional_register_match=register_match,
                is_repetitive=is_repetitive,
                suggestions=suggestions,
                should_regenerate=should_regenerate,
                processing_time_ms=processing_time
            )

            if should_regenerate:
                logger.warning(
                    f"Response critique: score={score} (threshold={COMPANION_QUALITY_THRESHOLD}) - "
                    f"REGENERATION TRIGGERED. Suggestions: {suggestions}"
                )
            else:
                logger.info(f"Response critique: score={score} ({processing_time}ms)")

            return critique

        except Exception as e:
            processing_time = int((time.time() - start) * 1000)
            logger.warning(f"Response critique failed ({processing_time}ms): {e}")
            # Return passing critique on failure — don't block the response
            return QualityCritique(
                score=7,
                is_generic=False,
                addresses_message=True,
                emotional_register_match=True,
                is_repetitive=False,
                should_regenerate=False,
                processing_time_ms=processing_time
            )


# ---------------------------------------------------------------------------
# Quality hints — appended to prompt when regeneration is triggered
# ---------------------------------------------------------------------------

def format_quality_hints(critique: QualityCritique) -> str:
    """Format critique suggestions as regeneration hints for the LLM.

    These are appended to the system prompt before the retry call,
    giving the LLM specific guidance on what to improve.
    """
    hints = ["\n\n[QUALITY IMPROVEMENT - Your previous response scored low. Please improve:]"]

    if critique.is_generic:
        hints.append("- Sound more like yourself, not a generic chatbot")
    if not critique.addresses_message:
        hints.append("- Actually address what the user said instead of deflecting")
    if not critique.emotional_register_match:
        hints.append("- Match the emotional tone of what the user shared")
    if critique.is_repetitive:
        hints.append("- Avoid repeating phrases you've used recently")
    for suggestion in critique.suggestions:
        hints.append(f"- {suggestion}")

    return '\n'.join(hints)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_critic: Optional[ResponseCritic] = None


def get_response_critic() -> ResponseCritic:
    global _critic
    if _critic is None:
        _critic = ResponseCritic()
    return _critic
