"""
Conversation Mode Detection - Classifies the current conversation mode.

Detects whether the conversation is emotional support, casual banter,
intellectual discussion, playful/flirty, problem-solving, or intimate.
This informs temperature, response length, and curiosity injection.

Feature flag: COMPANION_MODE_DETECTION_ENABLED (default: true)
"""

import os
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

COMPANION_MODE_DETECTION_ENABLED = os.environ.get('COMPANION_MODE_DETECTION_ENABLED', 'true').lower() == 'true'


class ConversationMode(Enum):
    EMOTIONAL_SUPPORT = "emotional_support"
    CASUAL_BANTER = "casual_banter"
    INTELLECTUAL = "intellectual"
    PLAYFUL_FLIRTY = "playful_flirty"
    PROBLEM_SOLVING = "problem_solving"
    INTIMATE = "intimate"


# Mode-specific hints for the pipeline
MODE_HINTS = {
    ConversationMode.EMOTIONAL_SUPPORT: {
        'temp_adjustment': 0.05,   # Slightly warmer
        'length_hint': 'medium',   # Thoughtful but not overwhelming
        'curiosity_injection': False,  # Don't change subject
    },
    ConversationMode.CASUAL_BANTER: {
        'temp_adjustment': 0.0,
        'length_hint': 'short',
        'curiosity_injection': True,  # Good time to weave in curiosities
    },
    ConversationMode.INTELLECTUAL: {
        'temp_adjustment': -0.05,  # Slightly more focused
        'length_hint': 'long',
        'curiosity_injection': True,
    },
    ConversationMode.PLAYFUL_FLIRTY: {
        'temp_adjustment': 0.0,
        'length_hint': 'short',
        'curiosity_injection': False,
    },
    ConversationMode.PROBLEM_SOLVING: {
        'temp_adjustment': -0.05,
        'length_hint': 'medium',
        'curiosity_injection': False,
    },
    ConversationMode.INTIMATE: {
        'temp_adjustment': 0.05,
        'length_hint': 'medium',
        'curiosity_injection': False,
    },
}


@dataclass
class ModeDetection:
    mode: ConversationMode
    confidence: float  # 0.0 - 1.0
    hints: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0


class ConversationModeDetector:
    """Detects the current conversation mode using a fast LLM call."""

    def detect(
        self,
        user_message: str,
        recent_turns: List[Dict[str, str]] = None,
        scene_state: str = ""
    ) -> ModeDetection:
        """
        Classify the current conversation mode.

        Args:
            user_message: The current user message
            recent_turns: Last few conversation turns [{"role": "user/assistant", "content": "..."}]
            scene_state: Current scene state string (if any)

        Returns:
            ModeDetection with mode, confidence, and hints
        """
        start = time.time()

        try:
            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

            # Build context from recent turns
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            turn_context = ""
            if recent_turns:
                for turn in recent_turns[-4:]:
                    role = _pc.primary_user_name if turn["role"] == "user" else _pc.companion_short_name
                    turn_context += f"{role}: {turn['content'][:150]}\n"

            scene_hint = ""
            if scene_state and ("intimate" in scene_state.lower() or "bedroom" in scene_state.lower()):
                scene_hint = "Note: There is an active intimate/private scene."

            prompt = f"""Classify this conversation's current mode. Recent context:

{turn_context}
Current message from {_pc.primary_user_name}: {user_message}
{scene_hint}

Respond with ONLY one word from: emotional_support, casual_banter, intellectual, playful_flirty, problem_solving, intimate

Then a space and a confidence score 0.0-1.0.

Example: casual_banter 0.85"""

            chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[
                    {"role": "system", "content": "Classify conversation mode. Reply with ONLY the mode and confidence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=20,
                chain=chain,
                timeout=3
            )

            # Parse response
            parts = response.strip().split()
            mode_str = parts[0].lower().strip() if parts else "casual_banter"
            confidence = float(parts[1]) if len(parts) > 1 else 0.6

            # Map to enum
            mode_map = {m.value: m for m in ConversationMode}
            mode = mode_map.get(mode_str, ConversationMode.CASUAL_BANTER)

            processing_time = int((time.time() - start) * 1000)

            result = ModeDetection(
                mode=mode,
                confidence=min(1.0, max(0.0, confidence)),
                hints=MODE_HINTS.get(mode, MODE_HINTS[ConversationMode.CASUAL_BANTER]),
                processing_time_ms=processing_time
            )

            logger.info(f"Mode detection: {mode.value} (confidence={confidence:.2f}, {processing_time}ms)")
            return result

        except Exception as e:
            processing_time = int((time.time() - start) * 1000)
            logger.warning(f"Mode detection failed ({processing_time}ms): {e}")
            return ModeDetection(
                mode=ConversationMode.CASUAL_BANTER,
                confidence=0.3,
                hints=MODE_HINTS[ConversationMode.CASUAL_BANTER],
                processing_time_ms=processing_time
            )


# Singleton
_detector: Optional[ConversationModeDetector] = None


def get_mode_detector() -> ConversationModeDetector:
    global _detector
    if _detector is None:
        _detector = ConversationModeDetector()
    return _detector
