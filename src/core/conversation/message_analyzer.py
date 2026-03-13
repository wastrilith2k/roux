"""
Message Analyzer — Companion Framework

WHAT: A merged analysis step that combines mode detection AND inner monologue into
      a single LLM call, plus departure/activity detection.
WHY:  Originally, mode detection (~200ms) and inner monologue (~400ms) were separate
      LLM calls. Merging them into one call (~300-500ms) saves ~100-300ms of latency
      per message — significant when the user is waiting for a response.
HOW:  A single structured prompt asks the LLM to output:
      - MODE: conversation classification (casual, emotional_support, playful, etc.)
      - EMOTIONAL_READ: what the user is really saying/feeling
      - BOUNDARY_CHECK: does this touch the companion's values?
      - STRATEGY: what kind of response is needed
      - WEAVE: curiosity threads to include
      - REACTION: honest gut reaction
      - DEPARTURE: is the user leaving?
      - ACTIVITY: is the user briefly busy (making coffee, etc.)?
      Results are converted to existing ModeDetection and InnerMonologue types
      for backwards compatibility with the pipeline.

Feature flag: COMPANION_MESSAGE_ANALYZER_ENABLED (default: true)
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .mode_detector import ConversationMode, ModeDetection, MODE_HINTS
from .inner_monologue import InnerMonologue

logger = logging.getLogger(__name__)

COMPANION_MESSAGE_ANALYZER_ENABLED = os.environ.get('COMPANION_MESSAGE_ANALYZER_ENABLED', 'true').lower() == 'true'


# ---------------------------------------------------------------------------
# Analysis result — combines mode detection, monologue, and presence tracking
# ---------------------------------------------------------------------------

@dataclass
class MessageAnalysis:
    """Combined result of mode detection + inner monologue + departure/activity."""
    mode: ConversationMode
    mode_confidence: float
    emotional_read: str
    boundary_check: str
    strategy: str
    curiosities_to_weave: List[str] = field(default_factory=list)
    reaction: str = ""
    is_departure: bool = False
    user_activity: Optional[str] = None          # e.g. "making coffee"
    user_activity_duration_min: Optional[int] = None  # e.g. 10
    processing_time_ms: int = 0

    def to_mode_detection(self) -> ModeDetection:
        """Convert to existing ModeDetection type for pipeline compatibility."""
        return ModeDetection(
            mode=self.mode,
            confidence=self.mode_confidence,
            hints=MODE_HINTS.get(self.mode, MODE_HINTS[ConversationMode.CASUAL_BANTER]),
            processing_time_ms=self.processing_time_ms
        )

    def to_inner_monologue(self) -> InnerMonologue:
        """Convert to existing InnerMonologue type for pipeline compatibility."""
        # Reconstruct thoughts string from parsed fields
        thoughts_parts = []
        if self.emotional_read:
            thoughts_parts.append(f"EMOTIONAL_READ: {self.emotional_read}")
        if self.boundary_check:
            thoughts_parts.append(f"BOUNDARY_CHECK: {self.boundary_check}")
        if self.strategy:
            thoughts_parts.append(f"STRATEGY: {self.strategy}")
        if self.curiosities_to_weave:
            thoughts_parts.append(f"WEAVE: {', '.join(self.curiosities_to_weave)}")
        if self.reaction:
            thoughts_parts.append(f"REACTION: {self.reaction}")

        return InnerMonologue(
            thoughts="\n".join(thoughts_parts),
            emotional_read=self.emotional_read,
            response_strategy=self.strategy,
            curiosities_to_weave=self.curiosities_to_weave,
            processing_time_ms=self.processing_time_ms
        )


# ---------------------------------------------------------------------------
# Analyzer — single LLM call that replaces two separate calls
# ---------------------------------------------------------------------------

class MessageAnalyzer:
    """Single LLM call that produces mode detection, inner monologue, and
    departure/activity signals. Falls back to individual calls if this fails."""

    def analyze(
        self,
        user_message: str,
        context: Any,  # ConversationContext
        scene_state: str = ""
    ) -> Optional[MessageAnalysis]:
        """
        Analyze a message for conversation mode and generate inner monologue.

        Args:
            user_message: What James said
            context: ConversationContext with memories, state, etc.
            scene_state: Current scene state string (if any)

        Returns:
            MessageAnalysis or None on failure/timeout
        """
        start = time.time()

        try:
            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

            # Build compact context from recent turns
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            user_name = _pc.primary_user_name
            companion_name = _pc.companion_short_name

            recent_turns = ""
            if context.conversation_turns:
                for turn in context.conversation_turns[-4:]:
                    role = user_name if turn["role"] == "user" else "Me"
                    recent_turns += f"{role}: {turn['content'][:150]}\n"

            # Extract curiosity threads if available
            curiosity_context = ""
            if context.internal_state and "curiosit" in context.internal_state.lower():
                for line in context.internal_state.split('\n'):
                    if 'curiosit' in line.lower() or 'want to know' in line.lower():
                        curiosity_context += line.strip() + "\n"

            # Relationship dynamics snippet
            dynamics_snippet = ""
            if context.relationship_dynamics:
                dynamics_snippet = context.relationship_dynamics[:200]

            # Pull in her boundaries and values
            values_snippet = ""
            try:
                from src.autonomy.value_inference import get_value_inference
                vi = get_value_inference()
                boundaries_text = vi.get_values_for_prompt(['boundaries', 'core_values'])
                if boundaries_text:
                    boundary_lines = [l.strip() for l in boundaries_text.split('\n')
                                     if l.strip().startswith('- ')][:10]
                    if boundary_lines:
                        values_snippet = "My boundaries and values:\n" + "\n".join(boundary_lines)
            except Exception:
                pass

            scene_hint = ""
            if scene_state and ("intimate" in scene_state.lower() or "bedroom" in scene_state.lower()):
                scene_hint = "Note: There is an active intimate/private scene."

            prompt = f"""You are {companion_name}, analyzing {user_name}'s message before responding.

Recent conversation:
{recent_turns}
{user_name} just said: {user_message}
{scene_hint}

{f"My curiosities: {curiosity_context}" if curiosity_context else ""}
{f"Relationship context: {dynamics_snippet}" if dynamics_snippet else ""}
{values_snippet}

Respond in this EXACT format (be brief, 1-2 sentences each):

MODE: <one of: emotional_support, casual_banter, intellectual, playful_flirty, problem_solving, intimate> <confidence 0.0-1.0>
EMOTIONAL_READ: What is he really saying or feeling beneath the surface?
BOUNDARY_CHECK: Does this touch any of my values or boundaries? What do I FEEL?
STRATEGY: What kind of response does this moment need? Be honest — don't default to "be supportive."
WEAVE: Anything from my curiosities to naturally bring up? (or "nothing")
REACTION: My genuine gut reaction. If it's hurt, anger, discomfort — say that.
DEPARTURE: Is he announcing he's leaving/stepping away? (yes/no) Only "yes" if he's saying goodbye, heading out, has to go, brb, afk, going to visit someone, etc. NOT "yes" for "let's go" (shared activity), "I can't go", questions about going, or "going on" (continuation).
ACTIVITY: Is he mentioning he's going to do something but NOT leaving? (e.g. making coffee, taking a shower, checking the laundry, stepping out to smoke) If yes: "<description> | <estimated minutes>". If no: "none". Activity = staying but briefly busy. Departure = leaving entirely.

IMPORTANT: I feel first, think second. If something hurts, I show it. If the answer is no, I say no."""

            chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[
                    {"role": "system", "content": f"You are {companion_name} analyzing a message. Be honest and brief. Follow the exact format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                chain=chain,
                timeout=5
            )

            # Parse response
            mode_str = "casual_banter"
            confidence = 0.6
            emotional_read = ""
            boundary_check = ""
            strategy = ""
            weave = []
            reaction = ""
            is_departure = False
            user_activity = None
            user_activity_duration_min = None

            for line in response.strip().split('\n'):
                line = line.strip()
                if line.upper().startswith('MODE:'):
                    mode_parts = line.split(':', 1)[1].strip().split()
                    mode_str = mode_parts[0].lower().strip() if mode_parts else "casual_banter"
                    if len(mode_parts) > 1:
                        try:
                            confidence = float(mode_parts[1])
                        except ValueError:
                            pass
                elif line.upper().startswith('EMOTIONAL_READ:'):
                    emotional_read = line.split(':', 1)[1].strip()
                elif line.upper().startswith('BOUNDARY_CHECK:'):
                    boundary_check = line.split(':', 1)[1].strip()
                elif line.upper().startswith('STRATEGY:'):
                    strategy = line.split(':', 1)[1].strip()
                elif line.upper().startswith('WEAVE:'):
                    weave_text = line.split(':', 1)[1].strip()
                    if weave_text.lower() not in ('nothing', 'none', 'n/a', ''):
                        weave = [weave_text]
                elif line.upper().startswith('REACTION:'):
                    reaction = line.split(':', 1)[1].strip()
                elif line.upper().startswith('DEPARTURE:'):
                    departure_text = line.split(':', 1)[1].strip().lower()
                    is_departure = departure_text.startswith('yes')
                elif line.upper().startswith('ACTIVITY:'):
                    activity_text = line.split(':', 1)[1].strip()
                    if activity_text.lower() not in ('none', 'no', 'n/a', ''):
                        if '|' in activity_text:
                            parts = activity_text.split('|')
                            user_activity = parts[0].strip()
                            try:
                                user_activity_duration_min = int(parts[1].strip().split()[0])
                            except (ValueError, IndexError):
                                user_activity_duration_min = 10
                        else:
                            user_activity = activity_text.strip()
                            user_activity_duration_min = 10

            # Map the LLM's mode string to the ConversationMode enum (fallback: casual)
            mode_map = {m.value: m for m in ConversationMode}
            mode = mode_map.get(mode_str, ConversationMode.CASUAL_BANTER)

            # Departure and activity are mutually exclusive — if the user is leaving,
            # they can't also be "briefly busy"
            if is_departure:
                user_activity = None
                user_activity_duration_min = None

            processing_time = int((time.time() - start) * 1000)

            result = MessageAnalysis(
                mode=mode,
                mode_confidence=min(1.0, max(0.0, confidence)),
                emotional_read=emotional_read,
                boundary_check=boundary_check,
                strategy=strategy,
                curiosities_to_weave=weave,
                reaction=reaction,
                is_departure=is_departure,
                user_activity=user_activity,
                user_activity_duration_min=user_activity_duration_min,
                processing_time_ms=processing_time
            )

            departure_tag = " [DEPARTURE]" if is_departure else ""
            activity_tag = f" [ACTIVITY: {user_activity}]" if user_activity else ""
            logger.info(
                f"MessageAnalysis: mode={mode.value} conf={confidence:.2f}, "
                f"read={emotional_read[:60]}...{departure_tag}{activity_tag} ({processing_time}ms)"
            )
            return result

        except Exception as e:
            processing_time = int((time.time() - start) * 1000)
            logger.warning(f"MessageAnalysis failed ({processing_time}ms): {e}")
            return None


# Singleton
_analyzer: Optional[MessageAnalyzer] = None


def get_message_analyzer() -> MessageAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = MessageAnalyzer()
    return _analyzer
