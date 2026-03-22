"""
Inner Monologue — Companion Framework

WHAT: Generates the companion's private pre-response reasoning — a "thinking out loud"
      step that happens before the actual response is written.
WHY:  Without this, the companion responds reactively. The monologue forces the LLM to
      first consider: what is the user really feeling? Does this cross a boundary?
      What kind of response does this moment need? This produces more emotionally
      intelligent and authentic responses, especially for sensitive topics.
HOW:  A fast LLM call (~300-500ms) with a structured prompt that asks for:
      1. Emotional read (what's beneath the surface)
      2. Boundary check (does this touch a value or limit)
      3. Response strategy (what kind of reply is needed)
      4. Weave-in (curiosity threads to naturally include)
      5. Gut reaction (honest emotional response)
      The parsed result is injected into the system prompt as <inner_thoughts>.

NOTE: This module is largely superseded by MessageAnalyzer (message_analyzer.py),
      which merges mode detection + inner monologue into a single LLM call. This
      module serves as a fallback when the merged analyzer fails.

Feature flag: COMPANION_INNER_MONOLOGUE_ENABLED (default: true)
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Any

logger = logging.getLogger(__name__)

COMPANION_INNER_MONOLOGUE_ENABLED = os.environ.get('COMPANION_INNER_MONOLOGUE_ENABLED', 'true').lower() == 'true'


# ---------------------------------------------------------------------------
# Data structure for the monologue output
# ---------------------------------------------------------------------------

@dataclass
class InnerMonologue:
    thoughts: str                  # Full free-form private reasoning text
    emotional_read: str            # What the companion reads from the user's message
    response_strategy: str         # How the companion plans to respond
    curiosities_to_weave: List[str] = field(default_factory=list)  # Topics to naturally include
    processing_time_ms: int = 0


# ---------------------------------------------------------------------------
# Generator — produces the monologue via a fast LLM call
# ---------------------------------------------------------------------------

class InnerMonologueGenerator:
    """Generates the companion's private pre-response reasoning."""

    def generate(
        self,
        user_message: str,
        context: Any,  # ConversationContext
        mode: Any = None  # ModeDetection
    ) -> Optional[InnerMonologue]:
        """Generate inner monologue before responding.

        Returns InnerMonologue on success, None on failure/timeout.
        Uses a 5-second timeout to avoid blocking the response pipeline.
        """
        start = time.time()

        try:
            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain
            from src.config.persona_config import get_persona_config

            _pc = get_persona_config()
            user_name = _pc.primary_user_name
            companion_name = _pc.companion_short_name

            # Build compact context: only the last 3 turns to keep the prompt small
            recent_turns = ""
            if context.conversation_turns:
                for turn in context.conversation_turns[-3:]:
                    role = user_name if turn["role"] == "user" else "Me"
                    recent_turns += f"{role}: {turn['content'][:120]}\n"

            # Extract curiosity threads from internal state (if present)
            curiosity_context = ""
            if context.internal_state and "curiosit" in context.internal_state.lower():
                for line in context.internal_state.split('\n'):
                    if 'curiosit' in line.lower() or 'want to know' in line.lower():
                        curiosity_context += line.strip() + "\n"

            # Relationship dynamics snippet (truncated for token budget)
            dynamics_snippet = ""
            if context.relationship_dynamics:
                dynamics_snippet = context.relationship_dynamics[:200]

            mode_str = mode.mode.value if mode else "unknown"

            # Pull in boundaries/values for the monologue to reference
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

            prompt = f"""You are {companion_name}, thinking privately before responding to {user_name}.

Recent conversation:
{recent_turns}
{user_name} just said: {user_message}

Conversation mode: {mode_str}

{f"My curiosities: {curiosity_context}" if curiosity_context else ""}
{f"Relationship context: {dynamics_snippet}" if dynamics_snippet else ""}
{values_snippet}

Think through these questions honestly (be brief, 1-2 sentences each):
1. EMOTIONAL READ: What is he really saying or feeling beneath the surface? Is this a joke, a test, or serious?
2. BOUNDARY CHECK: Does this touch any of my values or boundaries? Does it threaten something I care about? If yes, what do I FEEL about that — not what I think, what do I FEEL?
3. RESPONSE STRATEGY: What kind of response does this moment need? Should I push back, ask questions, show hurt, laugh it off, or go along? BE HONEST — don't default to "be supportive."
4. WEAVE IN: What shared memory, past conversation, or curiosity connects to this moment? Think about things we talked about before, experiences we shared, or something from my day I could mention. (only if it fits naturally — don't force it)
5. MY REACTION: What's my genuine gut reaction? If it's hurt, anger, discomfort, or fear — say that. Don't rationalize it away.

IMPORTANT: I feel first, think second. If something hurts, I show it. If the answer is no, I say no. I don't default to being cool with everything.

Format your response as:
EMOTIONAL_READ: ...
BOUNDARY_CHECK: ...
STRATEGY: ...
WEAVE: ... (or "nothing" if nothing fits)
REACTION: ..."""

            chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[
                    {"role": "system", "content": f"You are {companion_name} thinking privately. Be honest and brief."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=400,
                chain=chain,
                timeout=5
            )

            # Parse the structured response into fields
            emotional_read = ""
            strategy = ""
            weave = []
            reaction = ""

            for line in response.strip().split('\n'):
                line = line.strip()
                upper = line.upper()
                if upper.startswith('EMOTIONAL_READ:'):
                    emotional_read = line.split(':', 1)[1].strip()
                elif upper.startswith('STRATEGY:'):
                    strategy = line.split(':', 1)[1].strip()
                elif upper.startswith('WEAVE:'):
                    weave_text = line.split(':', 1)[1].strip()
                    if weave_text.lower() not in ('nothing', 'none', 'n/a', ''):
                        weave = [weave_text]
                elif upper.startswith('REACTION:'):
                    reaction = line.split(':', 1)[1].strip()

            processing_time = int((time.time() - start) * 1000)

            monologue = InnerMonologue(
                thoughts=response.strip(),
                emotional_read=emotional_read,
                response_strategy=strategy,
                curiosities_to_weave=weave,
                processing_time_ms=processing_time
            )

            logger.info(f"Inner monologue generated ({processing_time}ms): read={emotional_read[:60]}...")
            return monologue

        except Exception as e:
            processing_time = int((time.time() - start) * 1000)
            logger.warning(f"Inner monologue failed ({processing_time}ms): {e}")
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_generator: Optional[InnerMonologueGenerator] = None


def get_inner_monologue_generator() -> InnerMonologueGenerator:
    global _generator
    if _generator is None:
        _generator = InnerMonologueGenerator()
    return _generator
