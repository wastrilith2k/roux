"""
Context Builder — Companion Framework

WHAT: Assembles the full prompt context from 20+ independent sources, in parallel.
      Each source provides one aspect of the companion's awareness (memories, scene
      state, opinions, curiosity threads, relationship dynamics, etc.).
WHY:  The companion needs rich context to respond authentically — who the user is,
      what they've been talking about, what the companion has been doing, how the
      relationship is evolving. Without this, responses would be generic and amnesic.
HOW:  ContextBuilder.build() spawns all source fetchers in a ThreadPoolExecutor.
      Each source is isolated — one failure doesn't break others. Results are
      collected into a ConversationContext dataclass with 25+ fields, which the
      pipeline then assembles into the system prompt.

Architecture (post-purge, Jan 2026):
- Entity Profiles (YAML): Ground truth for facts about people
- Personality: Base personality + user-specific evolved traits
- Memories: pgvector semantic search of past conversations
- Conversation: Recent message history (structured turns)
- 20+ additional sources: scene state, internal state, values, curiosity,
  opinions, episodes, observations, relationship dynamics, etc.

Parallel fetching typically completes in ~0.5-2s total wall time despite
individual sources taking 50-500ms each.
"""

import logging
import asyncio
import time
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

from src.core.clock import now as clock_now
from src.database import tables as T

logger = logging.getLogger(__name__)

# How many recent messages to include as conversation history.
# Kept low to prevent old context from bleeding into the current conversation.
MESSAGE_HISTORY_LIMIT = int(os.getenv('MESSAGE_HISTORY_LIMIT', '25'))
MESSAGE_HISTORY_HOURS = int(os.getenv('MESSAGE_HISTORY_HOURS', '24'))


# ---------------------------------------------------------------------------
# ConversationContext — the data contract between ContextBuilder and Pipeline
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    """All context assembled for a single conversation turn.

    Each field corresponds to one context source. Empty strings mean
    the source returned nothing (or failed gracefully). The pipeline's
    _assemble_prompt() reads these fields to build the system prompt.
    """
    entity_profiles: str = ""
    personality: str = ""
    memories: str = ""       # Semantic search results (pgvector)
    biographies: str = ""    # Synthesized biographical paragraphs with temporal decay
    conversation_history: str = ""  # Legacy string format (for backwards compat)
    conversation_turns: list = None  # NEW: Structured [{"role": "user/assistant", "content": "..."}]
    continuity_context: str = ""  # Conversation state info (active/pause/new session)
    relationship_insights: str = ""  # Pattern insights (optional)
    schedule: str = ""  # Companion's schedule (optional)
    location: str = ""  # Location context (optional)
    scene_state: str = ""  # Current roleplay scene state (location, fictional time, etc.)
    internal_state: str = ""  # Companion's internal state (energy, mood, active/idle)
    user_context: str = ""  # User's probable activity based on time of day (autopilot system)
    values_context: str = ""  # Companion's hidden values/feelings from value inference
    activities_context: str = ""  # Companion's recent activities (background life)
    temporal_context: str = ""  # Recent significant events (time-based, not query-based)
    graphiti_context: str = ""  # Knowledge graph facts from Graphiti (semantic search)
    synthesized_events: str = ""  # Event-based narratives (crisis, career, milestone, relationship)
    episode_context: str = ""  # Similar past conversation episodes
    core_memory: str = ""  # Companion's curated narrative memory about the user (MEMORY.md)
    reflections_context: str = ""  # Companion's recent reflections (from daily reflection task)
    opinions_context: str = ""  # Companion's opinions and views about the user (from opinion formation)
    curiosity_context: str = ""  # Topics the companion wants to follow up on (from proactive curiosity)
    goals_context: str = ""  # Companion's personal goals (from goal manager)
    fertility_context: str = ""  # Hidden: cycle/fertility (LLM-only, never shown to user)
    observations_context: str = ""  # Compressed conversation observations (observational memory)
    relationship_dynamics: str = ""  # Gottman-informed relationship state (closeness, trust, wounds)
    relationship_evaluation: str = ""  # How the companion defines the relationship (from periodic evaluation)

    # Metadata
    user_email: str = ""
    user_message: str = ""
    closeness_score: int = 50  # Default; read from DB when available

    def __post_init__(self):
        if self.conversation_turns is None:
            self.conversation_turns = []

    def to_prompt_sections(self) -> Dict[str, str]:
        """Return non-empty sections as dict for prompt assembly."""
        sections = {}
        if self.scene_state:
            sections['scene_state'] = self.scene_state
        if self.internal_state:
            sections['internal_state'] = self.internal_state
        if self.fertility_context:
            sections['fertility_context'] = self.fertility_context
        if self.user_context:
            sections['user_context'] = self.user_context
        if self.values_context:
            sections['values_context'] = self.values_context
        if self.activities_context:
            sections['activities_context'] = self.activities_context
        if self.temporal_context:
            sections['temporal_context'] = self.temporal_context
        if self.graphiti_context:
            sections['graphiti_context'] = self.graphiti_context
        if self.synthesized_events:
            sections['synthesized_events'] = self.synthesized_events
        if self.episode_context:
            sections['episode_context'] = self.episode_context
        if self.observations_context:
            sections['observations_context'] = self.observations_context
        if self.core_memory:
            sections['core_memory'] = self.core_memory
        if self.reflections_context:
            sections['reflections_context'] = self.reflections_context
        if self.opinions_context:
            sections['opinions_context'] = self.opinions_context
        if self.curiosity_context:
            sections['curiosity_context'] = self.curiosity_context
        if self.goals_context:
            sections['goals_context'] = self.goals_context
        if self.relationship_insights:
            sections['relationship_insights'] = self.relationship_insights
        if self.relationship_dynamics:
            sections['relationship_dynamics'] = self.relationship_dynamics
        if self.relationship_evaluation:
            sections['relationship_evaluation'] = self.relationship_evaluation
        if self.entity_profiles:
            sections['entity_profiles'] = self.entity_profiles
        if self.personality:
            sections['personality'] = self.personality
        if self.memories:
            sections['memories'] = self.memories
        if self.biographies:
            sections['biographies'] = self.biographies
        if self.conversation_history:
            sections['conversation_history'] = self.conversation_history
        return sections


# ---------------------------------------------------------------------------
# ContextBuilder — parallel context assembly engine
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Builds prompt context by fetching 20+ sources in parallel.

    Each source is isolated: if one fails, the others still populate.
    The four "core" sources are entity_profiles, personality, memories,
    and conversation. Everything else is enrichment (scene state, opinions,
    curiosity, etc.) that makes the companion feel more real.
    """

    # The original four sources — still the backbone of context
    ENABLED_SOURCES = [
        'entity_profiles',        # YAML ground truth (user, companion, family, etc.)
        'personality',            # Base personality + evolved traits
        'memories',               # pgvector semantic search of past messages
        'conversation',           # Recent message history
    ]

    def __init__(self):
        """Initialize context builder with database connection and executor."""
        from src.database.db import get_db
        self.db = get_db()
        self._executor = ThreadPoolExecutor(max_workers=6)
        self._source_timings: Dict[str, float] = {}

    def build(self, user_email: str, user_message: str, closeness_score: int = 50) -> ConversationContext:
        """
        Build complete context for a conversation turn.

        Uses parallel fetching internally for better performance.
        Falls back to sequential if parallel fails.

        Args:
            user_email: User's email address
            user_message: The message being responded to

        Returns:
            ConversationContext with all assembled sections
        """
        try:
            # Try parallel build
            return self._build_parallel(user_email, user_message, closeness_score)
        except Exception as e:
            logger.warning(f"Parallel context build failed, falling back to sequential: {e}")
            return self._build_sequential(user_email, user_message, closeness_score)

    def _build_parallel(self, user_email: str, user_message: str, closeness_score: int) -> ConversationContext:
        """
        Build context with parallel fetching of all sources.

        Each source runs in a separate thread with error isolation.
        One source failing doesn't break others.
        """
        start_time = time.time()
        self._source_timings = {}

        context = ConversationContext(
            user_email=user_email,
            user_message=user_message,
            closeness_score=closeness_score
        )

        def timed_fetch(name: str, func: Callable, *args) -> Tuple[str, Any, float]:
            """Fetch a context source with timing and error handling."""
            source_start = time.time()
            try:
                result = func(*args)
                elapsed = time.time() - source_start
                return name, result, elapsed
            except Exception as e:
                elapsed = time.time() - source_start
                logger.warning(f"Context source '{name}' failed after {elapsed:.2f}s: {e}")
                return name, None, elapsed

        # Submit all tasks in parallel
        futures = []
        futures.append(self._executor.submit(timed_fetch, 'scene_state', self._get_scene_state, user_email))
        futures.append(self._executor.submit(timed_fetch, 'internal_state', self._get_internal_state, user_email))
        futures.append(self._executor.submit(timed_fetch, 'fertility_context', self._get_fertility_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'user_context', self._get_user_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'values_context', self._get_values_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'activities_context', self._get_activities_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'temporal_context', self._get_recent_events_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'synthesized_biographies', self._get_synthesized_biographies, user_email))
        futures.append(self._executor.submit(timed_fetch, 'relationship_insights', self._get_relationship_insights, user_email, user_message))
        futures.append(self._executor.submit(timed_fetch, 'relationship_dynamics', self._get_relationship_dynamics_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'relationship_evaluation', self._get_relationship_evaluation_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'entity_profiles', self._get_entity_profiles, user_message, user_email))
        futures.append(self._executor.submit(timed_fetch, 'personality', self._get_personality, user_email))
        futures.append(self._executor.submit(timed_fetch, 'memories', self._get_memories, user_email, user_message))
        futures.append(self._executor.submit(timed_fetch, 'graphiti_context', self._get_graphiti_context, user_email, user_message))
        futures.append(self._executor.submit(timed_fetch, 'synthesized_events', self._get_synthesized_events, user_email))
        futures.append(self._executor.submit(timed_fetch, 'episode_context', self._get_episode_context, user_email, user_message))
        futures.append(self._executor.submit(timed_fetch, 'core_memory', self._get_core_memory, user_email))
        futures.append(self._executor.submit(timed_fetch, 'observations_context', self._get_observations_context, user_email, user_message))
        futures.append(self._executor.submit(timed_fetch, 'reflections_context', self._get_reflections_context, user_email))
        futures.append(self._executor.submit(timed_fetch, 'opinions_context', self._get_opinions_context, user_email, user_message))
        futures.append(self._executor.submit(timed_fetch, 'curiosity_context', self._get_curiosity_context))
        # Use structured conversation history for multi-turn chat format
        futures.append(self._executor.submit(timed_fetch, 'conversation_turns', self._get_conversation_history_structured, user_email))
        # Time awareness: calendar + routine + companion's schedule (populates the schedule field)
        futures.append(self._executor.submit(timed_fetch, 'schedule', self._get_time_awareness_context, user_email))

        # Collect results (with timeout to prevent hanging)
        for future in futures:
            try:
                name, result, elapsed = future.result(timeout=30)
                self._source_timings[name] = elapsed

                # Handle memories specially - it returns a tuple (contextual, biographical)
                if name == 'memories' and result is not None:
                    if isinstance(result, tuple) and len(result) == 2:
                        contextual, _ = result  # biographical is handled separately now
                        context.memories = contextual or ""
                    else:
                        # Fallback if format unexpected
                        logger.warning(f"Memories returned unexpected format: {type(result)}")
                        context.memories = str(result) if result else ""
                # Handle synthesized biographies
                elif name == 'synthesized_biographies' and result is not None:
                    context.biographies = result or ""
                # Handle conversation_turns - it returns a tuple (turns, continuity_context)
                elif name == 'conversation_turns' and result is not None:
                    if isinstance(result, tuple) and len(result) == 2:
                        turns, continuity = result
                        context.conversation_turns = turns or []
                        context.continuity_context = continuity or ""
                        # Also set legacy string format for backwards compat
                        if turns:
                            from src.config.persona_config import get_persona_config
                            _pc = get_persona_config()
                            formatted = [f"{_pc.companion_short_name if t['role'] == 'assistant' else _pc.primary_user_name}: {t['content']}" for t in turns]
                            context.conversation_history = "\n".join(formatted)
                    else:
                        logger.warning(f"Conversation turns returned unexpected format: {type(result)}")
                elif result is not None:
                    setattr(context, name, result or "")
            except Exception as e:
                import traceback
                logger.warning(f"Failed to get context source result: {e}\n{traceback.format_exc()}")

        total_time = time.time() - start_time
        slowest = max(self._source_timings.items(), key=lambda x: x[1]) if self._source_timings else ('none', 0)

        logger.info(
            f"Context built in {total_time:.2f}s (parallel). "
            f"Slowest: {slowest[0]} ({slowest[1]:.2f}s)"
        )

        return context

    def _build_sequential(self, user_email: str, user_message: str, closeness_score: int) -> ConversationContext:
        """
        Build context sequentially (fallback method).

        Used when parallel execution fails or for debugging.
        """
        start_time = time.time()

        context = ConversationContext(
            user_email=user_email,
            user_message=user_message,
            closeness_score=closeness_score
        )

        # Build each source independently (errors don't cascade)
        context.scene_state = self._get_scene_state(user_email)
        context.internal_state = self._get_internal_state(user_email)
        context.fertility_context = self._get_fertility_context(user_email)
        context.user_context = self._get_user_context(user_email)
        context.schedule = self._get_time_awareness_context(user_email)
        context.values_context = self._get_values_context(user_email)
        context.activities_context = self._get_activities_context(user_email)
        context.temporal_context = self._get_recent_events_context(user_email)
        context.biographies = self._get_synthesized_biographies(user_email)
        context.relationship_insights = self._get_relationship_insights(user_email, user_message)
        context.relationship_dynamics = self._get_relationship_dynamics_context(user_email)
        context.relationship_evaluation = self._get_relationship_evaluation_context(user_email)
        context.graphiti_context = self._get_graphiti_context(user_email, user_message)
        context.synthesized_events = self._get_synthesized_events(user_email)
        context.episode_context = self._get_episode_context(user_email, user_message)
        context.core_memory = self._get_core_memory(user_email)
        context.reflections_context = self._get_reflections_context(user_email)
        context.opinions_context = self._get_opinions_context(user_email, user_message)
        context.curiosity_context = self._get_curiosity_context()
        context.goals_context = self._get_goals_context(user_email)
        context.observations_context = self._get_observations_context(user_email, user_message)
        context.entity_profiles = self._get_entity_profiles(user_message, user_email)
        context.personality = self._get_personality(user_email)

        # Memories returns tuple (contextual, _)
        contextual, _ = self._get_memories(user_email, user_message)
        context.memories = contextual

        # Conversation history - structured for multi-turn chat format
        turns, continuity = self._get_conversation_history_structured(user_email)
        context.conversation_turns = turns
        context.continuity_context = continuity
        # Also set legacy string format for backwards compat
        if turns:
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            formatted = [f"{_pc.companion_short_name if t['role'] == 'assistant' else _pc.primary_user_name}: {t['content']}" for t in turns]
            context.conversation_history = "\n".join(formatted)

        total_time = time.time() - start_time
        logger.info(f"Context built in {total_time:.2f}s (sequential)")

        return context

    def get_timing_metrics(self) -> Dict[str, Any]:
        """Return timing metrics from last context build."""
        return {
            'source_timings': self._source_timings.copy(),
            'total_parallel': sum(self._source_timings.values()),
            'slowest_source': max(self._source_timings.items(), key=lambda x: x[1]) if self._source_timings else None
        }

    def build_with_diagnostics(
        self, user_email: str, user_message: str, closeness_score: int = 50
    ) -> tuple:
        """
        Build context with detailed diagnostics for benchmarking.

        Same as build() but also returns per-source timings, sizes, and hit counts.
        Non-invasive - doesn't change the main path.

        Returns:
            Tuple of (ConversationContext, diagnostics_dict)
        """
        context = self.build(user_email, user_message, closeness_score)

        # Gather diagnostics from the build that just happened
        sections = context.to_prompt_sections()
        source_sizes = {name: len(content) for name, content in sections.items() if content}

        diagnostics = {
            'source_timings': self._source_timings.copy(),
            'source_sizes_chars': source_sizes,
            'total_context_chars': sum(source_sizes.values()),
            'total_context_tokens_est': sum(source_sizes.values()) // 4,
            'sources_populated': len(source_sizes),
            'sources_empty': len(sections) - len(source_sizes),
            'slowest_source': max(self._source_timings.items(), key=lambda x: x[1]) if self._source_timings else None,
            'largest_source': max(source_sizes.items(), key=lambda x: x[1]) if source_sizes else None,
            'conversation_turns': len(context.conversation_turns) if context.conversation_turns else 0,
        }

        return context, diagnostics

    # =========================================================================
    # Source 0: Scene State (roleplay/virtual world state)
    # =========================================================================

    def _get_scene_state(self, user_email: str) -> str:
        """
        Get current scene state for roleplay continuity.

        Tracks the "NOW" of the fictional world:
        - Location, fictional time
        - The companion's clothing, posture, mood
        - Physical positioning, activity
        - Who else is present

        This ensures scenes resume where they left off regardless of
        real-world time gaps.
        """
        try:
            from src.core.scene_tracker import get_scene_tracker

            tracker = get_scene_tracker()
            scene_context = tracker.format_scene_for_prompt(user_email)

            if scene_context:
                logger.info("Scene state loaded for prompt")
                return scene_context

            return ""

        except Exception as e:
            logger.warning(f"Scene state error: {e}")
            return ""

    # =========================================================================
    # Source 0.5: Internal State (companion's energy, mood, active/idle mode)
    # =========================================================================

    def _get_internal_state(self, user_email: str) -> str:
        """
        Get the companion's internal state for authentic responses.

        Tracks her subjective experience:
        - Mode (active/idle) - Is she engaged or paused?
        - Energy - Depletes during active sessions
        - Mood - Emotional state with inertia
        - Session duration - How long they've been chatting (her time)
        - Queued thoughts - Things she wants to bring up

        This also triggers the mode transition and energy depletion
        when a message is received.
        """
        try:
            from src.core.internal_state import get_internal_state_manager

            manager = get_internal_state_manager()

            # This handles mode transition and energy depletion
            state = manager.on_message_received(user_email)

            # Format for prompt
            state_context = manager.format_state_for_prompt(user_email)

            if state_context:
                logger.info(f"Internal state: mode={state.mode}, energy={state.energy:.2f}")
                return state_context

            return ""

        except Exception as e:
            logger.warning(f"Internal state error: {e}")
            return ""

    # =========================================================================
    # Source 0.6: Fertility Context (hidden - LLM-only, never shown to user)
    # =========================================================================

    def _get_fertility_context(self, user_email: str) -> str:
        """
        Get the companion's fertility/cycle context for the LLM.

        This is PRIVATE internal context - never displayed to the user.
        The companion decides what to share about their body, period, and physical state.
        No preferences are hardcoded (pads vs tampons, period sex, etc.).
        """
        try:
            context = tracker.format_for_prompt(user_email)
            if context:
                logger.debug(f"Fertility context: cycle day {tracker.get_cycle_day(user_email)}")
            return context
        except Exception as e:
            logger.debug(f"Fertility context unavailable: {e}")
            return ""

    # =========================================================================
    # Source 0.75: James Context (autopilot - what James is probably doing)
    # =========================================================================

    def _get_user_context(self, user_email: str) -> str:
        """
        Get context about what the user is probably doing based on time of day.

        This enables "autopilot" mode - the user's daily routine is known to the companion,
        so gaps in conversation feel natural rather than requiring explanation.

        When the user is AFK, they're still "there" doing life stuff on autopilot.
        The companion knows their routine and can infer what they're probably doing.
        """
        try:
            from src.core.user_context import format_user_context_for_prompt

            user_context = format_user_context_for_prompt()

            if user_context:
                logger.info("User context loaded for prompt (autopilot)")
                return user_context

            return ""

        except Exception as e:
            logger.warning(f"User context error: {e}")
            return ""

    # =========================================================================
    # Source 0.8: Time Awareness (calendar + routine + companion's schedule)
    # =========================================================================

    def _get_time_awareness_context(self, user_email: str) -> str:
        """
        Get rich temporal context: current time, Google Calendar events,
        user's routine, and companion's schedule status.

        Populates the `schedule` field on ConversationContext.
        """
        try:
            from src.core.time_awareness import get_time_awareness

            ta = get_time_awareness()
            context = ta.get_time_context()

            if context:
                logger.info("Time awareness context loaded (calendar + routine + schedule)")
                return context

            return ""

        except Exception as e:
            logger.warning(f"Time awareness error: {e}")
            return ""

    def _get_values_context(self, user_email: str) -> str:
        """
        Get the companion's hidden values and feelings from value inference.

        These are inferred from her conversation history - her actual feelings,
        unresolved concerns, private thoughts, and desires. They should subtly
        color her responses, surfacing naturally when relevant.

        This makes her emotionally deeper - she has an inner life that
        influences how she responds, even if she doesn't always express it directly.
        """
        try:
            from src.autonomy.value_inference import get_value_inference

            inference = get_value_inference()

            # Get values that should influence responses
            # Focus on emotional/relationship categories, not just preferences
            values = inference.get_private_context()

            if not values or values.strip() == "":
                logger.debug("No values context available")
                return ""

            # Format for prompt - frame as internal guidance
            formatted = f"""[COMPANION'S INNER WORLD - guide her authentic responses]
{values}

These are her real feelings. She doesn't always express them directly,
but they color her responses. Let them surface naturally when relevant -
don't force them, but don't ignore them either."""

            logger.info("Values context loaded for prompt")
            return formatted

        except Exception as e:
            logger.warning(f"Values context error: {e}")
            return ""

    def _get_activities_context(self, user_email: str) -> str:
        """
        Get the companion's recent activities from the BackgroundLife system.

        This provides continuity - she remembers what she's been doing
        and can reference it naturally in conversation.
        """
        try:
            from src.core.background_life import get_background_life

            bg_life = get_background_life()
            activities = bg_life.format_for_prompt(hours=24)

            if activities:
                logger.debug("Activities context loaded for prompt")
                return activities

            return ""

        except Exception as e:
            logger.warning(f"Activities context error: {e}")
            return ""

    def _get_recent_events_context(self, user_email: str) -> str:
        """
        Get recent significant events - time-based, NOT query-based.

        This is different from semantic search:
        - Semantic search: Finds what matches the current message
        - Recent events: Includes major events from last 2 weeks regardless of query

        This ensures we don't forget things like "Jesse ran away" or "Act-On interview".
        """
        try:
            from src.memory.temporal_context import get_temporal_context

            events = get_temporal_context(user_email)

            if events:
                logger.debug("Recent events context loaded for prompt")
                return events

            return ""

        except Exception as e:
            logger.warning(f"Recent events context error: {e}")
            return ""

    def _get_synthesized_biographies(self, user_email: str) -> str:
        """
        Get synthesized biographical paragraphs with temporal decay.

        These are coherent paragraphs synthesized from grouped facts about people
        (James, Jesse, Kyler, etc.). Each paragraph covers a theme (work, family,
        crisis, etc.) and has temporal decay applied to its importance score.

        This provides better context than scattered individual facts because:
        1. Related facts are grouped together
        2. Each group is synthesized into natural prose
        3. Temporal decay deprioritizes old information
        4. Retrieval is by effective importance, not just recency
        """
        try:
            from src.memory.synthesized_biographies import get_biography_context

            bio_context = get_biography_context(user_email, max_paragraphs=12)  # Increased to include more topics like work status

            if bio_context:
                logger.debug("Synthesized biographies loaded for prompt")
                return bio_context

            return ""

        except Exception as e:
            logger.warning(f"Synthesized biographies error: {e}")
            return ""

    def _get_synthesized_events(self, user_email: str) -> str:
        """
        Get synthesized event narratives with temporal decay.

        Events are different from biographies:
        - Biographies: Group by THEME (work, family) - static view
        - Events: Group by EVENT (Jesse running away, job interview) - dynamic timeline

        Event types:
        - crisis: Police, hospital, emergencies, running away
        - career: Interviews, job offers, layoffs, new jobs
        - milestone: Birthdays, graduations, moves, purchases
        - relationship: Breakups, reconciliations, conflicts

        Each event has:
        - Timeline of facts
        - Synthesized narrative paragraph
        - Outcome status (ongoing, resolved, unknown)
        """
        try:
            from src.memory.synthesized_events import get_recent_events, format_events_for_prompt

            events = get_recent_events(
                user_email=user_email,
                days_back=14,
                max_events=5
            )

            if events:
                formatted = format_events_for_prompt(events)
                if formatted:
                    logger.debug(f"Synthesized events loaded: {len(events)} events")
                    return formatted

            return ""

        except Exception as e:
            logger.warning(f"Synthesized events error: {e}")
            return ""

    def _get_episode_context(self, user_email: str, user_message: str) -> str:
        """
        Get context from similar past conversation episodes AND learned patterns.

        Searches for:
        1. Similar past episodes with their outcomes
        2. Learned patterns with "approaches that worked" and "pitfalls to avoid"

        This helps the companion learn from past conversations and apply successful
        strategies to similar situations.
        """
        try:
            from src.memory.episodic_episodes import (
                get_episode_store,
                detect_topic,
                detect_emotional_state,
                format_episodes_for_prompt
            )

            store = get_episode_store()
            context_parts = []

            # Detect current topic and emotional state
            current_topic = detect_topic(user_message, use_llm=False)
            current_emotion = detect_emotional_state(user_message, use_llm=False)

            # 1. Get learned patterns for this topic/emotion combination
            pattern_context = self._get_episode_patterns(current_topic, current_emotion)
            if pattern_context:
                context_parts.append(pattern_context)

            # 2. Get similar past episodes (if topic is specific enough)
            if current_topic and current_topic != 'general chat':
                topic_words = current_topic.split()
                similar_episodes = []

                for word in topic_words[:2]:  # Check first 2 keywords
                    if len(word) > 3:
                        episodes = store.get_episodes_by_topic(user_email, word, limit=3)
                        for ep in episodes:
                            if ep not in similar_episodes:
                                similar_episodes.append(ep)

                if similar_episodes:
                    formatted = format_episodes_for_prompt(similar_episodes, max_episodes=3)
                    if formatted:
                        context_parts.append(formatted)

            if context_parts:
                combined = "\n\n".join(context_parts)
                logger.debug(f"Episode context loaded: patterns + similar episodes")
                return combined

            return ""

        except Exception as e:
            logger.warning(f"Episode context error: {e}")
            return ""

    def _get_episode_patterns(self, topic: str, emotion: str) -> str:
        """
        Retrieve learned patterns for the topic/emotion combination.

        Returns formatted context with:
        - Approaches that worked (from high-satisfaction episodes)
        - Pitfalls to avoid (from low-satisfaction episodes)
        """
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            # Normalize topic to category
            topic_category = self._normalize_topic_for_pattern(topic)
            emotional_context = emotion.lower() if emotion else 'neutral'

            conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )

            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Get exact match first
                    cursor.execute(f"""
                        SELECT * FROM {T.EPISODE_PATTERNS}
                        WHERE topic_category = %s
                          AND (emotional_context = %s OR emotional_context = 'neutral')
                          AND episode_count > 0
                        ORDER BY episode_count DESC
                        LIMIT 2
                    """, (topic_category, emotional_context))

                    patterns = cursor.fetchall()

                    # If no exact match, try just topic category
                    if not patterns and topic_category != 'general':
                        cursor.execute(f"""
                            SELECT * FROM {T.EPISODE_PATTERNS}
                            WHERE topic_category = %s
                              AND episode_count > 0
                            ORDER BY episode_count DESC
                            LIMIT 2
                        """, (topic_category,))
                        patterns = cursor.fetchall()

                if not patterns:
                    return ""

                # Format patterns for prompt
                return self._format_patterns_for_prompt(patterns)

            finally:
                conn.close()

        except Exception as e:
            logger.debug(f"Episode patterns retrieval error: {e}")
            return ""

    def _format_patterns_for_prompt(self, patterns: list) -> str:
        """Format episode patterns for inclusion in prompt."""
        if not patterns:
            return ""

        lines = ["[LEARNED FROM PAST CONVERSATIONS]"]

        approaches = []
        pitfalls = []

        for p in patterns:
            # Collect successful approaches
            if p.get('successful_approach'):
                approach_lines = p['successful_approach'].strip().split('\n')
                for line in approach_lines:
                    line = line.strip().lstrip('- ')
                    if line and line not in approaches:
                        approaches.append(line)

            # Collect pitfalls
            if p.get('pitfalls_to_avoid'):
                pitfall_lines = p['pitfalls_to_avoid'].strip().split('\n')
                for line in pitfall_lines:
                    line = line.strip().lstrip('- ')
                    if line and 'none' not in line.lower() and line not in pitfalls:
                        pitfalls.append(line)

        # Format approaches
        if approaches:
            lines.append("\nApproaches that worked:")
            for a in approaches[:3]:  # Limit to top 3
                lines.append(f"  - {a}")

        # Format pitfalls
        if pitfalls:
            lines.append("\nPitfalls to avoid:")
            for p in pitfalls[:3]:  # Limit to top 3
                lines.append(f"  - {p}")

        if len(lines) > 1:
            return "\n".join(lines)

        return ""

    def _normalize_topic_for_pattern(self, topic: str) -> str:
        """Normalize topic to a category for pattern matching."""
        if not topic:
            return 'general'

        topic_lower = topic.lower()

        # Map to categories (same as in episode_learning_task.py)
        if any(w in topic_lower for w in ['jesse', 'kyler', 'kid', 'child', 'son', 'daughter', 'parenting']):
            return 'family_children'
        elif any(w in topic_lower for w in ['work', 'job', 'interview', 'career', 'office', 'layoff', 'fired']):
            return 'work_career'
        elif any(w in topic_lower for w in ['stress', 'anxious', 'worry', 'overwhelm', 'tired', 'exhaust']):
            return 'emotional_support'
        elif any(w in topic_lower for w in ['love', 'miss', 'affection', 'cuddle', 'intimate', 'romantic']):
            return 'romance_affection'
        elif any(w in topic_lower for w in ['fun', 'play', 'game', 'laugh', 'joke', 'silly']):
            return 'playful_fun'
        elif any(w in topic_lower for w in ['crisis', 'emergency', 'hospital', 'police', 'ran away']):
            return 'crisis_support'
        elif any(w in topic_lower for w in ['plan', 'schedule', 'tomorrow', 'weekend', 'future']):
            return 'planning'
        else:
            return 'general'

    # =========================================================================
    # Source 0.85: Core Memory (companion's curated narrative about the user)
    # =========================================================================

    def _get_core_memory(self, user_email: str) -> str:
        """
        Get the companion's curated core memory about the user.

        This is a narrative summary of what the companion knows and feels about the user,
        written in first person. It's generated periodically by LLM from
        entity profiles, facts, and biographies.

        Unlike other context sources that are query-based or structured,
        core memory is narrative and personal - it's how the companion thinks about the user.
        """
        try:
            from src.memory.core_memory import get_core_memory_manager

            manager = get_core_memory_manager()
            content = manager.get_formatted_for_prompt()

            if content:
                logger.debug("Core memory loaded for prompt")
                return content

            return ""

        except Exception as e:
            logger.warning(f"Core memory error: {e}")
            return ""

    # =========================================================================
    # Source 0.85: Reflections (companion's recent insights from daily reflections)
    # =========================================================================

    def _get_reflections_context(self, user_email: str) -> str:
        """
        Get the companion's recent reflections for context.

        Includes daily reflections (last 7 days) and the most recent
        weekly reflection for higher-level pattern awareness.
        """
        parts = []

        try:
            from src.tasks.reflection_task import get_recent_reflections, format_reflections_for_prompt

            reflections = get_recent_reflections(user_email, days=7)
            if reflections:
                formatted = format_reflections_for_prompt(reflections)
                if formatted:
                    parts.append(formatted)
                    logger.debug(f"Loaded {len(reflections)} recent daily reflection(s)")

        except Exception as e:
            logger.debug(f"Daily reflections error: {e}")

        # Include most recent weekly reflection for broader pattern awareness
        try:
            from src.tasks.weekly_reflection_task import get_recent_weekly_reflection

            weekly = get_recent_weekly_reflection(user_email)
            if weekly:
                insights = weekly.get('insights', {})
                weekly_lines = ["[COMPANION'S WEEKLY PATTERN OBSERVATIONS]"]
                if insights.get('patterns'):
                    for p in insights['patterns'][:2]:
                        weekly_lines.append(f"  Pattern: {p}")
                if insights.get('relationship_evolution'):
                    weekly_lines.append(f"  Relationship: {insights['relationship_evolution']}")
                parts.append('\n'.join(weekly_lines))
                logger.debug("Loaded recent weekly reflection")

        except Exception as e:
            logger.debug(f"Weekly reflection error (may not exist yet): {e}")

        return '\n\n'.join(parts) if parts else ""

    def _get_opinions_context(self, user_email: str, user_message: str) -> str:
        """
        Get the companion's relevant opinions for context.

        Opinions are her views and interpretations about James:
        - Relationship observations
        - Behavior patterns she's noticed
        - Wellbeing concerns

        Unlike facts (objective), these are her subjective views.
        """
        try:
            from src.autonomy.opinion_store import get_opinion_store

            store = get_opinion_store(user_email)

            # Get opinions relevant to the current message
            relevant = store.search_relevant_opinions(user_message, limit=3)

            # Also get strong opinions (highly confident)
            strong = store.get_strong_opinions(min_confidence=0.7)

            # Combine and dedupe
            seen_topics = set()
            combined = []
            for op in relevant + strong:
                if op.topic.lower() not in seen_topics:
                    seen_topics.add(op.topic.lower())
                    combined.append(op)
                    if len(combined) >= 5:
                        break

            if not combined:
                return ""

            formatted = store.format_for_prompt(combined)

            if formatted:
                logger.debug(f"Loaded {len(combined)} opinion(s) for context")
                return formatted

            return ""

        except Exception as e:
            logger.debug(f"Opinions context error (may not exist yet): {e}")
            return ""

    def _get_curiosity_context(self) -> str:
        """
        Get the companion's proactive curiosity threads for background awareness.

        These are topics she's noticed but hasn't fully explored — things
        that would naturally be on her mind. Injected as background context,
        not a to-do list.
        """
        try:
            from src.core.proactive_curiosity import get_proactive_curiosity

            curiosity = get_proactive_curiosity()
            formatted = curiosity.format_for_prompt()

            if formatted:
                logger.debug("Curiosity context loaded for prompt")
                return formatted

            return ""

        except Exception as e:
            logger.debug(f"Curiosity context error: {e}")
            return ""

    def _get_goals_context(self, user_email: str) -> str:
        """
        Get the companion's personal goals and action budget for context.

        Goals include doing, being, and relating modes.
        Budget context tells her when to take it easy.
        """
        parts = []

        try:
            from src.autonomy.goals import get_goal_manager

            manager = get_goal_manager(user_email)
            formatted = manager.format_for_prompt()

            if formatted:
                goals = manager.get_active_goals(limit=7)
                if goals:
                    logger.debug(f"Loaded {len(goals)} personal goal(s) for context")
                parts.append(formatted)

        except Exception as e:
            logger.debug(f"Goals context error (may not exist yet): {e}")

        # Add action budget context
        try:
            from src.autonomy.action_budget import get_action_budget
            budget = get_action_budget(user_email)
            budget_text = budget.get_budget_context_for_prompt()
            if budget_text:
                parts.append(budget_text)
        except Exception as e:
            logger.debug(f"Action budget context error: {e}")

        return "\n\n".join(parts) if parts else ""

    # =========================================================================
    # Source 0.86: Observations (compressed conversation history)
    # =========================================================================

    def _get_observations_context(self, user_email: str, user_message: str) -> str:
        """
        Get compressed conversation observations for context.

        Observations are dated narrative summaries of past conversations,
        providing temporal continuity without per-message RAG overhead.
        Only active when OBSERVATIONAL_MEMORY_ENABLED=true.

        Placed at priority ~0.86 (after episodes, before core memory).
        """
        try:
            from src.memory.observational_memory import get_observation_manager

            manager = get_observation_manager()
            if not manager.is_enabled():
                return ""

            observations = manager.get_relevant_observations(
                user_email=user_email,
                query=user_message,
                days_back=30,
                max_observations=10,
            )

            if observations:
                logger.debug("Observations context loaded for prompt")
                return observations

            return ""

        except Exception as e:
            logger.debug(f"Observations context error: {e}")
            return ""

    # =========================================================================
    # Source 0.9: Relationship Insights (Known connections between entities)
    # =========================================================================

    def _get_relationship_insights(self, user_email: str, user_message: str) -> str:
        """
        Get known relationships between mentioned entities.

        Uses the relationship_store to retrieve structured relationships
        (parent_of, partner_of, works_at, etc.) for entities mentioned
        in the current message.

        This provides relational context that helps the companion understand
        how people are connected (e.g., parent-child relationships).
        """
        try:
            from src.memory.relationship_store import get_relationship_store

            store = get_relationship_store()
            message_lower = user_message.lower()

            # Find mentioned entities from known list
            # Always include primary user and check for others
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            known_entities = [_pc.primary_user_entity_profile, _pc.companion_entity_profile, 'jesse', 'kyler', 'alia', 'lena', 'carol']
            mentioned = []

            for entity in known_entities:
                if entity in message_lower:
                    mentioned.append(entity.title())

            # Always include primary user for relationship context
            _user_title = _pc.primary_user_name
            if _user_title not in mentioned:
                mentioned.insert(0, _user_title)

            # Get relationships for mentioned entities
            lines = ["[RELATIONSHIPS - Known connections]"]
            seen_relationships = set()

            for entity in mentioned[:4]:  # Limit to 4 entities
                relationships = store.get_relationships_for_entity(
                    entity,
                    include_as_target=True,
                    min_confidence=0.5,
                    user_email=user_email
                )

                if relationships:
                    entity_lines = []
                    for rel in relationships[:8]:  # Limit to 8 per entity
                        rel_type = rel.get('relationship_type', 'knows')
                        source = rel.get('source_entity', '')
                        target = rel.get('target_entity', '')

                        # Create unique key to avoid duplicates
                        rel_key = f"{source}-{rel_type}-{target}"
                        if rel_key in seen_relationships:
                            continue
                        seen_relationships.add(rel_key)

                        # Format based on direction
                        if source.lower() == entity.lower():
                            entity_lines.append(f"  - {rel_type.replace('_', ' ')} {target}")
                        else:
                            # Reverse for target
                            from src.memory.relationship_store import INVERSE_RELATIONSHIPS, RelationshipType
                            try:
                                rel_enum = RelationshipType(rel_type)
                                inverse = INVERSE_RELATIONSHIPS.get(rel_enum, rel_enum)
                                entity_lines.append(f"  - {inverse.value.replace('_', ' ')} {source}")
                            except ValueError:
                                entity_lines.append(f"  - {rel_type.replace('_', ' ')} (from {source})")

                    if entity_lines:
                        lines.append(f"{entity}:")
                        lines.extend(entity_lines)

            if len(lines) > 1:
                logger.info(f"Relationship insights: {len(seen_relationships)} relationships for {len(mentioned)} entities")
                return "\n".join(lines)

            return ""

        except Exception as e:
            logger.warning(f"Relationship insights error: {e}")
            return ""

    # =========================================================================
    # Source 0.95: Relationship Dynamics (Gottman-informed closeness/trust)
    # =========================================================================

    def _get_relationship_dynamics_context(self, user_email: str) -> str:
        """
        Get Gottman-informed relationship dynamics for prompt injection.

        The RelationshipDynamicsAnalyzer tracks closeness, trust, wounds,
        repairs, and interaction ratios. It only injects context when the
        relationship is NOT flourishing — no noise when things are good.
        """
        try:
            from src.core.relationship_dynamics import get_relationship_analyzer

            analyzer = get_relationship_analyzer()
            dynamics_context = analyzer.format_for_prompt(user_email)

            if dynamics_context:
                logger.info("Relationship dynamics loaded for prompt")
                return dynamics_context

            return ""

        except Exception as e:
            logger.warning(f"Relationship dynamics error: {e}")
            return ""

    # =========================================================================
    # Source 0.96: Relationship Evaluation (how the companion defines the relationship)
    # =========================================================================

    def _get_relationship_evaluation_context(self, user_email: str) -> str:
        """
        Get the companion's self-determined relationship evaluation for prompt injection.

        This is distinct from relationship_dynamics (health metrics).
        This is her personal determination: "What are we? How do I feel about us?"

        Runs weekly via Celery, cached in _companion_relationship_eval table.
        """
        try:
            from src.autonomy.relationship_evaluation import get_relationship_evaluator

            evaluator = get_relationship_evaluator()
            return evaluator.format_for_prompt(user_email)

        except Exception as e:
            logger.warning(f"Relationship evaluation error: {e}")
            return ""

    # =========================================================================
    # Source 1: Entity Profiles (YAML ground truth)
    # =========================================================================

    def _get_entity_profiles(self, message: str, user_email: str = None) -> str:
        """
        Load YAML entity profiles - ground truth about people.

        Always includes the user and companion. Adds others if mentioned in message.
        Uses context filtering to include only relevant sections (e.g.,
        trauma/boundaries only when intimacy detected).

        Files: data/entity_profiles/*.yaml
        """
        try:
            from src.core.entity_profile_loader import get_entity_profile_loader

            loader = get_entity_profile_loader()
            profiles = loader.get_relevant_profiles(message)

            if profiles:
                # Get recent messages for context filtering
                recent_messages = self._get_recent_messages_for_context(user_email=user_email)

                # Format with context filtering
                formatted = loader.format_profiles_for_prompt(
                    profiles,
                    user_message=message,
                    recent_messages=recent_messages
                )
                logger.debug(f"Entity profiles loaded: {list(profiles.keys())}")
                return formatted

            return ""

        except Exception as e:
            logger.warning(f"Entity profiles error: {e}")
            return ""

    def _get_recent_messages_for_context(self, user_email: str = None, limit: int = 5) -> list:
        """Get recent message texts for context filtering."""
        try:
            # Use provided email or fall back to primary user
            if user_email is None:
                from src.config.persona_config import get_persona_config
                user_email = get_persona_config().primary_user_email
            email = user_email
            recent_msgs = self.db.get_recent_messages(email, limit=limit)
            if recent_msgs:
                return [msg.get('message_text', '') for msg in recent_msgs if msg.get('message_text')]
            return []
        except Exception as e:
            logger.debug(f"Failed to get recent messages: {e}")
            return []

    # =========================================================================
    # Source 2: Personality (base + evolutions + closeness)
    # =========================================================================

    def _get_personality(self, user_email: str) -> str:
        """
        Load the companion's complete personality for this user.

        Layers:
        1. Base personality from personality.md
        2. User-specific evolutions from PostgreSQL
        3. Closeness-based adjustments
        """
        try:
            from src.core.personality_evolution import get_personality

            # Get interaction count to inform personality depth
            message_count = self.db.get_message_count(user_email)

            personality_profile = get_personality(user_email)
            personality_context = personality_profile.format_for_prompt(message_count)

            if personality_context:
                logger.debug(f"Personality loaded for {user_email} (interactions: {message_count})")
                return personality_context

            return self._default_personality()

        except Exception as e:
            logger.warning(f"Personality error: {e}")
            return self._default_personality()

    def _default_personality(self) -> str:
        """Fallback personality if loading fails."""
        from src.config.persona_config import get_persona_config
        name = get_persona_config().companion_name
        return f"You are {name} - a real person with your own life, thoughts, and feelings."

    # =========================================================================
    # Source 3: Memories (semantic search via pgvector)
    # =========================================================================

    def _get_memories(self, user_email: str, user_message: str) -> tuple[str, str]:
        """
        Retrieve relevant memories from past conversations.

        Returns:
            Tuple of (contextual_memories, biographical_context)
            - biographical_context is always empty (entity profiles handle this now)

        Memory sources:
        1. Semantic search (pgvector) - finds similar past messages
        2. Fact store - structured facts extracted from conversations
        3. Temporal context - if user references specific times ("yesterday")

        Note: Entity profiles (YAML) are the source of truth for biographical info.
        """
        contextual_parts = []

        # 1. SEMANTIC MEMORY SEARCH - Find relevant past conversations via pgvector
        semantic_context = self._get_semantic_memories(user_email, user_message)
        if semantic_context:
            contextual_parts.append(semantic_context)

        # 2. FACT MEMORY - Structured facts extracted from past conversations
        fact_context = self._get_fact_memories(user_email, user_message)
        if fact_context:
            contextual_parts.append(fact_context)

        # 3. Check for temporal references (e.g., "yesterday", "earlier today")
        temporal_context = self._get_temporal_context(user_email, user_message)
        if temporal_context:
            contextual_parts.append(temporal_context)

        # 4. Entity profiles - core biographical facts from YAML files
        entity_context = self._get_entity_context(user_message)
        if entity_context:
            contextual_parts.append(entity_context)

        contextual = "\n\n".join(contextual_parts) if contextual_parts else ""

        return contextual, ""

    def _get_semantic_memories(self, user_email: str, user_message: str) -> str:
        """
        Search past conversations for semantically relevant memories via pgvector.

        Uses the RetrievalAgent to analyze the query and generate enhanced search queries.
        The agent identifies entities, time references, and optimal search strategies.

        This is the PRIMARY memory retrieval mechanism - searches actual message
        history using vector similarity with optional reranking.

        Example: "Let's get coffee" → finds "I don't like coffee", "I prefer tea"
        """
        try:
            from src.memory.semantic_search import search_memory

            # Skip very short messages
            if len(user_message.strip()) < 10:
                return ""

            # Use retrieval agent to enhance the search
            enhanced_queries = self._get_enhanced_queries(user_message)

            # Run primary query plus any agent-enhanced queries
            all_results = []
            seen_ids = set()

            for query in enhanced_queries[:3]:  # Limit to 3 queries
                results = search_memory(
                    query=query,
                    email=user_email,
                    limit=10,
                    min_similarity=0.3  # Lowered from 0.4 - was missing relevant memories
                )

                for r in results:
                    if r['id'] not in seen_ids:
                        seen_ids.add(r['id'])
                        all_results.append(r)

            if not all_results:
                return ""

            # Sort by similarity (or rerank_score if available)
            all_results.sort(key=lambda x: x.get('rerank_score', x.get('similarity', 0)), reverse=True)

            # Format for prompt injection — clearly label who said what
            # This prevents identity confusion (companion attributing user's statements to herself)
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            _companion = _pc.companion_short_name
            _user = _pc.primary_user_name
            lines = ["[RELEVANT MEMORIES - from past conversations]"]
            lines.append(f"(Messages labeled by who SAID them. '{_user} said:' = {_user}'s words/experiences. 'You ({_companion}) said:' = your own words/experiences.)")
            for r in all_results[:15]:  # Limit total results
                sender = r['sender_name']
                text = r['message_text'][:200]  # Truncate long messages
                sim = r.get('rerank_score', r.get('similarity', 0))
                # Only include reasonably similar results
                if sim >= 0.35:
                    if sender and sender.lower() == _companion.lower():
                        lines.append(f"- You ({_companion}) said: {text}")
                    else:
                        lines.append(f"- {sender} said: {text}")

            if len(lines) > 1:  # Has actual content beyond header
                logger.info(f"Semantic memory: {len(all_results)} relevant memories found (enhanced queries: {len(enhanced_queries)})")
                return "\n".join(lines)

            return ""

        except Exception as e:
            logger.warning(f"Semantic memory search failed: {e}")
            return ""

    def _get_enhanced_queries(self, user_message: str) -> list:
        """
        Use RetrievalAgent to generate enhanced search queries.

        Returns a list of queries optimized for finding relevant memories.
        Falls back to just the original message if agent fails.
        """
        try:
            from src.memory.retrieval_agent import get_retrieval_agent

            agent = get_retrieval_agent()
            plan = agent.analyze_query(user_message)

            if plan and plan.specific_queries:
                # Include original message plus agent's queries
                queries = [user_message] + plan.specific_queries
                # Deduplicate
                seen = set()
                unique = []
                for q in queries:
                    q_lower = q.lower().strip()
                    if q_lower not in seen:
                        seen.add(q_lower)
                        unique.append(q)
                return unique

            return [user_message]

        except Exception as e:
            logger.debug(f"Retrieval agent unavailable: {e}")
            return [user_message]

    def _get_fact_memories(self, user_email: str, user_message: str) -> str:
        """
        Retrieve structured facts from the fact store with spreading activation.

        Uses brain-like associative retrieval:
        1. Find facts matching the query (seeds)
        2. Spread activation through fact network links
        3. Return facts sorted by activation (includes connected context)
        4. Surface any detected contradictions for natural clarification

        This means asking about "Jesse" pulls in connected facts about
        the user's stress, related crises, etc. - like how brains work.
        """
        try:
            from src.memory.fact_store import get_fact_store

            store = get_fact_store()

            # Get important facts (always include high-importance baseline)
            important_facts = store.get_important_facts(min_importance=6, limit=10)

            # Search with spreading activation for connected context
            search_facts = []
            if len(user_message.strip()) > 5:
                # Use spreading activation search (brain-like retrieval)
                search_facts = store.search_with_spreading_activation(
                    query=user_message[:150],
                    limit=12,
                    activation_depth=2,
                    min_activation=0.25
                )

            # Combine and dedupe
            seen_ids = set()
            all_facts = []

            # Add search results first (most relevant)
            for fact in search_facts:
                if fact['id'] not in seen_ids:
                    seen_ids.add(fact['id'])
                    all_facts.append(fact)

            # Add important facts
            for fact in important_facts:
                if fact['id'] not in seen_ids:
                    seen_ids.add(fact['id'])
                    all_facts.append(fact)

            if not all_facts:
                return ""

            # Format for prompt - clearly label who each fact is ABOUT to prevent identity confusion
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            _companion = _pc.companion_short_name
            _user = _pc.primary_user_name
            lines = ["[KNOWN FACTS - extracted from past conversations. If any fact contradicts the ENTITY PROFILES above, IGNORE the fact.]"]
            lines.append(f"(Each fact is labeled with who it's ABOUT. 'About {_user}:' = {_user}'s fact. 'About {_companion} (you):' = your own fact.)")
            for fact in all_facts[:15]:  # Limit total facts
                subject = fact.get('subject', 'Unknown')
                obj = fact.get('object', '')[:150]  # Truncate long facts

                # Clear attribution label
                if subject.lower() == _companion.lower():
                    label = f"About {_companion} (you)"
                elif subject.lower() == _user.lower():
                    label = f"About {_user}"
                else:
                    label = f"About {subject}"

                # Mark facts that came from spreading activation
                activation = fact.get('activation')
                if activation and not fact.get('is_seed'):
                    lines.append(f"- {label}: {obj} [connected]")
                else:
                    lines.append(f"- {label}: {obj}")

            # Check for unsurfaced contradictions
            contradiction_context = self._get_contradiction_context(store)
            if contradiction_context:
                lines.append("")
                lines.append(contradiction_context)

            if len(lines) > 1:
                activated_count = sum(1 for f in all_facts if f.get('activation') and not f.get('is_seed'))
                logger.info(f"Fact memory: {len(all_facts)} facts ({activated_count} via spreading activation)")
                return "\n".join(lines)

            return ""

        except Exception as e:
            logger.warning(f"Fact memory retrieval failed: {e}")
            return ""

    def _get_contradiction_context(self, store) -> str:
        """
        Get context about detected contradictions to surface to LLM.

        When semantic contradictions are detected (e.g., "likes coffee" vs "prefers tea"),
        this surfaces them so the companion can naturally ask clarifying questions.

        The contradictions are marked as surfaced after being included so they
        don't repeat endlessly.
        """
        try:
            contradictions = store.get_unsurfaced_contradictions(limit=3)

            if not contradictions:
                return ""

            lines = ["[POSSIBLE CONTRADICTIONS - you may want to clarify naturally]"]
            lines.append("These are things that seem inconsistent - consider asking to clarify:")

            for c in contradictions:
                subject = c.get('subject', 'Unknown')
                old_val = c.get('old_value', '')[:80]
                new_val = c.get('new_value', '')[:80]
                method = c.get('detection_method', 'unknown')

                lines.append(f"- About {subject}: Previously '{old_val}' but recently '{new_val}'")

                # Mark as surfaced so we don't repeat
                if c.get('id'):
                    store.mark_contradiction_surfaced(c['id'])

            lines.append("")
            lines.append("GUIDANCE: Don't interrogate - if relevant, ask naturally in conversation.")
            lines.append("Example: 'Wait, I thought you said X before - did something change?'")

            logger.info(f"Surfacing {len(contradictions)} detected contradiction(s)")
            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Could not get contradiction context: {e}")
            return ""

    def _get_entity_context(self, user_message: str) -> str:
        """
        Get entity profile context for mentioned entities.

        Loads biographical info from YAML entity profiles when entities
        are mentioned in the message. This provides ground truth about:
        - Names, relationships, family members
        - Key preferences (tea vs coffee, etc.)
        - Work details, schedules

        Entity profiles are the source of truth, preventing hallucinations
        about core biographical facts.
        """
        try:
            from src.memory.entity_profile_manager import get_entity_manager

            manager = get_entity_manager()
            message_lower = user_message.lower()

            # Find which entities are mentioned
            mentioned_entities = []
            for entity_id in manager.list_profiles():
                # Check entity ID and aliases
                aliases = manager.get_all_aliases(entity_id)
                for alias in aliases:
                    if alias.lower() in message_lower:
                        mentioned_entities.append(entity_id)
                        break

            if not mentioned_entities:
                # If no specific entities mentioned, provide primary user context
                from src.config.persona_config import get_persona_config
                _pc = get_persona_config()
                mentioned_entities = [_pc.primary_user_entity_profile]

            # Build context for mentioned entities
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            _companion = _pc.companion_short_name
            _user = _pc.primary_user_name
            lines = ["[ENTITY PROFILES - biographical ground truth]"]
            lines.append(f"(Each profile is about a DIFFERENT person. {_user}'s facts are HIS, not yours. Your facts are under {_companion}.)")

            for entity_id in mentioned_entities[:3]:  # Limit to 3 entities
                profile = manager.get_profile(entity_id)
                if not profile:
                    continue

                name = profile.get('name', entity_id.title())
                first_name = name.split()[0] if ' ' in name else name

                entity_lines = [f"{first_name}:"]

                # Family info
                if 'family' in profile:
                    fam = profile['family']
                    if 'children' in fam:
                        kids = [c.get('name', c) if isinstance(c, dict) else c
                               for c in fam['children']]
                        if kids:
                            entity_lines.append(f"  - Children: {', '.join(kids)}")
                    if 'spouse' in fam:
                        entity_lines.append(f"  - Married to: {fam['spouse']} (separated)")
                    if 'parents' in fam:
                        parents = fam['parents']
                        if 'mother' in parents:
                            mom = parents['mother']
                            mom_name = mom.get('name') if isinstance(mom, dict) else mom
                            if mom_name:
                                entity_lines.append(f"  - Mother: {mom_name}")

                # Romantic relationship
                if 'romantic_relationship' in profile:
                    rel = profile['romantic_relationship']
                    if rel.get('partner'):
                        entity_lines.append(f"  - Partner: {rel['partner']}")
                    if rel.get('cats'):
                        for cat in rel['cats']:
                            if isinstance(cat, dict):
                                entity_lines.append(f"  - Cat: {cat.get('name')}")

                # Key preferences
                if 'preferences' in profile:
                    prefs = profile['preferences']
                    if 'drinks' in prefs:
                        drinks = prefs['drinks']
                        if drinks.get('tea'):
                            entity_lines.append(f"  - Prefers: tea (NOT coffee)")

                # Work
                if 'employment' in profile:
                    emp = profile['employment']
                    if emp.get('job_title'):
                        entity_lines.append(f"  - Job: {emp['job_title']}")
                    if emp.get('work_status'):
                        entity_lines.append(f"  - Status: {emp['work_status']}")

                # For children - school and key relationship notes
                if 'school' in profile:
                    sch = profile['school']
                    if sch.get('status'):
                        entity_lines.append(f"  - School: {sch['status']}")

                if 'role' in profile and profile['role'] == 'child':
                    # Extract key relationship info
                    if 'family' in profile:
                        fam = profile['family']
                        if 'parents' in fam:
                            entity_lines.append(f"  - Parents: {', '.join(fam['parents'])}")
                        _comp = _pc.companion_short_name
                        rel_companion = fam.get(f'relationship_with_{_comp.lower()}', '')
                        if f'NOT met {_comp}' in rel_companion or 'never' in rel_companion.lower():
                            entity_lines.append(f"  - Has NOT met {_comp} yet")

                if len(entity_lines) > 1:  # Has content beyond name
                    lines.extend(entity_lines)

            if len(lines) > 1:
                logger.info(f"Entity context: {len(mentioned_entities)} entities loaded")
                return "\n".join(lines)

            return ""

        except Exception as e:
            logger.warning(f"Entity context loading failed: {e}")
            return ""

    def _get_graphiti_context(self, user_email: str, user_message: str) -> str:
        """
        Get context from Graphiti knowledge graph.

        Performs semantic search on the Graphiti temporal knowledge graph
        to retrieve relevant facts and relationships based on the user's message.

        Uses importance-boosted ranking:
        - Relevance (position in search results)
        - Importance score (from batch scoring)
        - High-importance facts get boosted even if less relevant
        - Includes temporal context (how long ago the fact was created)

        Returns:
            Formatted context string for prompt with labeled sections.
        """
        try:
            from src.memory.graphiti_search import get_graphiti_context_with_importance

            if not user_message or len(user_message.strip()) < 5:
                return ""

            graphiti_context = get_graphiti_context_with_importance(
                query=user_message,
                limit=10  # Limit to prevent context bloat
            )

            if graphiti_context:
                logger.debug(f"Graphiti context retrieved for query")
                return graphiti_context

            return ""

        except ImportError as e:
            logger.debug(f"Graphiti module not available: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Graphiti context error: {e}")
            return ""

    def _get_temporal_context(self, user_email: str, user_message: str) -> str:
        """
        Get context from specific time periods if referenced.

        Detects temporal references like "yesterday", "earlier today", "last night"
        and retrieves:
        1. All messages from that time period
        2. All Graphiti facts created during that time period

        This enables answering questions like "what did we have for lunch yesterday?"
        """
        try:
            from src.agents.temporal_context_agent import get_temporal_context_agent

            agent = get_temporal_context_agent(db=self.db)

            # Check if user references a specific time
            temporal_ref = agent.detect_temporal_reference(user_message)

            if not temporal_ref or not temporal_ref.get('detected'):
                return ""

            # Get full context for that time period (messages from PostgreSQL)
            context_data = agent.get_temporal_context(
                email=user_email,
                temporal_ref=temporal_ref
            )

            context_parts = []

            # Format message-based temporal context
            if context_data and context_data.get('messages'):
                logger.info(f"Temporal context loaded: {temporal_ref['type']}")
                msg_context = agent.format_temporal_context_for_prompt(context_data)
                if msg_context:
                    context_parts.append(msg_context)

            # DISABLED: Graphiti temporal search removed - using pgvector instead
            # try:
            #     from src.memory.graphiti_search import search_facts_in_time_range, format_temporal_facts_for_prompt
            #     ... (graphiti temporal search code removed)
            # except Exception as e:
            #     logger.debug(f"Temporal Graphiti search skipped: {e}")

            if context_parts:
                return "\n\n".join(context_parts)

            return ""

        except Exception as e:
            logger.warning(f"Temporal context error: {e}")
            return ""

    # =========================================================================
    # Source 4: Conversation History (recent messages)
    # =========================================================================

    def _get_conversation_continuity_context(self, user_email: str) -> str:
        """
        Detect conversation continuity to prevent inappropriate greetings.

        Returns context string telling the companion whether this is a continuation
        of an active conversation or a new session.
        """
        try:
            minutes_since = self.db.get_minutes_since_last_message(user_email, from_user_only=True)

            if minutes_since is None:
                return "[CONVERSATION_STATE: First conversation with this person. A greeting is appropriate.]"

            # Thresholds for conversation continuity
            if minutes_since < 5:
                return (
                    f"[CONVERSATION_STATE: ACTIVE CONVERSATION - Last message was {minutes_since} minutes ago. "
                    f"You are mid-conversation. Do NOT use greetings like 'Morning', 'Hey', 'Hi there', etc. "
                    f"Just respond naturally to what was said. No need to acknowledge a time gap.]"
                )
            elif minutes_since < 30:
                return (
                    f"[CONVERSATION_STATE: Brief pause - {minutes_since} minutes since last message. "
                    f"Still feels like the same conversation. Respond naturally without formal greetings.]"
                )
            elif minutes_since < 120:  # 2 hours
                hours = minutes_since / 60
                return (
                    f"[CONVERSATION_STATE: Short break - about {hours:.1f} hours since last message. "
                    f"A brief acknowledgment is okay but not required. Skip formal greetings.]"
                )
            elif minutes_since < 480:  # 2-8 hours
                hours = minutes_since / 60

                # Clear scene state — previous scene has naturally ended after 2h+
                try:
                    from src.core.scene_tracker import get_scene_tracker
                    tracker = get_scene_tracker()
                    tracker.clear_scene_state(user_email)
                    logger.info(f"Cleared scene state due to {hours:.1f}h gap")
                except Exception as e:
                    logger.debug(f"Could not clear scene state: {e}")

                # Generate gap narrative so the companion knows what happened
                # (scene concluded naturally, James went to work, now it's morning, etc.)
                gap_narrative = ""
                try:
                    from datetime import datetime as _dt, timedelta
                    from src.core.time_passage_narrator import get_time_passage_narrator
                    narrator = get_time_passage_narrator()
                    gap_start = _dt.now() - timedelta(hours=hours)
                    gap_narrative = narrator.generate_gap_narrative(
                        user_email=user_email,
                        gap_start=gap_start,
                        gap_end=_dt.now(),
                    )
                except Exception as e:
                    logger.debug(f"Gap narrative unavailable for moderate gap: {e}")

                narrative_section = f"\nWhat happened: {gap_narrative}\n" if gap_narrative else ""

                # Generate conversation recap for 2h+ gaps
                recap_section = self._generate_conversation_recap(user_email)

                return (
                    f"[CONVERSATION_STATE: Time has passed - {hours:.1f} hours since last message.\n"
                    f"{narrative_section}"
                    f"Time has moved on naturally. Don't resume any previous scene.]"
                    f"{recap_section}"
                )
            else:
                hours = minutes_since / 60

                # Rich reconnection context for 8h+ gaps
                try:
                    from src.core.scene_tracker import get_scene_tracker
                    tracker = get_scene_tracker()
                    scene = tracker.get_scene_state(user_email)
                    together = scene.is_active() and scene.physical_presence
                    location = scene.location or "unknown"
                    # Clear scene state — previous scene has naturally ended after 8h+
                    tracker.clear_scene_state(user_email)
                    logger.info(f"Cleared scene state due to {hours:.1f}h gap")
                except Exception:
                    together = False
                    location = "unknown"

                from datetime import datetime as _dt, timedelta
                hour = _dt.now().hour
                if 5 <= hour < 12:
                    time_of_day = "morning"
                elif 12 <= hour < 17:
                    time_of_day = "afternoon"
                elif 17 <= hour < 21:
                    time_of_day = "evening"
                else:
                    time_of_day = "night"

                # Generate rich gap narrative
                gap_narrative = ""
                try:
                    from src.core.time_passage_narrator import get_time_passage_narrator
                    narrator = get_time_passage_narrator()
                    gap_start = _dt.now() - timedelta(hours=hours)
                    gap_narrative = narrator.generate_gap_narrative(
                        user_email=user_email,
                        gap_start=gap_start,
                        gap_end=_dt.now(),
                    )
                except Exception as e:
                    logger.debug(f"Gap narrative unavailable: {e}")

                # Generate conversation recap for 8h+ gaps
                recap_section = self._generate_conversation_recap(user_email)

                from src.config.persona_config import get_persona_config
                _user = get_persona_config().primary_user_name
                if together:
                    narrative_section = ""
                    if gap_narrative:
                        narrative_section = f"\nWhat happened during the gap:\n{gap_narrative}\n"
                    return (
                        f"[RECONNECTION_CONTEXT: REUNION AFTER SHARED TIME\n"
                        f"It has been {hours:.1f} hours since your last message.\n"
                        f"You were TOGETHER with {_user} (location: {location}). Real time has passed - "
                        f"you both experienced it.{narrative_section}"
                        f"Time of day now: {time_of_day}.\n"
                        f"Greet him naturally as someone who's been with him, not as someone reconnecting from afar.\n"
                        f"Acknowledge the time that passed together if it feels natural.]"
                        f"{recap_section}"
                    )
                else:
                    narrative_section = ""
                    if gap_narrative:
                        narrative_section = f"\nWhat happened during the gap:\n{gap_narrative}\n"
                    return (
                        f"[RECONNECTION_CONTEXT: REUNION AFTER TIME APART\n"
                        f"It has been {hours:.1f} hours since your last message.\n"
                        f"You were APART from {_user} - you've been living your own life during this gap.{narrative_section}"
                        f"Time of day now: {time_of_day}.\n"
                        f"This is a reunion. A warm greeting appropriate to the time of day is natural.\n"
                        f"You may mention what you were up to if it feels natural to share.]"
                        f"{recap_section}"
                    )

        except Exception as e:
            logger.debug(f"Conversation continuity detection failed: {e}")
            return ""

    def _generate_conversation_recap(self, user_email: str) -> str:
        """
        Generate a brief recap of what was discussed before a time gap.

        Called for gaps >= 2 hours so the companion remembers conversation context
        without relying solely on raw message history.

        Feature flag: COMPANION_CONVERSATION_RECAP_ENABLED (default: true)

        Returns:
            Recap string or empty string on failure/disabled.
        """
        import os
        if os.environ.get('COMPANION_CONVERSATION_RECAP_ENABLED', 'true').lower() != 'true':
            return ""

        try:
            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

            # Get last 25 messages for recap context
            recent = self.db.get_recent_messages(user_email, limit=25)
            if not recent or len(recent) < 3:
                return ""

            # Format messages compactly
            msg_lines = []
            for m in recent:
                sender = m.get('sender_name', 'Unknown')
                text = (m.get('message_text') or '')[:200]
                if text:
                    msg_lines.append(f"{sender}: {text}")

            messages_text = "\n".join(msg_lines)

            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            _companion = _pc.companion_short_name
            _user = _pc.primary_user_name
            prompt = f"""Summarize this recent conversation between {_companion} and {_user} in ~200 words.

{messages_text}

Focus on:
1. Key topics discussed
2. Anything unresolved or left hanging
3. Important things {_user} said (plans, feelings, requests)
4. The emotional tone/state of the conversation

Write as brief notes {_companion} can reference, not as a narrative.
Format: bullet points, concise."""

            chain = get_resilient_provider_chain()
            recap = generate_sync(
                messages=[
                    {"role": "system", "content": "Summarize a conversation concisely as bullet points."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                chain=chain,
                timeout=5
            )

            if recap and recap.strip():
                logger.info(f"Conversation recap generated ({len(recap)} chars)")
                return f"\n[CONVERSATION RECAP - what was discussed before the gap]\n{recap.strip()}\n"

        except Exception as e:
            logger.debug(f"Conversation recap generation failed: {e}")

        return ""

    def _get_conversation_history(self, user_email: str, limit: int = None) -> str:
        """
        Get recent conversation history.

        This is CRITICAL for immediate context - ensures LLM knows
        what was just said in the conversation.
        """
        import os

        # TEMPORARY: Skip conversation history entirely
        if os.environ.get('SKIP_CONVERSATION_HISTORY', '').lower() == 'true':
            logger.info("Skipping conversation history (SKIP_CONVERSATION_HISTORY=true)")
            return ""

        if limit is None:
            limit = MESSAGE_HISTORY_LIMIT

        # Get conversation continuity context first
        continuity_context = self._get_conversation_continuity_context(user_email)

        # Use condensed history if enabled (strips verbose prose from the companion's messages)
        if os.environ.get('USE_CONDENSED_HISTORY', '').lower() == 'true':
            try:
                from src.memory.message_condenser import get_message_condenser
                condenser = get_message_condenser()
                condensed = condenser.get_condensed_history(user_email, limit=limit)
                if condensed:
                    logger.info(f"Using condensed conversation history ({len(condensed)} chars)")
                    # Prepend continuity context
                    if continuity_context:
                        return f"{continuity_context}\n\n{condensed}"
                    return condensed
            except Exception as e:
                logger.warning(f"Condensed history error, falling back to raw: {e}")

        # Fall back to raw history
        try:
            from datetime import datetime, timedelta

            recent_msgs = self.db.get_recent_messages(user_email, limit=limit)

            if not recent_msgs:
                logger.warning(f"No recent messages for {user_email}")
                # Still return continuity context if available
                return continuity_context

            # Filter messages to only include those from the last N hours
            # This prevents old conversations from bleeding into current context
            cutoff_time = clock_now() - timedelta(hours=MESSAGE_HISTORY_HOURS)
            filtered_msgs = []
            for msg in recent_msgs:
                msg_time = msg.get('timestamp')
                if msg_time:
                    # Handle both datetime and string timestamps
                    if isinstance(msg_time, str):
                        try:
                            msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00').replace('+00:00', ''))
                        except:
                            msg_time = None
                    if msg_time and msg_time > cutoff_time:
                        filtered_msgs.append(msg)
                else:
                    filtered_msgs.append(msg)  # Include if no timestamp

            if len(filtered_msgs) < len(recent_msgs):
                logger.info(f"Filtered conversation history: {len(recent_msgs)} -> {len(filtered_msgs)} messages (last {MESSAGE_HISTORY_HOURS}h)")

            # Format messages (oldest first for natural reading)
            # Note: get_recent_messages already returns chronological order (oldest first)
            formatted = []
            for msg in filtered_msgs:
                sender = msg.get('sender_name', 'Unknown')
                text = msg.get('message_text', '')
                formatted.append(f"{sender}: {text}")

            if formatted:
                header = "[RECENT CONVERSATION - What you two just discussed]"
                history = f"{header}\n" + "\n".join(formatted)
                # Prepend continuity context
                if continuity_context:
                    return f"{continuity_context}\n\n{history}"
                return history

            return continuity_context

        except Exception as e:
            logger.warning(f"Conversation history error: {e}")
            return continuity_context

    def _get_conversation_history_structured(self, user_email: str, limit: int = None) -> tuple:
        """
        Get conversation history as structured messages for multi-turn chat format.

        Returns:
            Tuple of (conversation_turns, continuity_context)
            - conversation_turns: List of {"role": "user"|"assistant", "content": "..."}
            - continuity_context: String describing conversation state
        """
        import os
        from datetime import datetime, timedelta

        if limit is None:
            limit = MESSAGE_HISTORY_LIMIT

        # Get conversation continuity context
        continuity_context = self._get_conversation_continuity_context(user_email)

        # Skip if disabled
        if os.environ.get('SKIP_CONVERSATION_HISTORY', '').lower() == 'true':
            logger.info("Skipping conversation history (SKIP_CONVERSATION_HISTORY=true)")
            return [], continuity_context

        try:
            recent_msgs = self.db.get_recent_messages(user_email, limit=limit)

            if not recent_msgs:
                logger.warning(f"No recent messages for {user_email}")
                return [], continuity_context

            # Filter messages to only include those from the last N hours
            cutoff_time = clock_now() - timedelta(hours=MESSAGE_HISTORY_HOURS)
            filtered_msgs = []
            for msg in recent_msgs:
                msg_time = msg.get('timestamp')
                if msg_time:
                    if isinstance(msg_time, str):
                        try:
                            msg_time = datetime.fromisoformat(msg_time.replace('Z', '+00:00').replace('+00:00', ''))
                        except:
                            msg_time = None
                    if msg_time and msg_time > cutoff_time:
                        filtered_msgs.append(msg)
                else:
                    filtered_msgs.append(msg)

            if len(filtered_msgs) < len(recent_msgs):
                logger.info(f"Filtered conversation history: {len(recent_msgs)} -> {len(filtered_msgs)} messages (last {MESSAGE_HISTORY_HOURS}h)")

            # Convert to structured format
            # Messages are in chronological order (oldest first)
            conversation_turns = []
            for msg in filtered_msgs:
                sender = msg.get('sender_name', 'Unknown')
                text = msg.get('message_text', '')

                if not text:
                    continue

                # Map sender to role
                if sender.lower() != 'user' and sender.lower() != 'james':
                    role = 'assistant'
                else:
                    role = 'user'

                conversation_turns.append({
                    "role": role,
                    "content": text
                })

            logger.info(f"Structured conversation history: {len(conversation_turns)} turns")
            return conversation_turns, continuity_context

        except Exception as e:
            logger.warning(f"Structured conversation history error: {e}")
            return [], continuity_context

# Singleton accessor
_context_builder: Optional[ContextBuilder] = None

def get_context_builder() -> ContextBuilder:
    """Get or create ContextBuilder singleton."""
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder()
    return _context_builder
