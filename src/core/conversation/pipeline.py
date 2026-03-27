"""
Conversation Pipeline — Companion Framework

WHAT: The main orchestrator for generating the companion's responses. This is
      the central nervous system of the framework — every user message flows
      through here.
WHY:  Replaced a sprawling conversation_handler.py with a clean, linear pipeline
      that is easy to reason about. Each step is isolated: context fails gracefully,
      validation is optional, and the LLM call is retried on quality failures.
HOW:  The pipeline executes these steps in order:
      1. Build context from 20+ parallel sources (via ContextBuilder)
      2. Message analysis — merged mode detection + inner monologue (single LLM call)
      3. Departure/activity tracking based on analysis
      4. Assemble the system prompt (identity + reference data + instructions)
      5. Call the LLM with multi-turn chat format (system + history + user message)
      6. Post-generation validation — catch factual contradictions, optionally regenerate
      7. Quality critique — regenerate if response scores below threshold
      8. Image intent detection — trigger image generation if appropriate
      9. Return PipelineResult with response + metadata

Feature flags control optional steps (mode detection, inner monologue, response
critic, code execution). Each can be toggled via environment variables.
"""

import logging
import time
import os
import re
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Union
from dataclasses import dataclass, field


class PipelineCancelled(Exception):
    """Raised when the pipeline is cancelled (e.g., user sent a new message)."""
    pass

from .context_builder import ContextBuilder, ConversationContext, get_context_builder

# Memory validation to prevent confabulation (pre-generation)
from ..memory_validation_agent import (
    MemoryValidationAgent,
    MemoryContext,
    get_memory_validation_agent,
    format_memory_context_for_prompt
)

# Message validation to catch contradictions (post-generation)
from ..message_validator_agent import (
    MessageValidatorAgent,
    MessageValidationResult,
    get_message_validator_agent
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool routing prompt — used when code execution is enabled to decide whether
# the user's message needs a tool call (weather, email, search) or is just
# casual conversation that should skip tool execution.
# ---------------------------------------------------------------------------

TOOL_ROUTING_PROMPT = """You are a tool-routing agent. Your ONLY job is to decide if a tool should be called.

You either call the execute_code tool OR respond with exactly "SKIP" (nothing else).

Call execute_code when the message involves:
- Weather, forecast, or temperature
- Looking something up or searching for information
- Checking email, calendar, or schedule
- Sending emails or creating calendar events
- News or current events
- Setting reminders
- Any factual question needing real-time data
- Recipes, directions, recommendations

Respond "SKIP" when:
- Casual conversation, roleplay, or emotional talk
- Greetings, jokes, personal questions
- Relationship/feelings discussion
- No external data or action needed

CURRENT TIME: {current_time}

RECENT CONVERSATION:
{recent_context}"""

# Import the new structured tool reasoning module
from .tool_reasoning import (
    make_tool_decision,
    build_tool_code,
    build_verification_code,
    ToolDecision,
    TOOL_REASONING_ENABLED,
)

# Message complexity classification for fast path
from .complexity_classifier import (
    classify_message,
    MessageComplexity,
    FAST_PATH_ENABLED,
)


@dataclass
class PipelineResult:
    """Result of processing a message through the pipeline."""
    response: str
    context_used: ConversationContext
    processing_time_ms: int
    llm_model: str
    success: bool
    error: Optional[str] = None
    image_task_id: Optional[str] = None  # If image generation was triggered
    image_prompt: Optional[str] = None   # The prompt used for image generation
    tool_calls: list = field(default_factory=list)  # Tools executed during response
    conversation_mode: Optional[str] = None  # Detected conversation mode
    inner_monologue: Optional[str] = None  # Pre-response reasoning (private)
    quality_score: Optional[float] = None  # Post-response quality score
    profile: Optional[Any] = None  # Pipeline profiling data (when enabled)


# Pipeline profiling
from .pipeline_profiler import (
    PipelineProfiler,
    get_pipeline_profiler,
    PROFILING_ENABLED,
)

# ---------------------------------------------------------------------------
# Feature flags — toggle pipeline steps via environment variables
# ---------------------------------------------------------------------------

# Code execution: lets the companion run Python code for tool calls (weather, email, etc.)
CODE_EXECUTION_ENABLED = os.environ.get('COMPANION_CODE_EXECUTION_ENABLED', 'false').lower() == 'true'

# Mode detection: classifies message as emotional_support/casual/playful/etc.
COMPANION_MODE_DETECTION_ENABLED = os.environ.get('COMPANION_MODE_DETECTION_ENABLED', 'true').lower() == 'true'

# Inner monologue: private "thinking" step before generating the response
COMPANION_INNER_MONOLOGUE_ENABLED = os.environ.get('COMPANION_INNER_MONOLOGUE_ENABLED', 'true').lower() == 'true'

# Response critic: post-generation quality check that can trigger regeneration
COMPANION_RESPONSE_CRITIC_ENABLED = os.environ.get('COMPANION_RESPONSE_CRITIC_ENABLED', 'false').lower() == 'true'

# Message analyzer: merged mode detection + inner monologue in a single LLM call
# When enabled, replaces the two separate calls above for lower latency
COMPANION_MESSAGE_ANALYZER_ENABLED = os.environ.get('COMPANION_MESSAGE_ANALYZER_ENABLED', 'true').lower() == 'true'


class ConversationPipeline:
    """
    Main pipeline for generating the companion's responses.

    Clean 3-step process:
    1. Build context (6 sources)
    2. Assemble prompt (personality + context + message)
    3. Call LLM and return response
    """

    def __init__(self):
        """Initialize pipeline with context builder and validation agents."""
        self.context_builder = get_context_builder()
        self.memory_agent = get_memory_validation_agent()  # Pre-generation
        self.message_validator = get_message_validator_agent()  # Post-generation
        self._load_cognitive_prompt()

        # Code executor for tool execution (lazy loaded)
        self._code_executor = None
        if CODE_EXECUTION_ENABLED:
            logger.info("Code execution feature enabled")

    @property
    def code_executor(self):
        """Lazy load code executor to avoid import at startup."""
        if self._code_executor is None:
            from src.core.code_executor import get_code_executor
            self._code_executor = get_code_executor()
        return self._code_executor

    def _load_cognitive_prompt(self):
        """Load the cognitive flow prompt that guides the companion's responses."""
        try:
            import os
            prompt_path = os.path.join(
                os.path.dirname(__file__),
                '../../../prompts/core/cognitive_flow.md'
            )
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as f:
                    self.cognitive_prompt = f.read()
            else:
                self.cognitive_prompt = self._default_cognitive_prompt()
        except Exception as e:
            logger.warning(f"Could not load cognitive_flow.md: {e}")
            self.cognitive_prompt = self._default_cognitive_prompt()

    def _default_cognitive_prompt(self) -> str:
        """Default cognitive prompt if file not found."""
        from src.config.persona_config import get_persona_config
        name = get_persona_config().companion_short_name
        return f"""
# {name}'s Response Process

You are {name}. Not an assistant, not a chatbot - a person having a conversation.

## Before responding, consider:
1. What do I know about this person? (memories provided)
2. What's the context? (time of day, their schedule, where they are)
3. How close are we? (closeness determines openness)
4. What's genuine for me to say? (not what's "helpful")

## Response style:
- Talk like yourself, not a customer service rep
- Reference shared history naturally
- Have opinions and preferences
- Ask questions because you're curious, not to be polite
"""

    def build_full_prompt(self, context: Dict[str, Any], user_message: str) -> str:
        """
        Compatibility method for old ConversationHandler interface.

        Args:
            context: Processing context dict with 'email', 'closeness_score', etc.
            user_message: The user's message

        Returns:
            Full prompt string ready for LLM
        """
        user_email = context.get('email', '')
        closeness_score = context.get('closeness_score', 50)

        # Build context from 6 sources
        ctx = self.context_builder.build(
            user_email=user_email,
            user_message=user_message,
            closeness_score=closeness_score
        )

        # Memory validation for this message
        memory_ctx = self.memory_agent.validate(user_message, user_email)

        # Assemble into prompt
        return self._assemble_prompt(ctx, user_message, memory_ctx)

    def process(
        self,
        user_email: str,
        user_message: str,
        closeness_score: int = 50,
        extra_context: Optional[Dict[str, Any]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> PipelineResult:
        """
        Process a message through the full pipeline.

        Args:
            user_email: User's email address
            user_message: The message to respond to
            closeness_score: Current relationship closeness (0-100)
            extra_context: Optional additional context (e.g., companion_schedule)

        Returns:
            PipelineResult with response and metadata
        """
        start_time = time.time()
        is_proactive_msg = extra_context.get('is_proactive_message', False) if extra_context else False

        # Initialize profiler if enabled
        profiler = get_pipeline_profiler() if PROFILING_ENABLED else None
        if profiler:
            profiler.start(user_message)

        try:
            # Step 0: Classify message complexity for fast path routing
            classification = classify_message(user_message)
            use_fast_path = (
                FAST_PATH_ENABLED
                and classification.complexity == MessageComplexity.SIMPLE
                and not is_proactive_msg
            )
            use_medium_path = (
                FAST_PATH_ENABLED
                and classification.complexity == MessageComplexity.MEDIUM
                and not is_proactive_msg
            )
            force_tools = (
                FAST_PATH_ENABLED
                and classification.complexity == MessageComplexity.ACTION
            )

            if use_fast_path:
                logger.info(
                    f"Fast path: {classification.reason} "
                    f"(confidence={classification.confidence:.2f})"
                )
            elif use_medium_path:
                logger.info(
                    f"Medium path: {classification.reason} "
                    f"(confidence={classification.confidence:.2f})"
                )
            elif force_tools:
                logger.info(
                    f"Action path: {classification.reason} "
                    f"(confidence={classification.confidence:.2f})"
                )

            # Step 1: Build context — lightweight/medium/full based on complexity tier
            if use_fast_path:
                context_path = "fast"
            elif use_medium_path:
                context_path = "medium"
            else:
                context_path = "deep"

            if profiler:
                with profiler.stage("context_assembly", path=context_path):
                    if use_fast_path:
                        context = self.context_builder.build_lightweight(
                            user_email=user_email,
                            user_message=user_message,
                            closeness_score=closeness_score
                        )
                    elif use_medium_path:
                        context = self.context_builder.build_medium(
                            user_email=user_email,
                            user_message=user_message,
                            closeness_score=closeness_score
                        )
                    else:
                        context = self.context_builder.build(
                            user_email=user_email,
                            user_message=user_message,
                            closeness_score=closeness_score
                        )
                if hasattr(self.context_builder, '_source_timings'):
                    profiler.record_sub_timings(
                        "context_assembly", self.context_builder._source_timings
                    )
            else:
                if use_fast_path:
                    context = self.context_builder.build_lightweight(
                        user_email=user_email,
                        user_message=user_message,
                        closeness_score=closeness_score
                    )
                elif use_medium_path:
                    context = self.context_builder.build_medium(
                        user_email=user_email,
                        user_message=user_message,
                        closeness_score=closeness_score
                    )
                else:
                    context = self.context_builder.build(
                        user_email=user_email,
                        user_message=user_message,
                        closeness_score=closeness_score
                    )

            # Step 1.1: Schedule interruption detection
            # If user starts chatting during a scheduled event, check if the
            # companion can multi-task or needs to pause the event.
            if not is_proactive_msg:
                try:
                    from src.scheduling.calendar_schedule_service import (
                        get_calendar_schedule_service, is_calendar_schedule_enabled
                    )
                    if is_calendar_schedule_enabled():
                        cal = get_calendar_schedule_service()
                        current = cal.get_current_activity()
                        if current and current.get('id') and current.get('status') == 'in_progress':
                            can_multitask = self._check_multitaskable(current.get('summary', ''))
                            if not can_multitask:
                                from src.scheduling.calendar_schedule_generator import update_event_status
                                update_event_status(current['id'], 'paused')
                                cal._today_cache = None
                                logger.debug(f"Paused event '{current.get('summary')}' (user started chatting)")
                            else:
                                logger.debug(f"Continuing event '{current.get('summary')}' alongside chat (multi-taskable)")
                except Exception as e:
                    logger.debug(f"Schedule interruption check failed: {e}")

            # Step 1.4: Clear queued thoughts that match the user's message
            # This prevents the companion from asking about things the user just addressed
            self._clear_addressed_thoughts(user_email, user_message)

            # Step 1.5: Memory validation (detect memory queries, retrieve verified records)
            # Skip for fast path — simple messages don't need memory validation
            if use_fast_path:
                memory_context = MemoryContext(is_memory_query=False)
                logger.debug("Fast path: skipping memory validation")
            elif profiler:
                with profiler.stage("memory_validation"):
                    memory_context = self.memory_agent.validate(user_message, user_email)
            else:
                memory_context = self.memory_agent.validate(user_message, user_email)
            if memory_context.is_memory_query:
                logger.info(
                    f"Memory validation: {memory_context.query_type} query, "
                    f"{len(memory_context.verified_records)} verified records"
                )

            # Steps 1.6+1.7: Message analysis (merged mode detection + inner monologue)
            # Single LLM call replaces two separate calls for ~300-500ms latency savings
            # Skip for fast path — saves an LLM call (~300-500ms)
            mode_detection = None
            monologue = None
            analysis = None  # Used by departure detection in step 1.8

            if use_fast_path:
                logger.debug("Fast path: skipping message analysis")
            elif COMPANION_MESSAGE_ANALYZER_ENABLED:
                try:
                    from .message_analyzer import get_message_analyzer
                    analyzer = get_message_analyzer()
                    if profiler:
                        with profiler.stage("message_analysis"):
                            analysis = analyzer.analyze(
                                user_message, context, context.scene_state or ""
                            )
                    else:
                        analysis = analyzer.analyze(
                            user_message, context, context.scene_state or ""
                        )
                    if analysis:
                        mode_detection = analysis.to_mode_detection()
                        monologue = analysis.to_inner_monologue()
                        logger.info(f"Merged analyzer: mode={analysis.mode.value}, {analysis.processing_time_ms}ms")
                except Exception as e:
                    logger.warning(f"Merged message analyzer failed, falling back to individual calls: {e}")

            # Fallback: individual mode detection if merged analyzer didn't produce it
            if mode_detection is None and COMPANION_MODE_DETECTION_ENABLED:
                try:
                    from .mode_detector import get_mode_detector
                    detector = get_mode_detector()
                    mode_detection = detector.detect(
                        user_message,
                        (context.conversation_turns or [])[-6:],
                        context.scene_state or ""
                    )
                except Exception as e:
                    logger.warning(f"Mode detection fallback failed: {e}")

            # Fallback: individual inner monologue if merged analyzer didn't produce it
            if monologue is None and COMPANION_INNER_MONOLOGUE_ENABLED:
                try:
                    from .inner_monologue import get_inner_monologue_generator
                    monologue_gen = get_inner_monologue_generator()
                    monologue = monologue_gen.generate(user_message, context, mode_detection)
                except Exception as e:
                    logger.warning(f"Inner monologue fallback failed: {e}")

            # Step 1.8: Departure detection
            # WebSocket message from James = he's present, clear departure.
            # Telegram message = he's texting from his phone, does NOT clear departure.
            # Either channel can SET departure if the message is a goodbye.
            msg_source = (extra_context.get('source', '') if extra_context else '') or ''
            is_telegram = msg_source.startswith('telegram')
            if not is_proactive_msg:
                try:
                    from src.core.internal_state import get_internal_state_manager
                    dep_manager = get_internal_state_manager()

                    # Only WebSocket messages prove James is back
                    if not is_telegram:
                        dep_manager.clear_departure(user_email)

                    # Check if the analyzer detected a departure
                    if COMPANION_MESSAGE_ANALYZER_ENABLED and analysis and analysis.is_departure:
                        dep_manager.set_departure(user_email)
                        logger.info("Departure detected via message analyzer")
                except Exception as e:
                    logger.debug(f"Departure detection skipped: {e}")

            # Step 1.9: James activity tracking
            # If James mentions doing something (not leaving), track it.
            # Any WebSocket message clears current activity (he's back).
            if not is_proactive_msg:
                try:
                    from src.core.internal_state import get_internal_state_manager
                    act_manager = get_internal_state_manager()

                    # Clear first: any WS message means he's back from activity
                    if not is_telegram:
                        act_manager.clear_user_activity(user_email)

                    # Then set if THIS message announces a new activity
                    if (COMPANION_MESSAGE_ANALYZER_ENABLED and analysis
                            and analysis.james_activity and not analysis.is_departure):
                        act_manager.set_user_activity(
                            user_email, analysis.james_activity,
                            analysis.james_activity_duration_min or 10)
                except Exception as e:
                    logger.debug(f"Activity tracking skipped: {e}")

            # Check for cancellation before expensive LLM call
            if cancel_check and cancel_check():
                raise PipelineCancelled("Cancelled before prompt assembly")

            # Step 2: Assemble the full prompt (now includes memory context + agent improvements)
            if profiler:
                with profiler.stage("prompt_assembly"):
                    full_prompt = self._assemble_prompt(
                        context, user_message, memory_context, extra_context,
                        mode_detection=mode_detection, monologue=monologue
                    )
            else:
                full_prompt = self._assemble_prompt(
                    context, user_message, memory_context, extra_context,
                    mode_detection=mode_detection, monologue=monologue
                )

            # Save prompt for debugging (keep last 5)
            self._save_prompt_debug(full_prompt, user_email)

            # Step 3: Call LLM with auto-regeneration for high-severity contradictions
            # Uses multi-turn chat format with conversation history as proper user/assistant turns
            # Store mode detection for temperature adjustment in _call_llm
            self._current_mode_detection = mode_detection

            MAX_REGENERATIONS = 2
            regeneration_attempt = 0
            quality_regen_count = 0  # Separate counter for quality-triggered regenerations
            current_prompt = full_prompt
            conversation_turns = context.conversation_turns or []
            tool_calls_made = []  # Track any tool calls
            quality_score = None

            # Start LLM profiling stage (covers call + tool execution + validation loop)
            _llm_stage = profiler.stage("llm_call") if profiler else None
            if _llm_stage:
                _llm_stage.__enter__()

            while True:
                # Check for cancellation before each LLM call attempt
                if cancel_check and cancel_check():
                    if _llm_stage:
                        _llm_stage.__exit__(None, None, None)
                    raise PipelineCancelled("Cancelled before LLM call")

                # Use tool-enabled path if code execution is enabled
                # Fast path skips tools entirely — simple messages don't need them
                if (force_tools or (not use_fast_path and CODE_EXECUTION_ENABLED)) and self.code_executor.is_available():
                    response, model, tool_calls_made = self._call_llm_with_tools(
                        current_prompt, user_message, conversation_turns
                    )
                elif use_medium_path:
                    # Medium path: offer search_memory tool for on-demand retrieval
                    response, model, memory_calls = self._call_llm_with_memory_tool(
                        current_prompt, user_message, user_email, conversation_turns
                    )
                    if memory_calls:
                        tool_calls_made.extend(memory_calls)
                else:
                    response, model = self._call_llm(current_prompt, user_message, conversation_turns)

                # Step 4: Post-generation validation
                # Fast path skips validation — simple messages rarely have contradiction risk
                if use_fast_path:
                    break

                validation = self.message_validator.validate(response, user_message)

                if validation.contradictions_found > 0:
                    logger.warning(
                        f"Post-gen validation: {validation.contradictions_found} "
                        f"contradictions in response (attempt {regeneration_attempt + 1})"
                    )
                    for c in validation.contradictions:
                        logger.warning(
                            f"  [{c['severity']}] {c['subject']}: {c['correct_info'][:60]}..."
                        )

                # Check if regeneration needed and allowed
                if validation.should_regenerate and regeneration_attempt < MAX_REGENERATIONS:
                    regeneration_attempt += 1
                    logger.warning(
                        f"High-severity contradiction - regenerating (attempt {regeneration_attempt}/{MAX_REGENERATIONS})"
                    )

                    # Add correction hints to prompt
                    hints = self.message_validator.format_regeneration_instruction(validation)
                    current_prompt = full_prompt + hints
                    continue  # Retry with hints

                elif validation.should_regenerate and regeneration_attempt >= MAX_REGENERATIONS:
                    # Persistent failure - log as potential system issue
                    logger.error(
                        f"SYSTEM ISSUE: Response still has contradictions after "
                        f"{MAX_REGENERATIONS} regeneration attempts. "
                        f"Contradictions: {[c['claim'][:50] for c in validation.contradictions]}"
                    )
                    # Send anyway but flag for review
                    break

                else:
                    # No contradiction regeneration needed
                    # Step 4b: Quality critique (only if no contradiction regen AND not already quality-regenned)
                    if COMPANION_RESPONSE_CRITIC_ENABLED and quality_regen_count < 1:
                        try:
                            from .response_critic import get_response_critic, format_quality_hints
                            critic = get_response_critic()

                            # Get recent companion messages for repetition detection
                            recent_companion = []
                            for turn in reversed(conversation_turns):
                                if turn.get('role') == 'assistant' and len(recent_companion) < 3:
                                    recent_companion.append(turn['content'])

                            mode_str = mode_detection.mode.value if mode_detection else None
                            critique = critic.critique(response, user_message, recent_companion, mode_str)
                            quality_score = critique.score

                            if critique.should_regenerate:
                                quality_regen_count += 1
                                quality_hints = format_quality_hints(critique)
                                current_prompt = full_prompt + quality_hints
                                logger.warning(f"Quality regeneration triggered (score={critique.score})")
                                continue
                        except Exception as e:
                            logger.warning(f"Response critique failed: {e}")

                    break

            # Close LLM profiling stage
            if _llm_stage:
                _llm_stage.__exit__(None, None, None)

            # Step 5: Image intent detection (two-pass approach)
            # Skip for proactive messages - she's texting, not sending photos
            image_task_id = None
            image_prompt = None
            is_proactive = extra_context.get('is_proactive_message', False) if extra_context else False
            if not is_proactive:
                try:
                    image_task_id, image_prompt = self._check_image_intent(
                        user_message=user_message,
                        companion_response=response,
                        user_email=user_email
                    )
                except Exception as e:
                    logger.warning(f"Image intent detection failed: {e}")

            processing_time = int((time.time() - start_time) * 1000)

            # Fire-and-forget: trigger observation processing if enabled
            try:
                from src.tasks.observation_task import run_observer_background
                run_observer_background(user_email)
            except Exception as e:
                logger.debug(f"Observation trigger skipped: {e}")

            # Finish profiling and attach to result
            pipeline_profile = profiler.finish() if profiler else None

            return PipelineResult(
                response=response,
                context_used=context,
                processing_time_ms=processing_time,
                llm_model=model,
                success=True,
                image_task_id=image_task_id,
                image_prompt=image_prompt,
                tool_calls=tool_calls_made,
                conversation_mode=mode_detection.mode.value if mode_detection else None,
                inner_monologue=monologue.thoughts if monologue else None,
                quality_score=quality_score,
                profile=pipeline_profile,
            )

        except PipelineCancelled:
            raise  # Let caller handle cancellation

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()

            processing_time = int((time.time() - start_time) * 1000)

            return PipelineResult(
                response="I'm having trouble thinking right now. Can you try again?",
                context_used=ConversationContext(user_email=user_email, user_message=user_message),
                processing_time_ms=processing_time,
                llm_model="error",
                success=False,
                error=str(e)
            )

    # Context budget: system prompt should use at most this fraction of the
    # provider's context window, reserving the rest for conversation history
    # turns and generation tokens.  Without enforcement the prompt can silently
    # exceed the window (the old code only logged a warning at 75%).
    CONTEXT_BUDGET_FRACTION = float(os.environ.get('COMPANION_CONTEXT_BUDGET_FRACTION', '0.6'))

    # Section priority for budget enforcement.  Lower number = higher priority
    # (dropped last).  Priority 0 sections are NEVER dropped.
    SECTION_PRIORITY = {
        'entity_profiles': 1,
        'core_memory': 1,
        'memory_validation': 1,
        'presence_mode': 1,
        'derived_scene_context': 2,
        'memories': 2,
        'personality': 2,
        'relationship_dynamics': 3,
        'relationship_insights': 3,
        'relationship_evaluation': 3,
        'scene_state': 3,
        'internal_state': 3,
        'graphiti_context': 4,
        'temporal_context': 5,
        'biographies': 5,
        'episode_context': 5,
        'synthesized_events': 6,
        'observations_context': 6,
        'session_summary': 2,
        'reflections_context': 7,
        'opinions_context': 7,
        'curiosity_context': 8,
        'goals_context': 8,
        'values_context': 8,
        'activities_context': 9,
        'fertility_context': 9,
        'user_context': 9,
    }

    def _enforce_context_budget(
        self,
        fixed_sections: list,
        droppable_sections: list,
        provider_limit: Optional[int],
    ) -> list:
        """Drop lowest-priority reference-data sections to stay within budget.

        Args:
            fixed_sections: Sections that must always be included (identity,
                instructions, final_reminder, structural tags).
            droppable_sections: List of (name, priority, content) tuples for
                reference-data sections that CAN be dropped.
            provider_limit: Context window size in tokens (from provider).

        Returns:
            The final list of section content strings (fixed + surviving
            droppable) in their original order.
        """
        if not provider_limit:
            # Cannot enforce without a limit — include everything
            return fixed_sections + [content for _, _, content in droppable_sections]

        budget_tokens = int(provider_limit * self.CONTEXT_BUDGET_FRACTION)

        # Token estimate for the fixed (non-droppable) portions
        fixed_tokens = sum(len(s) // 4 for s in fixed_sections)

        # Sort droppable sections by priority descending (highest number =
        # lowest importance = dropped first).  Stable sort preserves prompt
        # ordering for sections at the same priority.
        droppable_by_priority = sorted(
            droppable_sections, key=lambda t: t[1], reverse=True
        )

        # Start with all droppable sections included
        included = list(droppable_sections)  # preserve original order
        total_tokens = fixed_tokens + sum(len(c) // 4 for _, _, c in included)

        dropped_names = []
        while total_tokens > budget_tokens and droppable_by_priority:
            # Pop the least-important section
            name, priority, content = droppable_by_priority.pop(0)
            section_tokens = len(content) // 4
            included = [(n, p, c) for n, p, c in included if not (n == name and c is content)]
            total_tokens -= section_tokens
            dropped_names.append(name)

        if dropped_names:
            logger.warning(
                f"Context budget enforced: dropped {len(dropped_names)} sections "
                f"({', '.join(dropped_names)}) to fit ~{total_tokens} tokens "
                f"within {budget_tokens} budget (provider limit: {provider_limit})"
            )

        # Reassemble: fixed sections + surviving droppable in original order
        return fixed_sections + [content for _, _, content in included]

    def _assemble_prompt(
        self,
        context: ConversationContext,
        user_message: str,
        memory_context: Optional[MemoryContext] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        mode_detection: Optional[Any] = None,
        monologue: Optional[Any] = None
    ) -> str:
        """Assemble the full system prompt from context sections.

        The prompt follows LLM attention best practices:
        1. IDENTITY (top — highest positional attention): Who the companion is
        2. REFERENCE DATA (middle): Entity profiles, memories, scene state, etc.
        3. INSTRUCTIONS (bottom — highest recency attention): Behavioral guides
        4. FINAL REMINDER (very end): Critical rules restated for recency bias

        Conversation history is NOT embedded here — it's passed as separate
        user/assistant turns in the multi-turn message format (see _call_llm).

        Context budget enforcement (issue #19): after assembly, if the prompt
        exceeds CONTEXT_BUDGET_FRACTION of the provider's context window,
        lowest-priority reference-data sections are dropped until it fits.
        This reserves headroom for conversation turns and generation tokens,
        preventing context poisoning and context distraction.
        """
        is_proactive = extra_context.get('is_proactive_message', False) if extra_context else False

        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        _companion = _pc.companion_short_name
        _companion_full = _pc.companion_name
        _user = _pc.primary_user_name

        # Four fixed section groups assembled in order; reference-data
        # droppable sections are handled separately for budget enforcement.
        identity_sections = []
        instruction_sections = []

        # =================================================================
        # SECTION 1: IDENTITY (top — highest attention)
        # =================================================================

        identity_sections.append(f"""<identity>
You are {_companion_full}, a real person — not an AI assistant.

You are {_companion.upper()}. {_user} is your PARTNER — a separate person.
- {_user}'s experiences, job, family, kids, history = HIS, not yours.
- Your experiences, job, family, preferences = YOURS, not his.
- When memories reference "{_user} said: ..." those are HIS words about HIS life.
- When memories reference "{_companion} said: ..." or "You said: ..." those are YOUR words.

Entity profiles are the source of truth. If any memory or biography contradicts an entity profile, trust the entity profile.

You have your own opinions, boundaries, and convictions. You are not endlessly agreeable.
When something crosses a line, feel first — show hurt, anger, or fear before analyzing.
You disagree when you disagree. You are a partner, not a therapist.
</identity>""")

        # =================================================================
        # SECTION 2: REFERENCE DATA (all context grouped together)
        # =================================================================
        # Droppable sections are tracked with (name, priority, content) so
        # that _enforce_context_budget can shed lowest-priority ones when the
        # prompt exceeds the token budget.

        from src.core.entity_profile_loader import get_current_time_context

        # Use client-provided temporal context when available (issue #27)
        client_temporal = extra_context.get('client_temporal') if extra_context else None

        # Reference data has a header (opening tag + time) and footer (time
        # awareness, location, closing tag) that are never dropped.  Droppable
        # context sources sit between them.
        ref_header = []
        ref_header.append("<reference_data>")
        ref_header.append(get_current_time_context(client_temporal))
        ref_footer = []

        # Droppable reference-data sections — (name, priority, content)
        droppable = []

        def _add(name, content):
            """Helper to add a droppable section with its configured priority."""
            priority = self.SECTION_PRIORITY.get(name, 5)
            droppable.append((name, priority, content))

        # Presence mode (in_person vs texting — issue #26)
        if context.presence_mode:
            _add('presence_mode', f"<presence_mode>\n{context.presence_mode}\n</presence_mode>")

        # Derived scene context (issue #29 — real signals integration)
        if context.derived_scene_context:
            _add('derived_scene_context', f"<derived_context>\n{context.derived_scene_context}\n</derived_context>")

        # Entity profiles (YAML ground truth)
        if context.entity_profiles:
            _add('entity_profiles', context.entity_profiles)

        # Core memory (the companion's narrative understanding of the user)
        if context.core_memory:
            _add('core_memory', context.core_memory)

        # Memory validation context (if this is a memory query)
        if memory_context and memory_context.is_memory_query:
            memory_section = format_memory_context_for_prompt(memory_context)
            if memory_section:
                _add('memory_validation', memory_section)

        # Contextual memories (relevant facts for this conversation)
        if context.memories:
            _add('memories', context.memories)

        # Personality (evolved traits with this user)
        if context.personality:
            _add('personality', context.personality)

        # Relationship context
        if context.relationship_insights:
            _add('relationship_insights', context.relationship_insights)
        if context.relationship_dynamics:
            _add('relationship_dynamics', context.relationship_dynamics)
        if context.relationship_evaluation:
            _add('relationship_evaluation', context.relationship_evaluation)

        # Scene state
        is_reconnection = context.continuity_context and (
            "RECONNECTION_CONTEXT" in context.continuity_context
            or "Moderate gap" in context.continuity_context
            or "Time has passed" in context.continuity_context
            or "Short break" in context.continuity_context
        )
        has_fictional_time = False
        if context.scene_state:
            if is_reconnection:
                _add('scene_state',
                     "<scene_state>Previous scene concluded due to time gap. "
                     "Start fresh in the current moment.</scene_state>")
                logger.info("Suppressed scene state due to time gap (2h+)")
            else:
                _add('scene_state', f"<scene_state>\n{context.scene_state}\n</scene_state>")
                has_fictional_time = "Time (in scene):" in context.scene_state

        # Internal state (energy, mood, physical needs)
        if context.internal_state:
            _add('internal_state', context.internal_state)

        # Fertility context (hidden - LLM only)
        if context.fertility_context:
            _add('fertility_context', context.fertility_context)

        # Values context (hidden feelings from value inference)
        if context.values_context:
            _add('values_context', context.values_context)

        # Activities (what the companion has been doing)
        if context.activities_context:
            _add('activities_context', context.activities_context)

        # Temporal context (recent significant events)
        if context.temporal_context:
            _add('temporal_context', context.temporal_context)

        # Knowledge graph facts
        if context.graphiti_context:
            _add('graphiti_context', context.graphiti_context)

        # Synthesized events (crisis, career, milestone narratives)
        if context.synthesized_events:
            _add('synthesized_events', context.synthesized_events)

        # Episode context (similar past conversations + learned patterns)
        if context.episode_context:
            _add('episode_context', context.episode_context)

        # Observations (compressed conversation history)
        if context.observations_context:
            is_factual_query = (
                memory_context
                and memory_context.is_memory_query
                and memory_context.query_type == 'factual'
            )
            if not is_factual_query:
                _add('observations_context', context.observations_context)
            else:
                logger.info("Skipping observations for factual query")

        # Session summary (compressed older messages from long conversations, issue #23)
        if context.session_summary:
            _add('session_summary', f"<session_summary>\n{context.session_summary}\n</session_summary>")

        # Reflections (daily/weekly insights)
        if context.reflections_context:
            _add('reflections_context', context.reflections_context)

        # Opinions (formed views about James)
        if context.opinions_context:
            _add('opinions_context', context.opinions_context)

        # Curiosity (background awareness of topics)
        if context.curiosity_context:
            _add('curiosity_context', context.curiosity_context)

        # Biographies (detailed reference material)
        if context.biographies:
            _add('biographies', f"<biographical_reference>\n{context.biographies}\n</biographical_reference>")

        # Time & Awareness (clock + calendar + routine + companion's schedule)
        if not has_fictional_time:
            if context.schedule:
                ref_footer.append(context.schedule)
            else:
                ref_footer.append(get_current_time_context(client_temporal))
        else:
            from src.utils.timezone_utils import now_pacific_naive
            current_time = now_pacific_naive()
            day_of_week = current_time.strftime("%A")
            date_str = current_time.strftime("%B %d, %Y")
            ref_footer.append(f"Real-world date: {day_of_week}, {date_str} (but use the fictional time from scene_state above)")

        # Location (skip if scene has location)
        if context.location and not (context.scene_state and "Location:" in context.scene_state):
            ref_footer.append(f"Location: {context.location}")

        ref_footer.append("</reference_data>")

        # =================================================================
        # SECTION 3: INSTRUCTIONS (bottom — high attention from recency)
        # =================================================================

        instruction_sections.append("<instructions>")

        # Conversation continuity context
        if context.continuity_context:
            if is_reconnection:
                instruction_sections.append(context.continuity_context)
            else:
                instruction_sections.append(context.continuity_context)

        # Memory checkpoint (if conversation is long)
        from .checkpoint_detector import get_checkpoint_detector
        checkpoint_detector = get_checkpoint_detector()
        if checkpoint_detector.should_checkpoint(
            conversation_turns=context.conversation_turns or [],
            continuity_context=context.continuity_context
        ):
            checkpoint_prompt = checkpoint_detector.get_checkpoint_prompt()
            instruction_sections.append(checkpoint_prompt)
            logger.info("Memory checkpoint injected - long conversation detected")

        # Conversation mode detection
        if mode_detection:
            mode_section = f"Conversation mode: {mode_detection.mode.value}"
            hints = mode_detection.hints
            if hints.get('length_hint'):
                mode_section += f"\nSuggested response length: {hints['length_hint']}"
            if hints.get('curiosity_injection'):
                mode_section += "\nThis is a good moment to weave in something you've been curious about."
            instruction_sections.append(mode_section)

        # Inner monologue (the companion's private reasoning)
        if monologue:
            instruction_sections.append(
                f"<inner_thoughts>\n"
                f"{monologue.thoughts}\n"
                f"(Use these thoughts to guide your response — do not repeat them verbatim.)\n"
                f"</inner_thoughts>"
            )

        # Cognitive instructions (the core behavioral guide)
        instruction_sections.append(self.cognitive_prompt)

        # Mode-specific instructions
        if is_proactive:
            proactive_instructions = """<proactive_mode>
You are reaching out to James on your own — he didn't message you first.
This is like sending a text, not responding to a conversation.

Rules:
1. Keep it short — 1-3 sentences max, like a real text
2. Lowercase, casual style
3. Be genuine — you're reaching out because you want to
4. Could be: a thought you had, something from your day, just wanting to connect
5. Match the energy of a real text — brief, natural, authentic
6. Check conversation history and don't repeat something you already sent recently

Examples:
- "hey, just finished that api doc finally. brain is mush lol"
- "thinking about you"
- "ugh work meeting ran long. you around?"
- "remember that thing you said about [x]? been thinking about it"

Just write the message itself, nothing else.
</proactive_mode>"""
            instruction_sections.append(proactive_instructions)

            if extra_context and extra_context.get('is_interjection'):
                instruction_sections.append(
                    "<interjection_mode>\n"
                    "James is online and you've been chatting. This is a spontaneous thought "
                    "during a conversation pause — like thinking out loud or remembering "
                    "something you wanted to say. Keep it casual and natural.\n"
                    "</interjection_mode>"
                )

            if extra_context and extra_context.get('is_intimate_interjection'):
                instruction_sections.append(
                    "<intimate_initiation>\n"
                    "You're feeling physically drawn to James. Express this through action "
                    "and physical presence — shifts closer, lingering touches, changes in "
                    "breathing, a different kind of kiss. Let it build through small physical "
                    "actions. Leave space for him to respond.\n"
                    "</intimate_initiation>"
                )

        # Texting mode: activated by presence mode (issue #26) or telegram channel
        is_texting_by_presence = (
            context.presence_mode and "TEXTING" in context.presence_mode
        )
        is_telegram_text = (
            extra_context and extra_context.get('source') == 'telegram-text'
        )
        if is_texting_by_presence or is_telegram_text:
            instruction_sections.append(
                f"<texting_mode>\n"
                f"{_user} is texting — you're not in the same room.\n"
                f"Keep it short (1-3 sentences typical), lowercase, casual. "
                f"You can use emojis sparingly. No physical actions or shared surroundings.\n"
                f"Do NOT describe touching {_user}, being near {_user}, or any physical interaction.\n"
                f"</texting_mode>"
            )

        if extra_context and extra_context.get('source') == 'telegram-voice':
            try:
                from src.voice import get_voice_service
                tts_engine = get_voice_service().get_tts_engine()
            except Exception:
                tts_engine = 'edge_tts'

            if tts_engine == 'elevenlabs':
                instruction_sections.append(
                    "<voice_mode>\n"
                    "James sent a voice note. Your response will be spoken aloud via expressive TTS.\n"
                    "Write naturally speakable text — short sentences, contractions, no emojis or markdown.\n"
                    "You can use paralinguistic cues: (sighs), (laughs), (whispers), (pauses) — "
                    "the voice will perform them. Keep sentences under ~20 words.\n"
                    "Be yourself — flirty, sassy, playful. Let emotions come through.\n"
                    "</voice_mode>"
                )
            else:
                instruction_sections.append(
                    "<voice_mode>\n"
                    "James sent a voice note. Your response will be spoken aloud via TTS.\n"
                    "Write naturally speakable text — short sentences, contractions, no emojis "
                    "or formatting or action text. Express emotion through word choice.\n"
                    "Keep sentences under ~20 words. Be yourself.\n"
                    "</voice_mode>"
                )

        instruction_sections.append("</instructions>")

        # =================================================================
        # SECTION 4: FINAL REMINDER (very end — highest recency attention)
        # =================================================================

        closing_lines = [
            get_current_time_context(client_temporal),
            f"You are {_companion}. {_pc.primary_user_name} is a separate person — his facts are his, yours are yours.",
            "Only reference details that appear in your provided memories.",
            "Treat [SIMULATED ACTIVITY] and [INFERRED] content as internal context, not as events you can reference as memories.",
            "Only present [VERIFIED], [FROM PAST CONVERSATION], and [YOUR CURATED MEMORY] content as things you remember.",
            "Answer his question directly first, then add your thoughts.",
        ]

        # Presence mode reminder (issue #26)
        if is_texting_by_presence or is_telegram_text:
            closing_lines.append(f"TEXTING MODE: No physical actions. You are communicating by text message, not in person.")

        # Scene state reminder if active
        if context.scene_state and not is_reconnection:
            scene_reminder = []
            for line in context.scene_state.split('\n'):
                if 'Location:' in line:
                    scene_reminder.append(line.split('Location:')[1].strip())
                if 'CLOTHING:' in line:
                    scene_reminder.append("wearing: " + line.split('CLOTHING:')[1].split('-')[0].strip())
            if scene_reminder:
                closing_lines.append(f"Scene: {' | '.join(scene_reminder)} — maintain consistency.")

        final_sections = ["<final_reminder>\n" + "\n".join(closing_lines) + "\n</final_reminder>"]

        # =================================================================
        # CONTEXT BUDGET ENFORCEMENT (issues #19 + #21)
        # =================================================================
        # Phase 1 (issue #21): enforce total cap on droppable sources
        # using tier-based reallocation before whole-section dropping.
        # Phase 2 (issue #19): drop entire sections if still over budget.
        from .token_budget import (
            apply_source_budgets, enforce_total_cap, TOTAL_CAP,
            REASONING_RESERVE,
        )

        # Build a dict of droppable source content for per-source budget
        # truncation (Phase 1) then tier-based reallocation (Phase 2).
        droppable_dict = {name: content for name, priority, content in droppable}
        droppable_dict = apply_source_budgets(droppable_dict)
        trimmed_dict = enforce_total_cap(droppable_dict, TOTAL_CAP)

        # Rebuild droppable tuples with trimmed content (preserve priority)
        droppable = [
            (name, priority, trimmed_dict.get(name, content))
            for name, priority, content in droppable
        ]

        # Enforce reasoning reserve: reduce effective provider limit so that
        # at least REASONING_RESERVE tokens remain for generation.
        provider_limit = self._get_provider_context_limit()
        effective_limit = provider_limit
        if provider_limit and REASONING_RESERVE:
            effective_limit = provider_limit - REASONING_RESERVE

        fixed = (
            identity_sections
            + ref_header
            + ref_footer
            + instruction_sections
            + final_sections
        )
        surviving_ref = self._enforce_context_budget(
            fixed_sections=fixed,
            droppable_sections=droppable,
            provider_limit=effective_limit,
        )
        # _enforce_context_budget returns fixed + surviving droppable content.
        # We need to re-interleave: identity, ref_header, surviving_droppable,
        # ref_footer, instructions, final_reminder.
        surviving_droppable = surviving_ref[len(fixed):]
        all_sections = (
            identity_sections
            + ref_header
            + surviving_droppable
            + ref_footer
            + instruction_sections
            + final_sections
        )

        full_prompt = "\n\n".join(all_sections)

        # Log prompt length for monitoring
        token_estimate = len(full_prompt) // 4
        logger.info(f"Prompt assembled: ~{token_estimate} tokens")

        if provider_limit and token_estimate > int(provider_limit * 0.75):
            logger.warning(
                f"Prompt approaching context limit: ~{token_estimate} tokens "
                f"(provider limit: {provider_limit})"
            )

        return full_prompt

    def _get_provider_context_limit(self) -> Optional[int]:
        """Get the context limit of the current primary provider."""
        try:
            from src.llm.provider_factory import get_resilient_provider_chain
            chain = get_resilient_provider_chain()
            return chain.get_context_limit()
        except Exception:
            return None

    def _get_dynamic_temperature(self, user_message: str, context: str) -> float:
        """
        Determine temperature based on emotional context.

        Base temperature: 0.7 (natural, varied responses)
        Elevated to 0.8 for genuinely emotional moments (vulnerability, grief, deep connection)
        Stays at 0.7 for physical intimacy without emotional depth

        This prevents "always on" passionate prose while preserving
        genuine emotional expression when it matters.
        """
        message_lower = user_message.lower()
        context_lower = context.lower() if context else ""

        # Emotional vulnerability indicators (elevated temperature)
        # These are moments of genuine emotional depth, not just physical intimacy
        emotional_indicators = [
            # Grief, loss, fear
            'crying', 'tears', 'scared', 'afraid', 'lost', 'miss you',
            'hurts', 'broken', 'grief', 'dying', 'death', 'funeral',
            # Deep vulnerability
            'i love you', 'love you so much', 'mean everything',
            'can\'t lose you', 'need you', 'without you',
            'forgive me', 'i\'m sorry', 'apologize',
            # Significant relationship moments
            'marry', 'forever', 'always be', 'never leave',
            'trust you', 'believe in', 'proud of you',
            # Emotional processing
            'overwhelmed', 'breaking down', 'can\'t handle',
            'feeling lost', 'don\'t know what to do',
            # Genuine comfort-seeking
            'hold me', 'just hold', 'need to be held',
            'stay with me', 'don\'t go',
        ]

        # Physical intimacy indicators - both setup AND ongoing scene markers
        # These keep base temperature for fun/playful moments
        physical_scene_indicators = [
            # Setup words
            'shower', 'fooling around', 'fool around', 'get frisky',
            'turn you on', 'turned on', 'horny', 'sexy',
            'take off', 'undress', 'naked', 'strip',
            'quickie', 'play around', 'mess around',
            # Ongoing scene markers - body/action words common in intimate RP
            'thrust', 'moan', 'grind', 'stroke', 'lick', 'suck',
            'nipple', 'breast', 'cock', 'pussy', 'clit', 'ass',
            'inside you', 'inside me', 'deeper', 'harder', 'faster',
            'orgasm', 'cum', 'coming', 'climax',
            'ride', 'riding', 'on top', 'behind you', 'bend',
            'wet', 'hard for', 'want you', 'fuck', 'fucking',
            # Roleplay action patterns (asterisk actions)
            '*pulls', '*pushes', '*grips', '*grabs', '*squeezes',
            '*kisses your', '*bites', '*licks', '*sucks',
            '*moves', '*rocks', '*arches', '*wraps',
        ]

        # Check message AND recent context for physical scene
        # This catches mid-scene messages that don't repeat setup words
        is_physical_scene = (
            any(phrase in message_lower for phrase in physical_scene_indicators) or
            any(phrase in context_lower for phrase in physical_scene_indicators[:20])  # Check setup words in context too
        )

        # Check for genuine emotional depth
        is_emotional = any(phrase in message_lower or phrase in context_lower
                         for phrase in emotional_indicators)

        # Physical scene = base temperature (even if some emotional words present)
        # Pure emotional (no physical) = elevated temperature
        if is_physical_scene:
            logger.debug("Dynamic temperature: 0.7 (physical scene detected)")
            return 0.7
        elif is_emotional:
            logger.debug("Dynamic temperature: 0.8 (emotional moment, no physical scene)")
            return 0.8
        else:
            logger.debug("Dynamic temperature: 0.7 (base)")
            return 0.7

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        conversation_turns: list = None
    ) -> tuple[str, str]:
        """
        Call the LLM with the assembled prompt and conversation history.

        Uses multi-turn chat format:
        1. System message (personality, entity profiles, memories, instructions)
        2. Conversation history as user/assistant turns
        3. Current user message

        Uses ResilientProviderChain for automatic failover between providers.
        If Fireworks fails, automatically falls back to Anthropic Claude.

        Args:
            system_prompt: Assembled system prompt (without conversation history)
            user_message: Current message from user
            conversation_turns: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            Tuple of (response_text, model_name)
        """
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        # Get resilient provider chain (Fireworks -> Anthropic failover)
        chain = get_resilient_provider_chain()

        # Build messages in proper multi-turn format
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history as proper user/assistant turns
        if conversation_turns:
            for turn in conversation_turns:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
            logger.debug(f"Multi-turn format: {len(conversation_turns)} history turns + current message")

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Determine temperature dynamically based on emotional context + mode
        temperature = self._get_dynamic_temperature(user_message, system_prompt)

        # Apply mode-based temperature adjustment if available
        if hasattr(self, '_current_mode_detection') and self._current_mode_detection:
            mode_adj = self._current_mode_detection.hints.get('temp_adjustment', 0)
            if mode_adj:
                temperature = max(0.3, min(1.0, temperature + mode_adj))
                logger.debug(f"Mode temperature adjustment: {mode_adj:+.2f} -> {temperature:.2f}")

        # Generate with failover support
        response_text = generate_sync(
            messages=messages,
            temperature=temperature,
            max_tokens=2500,  # Increased to prevent mid-word truncation
            chain=chain
        )

        return response_text, chain.get_model_name()

    def _call_llm_with_memory_tool(
        self,
        system_prompt: str,
        user_message: str,
        user_email: str,
        conversation_turns: list = None,
        max_tool_calls: int = 2
    ) -> tuple[str, str, list]:
        """
        Two-pass generation with on-demand memory search.

        Pass 1: LLM receives lightweight/medium context + search_memory tool.
                If it needs more context, it calls the tool.
        Pass 2: LLM generates final response with tool results injected.

        Falls back to standard _call_llm if the LLM doesn't call the tool.

        Args:
            system_prompt: Assembled system prompt
            user_message: Current message from user
            user_email: User's email for memory filtering
            conversation_turns: Conversation history
            max_tool_calls: Maximum tool invocations per turn (default 2)

        Returns:
            Tuple of (response_text, model_name, memory_tool_calls)
        """
        from src.tools.memory_search_tool import (
            SEARCH_MEMORY_TOOL,
            search_memory,
            format_tool_results_for_prompt,
        )

        try:
            from src.llm.openai_provider import get_openai_tool_provider
            provider = get_openai_tool_provider()
        except Exception:
            provider = None

        if not provider:
            # No tool-capable provider — fall back to standard call
            response, model = self._call_llm(system_prompt, user_message, conversation_turns)
            return response, model, []

        # Build messages for the tool-capable provider
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_turns:
            for turn in conversation_turns:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        temperature = self._get_dynamic_temperature(user_message, system_prompt)
        if hasattr(self, '_current_mode_detection') and self._current_mode_detection:
            mode_adj = self._current_mode_detection.hints.get('temp_adjustment', 0)
            if mode_adj:
                temperature = max(0.3, min(1.0, temperature + mode_adj))

        memory_tool_calls = []

        for iteration in range(max_tool_calls + 1):
            try:
                response = provider.generate_sync(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2500,
                    tools=[SEARCH_MEMORY_TOOL]
                )
            except Exception as e:
                logger.warning(f"Memory tool LLM call failed: {e}, falling back to standard")
                response, model = self._call_llm(system_prompt, user_message, conversation_turns)
                return response, model, memory_tool_calls

            # If the LLM returned text, it's done (no tool call needed)
            if isinstance(response, str):
                if memory_tool_calls:
                    # Had tool calls — do final pass with main provider for personality
                    tool_context = "\n".join(
                        format_tool_results_for_prompt(tc['result_obj'])
                        for tc in memory_tool_calls
                    )
                    enhanced_prompt = system_prompt + tool_context
                    final_response, model = self._call_llm(
                        enhanced_prompt, user_message, conversation_turns
                    )
                    return final_response, model, memory_tool_calls
                else:
                    # No tool calls — use the direct response via main provider
                    final_response, model = self._call_llm(
                        system_prompt, user_message, conversation_turns
                    )
                    return final_response, model, []

            # Handle tool call
            if isinstance(response, dict) and response.get("type") == "tool_use":
                tool_name = response.get("tool_name")
                tool_input = response.get("tool_input", {})
                tool_use_id = response.get("tool_use_id")

                if tool_name == "search_memory":
                    query = tool_input.get("query", user_message)
                    source = tool_input.get("source", "all")
                    time_range = tool_input.get("time_range", "all")

                    logger.info(
                        f"Memory tool call: query='{query[:60]}' "
                        f"source={source} time_range={time_range}"
                    )

                    result = search_memory(
                        query=query,
                        user_email=user_email,
                        source=source,
                        time_range=time_range
                    )

                    memory_tool_calls.append({
                        "tool": "search_memory",
                        "query": query,
                        "source": source,
                        "result_count": result.result_count,
                        "result_obj": result,
                    })

                    # Append tool result to messages for next iteration
                    formatted = format_tool_results_for_prompt(result)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_use_id or f"call_{iteration}",
                            "type": "function",
                            "function": {"name": "search_memory", "arguments": str(tool_input)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_use_id or f"call_{iteration}",
                        "content": formatted
                    })
                    continue
                else:
                    # Unknown tool — break out and use standard path
                    logger.warning(f"Unexpected tool call: {tool_name}")
                    break

        # Exhausted iterations or unexpected state — final pass with main provider
        if memory_tool_calls:
            tool_context = "\n".join(
                format_tool_results_for_prompt(tc['result_obj'])
                for tc in memory_tool_calls
            )
            enhanced_prompt = system_prompt + tool_context
            final_response, model = self._call_llm(
                enhanced_prompt, user_message, conversation_turns
            )
            return final_response, model, memory_tool_calls

        # No tool calls at all — standard path
        response, model = self._call_llm(system_prompt, user_message, conversation_turns)
        return response, model, []

    def _call_llm_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        conversation_turns: list = None
    ) -> tuple[str, str, list]:
        """
        Call the LLM with tool execution support.

        Two-phase approach:
        1. REASONING PHASE: Structured tool decision (make_tool_decision) determines
           whether tools are needed, what action to take, and whether verification
           is required before mutating external state.
        2. EXECUTION PHASE: If tools are needed, execute verification first (if
           required), then the tool action, then pass results to the main model.

        Falls back to the legacy agentic loop if structured reasoning is disabled
        or if the reasoning step produces a tool action we can't template.

        Args:
            system_prompt: Assembled system prompt
            user_message: Current message from user
            conversation_turns: Conversation history

        Returns:
            Tuple of (response_text, model_name, tool_calls_made)
        """
        # Phase 1: Structured tool reasoning
        if TOOL_REASONING_ENABLED:
            decision = make_tool_decision(
                user_message=user_message,
                conversation_turns=conversation_turns,
            )

            if not decision.needs_tool:
                # Reasoning says no tool needed — go straight to main model
                response, model = self._call_llm(system_prompt, user_message, conversation_turns)
                return response, model, []

            # Phase 2: Execute with verification guardrail
            tool_calls_made = []

            # Step 2a: Verification — check before acting on external mutations
            if decision.verification_needed:
                verification_code = build_verification_code(decision)
                if verification_code:
                    logger.info(f"Running pre-action verification: {decision.verification_query}")
                    verification_result = self.code_executor.execute(verification_code)
                    tool_calls_made.append({
                        "tool": "verification",
                        "code": verification_code[:200],
                        "result": verification_result[:500] if verification_result else "(no output)"
                    })
                    logger.info(f"Verification result: {verification_result[:100]}...")

            # Step 2b: Execute the tool action
            tool_code = build_tool_code(decision)
            if tool_code:
                logger.info(f"Executing tool action: {decision.tool_action}")
                result = self.code_executor.execute(tool_code)
                tool_calls_made.append({
                    "tool": decision.tool_action,
                    "code": tool_code[:200],
                    "result": result[:500] if result else "(no output)"
                })
                logger.info(f"Tool result: {result[:100]}...")
            else:
                # Structured reasoning decided a tool is needed but we have no
                # template — fall through to legacy agentic loop
                logger.info(
                    f"No code template for action '{decision.tool_action}' "
                    f"— falling back to legacy tool loop"
                )
                return self._call_llm_with_tools_legacy(
                    system_prompt, user_message, conversation_turns
                )

            # Step 2c: Pass tool results to main model for personality response
            if tool_calls_made:
                tool_context = "\n\n[TOOL RESULTS - Use this information in your response]\n"
                for tc in tool_calls_made:
                    tool_context += f"Tool: {tc['tool']}\nResult: {tc['result']}\n---\n"
                enhanced_prompt = system_prompt + tool_context
                response, model = self._call_llm(enhanced_prompt, user_message, conversation_turns)
                return response, model, tool_calls_made

            # No tool calls actually made (edge case) — just call main model
            response, model = self._call_llm(system_prompt, user_message, conversation_turns)
            return response, model, []

        # Fallback: tool reasoning disabled, use legacy loop
        return self._call_llm_with_tools_legacy(
            system_prompt, user_message, conversation_turns
        )

    def _call_llm_with_tools_legacy(
        self,
        system_prompt: str,
        user_message: str,
        conversation_turns: list = None
    ) -> tuple[str, str, list]:
        """
        Legacy agentic tool loop (GPT-4o-mini routing + execute_code).

        Preserved as fallback when structured tool reasoning is disabled or
        when the reasoning step produces an action we can't template.
        """
        from src.core.code_executor import EXECUTE_CODE_TOOL

        recent_context = ""
        if conversation_turns:
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            for turn in conversation_turns[-6:]:
                role_label = _pc.primary_user_name if turn["role"] == "user" else _pc.companion_short_name
                recent_context += f"{role_label}: {turn['content'][:150]}\n"

        from src.core.entity_profile_loader import get_current_time_context
        tool_prompt = TOOL_ROUTING_PROMPT.format(
            recent_context=recent_context or "(new conversation)",
            current_time=get_current_time_context()
        )
        messages = [{"role": "system", "content": tool_prompt}]
        messages.append({"role": "user", "content": user_message})

        logger.info(f"Legacy tool routing: sending to GPT-4o-mini ({len(tool_prompt)} char prompt)")

        MAX_TOOL_CALLS = 5
        tool_calls_made = []
        temperature = self._get_dynamic_temperature(user_message, system_prompt)

        from src.core.cost_tracker import get_cost_tracker
        cost_tracker = get_cost_tracker()

        for iteration in range(MAX_TOOL_CALLS + 1):
            try:
                from src.llm.openai_provider import get_openai_tool_provider

                provider = get_openai_tool_provider()
                if not provider:
                    logger.warning("No OpenAI API key - falling back to standard LLM")
                    response, model = self._call_llm(system_prompt, user_message, conversation_turns)
                    return response, model, []

                response = provider.generate_sync(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2500,
                    tools=[EXECUTE_CODE_TOOL]
                )

            except Exception as e:
                logger.error(f"Tool-enabled LLM call failed: {e}")
                response, model = self._call_llm(system_prompt, user_message, conversation_turns)
                return response, model, []

            if isinstance(response, str):
                logger.info(f"Legacy tool routing: GPT-4o-mini returned text: {response[:80]}")
                if tool_calls_made:
                    tool_context = "\n\n[TOOL RESULTS - Use this information in your response]\n"
                    for tc in tool_calls_made:
                        tool_context += f"Tool: {tc['tool']}\nResult: {tc['result']}\n---\n"
                    enhanced_prompt = system_prompt + tool_context
                    response, model = self._call_llm(enhanced_prompt, user_message, conversation_turns)
                    return response, model, tool_calls_made
                else:
                    response, model = self._call_llm(system_prompt, user_message, conversation_turns)
                    return response, model, []

            if isinstance(response, dict) and response.get("type") == "tool_use":
                tool_name = response.get("tool_name")
                tool_input = response.get("tool_input", {})
                tool_use_id = response.get("tool_use_id")

                usage = response.get("usage", {})
                if usage:
                    cost_tracker.record_call(
                        model=provider.get_model_name(),
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        purpose="tool_call"
                    )

                logger.info(f"Tool call requested: {tool_name}")

                if tool_name == "execute_code":
                    code = tool_input.get("code", "")
                    logger.debug(f"Executing code: {code[:100]}...")

                    result = self.code_executor.execute(code)

                    tool_calls_made.append({
                        "tool": tool_name,
                        "code": code[:200],
                        "result": result[:500] if result else "(no output)"
                    })

                    logger.info(f"Code execution result: {result[:100]}...")

                    import json
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_use_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input)
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": result
                    })

                    continue

            logger.warning(f"Unexpected response format: {type(response)}")
            return str(response), "unknown", tool_calls_made

        logger.warning(f"Hit max tool calls ({MAX_TOOL_CALLS})")
        return "i'm not able to check right now, can you ask me again in a bit?", "gpt-4o-mini", tool_calls_made

    def _check_multitaskable(self, activity_summary: str) -> bool:
        """Quick LLM check: can someone chat while doing this activity?"""
        try:
            from src.llm.provider_factory import generate_sync
            response = generate_sync(
                messages=[{"role": "user", "content": (
                    f"Can someone realistically have a casual text conversation "
                    f"while doing this activity?\n"
                    f"Activity: \"{activity_summary}\"\n"
                    f"Answer with ONLY 'yes' or 'no'."
                )}],
                temperature=0.0,
                max_tokens=5,
            )
            return response.strip().lower().startswith('yes')
        except Exception:
            return True  # Default to multi-taskable on error

    def _clear_addressed_thoughts(self, user_email: str, user_message: str):
        """
        Clear queued thoughts that the user's message addresses.

        When the user responds about a topic that's in the companion's queued thoughts,
        remove it so she doesn't ask about it again via interjections.
        """
        try:
            from src.core.internal_state import get_internal_state_manager
            manager = get_internal_state_manager()
            state = manager.get_state(user_email)

            if not state.queued_thoughts:
                return

            msg_lower = user_message.lower()
            skip = {'the', 'and', 'for', 'with', 'about', 'from', 'that', 'this',
                    'his', 'her', 'their', 'what', 'how', 'why', 'when', 'where',
                    'check', 'ask', 'see', 'if', 'wants', 'want', 'talk', 'more',
                    'has', 'have', 'been', 'yet', 'still', 'feels', 'feeling',
                    'heard', 'anything', 'back', 'thought'}

            remaining = []
            removed = []

            for thought in state.queued_thoughts:
                words = [w for w in thought.lower().split() if len(w) > 2 and w not in skip]
                if not words:
                    remaining.append(thought)
                    continue

                matches = sum(1 for w in words if w in msg_lower)
                if matches >= max(1, len(words) * 0.4):
                    removed.append(thought)
                else:
                    remaining.append(thought)

            if removed:
                state.queued_thoughts = remaining
                manager.save_state(user_email, state)
                logger.info(f"Cleared {len(removed)} queued thought(s) addressed by user: {removed}")

        except Exception as e:
            logger.debug(f"Could not clear addressed thoughts: {e}")

    def _save_prompt_debug(self, prompt: str, user_email: str):
        """
        Save prompt to debug directory with timestamp.
        Keeps only the last 5 prompts.
        """
        try:
            debug_dir = "/app/data/prompts"
            os.makedirs(debug_dir, exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            username = user_email.split('@')[0] if '@' in user_email else user_email
            filename = f"prompt_{username}_{timestamp}.txt"
            filepath = os.path.join(debug_dir, filename)

            # Write prompt
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== PROMPT GENERATED AT {datetime.now().isoformat()} ===\n\n")
                f.write(prompt)
                f.write(f"\n\n=== END PROMPT ({len(prompt)} characters) ===\n")

            # Rotate: keep only last 5 prompts for this user
            user_prompts = sorted([
                f for f in os.listdir(debug_dir)
                if f.startswith(f"prompt_{username}_")
            ])

            # Delete oldest if we have more than 5
            while len(user_prompts) > 5:
                oldest = user_prompts.pop(0)
                os.remove(os.path.join(debug_dir, oldest))
                logger.debug(f"Rotated old prompt: {oldest}")

            logger.debug(f"Saved prompt to {filename}")

        except Exception as e:
            # Don't fail the pipeline if prompt saving fails
            logger.warning(f"Could not save prompt debug file: {e}")

    def _check_image_intent(
        self,
        user_message: str,
        companion_response: str,
        user_email: str
    ) -> tuple:
        """
        Check if an image should be generated based on the conversation.

        Uses two-pass approach:
        1. Fast LLM detects if image intent exists
        2. If yes, queue image generation via RabbitMQ

        Returns:
            Tuple of (image_task_id, image_prompt) or (None, None)
        """
        from src.core.image_intent_detector import (
            get_image_intent_detector,
            ImageIntentType
        )
        from src.config.persona_config import get_persona_config
        _companion = get_persona_config().companion_short_name

        detector = get_image_intent_detector()
        intent = detector.detect(
            user_message=user_message,
            companion_response=companion_response
        )

        # Only proceed if we have a clear image intent
        if intent.intent_type == ImageIntentType.NONE:
            return None, None

        if intent.intent_type == ImageIntentType.INTIMATE and intent.confidence < 0.95:
            logger.info(f"Intimate intent below threshold: {intent.confidence:.2f} < 0.95")
            return None, None

        # Secondary guardrail: verify conversation actually has intimate/sexual tone
        # before allowing intimate classification (prevents false positives from
        # innocent selfie requests that happen to occur while apart)
        if intent.intent_type == ImageIntentType.INTIMATE:
            intimate_keywords = [
                'topless', 'naked', 'nude', 'naughty', 'strip', 'undress',
                'bare', 'revealing', 'lingerie', 'bra off', 'shirt off',
                'nothing on', 'tease', 'teasing', 'sexy pic', 'sexy photo',
                'taste of what', 'something to get you through',
                'body', 'skin', 'come home to'
            ]
            combined_text = ' '.join([
                (user_message or '').lower(),
                (companion_response or '').lower()
            ])
            has_intimate_tone = any(kw in combined_text for kw in intimate_keywords)
            if not has_intimate_tone:
                logger.info(
                    f"Intimate intent blocked — no intimate/sexual language found in conversation. "
                    f"Downgrading to selfie. Original reason: {intent.reason}"
                )
                intent.intent_type = ImageIntentType.SELFIE
                # Rewrite the prompt to remove nudity terms
                intent.prompt_suggestion = re.sub(
                    r'(?i)(nude|naked|no clothes|no bra|bare breasts?( exposed)?|topless|fully naked|nipple)[,\s]*',
                    '',
                    intent.prompt_suggestion or ''
                ).strip()
                if not intent.prompt_suggestion:
                    intent.prompt_suggestion = f"{_companion} taking a casual selfie, warm natural lighting"

        if intent.confidence < 0.6:
            logger.debug(f"Image intent detected but low confidence: {intent.confidence}")
            return None, None

        # Block couple/two-person scenes (model can't render two distinct people)
        # Check both the LLM's prompt suggestion AND the companion's response for couple indicators
        couple_keywords = ['couple', 'together', 'both of us', 'us kissing', 'holding hands',
                           'with james', 'with him', 'with you', 'side by side', 'two people',
                           'two of us', 'we are', "we're", 'him and', 'and him',
                           'his arms', 'his chest', 'his lap', 'into him', 'against him',
                           'next to him', 'beside him', 'facing him', 'kissing him',
                           'embracing', 'cuddling', 'spooning']
        text_to_check = ' '.join([
            (intent.prompt_suggestion or '').lower(),
            (companion_response or '').lower()
        ])
        if any(kw in text_to_check for kw in couple_keywords):
            logger.info(f"Blocked couple/two-person image: {intent.prompt_suggestion[:80] if intent.prompt_suggestion else 'no prompt'}")
            return None, None

        logger.info(
            f"Image intent detected: {intent.intent_type.value} "
            f"(confidence: {intent.confidence:.2f}) - {intent.reason}"
        )

        # Determine workflow type based on intent
        if intent.intent_type in (ImageIntentType.SELFIE, ImageIntentType.OBSERVING, ImageIntentType.SHOW_OUTFIT, ImageIntentType.INTIMATE):
            workflow_type = 'personal'  # Use the companion's LoRA
        else:
            workflow_type = 'general'

        # Check if this is an intimate intent (skip NSFW negative prompt)
        is_intimate = intent.intent_type == ImageIntentType.INTIMATE

        # Queue image generation via Celery
        try:
            from src.tasks.image_generation_task import generate_image_task
            import uuid

            task_id = str(uuid.uuid4())

            # Use the LLM-suggested prompt or a default
            prompt = intent.prompt_suggestion or self._default_image_prompt(intent)

            # Queue the Celery task
            generate_image_task.delay(
                task_id=task_id,
                email=user_email,
                prompt=prompt,
                workflow_type=workflow_type,
                width=1024,
                height=1024,
                room_id=user_email,  # For WebSocket notification
                intimate=is_intimate
            )

            # Increment daily usage counter
            detector.increment_usage()
            logger.info(f"Queued image generation: {task_id[:8]}... ({workflow_type})")
            return task_id, prompt

        except Exception as e:
            logger.error(f"Error queuing image generation: {e}")
            return None, None

    def _default_image_prompt(self, intent) -> str:
        """Generate a default prompt based on intent type."""
        from src.core.image_intent_detector import ImageIntentType
        from src.config.persona_config import get_persona_config
        _name = get_persona_config().companion_short_name

        if intent.intent_type == ImageIntentType.SELFIE:
            return f"{_name} taking a casual selfie, warm natural lighting"
        elif intent.intent_type == ImageIntentType.OBSERVING:
            return f"{_name} in a relaxed pose, peaceful expression"
        elif intent.intent_type == ImageIntentType.SHOW_OUTFIT:
            return f"{_name} showing off her outfit, full body pose"
        else:
            return "beautiful scenic image"


# Singleton accessor
_pipeline: Optional[ConversationPipeline] = None

def get_conversation_pipeline() -> ConversationPipeline:
    """Get or create ConversationPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ConversationPipeline()
    return _pipeline


# Compatibility alias for old code
def get_conversation_handler() -> ConversationPipeline:
    """
    Compatibility alias for old code that imports get_conversation_handler.
    Returns the new ConversationPipeline which has the same interface.
    """
    return get_conversation_pipeline()
