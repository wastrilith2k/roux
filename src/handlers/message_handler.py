"""
Message Handler — Companion Framework

WHAT: The entry point for incoming user messages from WebSocket (or Telegram).
      Orchestrates the full message lifecycle: validate -> pipeline -> store -> dispatch tasks.
WHY:  Separates I/O concerns (WebSocket, database persistence, task dispatching) from
      the core response generation (which lives in pipeline.py). The pipeline focuses
      on building context and calling the LLM; this module handles everything before
      and after that.
HOW:  MessageProcessor.process_message() does:
      1. Validate the incoming message
      2. Get/create user profile and read closeness score
      3. Run the ConversationPipeline to generate a response
      4. Split multi-message responses (companion uses --- separator)
      5. Store user message + companion response(s) in the database
      6. Dispatch 10+ async Celery tasks for post-response processing:
         - Graphiti extraction, curiosity extraction, fact extraction
         - Episode tracking, scene extraction, internal state update
         - Relationship dynamics, interaction outcome analysis
         - Episodic embedding (pgvector), biography refresh
      7. Return the response payload to the WebSocket handler

The extensive task dispatching (step 6) is what gives the companion its
autonomous subsystems — each task runs independently via Celery/Redis.
"""
import time
import logging
from typing import Dict, Optional, Any
from flask_socketio import emit

logger = logging.getLogger(__name__)

# Database
from src.database.db import get_db
from src.utils.timezone_utils import now_pacific_naive
# Closeness read from user profile (defaults to 50)

# The new pipeline handles everything
from src.core.conversation.pipeline import get_conversation_handler


class MessageProcessor:
    """
    Processes incoming user messages through the ConversationPipeline.

    This is a simplified handler for stable-core that delegates most work
    to the ConversationPipeline.
    """

    def __init__(self):
        """Initialize the message processor"""
        self.db = get_db()
        self.pipeline = get_conversation_handler()

    def process_message(self, data: Dict[str, Any], email: str, sid: str, cancel_check=None) -> Optional[Dict[str, Any]]:
        """
        Main entry point for processing a user message.

        Args:
            data: Message data from WebSocket
            email: User's email
            sid: Socket.IO session ID
            cancel_check: Optional callable returning True if processing should abort

        Returns:
            Response data dict or None if processing failed
        """
        # Validate message
        message = data.get('message', '').strip()
        if not message:
            return {'error': 'Empty message'}

        message_type = data.get('message_type', 'chat')
        test_mode = message_type == 'test'
        if test_mode:
            logger.info(f"⚠️  TEST MESSAGE MODE - will not persist to database")
            logger.info(f"🧪 [TEST] USER: {message}")

        # Mark user as awake when they message (autopilot system)
        try:
            from src.core.user_context import on_user_message
            on_user_message()
        except Exception as e:
            logger.debug(f"Could not update user awake state: {e}")

        start_time = time.time()

        try:
            from src.core.conversation.pipeline import PipelineCancelled

            # Get or create user profile
            profile = self.db.get_profile(email)
            if not profile:
                logger.info(f"Creating profile for {email}")
                self.db.create_profile(email)
                profile = self.db.get_profile(email)

            # Read closeness from profile/DB; default to 50 if not available
            closeness = profile.get('closeness_score', 50) if profile else 50

            # IMPORTANT: User message is stored AFTER the pipeline runs.
            # If stored before, the pipeline's history fetch would include this message,
            # causing the companion to see the user's message twice in context.

            # Process through pipeline
            logger.info(f"📨 Processing message from {email}: {message[:100]}...")

            result = self.pipeline.process(
                user_email=email,
                user_message=message,
                closeness_score=closeness,
                cancel_check=cancel_check
            )

            if not result or not result.response:
                logger.error("Pipeline returned no response")
                return {'error': 'No response generated'}

            response_text = result.response

            # Log full exchange in test mode
            if test_mode:
                logger.info(f"🧪 [TEST] COMPANION: {response_text}")
                # Also log what context was used
                if result.context_used:
                    ctx = result.context_used
                    logger.info(f"🧪 [TEST] CONTEXT - Memories: {len(ctx.memories) if ctx.memories else 0} chars")
                    logger.info(f"🧪 [TEST] CONTEXT - Entity profiles: {len(ctx.entity_profiles) if ctx.entity_profiles else 0} chars")
                    logger.info(f"🧪 [TEST] CONTEXT - Scene state: {'Yes' if ctx.scene_state else 'No'}")

            # Split into multiple messages if the companion used --- separator
            messages = self._split_messages(response_text)

            if not messages:
                logger.error(f"No valid messages after split - response_text was: {response_text[:100] if response_text else 'None'}")
                return {'error': 'Empty response from companion'}

            # Save user message first (stored after pipeline to avoid duplication in history)
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            user_msg_id = None
            if message_type != 'test':
                user_msg_id = self.db.store_message(
                    email=email,
                    sender_name=_pc.primary_user_name,
                    message_text=message,
                    sentiment=0.5,
                    closeness=closeness
                )

            # Save companion's response(s) - track (msg_id, msg_text) pairs for embedding
            companion_messages_saved = []  # List of (msg_id, msg_text) tuples
            model_used = result.llm_model if result else None
            if message_type != 'test':
                for msg in messages:
                    msg_id = self.db.store_message(
                        email=email,
                        sender_name=_pc.companion_short_name,
                        message_text=msg,
                        sentiment=0.5,
                        closeness=closeness,
                        model=model_used  # Track which model generated this response
                    )
                    if msg_id:
                        companion_messages_saved.append((msg_id, msg))

            # Closeness system removed - no score updates

            # DISABLED: extraction_v3 removed - was producing garbage facts
            # Will be replaced with simpler PostgreSQL-based extraction
            # if message_type != 'test':
            #     try:
            #         from src.tasks.extraction_v3_task import process_exchange_v3
            #         process_exchange_v3.delay(email, message, response_text)
            #         logger.info("🧠 Entity extraction task queued")
            #     except Exception as e:
            #         logger.warning(f"Could not queue entity extraction: {e}")

            # =================================================================
            # Async task dispatch — each task runs independently via Celery.
            # These are fire-and-forget: failures are logged but don't block
            # the response. This is what powers the companion's autonomous
            # subsystems (memory, curiosity, opinions, episodes, etc.).
            # =================================================================
            if message_type != 'test':
                # Correction detection — learn when the user corrects the companion
                try:
                    from src.tasks.correction_task import queue_correction_detection
                    # Get the companion's previous message to check if user is correcting it
                    previous_companion = self._get_previous_companion_message(email)
                    if previous_companion:
                        queue_correction_detection(email, message, previous_companion)
                except Exception as e:
                    logger.warning(f"Could not queue correction detection: {e}")

                # RE-ENABLED: Graphiti temporal knowledge graph for memory overhaul
                try:
                    from src.tasks.graphiti_extraction_task import process_exchange_graphiti
                    process_exchange_graphiti.delay(email, message, response_text)
                    logger.info("🔗 Graphiti extraction task queued")
                except Exception as e:
                    logger.warning(f"Could not queue Graphiti extraction: {e}")

                # Curiosity extraction - detect topics the companion should follow up on
                try:
                    from src.tasks.curiosity_extraction_task import extract_curiosity_from_conversation
                    extract_curiosity_from_conversation.delay(email, message, response_text)
                    logger.info("💭 Curiosity extraction task queued")
                except Exception as e:
                    logger.warning(f"Could not queue curiosity extraction: {e}")

                # Curiosity resolution - mark topics as discussed if the companion or user mentioned them
                try:
                    from src.tasks.curiosity_extraction_task import resolve_discussed_curiosities
                    resolve_discussed_curiosities.delay(response_text, user_message=message)
                    logger.info("💭 Curiosity resolution task queued")
                except Exception as e:
                    logger.warning(f"Could not queue curiosity resolution: {e}")

                # Trigger async episodic embedding for new messages
                try:
                    from src.tasks.episodic_embedding_task import embed_single_message
                    embed_count = 0
                    # Embed user message
                    if user_msg_id and len(message) > 20:
                        embed_single_message.delay(user_msg_id, message, 'User')
                        embed_count += 1
                    # Embed companion's response(s) - each tuple has (msg_id, msg_text) paired correctly
                    from src.config.persona_config import get_persona_config
                    _companion_name = get_persona_config().companion_short_name
                    for msg_id, msg_text in companion_messages_saved:
                        if len(msg_text) > 20:
                            embed_single_message.delay(msg_id, msg_text, _companion_name)
                            embed_count += 1
                    if embed_count > 0:
                        logger.info(f"📚 Episodic embedding queued for {embed_count} message(s)")
                except Exception as e:
                    logger.warning(f"Could not queue episodic embedding: {e}")

                # Scene state extraction - track roleplay/virtual world state
                try:
                    from src.tasks.scene_extraction_task import extract_scene_state
                    extract_scene_state.delay(email, message, response_text, 'chat')
                    logger.info("🎬 Scene extraction task queued")
                except Exception as e:
                    logger.warning(f"Could not queue scene extraction: {e}")

                # Internal state update - track the companion's energy, mood, and transitions
                try:
                    from src.tasks.internal_state_task import update_internal_state
                    update_internal_state.delay(email, message, response_text)
                    logger.info("💭 Internal state update task queued")
                except Exception as e:
                    logger.warning(f"Could not queue internal state update: {e}")

                # Relationship dynamics - Gottman-informed closeness/trust tracking
                try:
                    from src.tasks.relationship_dynamics_task import analyze_relationship_dynamics
                    analyze_relationship_dynamics.delay(email, message, response_text)
                    logger.info("💕 Relationship dynamics task queued")
                except Exception as e:
                    logger.warning(f"Could not queue relationship dynamics: {e}")

                # Fact extraction - learn facts from user messages
                try:
                    from src.tasks.fact_extraction_task import extract_facts
                    extract_facts.delay(email, message, response_text, user_msg_id)
                    logger.info("📝 Fact extraction task queued")
                except Exception as e:
                    logger.warning(f"Could not queue fact extraction: {e}")

                # Relationship extraction - extract structured relationships
                try:
                    from src.tasks.relationship_extraction_task import extract_relationships
                    extract_relationships.delay(email, message, response_text, user_msg_id)
                    logger.info("🔗 Relationship extraction task queued")
                except Exception as e:
                    logger.warning(f"Could not queue relationship extraction: {e}")

                # Event synthesis - detect and synthesize crisis/career/milestone/relationship events
                try:
                    from src.tasks.event_synthesis_task import detect_and_synthesize_events
                    detect_and_synthesize_events.delay(email, message, response_text, user_msg_id)
                    logger.info("📅 Event synthesis task queued")
                except Exception as e:
                    logger.warning(f"Could not queue event synthesis: {e}")

                # Episode tracking - manage conversation episodes
                try:
                    from src.tasks.episode_tracking_task import process_message_episode
                    # Get companion's message ID (first one if multiple)
                    companion_msg_id = companion_messages_saved[0][0] if companion_messages_saved else None
                    process_message_episode.delay(email, message, response_text, user_msg_id, companion_msg_id)
                    logger.info("📖 Episode tracking task queued")
                except Exception as e:
                    logger.warning(f"Could not queue episode tracking: {e}")

                # Interaction outcome tracking - analyze how the user responds to the companion
                try:
                    from src.tasks.interaction_outcome_task import analyze_interaction_outcome
                    previous_companion = self._get_previous_companion_message(email)
                    if previous_companion:
                        # Get the companion message ID for the PREVIOUS message (before this exchange)
                        analyze_interaction_outcome.delay(
                            email, previous_companion, message, None, user_msg_id
                        )
                        logger.info("📊 Interaction outcome task queued")
                except Exception as e:
                    logger.warning(f"Could not queue interaction outcome: {e}")

                # Biography refresh when context window is filling up
                # Ensures important info from messages is preserved before they age out
                try:
                    self._check_biography_refresh_needed(email)
                except Exception as e:
                    logger.warning(f"Could not check biography refresh: {e}")

            elapsed = time.time() - start_time
            logger.info(f"✅ Response generated in {elapsed:.2f}s ({len(messages)} message(s))")

            # Get avatar (simple default for now)
            avatar_url = self._get_avatar_url()

            # Return multiple messages if split, single response for compatibility
            return {
                'response': response_text,  # Full response for compatibility
                'messages': messages,       # Split messages for multi-message display
                'avatar_url': avatar_url,
                'timestamp': now_pacific_naive().isoformat(),
                'processing_time': elapsed
            }

        except PipelineCancelled:
            logger.info(f"Pipeline cancelled for {email} (new message arrived)")
            return {'cancelled': True}

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            return {'error': str(e)}

    def _split_messages(self, response_text: str) -> list[str]:
        """
        Split response into multiple messages if the companion used --- separator.

        This allows the companion to respond to multiple topics with separate messages,
        which feels more natural in conversation.
        """
        # Handle empty/None input
        if not response_text or not response_text.strip():
            logger.warning("Empty response_text received in _split_messages")
            return []

        # Split on --- (with optional whitespace around it)
        import re
        parts = re.split(r'\n---\n|\n---$|^---\n', response_text.strip())

        # Clean up each part and filter out empty ones
        messages = [part.strip() for part in parts if part.strip()]

        # Return messages or original if somehow all parts were empty
        return messages if messages else [response_text.strip()]

    def _get_avatar_url(self) -> Optional[str]:
        """Get a default avatar URL"""
        # Status manager removed - avatar selection handled elsewhere
        return None

    def _get_previous_companion_message(self, email: str) -> Optional[str]:
        """
        Get the companion's most recent message to this user.

        Used for correction detection - we need to know what the companion said
        to detect if the user is correcting them.

        Args:
            email: User's email

        Returns:
            The companion's previous message text or None
        """
        try:
            # Get recent messages and find the last one from companion
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            messages = self.db.get_recent_messages(email, limit=5)
            for msg in messages:
                # Messages are ordered newest first
                if msg.get('sender_name') == _pc.companion_short_name:
                    return msg.get('message_text', '')
            return None
        except Exception as e:
            logger.warning(f"Could not get previous companion message: {e}")
            return None

    def _check_biography_refresh_needed(self, email: str) -> None:
        """
        Check if biography refresh is needed due to context window filling.

        Triggers biography refresh when message count since last refresh
        exceeds a threshold, ensuring important information from recent
        messages is preserved in biographies before aging out.

        Uses a simple counter stored in Redis to track messages since last refresh.
        """
        import os
        try:
            import redis

            # Context window is 25 messages - trigger refresh every 20 messages
            # This ensures facts are captured before messages age out
            REFRESH_THRESHOLD = int(os.environ.get('BIOGRAPHY_REFRESH_MESSAGE_THRESHOLD', '20'))

            redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
            r = redis.from_url(redis_url)

            # Increment message counter for this user
            counter_key = f"biography_refresh_counter:{email}"
            count = r.incr(counter_key)

            if count >= REFRESH_THRESHOLD:
                logger.info(f"📚 Context window filling ({count} messages) - triggering biography refresh")

                # Reset counter
                r.set(counter_key, 0)

                # Trigger async biography refresh
                from src.tasks.biography_refresh_task import refresh_biographies_task
                refresh_biographies_task.delay(email)

        except Exception as e:
            logger.debug(f"Biography refresh check skipped: {e}")


# Singleton instance
_processor: Optional[MessageProcessor] = None


def get_message_processor() -> MessageProcessor:
    """Get or create MessageProcessor singleton."""
    global _processor
    if _processor is None:
        _processor = MessageProcessor()
    return _processor
