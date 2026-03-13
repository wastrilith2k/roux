"""
Always-On Service -- The companion's "awake" background process.

WHAT: Runs a continuous dual-cadence loop that makes the companion feel present:
      1. Offline/Telegram path (3-25 min):  Checks if she should reach out via
         Telegram, using dynamic intervals driven by reach-out pressure.
      2. WebSocket path (2-5 min):  Checks if she should interject during an
         active web conversation.
      Also: processes incoming Telegram messages, dispatches async Celery tasks,
      tracks activity transitions, monitors James's calendar, and triggers
      autonomous goal actions (budget-gated).

WHY:  A companion that only replies feels transactional.  This service gives
      her initiative -- she notices when a meeting ends, when it's been too long
      since they talked, when she finishes a task and has something to share.
      The dual cadence avoids both spam (slow offline) and stale silence (fast
      when he's right there on the web UI).

HOW IT FITS:
  - Started by start_always_on() at app boot (called from app.py or worker).
  - Owns the TelegramBridge (start/stop lifecycle).
  - Delegates reach-out decisions to ReachOutEngine.
  - Delegates interjection decisions to InterjectionEngine.
  - Delegates goal execution to autonomous_action_task (budget-gated).
  - Uses ReachOutPressure for dynamic interval timing.
  - Publishes interjections to Redis pub/sub for WebSocket delivery.
  - Incoming Telegram messages route through the main conversation pipeline
    and dispatch the same async tasks as web chat (scene, facts, episodic, etc.).

Loop lifecycle:
  _run_loop()
    -> sleep(interval)
    -> _check_user_schedule()        # autopilot wake/sleep
    -> _check_activity_changes()     # companion schedule transitions
    -> _check_calendar_transitions() # James's calendar event endings
    -> if WebSocket: _check_interjection()
       else:        _check_reach_out_with_pressure()
    -> _check_goal_actions()         # always (budget prevents spam)
"""

import os
import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Optional
from threading import Thread
import time

from src.config.persona_config import get_persona_config

logger = logging.getLogger(__name__)


# =============================================================================
# AlwaysOnService
# =============================================================================

class AlwaysOnService:
    """
    Manages the companion's autonomous presence and proactive behavior.

    Dual-cadence loop:
    - When user is offline: dynamic reach-out intervals via Telegram (3-25 min based on pressure)
    - When user is on WebSocket: faster interjection checks (2-5 min)
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[Thread] = None
        self._telegram_bridge = None
        self._last_check = None
        self._last_interval_minutes: float = 15.0  # Track for decay calculations

        # Activity tracking state
        self._previous_activity_status: Optional[dict] = None
        self._activity_milestone_enabled = os.environ.get('COMPANION_ACTIVITY_MILESTONES', 'false').lower() == 'true'

        # Calendar transition tracking (user's Google Calendar events)
        self._user_was_in_event: bool = False
        self._user_last_event_name: Optional[str] = None

    def start(self):
        """Start the always-on service."""
        if self._running:
            logger.warning("AlwaysOnService already running")
            return

        self._running = True

        # Start Telegram bridge for autonomous messaging
        self._start_telegram()

        # Start the check loop
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info("AlwaysOnService started (dynamic intervals)")

    def stop(self):
        """Stop the always-on service."""
        self._running = False
        if self._telegram_bridge:
            self._telegram_bridge.stop()
        logger.info("AlwaysOnService stopped")

    # -----------------------------------------------------------------
    # Startup / shutdown
    # -----------------------------------------------------------------

    def _start_telegram(self):
        """Initialize Telegram bridge."""
        try:
            from src.autonomy.telegram_bridge import get_telegram_bridge

            self._telegram_bridge = get_telegram_bridge()

            # Set up message handler
            async def handle_telegram_message(content: str, user_id: str, source: str) -> Optional[str]:
                """Handle incoming Telegram messages from the user."""
                return await self._handle_incoming_message(content, user_id, source)

            started = self._telegram_bridge.start(on_message_callback=handle_telegram_message)

            if started:
                logger.info("Telegram bridge started (waiting for /start from user)")
            else:
                logger.warning("Telegram bridge not started (missing token?)")

        except Exception as e:
            logger.error(f"Failed to start Telegram bridge: {e}")

    # -----------------------------------------------------------------
    # Incoming Telegram message handling
    # -----------------------------------------------------------------

    async def _handle_incoming_message(self, content: str, user_id: str, source: str) -> Optional[tuple]:
        """
        Handle incoming messages from Telegram.

        Routes through the conversation pipeline for consistent responses,
        then dispatches async tasks (scene extraction, fact extraction, etc.)
        just like the web chat handler does.

        Returns:
            Tuple of (response_text, image_task_id) or None.
            image_task_id is None if no image was triggered.
        """
        try:
            # Import the pipeline
            from src.core.conversation.pipeline import get_conversation_handler

            pipeline = get_conversation_handler()

            # Process the message
            email = get_persona_config().primary_user_email

            # Telegram = not physically together. User is texting from their phone,
            # even if WebSocket is still connected (he forgot to log out).
            # Clear all "present" state so the companion treats this as a text conversation.
            if source.startswith('telegram'):
                # Clear WebSocket presence in Redis — he's texting, not "in the room"
                try:
                    import redis
                    r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379/0'))
                    ws_key = f'ws_connected:{email}'
                    if r.exists(ws_key):
                        r.delete(ws_key)
                        logger.info("📱 Telegram message — cleared WebSocket presence (treating as away)")
                except Exception as e:
                    logger.warning(f"Could not clear WebSocket presence for telegram: {e}")

                # Clear physical presence in scene state
                try:
                    from src.core.scene_tracker import get_scene_tracker
                    tracker = get_scene_tracker()
                    scene = tracker.get_scene_state(email)
                    if scene.physical_presence:
                        scene.physical_presence = False
                        scene.physical_state = None
                        scene.posture = None
                        scene.position_detail = None
                        scene.james_position = None
                        scene.companion_position = None
                        tracker.save_scene_state(email, scene)
                        logger.info("📱 Telegram message — cleared physical_presence before pipeline")
                except Exception as e:
                    logger.warning(f"Could not update scene for telegram: {e}")

            result = pipeline.process(
                user_email=email,
                user_message=content,
                closeness_score=100,  # High trust for direct messaging
                extra_context={'source': source}
            )

            if result and result.response:
                response_text = result.response

                # Store both messages in history (capture IDs for async tasks)
                from src.database.db import get_db
                db = get_db()

                user_msg_id = db.store_message(
                    email=email,
                    sender_name='User',
                    message_text=content,
                    sentiment=0.5,
                    closeness=100,
                    source=source
                )

                companion_msg_id = db.store_message(
                    email=email,
                    sender_name=get_persona_config().companion_short_name,
                    message_text=response_text,
                    sentiment=0.5,
                    closeness=100,
                    source=source
                )

                # Dispatch async tasks — same processing as web chat
                self._dispatch_async_tasks(
                    email, content, response_text,
                    user_msg_id, companion_msg_id, source
                )

                return (response_text, getattr(result, 'image_task_id', None))

            return None

        except Exception as e:
            logger.error(f"Error handling {source} message: {e}")
            return ("sorry, something went wrong on my end...", None)

    # -----------------------------------------------------------------
    # Async task dispatch (mirrors web chat handler)
    # -----------------------------------------------------------------

    def _dispatch_async_tasks(
        self, email: str, user_message: str, companion_response: str,
        user_msg_id: int, companion_msg_id: int, source: str
    ):
        """
        Dispatch async Celery tasks after a Telegram message exchange.

        Mirrors the task dispatching in message_handler.py so Telegram
        messages get the same processing as web chat messages.
        """
        # Scene state extraction
        try:
            from src.tasks.scene_extraction_task import extract_scene_state
            extract_scene_state.delay(email, user_message, companion_response, source)
            logger.info("🎬 Scene extraction task queued (telegram)")
        except Exception as e:
            logger.warning(f"Could not queue scene extraction: {e}")

        # Internal state update
        try:
            from src.tasks.internal_state_task import update_internal_state
            update_internal_state.delay(email, user_message, companion_response)
            logger.info("💭 Internal state update task queued (telegram)")
        except Exception as e:
            logger.warning(f"Could not queue internal state update: {e}")

        # Fact extraction
        try:
            from src.tasks.fact_extraction_task import extract_facts
            extract_facts.delay(email, user_message, companion_response, user_msg_id)
        except Exception as e:
            logger.warning(f"Could not queue fact extraction: {e}")

        # Graphiti knowledge graph
        try:
            from src.tasks.graphiti_extraction_task import process_exchange_graphiti
            process_exchange_graphiti.delay(email, user_message, companion_response)
        except Exception as e:
            logger.warning(f"Could not queue Graphiti extraction: {e}")

        # Relationship dynamics
        try:
            from src.tasks.relationship_dynamics_task import analyze_relationship_dynamics
            analyze_relationship_dynamics.delay(email, user_message, companion_response)
        except Exception as e:
            logger.warning(f"Could not queue relationship dynamics: {e}")

        # Curiosity extraction
        try:
            from src.tasks.curiosity_extraction_task import extract_curiosity_from_conversation
            extract_curiosity_from_conversation.delay(email, user_message, companion_response)
        except Exception as e:
            logger.warning(f"Could not queue curiosity extraction: {e}")

        # Curiosity resolution - check both companion's response and user's message
        try:
            from src.tasks.curiosity_extraction_task import resolve_discussed_curiosities
            resolve_discussed_curiosities.delay(companion_response, user_message=user_message)
        except Exception as e:
            logger.warning(f"Could not queue curiosity resolution: {e}")

        # Episodic embedding
        try:
            from src.tasks.episodic_embedding_task import embed_single_message
            if user_msg_id and len(user_message) > 20:
                embed_single_message.delay(user_msg_id, user_message, 'User')
            if companion_msg_id and len(companion_response) > 20:
                from src.config.persona_config import get_persona_config
                embed_single_message.delay(companion_msg_id, companion_response, get_persona_config().companion_short_name)
        except Exception as e:
            logger.warning(f"Could not queue episodic embedding: {e}")

        # Relationship extraction
        try:
            from src.tasks.relationship_extraction_task import extract_relationships
            extract_relationships.delay(email, user_message, companion_response, user_msg_id)
        except Exception as e:
            logger.warning(f"Could not queue relationship extraction: {e}")

        # Event synthesis
        try:
            from src.tasks.event_synthesis_task import detect_and_synthesize_events
            detect_and_synthesize_events.delay(email, user_message, companion_response, user_msg_id)
        except Exception as e:
            logger.warning(f"Could not queue event synthesis: {e}")

        # Episode tracking
        try:
            from src.tasks.episode_tracking_task import process_message_episode
            process_message_episode.delay(email, user_message, companion_response, user_msg_id, companion_msg_id)
        except Exception as e:
            logger.warning(f"Could not queue episode tracking: {e}")

    # =========================================================================
    # MAIN LOOP - Dynamic intervals with dual cadence
    # =========================================================================

    def _run_loop(self):
        """
        Main check loop with dynamic intervals.

        Interval is determined by reach-out pressure:
        - High pressure (she wants to say something) = shorter intervals
        - Low pressure (nothing happening) = longer intervals
        - When user is on WebSocket = fast interjection cadence (2-5 min)
        """
        logger.info("AlwaysOnService check loop started (dynamic intervals)")

        # Initialize activity tracking on first run
        self._initialize_activity_tracking()

        while self._running:
            try:
                # Determine sleep interval
                user_on_web = self._is_user_on_websocket()

                if user_on_web:
                    # Fast cadence for interjections during active conversations
                    interval = random.uniform(2.0, 5.0)
                else:
                    # Dynamic interval based on pressure
                    interval = self._get_dynamic_interval()

                self._last_interval_minutes = interval
                logger.info(f"Always-on: sleeping {interval:.1f}m (web={user_on_web})")

                time.sleep(interval * 60)

                if not self._running:
                    break

                # Check user's scheduled wake/sleep times (autopilot system)
                self._check_user_schedule()

                # Check for activity changes and record them
                self._check_activity_changes()

                # Check for user's calendar event transitions (event ended = good time to reach out)
                self._check_calendar_transitions()

                # Route to appropriate check based on user's presence
                if self._is_user_on_websocket():
                    self._check_interjection()
                else:
                    self._check_reach_out_with_pressure()

                # Goal actions run regardless of user's presence
                # Budget system (4/day, 2h gap) prevents spam
                self._check_goal_actions()

            except Exception as e:
                logger.error(f"Error in check loop: {e}")
                time.sleep(60)  # Wait a minute on error

    def _get_dynamic_interval(self) -> float:
        """Get the next check interval based on pressure state."""
        try:
            from src.autonomy.reach_out_pressure import get_reach_out_pressure
            pressure = get_reach_out_pressure()
            interval = pressure.get_next_interval_minutes()
            return interval
        except Exception as e:
            logger.debug(f"Could not get pressure interval: {e}")
            return 15.0  # Fallback to 15 min

    def _is_user_on_websocket(self) -> bool:
        """Check if user is connected via WebSocket (via Redis key)."""
        try:
            import redis
            r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379/0'))
            email = get_persona_config().primary_user_email
            return r.exists(f'ws_connected:{email}') > 0
        except Exception:
            return False

    # =========================================================================
    # REACH-OUT with pressure tracking (offline/Telegram path)
    # =========================================================================

    def _check_reach_out_with_pressure(self):
        """
        Check if the companion should reach out, tracking pressure for dynamic intervals.

        - Successful reach-out: pressure releases
        - External suppression (wanted to but can't): pressure builds
        - Internal "no" (chose not to): pressure decays
        """
        # Don't send Telegram messages when user is on WebSocket
        if self._is_user_on_websocket():
            logger.debug("User on WebSocket - skipping Telegram reach-out, using interjection path")
            return

        try:
            from src.autonomy.reach_out_engine import (
                get_reach_out_engine, classify_suppression, TELEGRAM_ENABLED
            )
            from src.autonomy.reach_out_pressure import get_reach_out_pressure, save_pressure

            engine = get_reach_out_engine()
            pressure = get_reach_out_pressure()

            logger.info(f"Checking reach-out (pressure: {pressure.level:.2f})...")

            should, reason, opener = engine.should_reach_out()
            suppression_type = classify_suppression(reason) if not should else "none"

            if should:
                # Telegram = not physically together. Clear scene before generating.
                self._clear_physical_presence_for_telegram()

                # Generate and send message
                message = engine.generate_message()

                if message and self._is_duplicate_proactive(message):
                    logger.info("Skipping duplicate proactive message, will try again next cycle")
                    pressure.decay(self._last_interval_minutes)
                    save_pressure(pressure)
                    return

                # Topic novelty gate — skip if topic was already discussed
                if message:
                    try:
                        from src.autonomy.topic_novelty_checker import get_topic_novelty_checker, COMPANION_TOPIC_NOVELTY_CHECK_ENABLED
                        if COMPANION_TOPIC_NOVELTY_CHECK_ENABLED:
                            checker = get_topic_novelty_checker()
                            from src.database.db import get_db
                            db = get_db()
                            recent = db.get_recent_messages(get_persona_config().primary_user_email, limit=25)
                            recent_texts = [m['message_text'] for m in recent if m.get('message_text')]
                            if not checker.is_novel(message, recent_texts, context_label="reach-out"):
                                logger.info("Reach-out suppressed - already discussed topic")
                                pressure.decay(self._last_interval_minutes)
                                save_pressure(pressure)
                                return
                    except Exception as e:
                        logger.debug(f"Reach-out novelty check failed (proceeding): {e}")

                if message:
                    # Final WebSocket check before sending (user may have come online)
                    if self._is_user_on_websocket():
                        logger.info("User came online during message generation - cancelling Telegram send")
                        return

                    success = self._send_via_telegram(message)

                    if success:
                        # Store in message history
                        self._store_proactive_message(message, 'telegram')
                        # Mark curiosity topics as asked (same fix as interjections)
                        self._mark_curiosity_topics_asked(message)
                        # Mark research findings as shared (only first 2 — matching
                        # the context limit in reach_out_engine.get_context())
                        try:
                            from src.tasks.autonomous_action_task import _load_findings, _save_findings
                            findings = _load_findings()
                            marked = 0
                            for finding in findings:
                                if not finding.get('shared', False):
                                    finding['shared'] = True
                                    marked += 1
                                    if marked >= 2:
                                        break
                            _save_findings(findings)
                        except Exception:
                            pass
                        # Release pressure
                        pressure.release()
                        save_pressure(pressure)
                        logger.info(f"Reached out, pressure released. Next interval: {pressure.get_next_interval_minutes():.1f} min")
                    else:
                        # Wanted to, generated message, but delivery failed
                        pressure.accumulate("delivery failed")
                        save_pressure(pressure)
                else:
                    logger.warning("Failed to generate reach-out message")
            else:
                if suppression_type == "external":
                    # She WANTED to but couldn't
                    pressure.accumulate(reason)
                    logger.info(
                        f"Pressure building: {pressure.level:.2f} "
                        f"(reason: {reason}, next: {pressure.get_next_interval_minutes():.1f} min)"
                    )
                else:
                    # She chose not to - decay pressure
                    pressure.decay(self._last_interval_minutes)
                    logger.debug(f"Internal 'no': {reason}. Pressure: {pressure.level:.2f}")

                save_pressure(pressure)

            self._last_check = datetime.now()

        except Exception as e:
            logger.error(f"Error in reach-out check: {e}")

    # -----------------------------------------------------------------
    # Deduplication & cleanup helpers
    # -----------------------------------------------------------------

    def _is_duplicate_proactive(self, message: str, threshold: float = 0.6) -> bool:
        """Check if this message is too similar to recent proactive messages (SequenceMatcher)."""
        from difflib import SequenceMatcher

        try:
            from src.database.db import get_db
            db = get_db()
            email = get_persona_config().primary_user_email
            recent = db.get_recent_messages(email, limit=10)

            # Match all autonomous companion messages (proactive reach-outs, interjections, milestones)
            autonomous_sources = {
                'telegram_proactive', 'websocket_interjection',
                'autonomous', 'telegram_milestone', 'proactive',
            }
            for msg in recent:
                source = msg.get('source', '')
                is_autonomous = (
                    msg.get('sender_name') == get_persona_config().companion_short_name
                    and (source in autonomous_sources or 'proactive' in source)
                )
                if is_autonomous:
                    similarity = SequenceMatcher(None, message.lower(), msg['message_text'].lower()).ratio()
                    if similarity > threshold:
                        logger.warning(f"Duplicate autonomous message detected ({similarity:.0%} similar, source={source})")
                        return True
            return False
        except Exception as e:
            logger.debug(f"Duplicate check failed: {e}")
            return False

    def _send_via_telegram(self, message: str) -> bool:
        """Send a message via Telegram. Returns True on success."""
        from src.autonomy.reach_out_engine import TELEGRAM_ENABLED

        if not TELEGRAM_ENABLED:
            return False

        try:
            from src.autonomy.telegram_bridge import send_to_telegram, get_telegram_bridge
            bridge = get_telegram_bridge()
            if bridge.is_ready():
                success = send_to_telegram(message)
                if success:
                    logger.info(f"Sent via Telegram: {message[:50]}...")
                return success
            else:
                logger.warning("Telegram bridge not ready")
                return False
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    def _store_proactive_message(self, message: str, channel: str):
        """Store a proactive message in the database."""
        try:
            from src.database.db import get_db
            db = get_db()
            email = get_persona_config().primary_user_email
            source = f'{channel}_proactive' if channel else 'proactive'
            db.store_message(
                email=email,
                sender_name=get_persona_config().companion_short_name,
                message_text=message,
                sentiment=0.5,
                closeness=100,
                source=source
            )
        except Exception as e:
            logger.warning(f"Failed to store proactive message: {e}")

    # =========================================================================
    # GOAL ACTIONS (event-driven, replaces cron schedule)
    # =========================================================================

    def _check_goal_actions(self):
        """
        Check if the companion should take an autonomous goal action.

        Called every loop iteration (offline branch). The budget gate
        (should_act()) is a cheap Redis check that returns "too soon" on
        most iterations, so this is safe to call frequently.
        """
        try:
            from src.tasks.autonomous_action_task import run_autonomous_action

            email = get_persona_config().primary_user_email
            result = run_autonomous_action(user_email=email)

            status = result.get('status', 'unknown')
            if status == 'success':
                action = result.get('action', '?')
                logger.info(f"Goal action executed: {action}")
            elif status == 'skipped':
                reason = result.get('reason', '?')
                logger.debug(f"Goal action skipped: {reason}")
            else:
                logger.warning(f"Goal action result: {result}")

        except Exception as e:
            logger.error(f"Error in goal action check: {e}")

    # =========================================================================
    # INTERJECTION (active WebSocket conversation path)
    # =========================================================================

    def _check_interjection(self):
        """
        Check if the companion should interject during an active WebSocket conversation.

        This runs on a fast cadence (2-5 min) when the user is connected.
        Uses the InterjectionEngine for LLM-driven decisions.
        """
        try:
            from src.autonomy.interjection_engine import get_interjection_engine

            engine = get_interjection_engine()
            should, reason, hint = engine.should_interject()

            if should:
                logger.info(f"Interjection triggered: {reason}")

                # Topic novelty gate — skip if topic was already discussed
                try:
                    from src.autonomy.topic_novelty_checker import get_topic_novelty_checker, COMPANION_TOPIC_NOVELTY_CHECK_ENABLED
                    if COMPANION_TOPIC_NOVELTY_CHECK_ENABLED and hint:
                        checker = get_topic_novelty_checker()
                        from src.database.db import get_db
                        db = get_db()
                        recent = db.get_recent_messages(get_persona_config().primary_user_email, limit=25)
                        recent_texts = [m['message_text'] for m in recent if m.get('message_text')]
                        if not checker.is_novel(hint, recent_texts, context_label="interjection hint"):
                            logger.info("Interjection suppressed - already discussed topic")
                            if reason == "deferred action return":
                                self._clear_deferred_action()
                            return
                except Exception as e:
                    logger.debug(f"Interjection novelty check failed (proceeding): {e}")

                message = self._generate_interjection_message(hint)

                if message and self._is_duplicate_proactive(message):
                    logger.info("Duplicate interjection detected, retrying with nudge")
                    message = self._generate_interjection_message(
                        f"{hint}\n\n[You already said something very similar to this recently. "
                        f"Say something DIFFERENT — a new thought, a follow-up, or just skip it "
                        f"and say something else entirely that's on your mind.]"
                    )
                    if message and self._is_duplicate_proactive(message):
                        logger.info("Retry still duplicate, aborting interjection")
                        # Still clear deferred action so it doesn't get stuck
                        if reason == "deferred action return":
                            self._clear_deferred_action()
                        return

                if message:
                    self._send_interjection(message)

                    if reason == "deferred action return":
                        self._clear_deferred_action()
                    else:
                        engine.record_interjection()

                    # Record intimate initiation (max 1 per session)
                    if getattr(engine, '_intimate_was_content_source', False):
                        engine.record_intimate_initiation()
                        engine._intimate_was_content_source = False

                    # Pop a queued thought after successful interjection
                    # This prevents the same thoughts from driving repeated interjections
                    self._pop_used_queued_thought(message)

                    # Mark any curiosity topics as asked (prevents repeat interjections)
                    self._mark_curiosity_topics_asked(message)

                    # Also release any built-up pressure since she got to say something
                    try:
                        from src.autonomy.reach_out_pressure import get_reach_out_pressure, save_pressure
                        pressure = get_reach_out_pressure()
                        pressure.release()
                        save_pressure(pressure)
                    except Exception:
                        pass
                else:
                    # Message generation failed — still clear deferred action
                    if reason == "deferred action return":
                        logger.warning("Deferred action message generation failed, clearing anyway")
                        self._clear_deferred_action()
            else:
                logger.debug(f"No interjection: {reason}")

        except Exception as e:
            logger.error(f"Error in interjection check: {e}")

    def _clear_deferred_action(self):
        """Clear the current deferred action from internal state."""
        try:
            from src.core.internal_state import get_internal_state_manager
            email = get_persona_config().primary_user_email
            get_internal_state_manager().clear_deferred_action(email)
        except Exception as e:
            logger.warning(f"Failed to clear deferred action: {e}")

    def _pop_used_queued_thought(self, interjection_message: str):
        """
        Remove queued thoughts that match the interjection message.

        When an interjection fires, any queued thoughts whose keywords appear
        in the generated message should be cleared so they don't drive
        repeated interjections on the same topic.
        """
        try:
            from src.core.internal_state import get_internal_state_manager
            email = get_persona_config().primary_user_email
            manager = get_internal_state_manager()
            state = manager.get_state(email)

            if not state.queued_thoughts:
                return

            msg_lower = interjection_message.lower()
            remaining = []
            removed = []

            for thought in state.queued_thoughts:
                # Extract meaningful words from the thought
                skip = {'the', 'and', 'for', 'with', 'about', 'from', 'that', 'this',
                        'his', 'her', 'their', 'what', 'how', 'why', 'when', 'where',
                        'check', 'ask', 'see', 'if', 'wants', 'want', 'talk', 'more',
                        'has', 'have', 'been', 'yet', 'still', 'feels', 'feeling',
                        'heard', 'anything', 'back', 'thought'}
                words = [w for w in thought.lower().split() if len(w) > 2 and w not in skip]

                if not words:
                    remaining.append(thought)
                    continue

                # If 40%+ of meaningful words appear in the message, it was addressed
                matches = sum(1 for w in words if w in msg_lower)
                if matches >= max(1, len(words) * 0.4):
                    removed.append(thought)
                else:
                    remaining.append(thought)

            if removed:
                state.queued_thoughts = remaining
                manager.save_state(email, state)
                logger.info(f"Cleared {len(removed)} queued thought(s) after interjection: {removed}")

        except Exception as e:
            logger.warning(f"Failed to pop queued thoughts: {e}")

    def _clear_physical_presence_for_telegram(self):
        """Clear physical presence when sending via Telegram (not physically together)."""
        try:
            from src.core.scene_tracker import get_scene_tracker
            email = get_persona_config().primary_user_email
            tracker = get_scene_tracker()
            scene = tracker.get_scene_state(email)
            if scene.physical_presence:
                scene.physical_presence = False
                scene.physical_state = None
                scene.posture = None
                scene.position_detail = None
                scene.james_position = None
                scene.companion_position = None
                tracker.save_scene_state(email, scene)
                logger.info("📱 Telegram outbound — cleared physical_presence before generation")
        except Exception as e:
            logger.warning(f"Could not clear scene for telegram: {e}")

    def _mark_curiosity_topics_asked(self, interjection_message: str):
        """
        Mark curiosity topics as 'asked' when they appear in an interjection.

        This fixes the bug where interjections never incremented times_asked,
        causing the same topic to fire repeatedly (e.g., Katie's birthday 7x).
        """
        try:
            from src.core.proactive_curiosity import get_proactive_curiosity

            curiosity = get_proactive_curiosity()
            msg_lower = interjection_message.lower()
            marked = []

            skip_words = {
                'the', 'and', 'for', 'with', 'about', 'from', 'that', 'this',
                'his', 'her', 'their', 'what', 'how', 'why', 'when', 'where',
                'hey', 'you', 'your', 'are', 'was', 'been',
            }

            for thread in curiosity.curiosity_threads:
                if thread.times_asked >= 3:
                    continue  # Already exhausted

                # Keyword match: check if topic keywords appear in the message
                topic_words = [
                    w for w in thread.topic.lower().split()
                    if len(w) > 2 and w not in skip_words
                ]
                if not topic_words:
                    continue

                matches = sum(1 for w in topic_words if w in msg_lower)
                # Be generous — any significant keyword overlap counts.
                # For 1-2 key words, require all. For 3+, require just 1.
                # Interjection text is free-form and may not use the exact topic words.
                threshold = len(topic_words) if len(topic_words) <= 2 else 1

                if matches >= threshold:
                    thread.times_asked += 1
                    thread.last_discussed = datetime.now()
                    marked.append(f"'{thread.topic}' (now asked {thread.times_asked}x)")

            if marked:
                curiosity._save_state()
                logger.info(f"Marked curiosity topics as asked via interjection: {', '.join(marked)}")

        except Exception as e:
            logger.warning(f"Failed to mark curiosity topics after interjection: {e}")

        # Also queue the formal resolution task as backup
        try:
            from src.tasks.curiosity_extraction_task import resolve_discussed_curiosities
            resolve_discussed_curiosities.delay(interjection_message)
        except Exception:
            pass

    def _generate_interjection_message(self, hint: str) -> Optional[str]:
        """Generate an interjection message via the full conversation pipeline."""
        try:
            from src.core.conversation.pipeline import get_conversation_pipeline

            pipeline = get_conversation_pipeline()
            email = get_persona_config().primary_user_email

            # Get pressure context so eagerness comes through
            eagerness_context = ""
            try:
                from src.autonomy.reach_out_pressure import get_reach_out_pressure
                pressure = get_reach_out_pressure()
                if pressure.level > 0.3:
                    eagerness_context = (
                        f"\n[EAGERNESS: You've been wanting to say this for a while "
                        f"(pressure: {pressure.level:.0%}). Let that eagerness come through "
                        f"naturally - you're excited/relieved to finally share this.]"
                    )
            except Exception:
                pass

            # Fetch recent autonomous messages for dedup context
            dedup_context = ""
            try:
                from src.database.db import get_db
                db = get_db()
                email_for_dedup = get_persona_config().primary_user_email
                recent = db.get_recent_messages(email_for_dedup, limit=10)
                autonomous_sources = {
                    'telegram_proactive', 'websocket_interjection',
                    'autonomous', 'telegram_milestone', 'proactive',
                }
                recent_auto = [
                    msg['message_text'] for msg in recent
                    if msg.get('sender_name') == get_persona_config().companion_short_name
                    and (msg.get('source', '') in autonomous_sources or 'proactive' in msg.get('source', ''))
                    and msg.get('message_text')
                ][:5]
                if recent_auto:
                    lines = "\n".join(f'- "{m[:100]}"' for m in recent_auto)
                    dedup_context = (
                        f"\n\nYOUR RECENT UNPROMPTED MESSAGES (DO NOT repeat or closely paraphrase):\n"
                        f"{lines}"
                    )
            except Exception:
                pass

            trigger_message = (
                f"[INTERJECTION_TRIGGER]\n"
                f"The companion wants to say something during a conversation pause.\n"
                f"What's on her mind: {hint}"
                f"{eagerness_context}{dedup_context}\n\n"
                f"Generate a natural, casual message - like texting a thought "
                f"mid-conversation."
            )

            # Check if this interjection is intimate-initiation driven
            from src.autonomy.interjection_engine import get_interjection_engine
            engine = get_interjection_engine()
            is_intimate_interjection = getattr(engine, '_intimate_was_content_source', False)

            result = pipeline.process(
                user_email=email,
                user_message=trigger_message,
                closeness_score=100,
                extra_context={
                    'is_proactive_message': True,
                    'is_interjection': True,
                    'is_intimate_interjection': is_intimate_interjection,
                }
            )

            if result.success and result.response:
                return result.response.strip()
            return None

        except Exception as e:
            logger.error(f"Failed to generate interjection: {e}")
            return None

    def _send_interjection(self, message: str):
        """
        Send an interjection via Redis pub/sub for WebSocket delivery.

        The subscriber in web_chat.py forwards it to the correct SocketIO room.
        """
        try:
            import redis

            email = get_persona_config().primary_user_email

            # 1. Store in database
            from src.database.db import get_db
            db = get_db()
            db.store_message(
                email=email,
                sender_name=get_persona_config().companion_short_name,
                message_text=message,
                sentiment=0.5,
                closeness=100,
                source='websocket_interjection'
            )

            # 2. Publish to Redis for WebSocket delivery
            r = redis.from_url(os.environ.get('REDIS_URL', 'redis://redis:6379/0'))
            r.publish('companion_interjection', json.dumps({
                'room_id': email,
                'response': message,
                'messages': [message],
                'timestamp': datetime.now().isoformat(),
                'source': 'interjection'
            }))

            logger.info(f"Interjection sent via WebSocket: {message[:50]}...")

        except Exception as e:
            logger.error(f"Failed to send interjection: {e}")

    # =========================================================================
    # LEGACY METHODS (kept for compatibility)
    # =========================================================================

    def _check_reach_out(self):
        """Legacy method - delegates to pressure-aware version."""
        self._check_reach_out_with_pressure()

    def force_check(self):
        """Force an immediate reach-out check."""
        self._check_reach_out_with_pressure()

    # =========================================================================
    # ACTIVITY TRACKING
    # =========================================================================

    def _check_user_schedule(self):
        """Check if user's scheduled wake/sleep time has been reached."""
        try:
            from src.core.user_context import check_scheduled_wake_sleep
            check_scheduled_wake_sleep()
        except Exception as e:
            logger.debug(f"Could not check user schedule: {e}")

    def _initialize_activity_tracking(self):
        """Initialize activity tracking on service start."""
        try:
            from src.scheduling.companion_schedule import get_companion_schedule

            schedule = get_companion_schedule()
            self._previous_activity_status = schedule.get_current_activity_status()
            logger.info(f"Activity tracking initialized: {self._previous_activity_status.get('status')}")

        except Exception as e:
            logger.warning(f"Could not initialize activity tracking: {e}")
            self._previous_activity_status = None

    def _check_activity_changes(self):
        """
        Check if the companion's activity has changed since last check.

        On change:
        1. Record the completed activity to background_life
        2. Update internal state
        3. Optionally trigger milestone-based message
        4. Bump reach-out pressure for natural messaging moments
        """
        try:
            from src.scheduling.companion_schedule import get_companion_schedule
            from src.core.background_life import get_background_life
            from src.core.internal_state import get_internal_state_manager

            schedule = get_companion_schedule()
            current_status = schedule.get_current_activity_status()

            # Skip if we don't have previous status
            if self._previous_activity_status is None:
                self._previous_activity_status = current_status
                return

            previous_status = self._previous_activity_status.get('status')
            new_status = current_status.get('status')

            # Check if status changed
            if previous_status != new_status:
                logger.info(f"Activity change detected: {previous_status} -> {new_status}")

                # 1. Record the completed activity
                self._record_activity_transition(
                    from_status=previous_status,
                    from_details=self._previous_activity_status.get('details', ''),
                    to_status=new_status,
                    to_details=current_status.get('details', '')
                )

                # 2. Update internal state based on transition
                self._update_internal_state_for_transition(
                    from_status=previous_status,
                    to_status=new_status
                )

                # 3. Check for milestone triggers (optional)
                if self._activity_milestone_enabled:
                    self._check_milestone_trigger(
                        from_status=previous_status,
                        to_status=new_status,
                        current_details=current_status.get('details', '')
                    )

                # 4. Bump pressure for natural messaging moments
                self._bump_pressure_for_transition(previous_status, new_status)

                # 5. Trigger goal action check on transitions to free time
                #    (natural moments to do background work)
                free_time_transitions = {
                    ('working', 'off'),
                    ('meeting', 'off'),
                    ('meeting', 'working'),
                    ('working', 'lunch'),
                }
                if (previous_status, new_status) in free_time_transitions:
                    logger.info(f"Activity transition {previous_status}->{new_status} — triggering goal action check")
                    self._check_goal_actions()

            # Update tracked status
            self._previous_activity_status = current_status

        except Exception as e:
            logger.error(f"Error checking activity changes: {e}")

    def _bump_pressure_for_transition(self, from_status: str, to_status: str):
        """
        Bump reach-out pressure when an activity transition creates a
        natural messaging moment.
        """
        natural_message_transitions = {
            ('working', 'off'): 0.10,       # End of work day - strong
            ('meeting', 'working'): 0.05,   # Meeting ended - mild
            ('asleep', 'working'): 0.08,    # Waking up - moderate
            ('working', 'lunch'): 0.06,     # Lunch break - moderate
        }

        bump_amount = natural_message_transitions.get((from_status, to_status))
        if bump_amount:
            try:
                from src.autonomy.reach_out_pressure import get_reach_out_pressure, save_pressure
                pressure = get_reach_out_pressure()
                pressure.bump(bump_amount, f"activity: {from_status}->{to_status}")
                save_pressure(pressure)
            except Exception as e:
                logger.debug(f"Could not bump pressure for transition: {e}")

    def _check_calendar_transitions(self):
        """
        Check if James just finished a calendar event.

        When an event ends, it's a natural moment to reach out
        (e.g., "how was your meeting?"). Bumps reach-out pressure.
        """
        try:
            from src.integrations.calendar_service import get_calendar_service, is_calendar_awareness_enabled

            if not is_calendar_awareness_enabled():
                return

            cal = get_calendar_service()
            busy_now, event_name = cal.is_user_busy_now()

            # Detect transition: was in event -> no longer in event
            if self._user_was_in_event and not busy_now:
                ended_event = self._user_last_event_name or "an event"
                logger.info(f"Calendar transition: James just finished '{ended_event}'")

                try:
                    from src.autonomy.reach_out_pressure import get_reach_out_pressure, save_pressure
                    pressure = get_reach_out_pressure()
                    pressure.bump(0.08, f"calendar: '{ended_event}' just ended")
                    save_pressure(pressure)
                except Exception as e:
                    logger.debug(f"Could not bump pressure for calendar transition: {e}")

            # Update tracking state
            self._user_was_in_event = busy_now
            self._user_last_event_name = event_name

        except Exception as e:
            logger.debug(f"Calendar transition check failed: {e}")

    def _record_activity_transition(
        self,
        from_status: str,
        from_details: str,
        to_status: str,
        to_details: str
    ):
        """Record the completed activity to background_life."""
        try:
            from src.core.background_life import get_background_life
            from src.core.internal_state import get_internal_state_manager

            bg_life = get_background_life()
            internal_state = get_internal_state_manager()

            # Get current energy/mood for context
            email = get_persona_config().primary_user_email
            state = internal_state.get_state(email)

            # Map status to activity type
            activity_type_map = {
                'working': 'work',
                'meeting': 'work',
                'lunch': 'personal',
                'off': 'personal',
                'asleep': 'rest'
            }

            activity_type = activity_type_map.get(from_status, 'personal')

            # Estimate duration (based on last interval)
            duration_hours = self._last_interval_minutes / 60

            # Record the activity that just ended
            bg_life.record_activity(
                activity_type=activity_type,
                description=from_details or f"Was {from_status}",
                duration_hours=duration_hours,
                mood=state.mood if state else "neutral",
                energy=state.energy if state else 0.5
            )

            logger.debug(f"Recorded activity: {activity_type} - {from_details[:50] if from_details else from_status}")

        except Exception as e:
            logger.warning(f"Could not record activity transition: {e}")

    def _update_internal_state_for_transition(self, from_status: str, to_status: str):
        """Update the companion's internal state based on activity transition."""
        try:
            from src.core.internal_state import get_internal_state_manager

            email = get_persona_config().primary_user_email
            manager = get_internal_state_manager()

            if from_status == 'working' and to_status == 'lunch':
                manager.adjust_energy(email, delta=0.05, reason="lunch break")
            elif from_status == 'meeting' and to_status in ('working', 'lunch', 'off'):
                manager.adjust_energy(email, delta=0.02, reason="meeting ended")
            elif from_status == 'asleep' and to_status != 'asleep':
                manager.adjust_energy(email, delta=0.3, reason="woke up")
            elif from_status == 'working' and to_status == 'off':
                manager.adjust_energy(email, delta=0.1, reason="work day ended")

        except Exception as e:
            logger.debug(f"Could not update internal state: {e}")

    def _check_milestone_trigger(self, from_status: str, to_status: str, current_details: str):
        """
        Check if this activity transition should trigger a proactive message.

        Only triggers on significant milestones and respects autonomy settings.
        """
        try:
            if os.environ.get('COMPANION_AUTONOMY_ENABLED', 'false').lower() != 'true':
                return

            milestone_transitions = {
                ('working', 'off'): 'work_day_ended',
                ('meeting', 'working'): 'meeting_finished',
                ('lunch', 'working'): None,
                ('working', 'lunch'): 'lunch_break',
            }

            trigger = milestone_transitions.get((from_status, to_status))

            if not trigger:
                return

            from src.database.db import get_db
            db = get_db()
            email = get_persona_config().primary_user_email

            minutes_since = db.get_minutes_since_last_message(email)
            if minutes_since is not None and minutes_since < 60:
                logger.debug(f"Skipping milestone trigger: messaged {minutes_since} min ago")
                return

            self._send_milestone_message(trigger, current_details)

        except Exception as e:
            logger.warning(f"Error in milestone trigger check: {e}")

    def _send_milestone_message(self, trigger_type: str, activity_details: str):
        """Send a milestone-triggered proactive message."""
        try:
            logger.info(f"Milestone trigger: {trigger_type}")

            os.environ['COMPANION_MILESTONE_TRIGGER'] = trigger_type
            os.environ['COMPANION_MILESTONE_DETAILS'] = activity_details

            try:
                from src.autonomy.reach_out_engine import check_and_reach_out

                success, message, channel = check_and_reach_out()

                if success:
                    logger.info(f"Milestone message sent via {channel}: {message[:50]}...")

                    from src.database.db import get_db
                    db = get_db()
                    email = get_persona_config().primary_user_email

                    db.store_message(
                        email=email,
                        sender_name=get_persona_config().companion_short_name,
                        message_text=message,
                        sentiment=0.5,
                        closeness=100,
                        source=f'{channel}_milestone' if channel else 'milestone'
                    )

            finally:
                os.environ.pop('COMPANION_MILESTONE_TRIGGER', None)
                os.environ.pop('COMPANION_MILESTONE_DETAILS', None)

        except Exception as e:
            logger.error(f"Failed to send milestone message: {e}")


# =============================================================================
# Singleton accessor & convenience functions
# =============================================================================

_service: Optional[AlwaysOnService] = None


def get_always_on_service() -> AlwaysOnService:
    """Get singleton AlwaysOnService instance."""
    global _service
    if _service is None:
        _service = AlwaysOnService()
    return _service


def start_always_on():
    """Start the always-on service."""
    service = get_always_on_service()
    service.start()
    return service


def stop_always_on():
    """Stop the always-on service."""
    service = get_always_on_service()
    service.stop()
