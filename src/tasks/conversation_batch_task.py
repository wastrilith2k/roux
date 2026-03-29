"""
Conversation Batch Task - Run conversation-level analyses once per conversation.

WHAT: Debounces three per-message tasks (interaction_outcome, event_synthesis,
      episode_tracking) so they fire once when a conversation goes idle,
      instead of on every message exchange.

WHY:  These tasks produce conversation-level results but were firing per-message,
      wasting ~7,700 LLM calls/month. Batching reduces this to ~60-80 calls
      (one per conversation), saving ~$1.75/month (20% of total cost).

HOW:  Each message calls schedule_conversation_batch(), which:
      1. Records the message timestamp in Redis
      2. Schedules a deferred Celery task with countdown=IDLE_SECONDS
      3. When the task fires, checks if newer messages arrived
      4. If idle: fetches full transcript, runs all three analyses
      5. If not idle: exits (newer message already scheduled its own task)
"""

import os
import time
import logging
from src.celery_app import celery_app

logger = logging.getLogger(__name__)

# How long (seconds) after the last message before we consider a conversation idle
# and run the batch tasks. Default: 5 minutes.
CONVERSATION_IDLE_SECONDS = int(os.environ.get('CONVERSATION_IDLE_SECONDS', '300'))

# Redis key prefixes for debounce tracking
_LAST_MSG_KEY = 'conv_batch:last_msg:{email}'
_LAST_COMPLETED_KEY = 'conv_batch:last_completed:{email}'


def _get_redis():
    """Get a Redis connection for batch tracking."""
    import redis
    redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
    return redis.from_url(redis_url)


def schedule_conversation_batch(user_email: str) -> None:
    """
    Schedule a deferred batch of conversation-level tasks.

    Called after each message. Uses Redis + Celery countdown to debounce:
    only the LAST scheduled task (after conversation goes idle) will execute.

    Args:
        user_email: User's email address
    """
    try:
        r = _get_redis()
        now = str(time.time())

        # Record this message's timestamp as the latest
        key = _LAST_MSG_KEY.format(email=user_email)
        r.set(key, now)

        # Schedule the batch task to run after the idle timeout.
        # Pass `now` so the task can verify no newer messages arrived.
        run_conversation_batch.apply_async(
            args=[user_email, now],
            countdown=CONVERSATION_IDLE_SECONDS
        )

        logger.debug(f"Conversation batch scheduled for {user_email} "
                      f"(fires in {CONVERSATION_IDLE_SECONDS}s if idle)")

    except Exception as e:
        logger.warning(f"Could not schedule conversation batch: {e}")


@celery_app.task(
    name='tasks.conversation_batch.run_batch',
    bind=True,
    max_retries=1,
    soft_time_limit=180,
    time_limit=240
)
def run_conversation_batch(self, user_email: str, scheduled_at: str):
    """
    Run batched conversation-level tasks if the conversation is idle.

    This task fires after CONVERSATION_IDLE_SECONDS. It checks whether
    any newer messages arrived since scheduling. If the conversation is
    still idle, it runs interaction_outcome, event_synthesis, and
    episode_tracking on the full conversation transcript.

    Args:
        user_email: User's email address
        scheduled_at: Timestamp string from when this task was scheduled
    """
    try:
        r = _get_redis()
        key = _LAST_MSG_KEY.format(email=user_email)
        current_last = r.get(key)

        # If a newer message arrived, skip — that message scheduled its own batch
        if current_last and current_last.decode() != scheduled_at:
            logger.debug(f"Conversation still active for {user_email}, skipping batch")
            return {'status': 'skipped', 'reason': 'conversation_still_active'}

        logger.info(f"Conversation idle for {user_email} — running batch tasks")

        # Determine the time window: from last completed batch to now
        completed_key = _LAST_COMPLETED_KEY.format(email=user_email)
        last_completed_raw = r.get(completed_key)
        last_completed_ts = float(last_completed_raw.decode()) if last_completed_raw else 0.0

        # Fetch conversation messages since the last batch
        messages = _get_conversation_messages(user_email, last_completed_ts)

        if not messages:
            logger.info(f"No messages to process for {user_email}")
            r.set(completed_key, str(time.time()))
            return {'status': 'no_messages'}

        # Build the full transcript for analysis
        transcript_lines = []
        user_messages = []
        companion_messages = []
        message_ids = []

        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()

        for msg in messages:
            sender = msg.get('sender_name', '')
            text = msg.get('message_text', '')
            msg_id = msg.get('id')

            if msg_id:
                message_ids.append(msg_id)

            if sender == _pc.primary_user_name:
                transcript_lines.append(f"{_pc.primary_user_name}: {text}")
                user_messages.append((msg_id, text))
            else:
                transcript_lines.append(f"{_pc.companion_short_name}: {text}")
                companion_messages.append((msg_id, text))

        full_transcript = "\n".join(transcript_lines)

        # --- Run the three batch tasks ---
        results = {}

        # 1. Episode tracking — batch-link messages and detect boundaries
        results['episode'] = _batch_episode_tracking(
            user_email, messages, user_messages, companion_messages, _pc
        )

        # 2. Event synthesis — scan full transcript for life events
        results['events'] = _batch_event_synthesis(
            user_email, full_transcript, message_ids
        )

        # 3. Interaction outcome — analyze the overall exchange
        results['outcome'] = _batch_interaction_outcome(
            user_email, user_messages, companion_messages
        )

        # Mark batch as completed
        r.set(completed_key, str(time.time()))

        logger.info(
            f"Conversation batch completed for {user_email}: "
            f"episode={results['episode'].get('status')}, "
            f"events={results['events'].get('status')}, "
            f"outcome={results['outcome'].get('status')}"
        )

        return {'status': 'success', 'results': results}

    except Exception as e:
        logger.error(f"Conversation batch failed for {user_email}: {e}")
        import traceback
        traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)
        return {'status': 'error', 'error': str(e)}


def _get_conversation_messages(user_email: str, since_timestamp: float) -> list:
    """
    Fetch messages from the database since a given Unix timestamp.

    Returns messages in chronological order (oldest first).
    """
    from src.database.db import get_db
    from src.database import tables as T
    from datetime import datetime
    from zoneinfo import ZoneInfo

    db = get_db()
    PST = ZoneInfo('America/Los_Angeles')

    # Convert Unix timestamp to datetime for SQL query
    if since_timestamp > 0:
        since_dt = datetime.fromtimestamp(since_timestamp, tz=PST)
    else:
        # If no previous batch, get messages from the last 24 hours
        from datetime import timedelta
        since_dt = datetime.now(PST) - timedelta(hours=24)

    try:
        with db._get_connection(user_email=user_email) as conn:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f"""
                SELECT id, sender_name, message_text, timestamp
                FROM {T.MESSAGES}
                WHERE email = %s AND timestamp > %s
                ORDER BY timestamp ASC, id ASC
            """, (user_email, since_dt))
            messages = [dict(row) for row in cursor.fetchall()]
            cursor.close()
        return messages
    except Exception as e:
        logger.error(f"Failed to fetch conversation messages: {e}")
        return []


def _batch_episode_tracking(
    user_email: str,
    messages: list,
    user_messages: list,
    companion_messages: list,
    persona_config
) -> dict:
    """
    Run episode tracking once for the full conversation.

    Creates/continues an episode and links all conversation messages to it.
    Detects emotional state once from the full conversation rather than
    per-message.
    """
    try:
        from src.memory.episodic_episodes import (
            get_episode_store,
            detect_episode_boundary,
            detect_topic,
            detect_emotional_state,
            EpisodeTrigger
        )

        store = get_episode_store()

        if not user_messages:
            return {'status': 'skipped', 'reason': 'no_user_messages'}

        # Use the first user message for boundary detection
        first_user_msg = user_messages[0][1]

        # Check for episode boundary using the first message
        is_new, reason, topic = detect_episode_boundary(
            user_email=user_email,
            current_message=first_user_msg,
            store=store
        )

        if is_new:
            # Detect emotional state once from the full conversation
            all_user_text = " ".join(text for _, text in user_messages)
            emotional_state = detect_emotional_state(all_user_text, use_llm=True)

            episode = store.create_episode(
                user_email=user_email,
                topic=topic,
                trigger=(EpisodeTrigger.TOPIC_SHIFT.value
                         if reason == 'topic_shift'
                         else EpisodeTrigger.USER_MESSAGE.value),
                emotional_state=emotional_state
            )
            if episode:
                logger.info(f"Created new episode: {episode.episode_id} - {topic} (reason: {reason})")
        else:
            episode = store.get_current_episode(user_email)

        if not episode:
            logger.warning(f"No episode available for {user_email}")
            return {'status': 'error', 'error': 'No episode'}

        # Batch-link all messages to the episode
        linked = 0
        for msg in messages:
            msg_id = msg.get('id')
            if msg_id:
                store.add_message_to_episode(episode.episode_id, msg_id)
                linked += 1

        # Update topic once if conversation was long enough
        if len(user_messages) >= 5:
            all_user_text = " ".join(text for _, text in user_messages)
            new_topic = detect_topic(all_user_text, use_llm=False)
            if new_topic and new_topic != episode.topic:
                combined = f"{episode.topic}, {new_topic}"
                if len(combined) < 200:
                    store.update_topic(episode.episode_id, combined)

        # Update emotional state once for the full conversation
        if len(user_messages) >= 3 and not is_new:
            all_user_text = " ".join(text for _, text in user_messages)
            current_emotion = detect_emotional_state(all_user_text, use_llm=True)
            if current_emotion and current_emotion != episode.emotional_state:
                store.update_emotional_state(episode.episode_id, current_emotion)

        return {
            'status': 'success',
            'episode_id': str(episode.episode_id),
            'is_new': is_new,
            'messages_linked': linked
        }

    except Exception as e:
        logger.error(f"Batch episode tracking failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _batch_event_synthesis(
    user_email: str,
    full_transcript: str,
    message_ids: list
) -> dict:
    """
    Run event synthesis once on the full conversation transcript.

    Scans the entire conversation for event triggers instead of
    checking each message individually.
    """
    try:
        from src.memory.synthesized_events import (
            detect_event_type,
            extract_event_subject,
            get_event_synthesizer,
            EVENT_TRIGGERS
        )
        from src.tasks.event_synthesis_task import (
            get_related_facts,
            get_related_messages,
            generate_event_title
        )

        # Check the full transcript for event triggers
        detection = detect_event_type(full_transcript)

        if not detection:
            logger.debug("No event triggers in conversation")
            return {'status': 'no_event'}

        event_type, keywords = detection
        subject = extract_event_subject(full_transcript, event_type)

        if subject == "SKIP":
            logger.info("Event skipped - subject not applicable")
            return {'status': 'skipped', 'reason': 'subject_not_applicable'}

        logger.info(f"Detected {event_type} event for {subject} (keywords: {keywords})")

        # Get related facts and messages for context
        related_facts = get_related_facts(
            user_email=user_email,
            event_type=event_type,
            subject=subject,
            keywords=keywords,
            days_back=30
        )

        related_messages = get_related_messages(
            user_email=user_email,
            keywords=keywords,
            days_back=14
        )

        # Build facts for synthesis
        from datetime import datetime
        from zoneinfo import ZoneInfo
        PST = ZoneInfo('America/Los_Angeles')

        facts_for_synthesis = []
        for fact in related_facts:
            facts_for_synthesis.append({
                'id': fact.get('id'),
                'object': fact.get('object', ''),
                'created_at': fact.get('created_at'),
                'importance': fact.get('importance', 5)
            })

        for msg in related_messages:
            facts_for_synthesis.append({
                'id': msg.get('id'),
                'object': f"{msg['sender_name']}: {msg['message_text'][:200]}",
                'created_at': msg.get('timestamp'),
                'importance': 5
            })

        if not facts_for_synthesis:
            # Use the first message ID from this conversation
            first_msg_id = message_ids[0] if message_ids else None
            facts_for_synthesis.append({
                'id': first_msg_id,
                'object': full_transcript[:500],
                'created_at': datetime.now(PST),
                'importance': 6
            })

        title = generate_event_title(event_type, subject, keywords)

        synthesizer = get_event_synthesizer()
        event = synthesizer.synthesize_event(
            event_type=event_type,
            subject=subject,
            title=title,
            facts=facts_for_synthesis,
            user_email=user_email
        )

        if event:
            logger.info(f"Synthesized event: {event.title} (id={event.id})")
            return {
                'status': 'success',
                'event_id': event.id,
                'event_type': event_type,
                'subject': subject
            }

        return {'status': 'failed', 'reason': 'synthesis_returned_no_event'}

    except Exception as e:
        logger.error(f"Batch event synthesis failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _batch_interaction_outcome(
    user_email: str,
    user_messages: list,
    companion_messages: list
) -> dict:
    """
    Run interaction outcome analysis once for the full conversation.

    Instead of analyzing each companion->user pair individually, produces
    a single outcome assessment for the entire exchange.
    """
    outcome_enabled = os.environ.get(
        'COMPANION_OUTCOME_TRACKING_ENABLED', 'true'
    ).lower() == 'true'

    if not outcome_enabled:
        return {'status': 'disabled'}

    if not user_messages or not companion_messages:
        return {'status': 'skipped', 'reason': 'insufficient_messages'}

    try:
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain
        from src.database import tables as T

        # Build a summary of the companion's messages and the user's responses
        companion_summary = "\n".join(
            f"- {text[:200]}" for _, text in companion_messages[:10]
        )
        user_summary = "\n".join(
            f"- {text[:200]}" for _, text in user_messages[:10]
        )

        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()

        prompt = f"""Analyze this conversation exchange between {_pc.companion_short_name} and {_pc.primary_user_name}:

{_pc.companion_short_name} said:
{companion_summary}

{_pc.primary_user_name} responded:
{user_summary}

Classify the OVERALL conversation:
1. ACTION_TYPE: What was {_pc.companion_short_name} primarily doing? (curiosity_followup, advice, emotional_support, topic_introduction, question, playful, general)
2. TOPIC: Brief topic/subject (max 5 words)
3. ENGAGEMENT: How engaged was {_pc.primary_user_name} overall? (enthusiastic, engaged, brief, deflected, ignored)
4. CONTINUED: Did topics carry through? (yes/no)
5. RESONANCE: Emotional resonance -1.0 to 1.0 (negative=disconnect, 0=neutral, positive=connection)
6. NOTES: Brief observation (1 sentence)

Format:
ACTION_TYPE: ...
TOPIC: ...
ENGAGEMENT: ...
CONTINUED: yes/no
RESONANCE: N.N
NOTES: ..."""

        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[
                {"role": "system", "content": "Classify conversation interaction outcomes. Be brief."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150,
            chain=chain
        )
        from src.services.cost_tracker import track_llm_call
        track_llm_call(chain, call_purpose='interaction_outcome', user_id=user_email)

        # Parse structured response (same logic as original task)
        action_type = 'general'
        topic = ''
        engagement = 'engaged'
        continued = False
        resonance = 0.0
        notes = ''

        for line in response.strip().split('\n'):
            line = line.strip()
            if line.upper().startswith('ACTION_TYPE:'):
                action_type = line.split(':', 1)[1].strip().lower()
            elif line.upper().startswith('TOPIC:'):
                topic = line.split(':', 1)[1].strip()[:255]
            elif line.upper().startswith('ENGAGEMENT:'):
                engagement = line.split(':', 1)[1].strip().lower()
            elif line.upper().startswith('CONTINUED:'):
                continued = 'yes' in line.lower()
            elif line.upper().startswith('RESONANCE:'):
                try:
                    resonance = float(line.split(':', 1)[1].strip().split()[0])
                    resonance = max(-1.0, min(1.0, resonance))
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith('NOTES:'):
                notes = line.split(':', 1)[1].strip()

        valid_levels = {'enthusiastic', 'engaged', 'brief', 'deflected', 'ignored'}
        if engagement not in valid_levels:
            engagement = 'engaged'

        # Store result — use the last companion and user message IDs
        from src.database.db import get_db

        last_companion_id = companion_messages[-1][0]
        last_user_id = user_messages[-1][0]
        companion_text = "\n".join(text[:200] for _, text in companion_messages)[:1000]
        user_text = "\n".join(text[:200] for _, text in user_messages)[:1000]

        db = get_db()
        with db._get_connection(user_email=user_email) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.INTERACTION_OUTCOMES} (
                        user_email, companion_message_id, user_response_id,
                        companion_message_text, user_response_text,
                        companion_action_type, companion_topic, engagement_level,
                        topic_continued, emotional_resonance, analysis_notes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_email, last_companion_id, last_user_id,
                    companion_text, user_text,
                    action_type, topic, engagement,
                    continued, resonance, notes
                ))

        logger.info(
            f"Batch outcome tracked: action={action_type}, engagement={engagement}, "
            f"resonance={resonance:.1f}, topic={topic[:40]}"
        )

        return {
            'status': 'success',
            'action_type': action_type,
            'engagement': engagement,
            'resonance': resonance
        }

    except Exception as e:
        logger.error(f"Batch interaction outcome failed: {e}")
        return {'status': 'error', 'error': str(e)}
