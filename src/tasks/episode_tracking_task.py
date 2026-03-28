"""
Episode Tracking Task - Manage conversation episode lifecycle.

WHAT: Detects episode boundaries (time gaps, topic shifts), creates new episodes,
      links messages to them, and updates topics/emotions as conversations evolve.
      Also closes stale episodes and generates approach summaries on close.

WHEN: process_message_episode fires after every message exchange.
      close_stale_episodes runs periodically (every few hours).
      summarize_episode fires when an episode is explicitly closed.

WHY:  Episodes group related messages into coherent conversation units. Without
      them, the companion has no concept of "that conversation about Jesse's
      school" vs "that late-night chat about work stress." Episodes enable
      pattern learning, satisfaction scoring, and topic-aware memory retrieval.
"""

import logging
from datetime import datetime

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)


@celery_app.task(
    name='tasks.episode_tracking.process_message',
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    soft_time_limit=30,
    time_limit=60
)
def process_message_episode(
    self,
    user_email: str,
    user_message: str,
    companion_response: str,
    user_msg_id: int = None,
    companion_msg_id: int = None
):
    """
    Process a message exchange for episode tracking.

    Called after each message to:
    1. Check if this starts a new episode
    2. Create episode if needed
    3. Link messages to current episode
    4. Update topic if it evolved

    Args:
        user_email: User's email
        user_message: User's message text
        companion_response: The companion's response text
        user_msg_id: ID of user message in database
        companion_msg_id: ID of the companion's response in database
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

        # Check for episode boundary
        is_new, reason, topic = detect_episode_boundary(
            user_email=user_email,
            current_message=user_message,
            store=store
        )

        if is_new:
            # Detect emotional state from the user's message
            emotional_state = detect_emotional_state(user_message, use_llm=True)

            # Create new episode
            episode = store.create_episode(
                user_email=user_email,
                topic=topic,
                trigger=EpisodeTrigger.TOPIC_SHIFT.value if reason == 'topic_shift' else EpisodeTrigger.USER_MESSAGE.value,
                emotional_state=emotional_state
            )
            if episode:
                logger.info(f"Created new episode: {episode.episode_id} - {topic} (reason: {reason})")
        else:
            # Get current episode
            episode = store.get_current_episode(user_email)

        if not episode:
            logger.warning(f"No episode available for {user_email}")
            return {'status': 'error', 'error': 'No episode'}

        # Link messages to episode
        if user_msg_id:
            store.add_message_to_episode(episode.episode_id, user_msg_id)
        if companion_msg_id:
            store.add_message_to_episode(episode.episode_id, companion_msg_id)

        # --- Periodic topic evolution check (every 5 messages) ---
        # Uses cheap keyword detection, not LLM. Appends new topics
        # rather than replacing, so "work stress" becomes "work stress, Jesse"
        if episode.message_count > 0 and episode.message_count % 5 == 0:
            new_topic = detect_topic(user_message, use_llm=False)
            if new_topic and new_topic != episode.topic:
                combined = f"{episode.topic}, {new_topic}"
                if len(combined) < 200:  # Prevent unbounded topic growth
                    store.update_topic(episode.episode_id, combined)

        # --- Periodic emotional state update (every 3 messages) ---
        # More frequent than topic checks because emotions shift faster
        if episode.message_count > 0 and episode.message_count % 3 == 0:
            current_emotion = detect_emotional_state(user_message, use_llm=True)
            if current_emotion and current_emotion != episode.emotional_state:
                store.update_emotional_state(episode.episode_id, current_emotion)
                logger.debug(f"Updated emotional state to: {current_emotion}")

        return {
            'status': 'success',
            'episode_id': episode.episode_id,
            'is_new': is_new,
            'reason': reason,
            'topic': topic
        }

    except Exception as e:
        logger.error(f"Episode tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


@celery_app.task(
    name='tasks.episode_tracking.close_stale_episodes',
    bind=True,
    soft_time_limit=120,
    time_limit=180
)
def close_stale_episodes(self, hours_threshold: int = 4):
    """
    Close episodes that have been inactive for too long.

    Run periodically to ensure episodes don't stay open forever.
    Scores satisfaction for each episode before closing.
    """
    try:
        from src.memory.episodic_episodes import (
            get_episode_store, EpisodeResolution, score_episode_satisfaction
        )
        from datetime import timedelta

        store = get_episode_store()
        conn = store._get_connection()

        # First, find stale episodes (don't close yet - we need to score them)
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT episode_id, user_email, topic
                FROM {T.EPISODES}
                WHERE resolution = 'ongoing'
                  AND updated_at < NOW() - INTERVAL '%s hours'
            """, (hours_threshold,))
            stale_episodes = cursor.fetchall()

        closed = []
        for ep_id, email, topic in stale_episodes:
            # Score satisfaction before closing
            satisfaction = score_episode_satisfaction(str(ep_id), store)

            # Close with satisfaction score
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.EPISODES}
                    SET resolution = %s,
                        ended_at = CURRENT_TIMESTAMP,
                        satisfaction = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id = %s::uuid
                """, (EpisodeResolution.INTERRUPTED.value, satisfaction, str(ep_id)))
                conn.commit()

            closed.append((ep_id, email, topic, satisfaction))
            logger.debug(f"  Closed: {ep_id} - {topic} (satisfaction: {satisfaction})")

        if closed:
            logger.info(f"Closed {len(closed)} stale episodes with satisfaction scoring")

        return {'status': 'success', 'closed_count': len(closed)}

    except Exception as e:
        logger.error(f"Failed to close stale episodes: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task(
    name='tasks.episode_tracking.summarize_episode',
    bind=True,
    soft_time_limit=60,
    time_limit=120
)
def summarize_episode(self, episode_id: str):
    """
    Generate a summary of what worked in an episode and score satisfaction.

    Called when an episode is closed to extract learnings.
    """
    try:
        from src.memory.episodic_episodes import (
            get_episode_store, score_episode_satisfaction
        )
        from src.llm.provider_factory import generate_sync

        store = get_episode_store()
        conn = store._get_connection()

        # Get episode messages
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT m.sender_name, m.message_text
                FROM {T.EPISODE_MESSAGES} em
                JOIN {T.MESSAGES} m ON m.id = em.message_id
                WHERE em.episode_id = %s::uuid
                ORDER BY em.turn_number
                LIMIT 20
            """, (episode_id,))
            messages = cursor.fetchall()

        if not messages:
            return {'status': 'no_messages'}

        # Format conversation
        conversation = "\n".join([
            f"{sender}: {text[:200]}" for sender, text in messages
        ])

        # Generate summary
        prompt = f"""Summarize what communication approach worked well in this conversation.
Focus on: tone, empathy, helpfulness, any strategies that seemed effective.

Conversation:
{conversation}

Write a 1-2 sentence summary of what worked:"""

        from src.llm.provider_factory import get_resilient_provider_chain
        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
            chain=chain
        )
        from src.services.cost_tracker import track_llm_call
        track_llm_call(chain, call_purpose='episode_tracking')

        if response:
            summary = response.strip()

            # Score satisfaction using the dedicated function
            satisfaction = score_episode_satisfaction(episode_id, store)
            if satisfaction is None:
                satisfaction = 0.7  # Fallback if scoring fails

            # Store the summary and satisfaction score
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.EPISODES}
                    SET approach_summary = %s,
                        satisfaction = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id = %s::uuid
                """, (summary, satisfaction, episode_id))
                conn.commit()

            logger.info(f"Summarized episode {episode_id}: {summary[:50]}... (satisfaction: {satisfaction})")
            return {'status': 'success', 'summary': summary, 'satisfaction': satisfaction}

        return {'status': 'no_summary'}

    except Exception as e:
        logger.error(f"Episode summarization failed: {e}")
        return {'status': 'error', 'error': str(e)}
