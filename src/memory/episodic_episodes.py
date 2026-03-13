"""
Episodic Episodes - Structured conversation episodes with topic tracking.

WHAT: Groups sequential chat messages into coherent "episodes" -- logical
conversation segments defined by topic and temporal boundaries. Each episode
tracks its topic, emotional state, resolution, satisfaction score, and the
approach that worked (or didn't).

WHY: Raw message streams have no structure. Episodes let the companion
reason at a higher level -- "the last time we talked about Jesse's school,
a supportive approach worked well (satisfaction 0.85)." This powers the
episode learning pipeline that distills conversational strategies.

HOW it fits:
  - The message handler calls detect_episode_boundary() on each incoming
    message to decide whether to continue the current episode or start a new
    one.
  - When an episode closes, score_episode_satisfaction() rates it and
    _analyze_episode_async() triggers the episode learning task in the
    background.
  - context_builder can use format_episodes_for_prompt() to include past
    episodes about the same topic.

Episode boundaries are detected by:
  1. Time gaps (>4 hours of silence = new episode)
  2. Explicit topic shift phrases ("by the way", "on another note", etc.)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# Configuration
TIME_GAP_HOURS = 4  # Hours of silence to trigger new episode


class EpisodeTrigger(str, Enum):
    USER_MESSAGE = "user_message"
    PROACTIVE = "proactive"
    TOPIC_SHIFT = "topic_shift"


class EpisodeResolution(str, Enum):
    ONGOING = "ongoing"
    RESOLVED = "resolved"
    TABLED = "tabled"
    INTERRUPTED = "interrupted"


@dataclass
class Episode:
    """A conversation episode with metadata."""
    id: Optional[int] = None
    episode_id: Optional[str] = None
    user_email: str = ""
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    topic: str = ""
    trigger: str = EpisodeTrigger.USER_MESSAGE.value
    emotional_state: str = ""
    resolution: str = EpisodeResolution.ONGOING.value
    satisfaction: Optional[float] = None
    approach_summary: str = ""
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'episode_id': self.episode_id,
            'user_email': self.user_email,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'topic': self.topic,
            'trigger': self.trigger,
            'emotional_state': self.emotional_state,
            'resolution': self.resolution,
            'satisfaction': self.satisfaction,
            'approach_summary': self.approach_summary,
            'message_count': self.message_count
        }


# =============================================================================
# Episode database operations
# =============================================================================

class EpisodeStore:
    """PostgreSQL-backed CRUD for episodes and their message associations."""

    def __init__(self):
        self._conn = None

    def _get_connection(self):
        import psycopg2
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def get_current_episode(self, user_email: str) -> Optional[Episode]:
        """Get the current ongoing episode for a user."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM episodes
                    WHERE user_email = %s AND resolution = 'ongoing'
                    ORDER BY started_at DESC
                    LIMIT 1
                """, (user_email,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_episode(row)
                return None
        except Exception as e:
            logger.warning(f"Failed to get current episode: {e}")
            return None

    def create_episode(
        self,
        user_email: str,
        topic: str = "",
        trigger: str = EpisodeTrigger.USER_MESSAGE.value,
        emotional_state: str = ""
    ) -> Optional[Episode]:
        """Create a new episode."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    INSERT INTO episodes (user_email, topic, trigger, emotional_state, started_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING *
                """, (user_email, topic, trigger, emotional_state))
                row = cursor.fetchone()
                conn.commit()
                if row:
                    logger.info(f"Created episode: {row['episode_id']} - {topic}")
                    return self._row_to_episode(row)
                return None
        except Exception as e:
            logger.error(f"Failed to create episode: {e}")
            conn.rollback()
            return None

    def close_episode(
        self,
        episode_id: str,
        resolution: str = EpisodeResolution.RESOLVED.value,
        satisfaction: float = None,
        approach_summary: str = ""
    ) -> bool:
        """Close an episode with outcome data."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE episodes
                    SET ended_at = CURRENT_TIMESTAMP,
                        resolution = %s,
                        satisfaction = %s,
                        approach_summary = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id = %s::uuid
                """, (resolution, satisfaction, approach_summary, episode_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to close episode: {e}")
            conn.rollback()
            return False

    def add_message_to_episode(self, episode_id: str, message_id: int) -> bool:
        """Link a message to an episode."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Get current turn count
                cursor.execute("""
                    SELECT COALESCE(MAX(turn_number), 0) + 1 as next_turn
                    FROM episode_messages
                    WHERE episode_id = %s::uuid
                """, (episode_id,))
                next_turn = cursor.fetchone()[0]

                # Add message
                cursor.execute("""
                    INSERT INTO episode_messages (episode_id, message_id, turn_number)
                    VALUES (%s::uuid, %s, %s)
                    ON CONFLICT (episode_id, message_id) DO NOTHING
                """, (episode_id, message_id, next_turn))

                # Update message count
                cursor.execute("""
                    UPDATE episodes
                    SET message_count = message_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id = %s::uuid
                """, (episode_id,))

                conn.commit()
                return True
        except Exception as e:
            logger.warning(f"Failed to add message to episode: {e}")
            conn.rollback()
            return False

    def update_topic(self, episode_id: str, topic: str) -> bool:
        """Update episode topic."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE episodes
                    SET topic = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id = %s::uuid
                """, (topic, episode_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Failed to update topic: {e}")
            conn.rollback()
            return False

    def update_emotional_state(self, episode_id: str, emotional_state: str) -> bool:
        """Update episode emotional state."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE episodes
                    SET emotional_state = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id = %s::uuid
                """, (emotional_state, episode_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Failed to update emotional state: {e}")
            conn.rollback()
            return False

    def get_recent_episodes(
        self,
        user_email: str,
        days_back: int = 14,
        limit: int = 10
    ) -> List[Episode]:
        """Get recent episodes for a user."""
        conn = self._get_connection()
        try:
            cutoff = datetime.now(PST) - timedelta(days=days_back)
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM episodes
                    WHERE user_email = %s
                      AND started_at > %s
                    ORDER BY started_at DESC
                    LIMIT %s
                """, (user_email, cutoff, limit))
                rows = cursor.fetchall()
                return [self._row_to_episode(row) for row in rows]
        except Exception as e:
            logger.warning(f"Failed to get recent episodes: {e}")
            return []

    def get_episodes_by_topic(
        self,
        user_email: str,
        topic_keyword: str,
        limit: int = 5
    ) -> List[Episode]:
        """Find episodes matching a topic keyword."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM episodes
                    WHERE user_email = %s
                      AND LOWER(topic) LIKE %s
                      AND resolution != 'ongoing'
                    ORDER BY satisfaction DESC NULLS LAST, started_at DESC
                    LIMIT %s
                """, (user_email, f'%{topic_keyword.lower()}%', limit))
                rows = cursor.fetchall()
                return [self._row_to_episode(row) for row in rows]
        except Exception as e:
            logger.warning(f"Failed to search episodes by topic: {e}")
            return []

    def get_time_since_last_message(self, user_email: str) -> Optional[timedelta]:
        """Get time since the user's last message."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT MAX(timestamp) FROM messages
                    WHERE email = %s AND sender_name = 'User'
                """, (user_email,))
                result = cursor.fetchone()
                if result and result[0]:
                    last_time = result[0]
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=PST)
                    return datetime.now(PST) - last_time
                return None
        except Exception as e:
            logger.warning(f"Failed to get time since last message: {e}")
            return None

    def _row_to_episode(self, row: Dict) -> Episode:
        return Episode(
            id=row.get('id'),
            episode_id=str(row.get('episode_id')),
            user_email=row.get('user_email', ''),
            started_at=row.get('started_at'),
            ended_at=row.get('ended_at'),
            topic=row.get('topic', ''),
            trigger=row.get('trigger', ''),
            emotional_state=row.get('emotional_state', ''),
            resolution=row.get('resolution', ''),
            satisfaction=row.get('satisfaction'),
            approach_summary=row.get('approach_summary', ''),
            message_count=row.get('message_count', 0)
        )


# =============================================================================
# Episode boundary detection and satisfaction scoring
# =============================================================================

def score_episode_satisfaction(episode_id: str, store: EpisodeStore = None) -> Optional[float]:
    """
    Score the satisfaction level of an episode using LLM analysis.

    Analyzes the conversation to determine:
    - Did the conversation reach a natural conclusion?
    - Was the emotional tone positive at the end?
    - Were the user's needs addressed?

    Args:
        episode_id: The episode to score
        store: Optional EpisodeStore instance

    Returns:
        Satisfaction score 0.0-1.0, or None if scoring failed
    """
    if store is None:
        store = get_episode_store()

    try:
        from src.llm.provider_factory import generate_sync

        conn = store._get_connection()

        # Get the episode's messages
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT m.sender_name, m.message_text
                FROM episode_messages em
                JOIN messages m ON m.id = em.message_id
                WHERE em.episode_id = %s::uuid
                ORDER BY em.turn_number
                LIMIT 15
            """, (episode_id,))
            messages = cursor.fetchall()

        if len(messages) < 2:
            # Too few messages to score meaningfully
            return 0.5  # Neutral default

        # Format conversation (last 15 messages)
        conversation = "\n".join([
            f"{sender}: {text[:150]}" for sender, text in messages
        ])

        prompt = f"""Rate the satisfaction level of this conversation on a scale from 0.0 to 1.0.

Consider:
- Did the conversation end naturally or abruptly?
- Was the emotional tone positive, neutral, or negative at the end?
- Did the user seem satisfied with the interaction?
- Were the user's needs/questions addressed?

Conversation:
{conversation}

Reply with ONLY a decimal number between 0.0 and 1.0 (e.g., 0.7 or 0.85).
- 0.0-0.3: Negative/unresolved (conflict, frustration, abrupt ending)
- 0.4-0.6: Neutral/incomplete (neither good nor bad)
- 0.7-1.0: Positive/satisfying (natural ending, needs addressed, good rapport)

Score:"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )

        if response:
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))  # Clamp to 0-1 range
            except ValueError:
                logger.warning(f"Invalid satisfaction score response: {response}")
                return 0.5

        return 0.5  # Default neutral

    except Exception as e:
        logger.warning(f"Failed to score episode satisfaction: {e}")
        return None


def detect_episode_boundary(
    user_email: str,
    current_message: str,
    store: EpisodeStore = None,
    analyze_on_close: bool = True
) -> Tuple[bool, str, str]:
    """
    Detect if the current message starts a new episode.

    Args:
        user_email: The user's email
        current_message: The current message text
        store: Optional EpisodeStore instance
        analyze_on_close: Whether to analyze the episode for lessons when closing

    Returns:
        Tuple of (is_new_episode, reason, detected_topic)
    """
    if store is None:
        store = get_episode_store()

    # Get current episode
    current_episode = store.get_current_episode(user_email)

    # First message or no ongoing episode
    if not current_episode:
        topic = detect_topic(current_message)
        return True, "first_message", topic

    # Time-based boundary
    time_since = store.get_time_since_last_message(user_email)
    if time_since and time_since > timedelta(hours=TIME_GAP_HOURS):
        logger.info(f"New episode due to time gap: {time_since}")
        # Score satisfaction before closing
        satisfaction = score_episode_satisfaction(current_episode.episode_id, store)
        # Close the old episode with satisfaction score
        store.close_episode(
            current_episode.episode_id,
            resolution=EpisodeResolution.INTERRUPTED.value,
            satisfaction=satisfaction
        )
        # Analyze for lessons (async to not block response)
        if analyze_on_close:
            _analyze_episode_async(current_episode.episode_id)
        topic = detect_topic(current_message)
        return True, "time_gap", topic

    # Topic shift detection (simplified - check for explicit topic changes)
    if _is_explicit_topic_change(current_message):
        topic = detect_topic(current_message)
        if topic and topic.lower() != current_episode.topic.lower():
            logger.info(f"New episode due to topic shift: {current_episode.topic} -> {topic}")
            # Score satisfaction before closing
            satisfaction = score_episode_satisfaction(current_episode.episode_id, store)
            # Close with satisfaction
            store.close_episode(
                current_episode.episode_id,
                resolution=EpisodeResolution.RESOLVED.value,
                satisfaction=satisfaction
            )
            # Analyze for lessons (async to not block response)
            if analyze_on_close:
                _analyze_episode_async(current_episode.episode_id)
            return True, "topic_shift", topic

    # Continue current episode
    return False, "continuation", current_episode.topic


def _analyze_episode_async(episode_id: str):
    """
    Trigger episode analysis asynchronously.

    Uses a thread to avoid blocking the response. Falls back to
    synchronous analysis if threading fails.
    """
    try:
        import threading

        def analyze():
            try:
                from src.tasks.episode_learning_task import analyze_episode_on_close
                analyze_episode_on_close(episode_id)
            except Exception as e:
                logger.warning(f"Async episode analysis failed: {e}")

        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
        logger.debug(f"Started async analysis for episode {episode_id}")

    except Exception as e:
        logger.warning(f"Failed to start async analysis: {e}")


def _is_explicit_topic_change(message: str) -> bool:
    """Check for explicit topic change indicators."""
    change_indicators = [
        'by the way', 'btw', 'anyway', 'changing topic',
        'on another note', 'speaking of', 'that reminds me',
        'i wanted to ask', 'can we talk about', 'let me tell you about'
    ]
    message_lower = message.lower()
    return any(ind in message_lower for ind in change_indicators)


def detect_topic(message: str, use_llm: bool = True) -> str:
    """
    Detect the topic of a message.

    Uses LLM for accurate detection, falls back to keyword extraction.
    """
    if not use_llm or len(message) < 20:
        return _extract_topic_keywords(message)

    try:
        from src.llm.provider_factory import generate_sync

        prompt = f"""Identify the main topic of this message in 2-5 words.
Focus on what the person wants to discuss.

Message: "{message[:500]}"

Reply with ONLY the topic (2-5 words), nothing else."""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.1
        )

        if response:
            topic = response.strip().strip('"\'')
            if len(topic) < 50:  # Sanity check
                return topic

        return _extract_topic_keywords(message)

    except Exception as e:
        logger.warning(f"LLM topic detection failed: {e}")
        return _extract_topic_keywords(message)


def _extract_topic_keywords(message: str) -> str:
    """Extract topic keywords from message (fallback)."""
    # Simple keyword extraction
    stopwords = {
        'i', 'me', 'my', 'you', 'your', 'we', 'the', 'a', 'an', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'so', 'if', 'then', 'that', 'this', 'these',
        'just', 'really', 'very', 'about', 'think', 'know', 'want', 'like'
    }

    words = message.lower().split()
    keywords = []
    for word in words:
        word = word.strip('.,!?()[]{}"\'-')
        if len(word) > 3 and word not in stopwords:
            keywords.append(word)
            if len(keywords) >= 3:
                break

    return ' '.join(keywords) if keywords else 'general chat'


# Valid emotional states for episodes
VALID_EMOTIONAL_STATES = {
    'neutral', 'happy', 'sad', 'stressed', 'anxious', 'frustrated',
    'affectionate', 'playful', 'tired', 'excited', 'relieved',
    'angry', 'worried', 'overwhelmed', 'hopeful', 'vulnerable'
}


def detect_emotional_state(message: str, use_llm: bool = True) -> str:
    """
    Detect the emotional state expressed in a message.

    Uses LLM for accurate detection, falls back to keyword matching.

    Args:
        message: The message text to analyze
        use_llm: Whether to use LLM (default True)

    Returns:
        One of the valid emotional states
    """
    if not message or len(message.strip()) < 10:
        return 'neutral'

    # Fast path: keyword-based detection for unambiguous emotional signals.
    # If a keyword matches and LLM is disabled, return immediately.
    # If LLM is enabled, the keyword match just confirms we should proceed
    # to LLM for a more nuanced classification.
    message_lower = message.lower()

    keyword_emotions = {
        'worried': ['worried', 'worry', 'anxious', 'nervous', 'scared', 'afraid'],
        'stressed': ['stressed', 'overwhelmed', 'exhausted', 'drained', 'burnt out', 'burnout'],
        'frustrated': ['frustrated', 'annoyed', 'irritated', 'fed up', 'pissed'],
        'sad': ['sad', 'depressed', 'down', 'crying', 'tears', 'heartbroken'],
        'angry': ['angry', 'furious', 'rage', 'livid', 'pissed off'],
        'happy': ['happy', 'excited', 'great', 'wonderful', 'amazing', 'thrilled'],
        'affectionate': ['love you', 'miss you', 'hug', 'kiss', 'cuddle', 'hold me'],
        'tired': ['tired', 'sleepy', 'exhausted', 'need sleep', 'so tired'],
        'hopeful': ['hopeful', 'optimistic', 'looking forward', 'excited about'],
        'vulnerable': ['vulnerable', 'scared', 'need you', 'dont know what to do'],
    }

    for emotion, keywords in keyword_emotions.items():
        if any(kw in message_lower for kw in keywords):
            if not use_llm:
                return emotion
            # Found keyword, but use LLM to confirm if enabled
            break

    if not use_llm:
        return 'neutral'

    try:
        from src.llm.provider_factory import generate_sync

        prompt = f"""What is the primary emotional state expressed in this message?

Message: "{message[:500]}"

Choose ONE from: neutral, happy, sad, stressed, anxious, frustrated, affectionate, playful, tired, excited, relieved, angry, worried, overwhelmed, hopeful, vulnerable

Consider:
- The overall tone and sentiment
- Any explicit emotion words
- Implicit emotional content

Reply with ONLY ONE WORD from the list above."""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )

        if response:
            state = response.strip().lower().strip('.,!?')
            if state in VALID_EMOTIONAL_STATES:
                return state

        return 'neutral'

    except Exception as e:
        logger.warning(f"LLM emotional state detection failed: {e}")
        return 'neutral'


def format_episodes_for_prompt(episodes: List[Episode], max_episodes: int = 3) -> str:
    """Format episodes for inclusion in prompt context."""
    if not episodes:
        return ""

    lines = ["[PAST CONVERSATIONS - Similar topics discussed before]"]

    for ep in episodes[:max_episodes]:
        status = f"[{ep.resolution.upper()}]" if ep.resolution != 'ongoing' else ""
        date_str = ep.started_at.strftime('%b %d') if ep.started_at else ''

        lines.append(f"\n**{ep.topic}** {status} ({date_str}, {ep.message_count} messages)")

        if ep.approach_summary:
            lines.append(f"What worked: {ep.approach_summary}")

        if ep.satisfaction and ep.satisfaction >= 0.7:
            lines.append("(This conversation went well)")

    return "\n".join(lines)


# Singleton
_episode_store: Optional[EpisodeStore] = None


def get_episode_store() -> EpisodeStore:
    """Get singleton episode store."""
    global _episode_store
    if _episode_store is None:
        _episode_store = EpisodeStore()
    return _episode_store
