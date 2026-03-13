"""
Episode Learning Task - Extract lessons from completed conversation episodes.

WHAT: Analyzes closed episodes to extract satisfaction scores, successful
      approaches, and pitfalls to avoid. Builds episode_patterns that group
      learnings by topic/emotion combos (e.g., "family_children + stressed").

WHEN: Runs daily at 4:00 AM Pacific via Celery beat. Can also be triggered
      immediately when an episode closes (analyze_episode_on_close).

WHY:  The companion needs to learn from past interactions. If a playful tone
      worked when James was stressed about work, that pattern should surface
      next time a similar situation arises. Without this, every conversation
      starts from scratch.

Based on the MarkTechPost episodic memory pattern:
- Outcome tracking (success/failure)
- Lesson extraction (what worked, what didn't)
- Pattern consolidation (similar episodes -> shared learnings)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import uuid

import psycopg2
from psycopg2.extras import RealDictCursor

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# --- Configuration ---
MIN_MESSAGES_FOR_ANALYSIS = 4  # Need enough back-and-forth to draw conclusions
MAX_EPISODES_PER_RUN = 10     # Cap per batch to avoid long-running LLM calls
ANALYSIS_LOOKBACK_DAYS = 7    # Ignore older episodes (likely stale)


@dataclass
class EpisodeLesson:
    """Extracted lesson from an episode."""
    episode_id: str
    topic: str
    emotional_context: str
    satisfaction: float  # 0.0 to 1.0
    successful_approach: str
    pitfalls_to_avoid: str
    summary: str


class EpisodeLearningExtractor:
    """Extracts lessons from completed episodes using LLM analysis."""

    def __init__(self):
        self._conn = None

    def _get_connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def get_unanalyzed_episodes(self, limit: int = MAX_EPISODES_PER_RUN) -> List[Dict]:
        """
        Get closed episodes that haven't been analyzed yet.

        An episode is considered unanalyzed if:
        - Resolution is not 'ongoing'
        - Has enough messages (MIN_MESSAGES_FOR_ANALYSIS)
        - satisfaction is NULL (not yet scored)
        - Created within the lookback window
        """
        conn = self._get_connection()
        cutoff = clock_now() - timedelta(days=ANALYSIS_LOOKBACK_DAYS)

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT e.*,
                           COUNT(em.id) as actual_message_count
                    FROM episodes e
                    LEFT JOIN episode_messages em ON e.episode_id = em.episode_id
                    WHERE e.resolution != 'ongoing'
                      AND e.satisfaction IS NULL
                      AND e.started_at > %s
                    GROUP BY e.id
                    HAVING COUNT(em.id) >= %s
                    ORDER BY e.ended_at DESC NULLS LAST
                    LIMIT %s
                """, (cutoff, MIN_MESSAGES_FOR_ANALYSIS, limit))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get unanalyzed episodes: {e}")
            return []

    def get_episode_messages(self, episode_id: str) -> List[Dict]:
        """Get all messages for an episode in chronological order."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT m.sender_name, m.message_text, m.timestamp
                    FROM episode_messages em
                    JOIN messages m ON em.message_id = m.id
                    WHERE em.episode_id = %s::uuid
                    ORDER BY em.turn_number ASC
                """, (episode_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to get episode messages: {e}")
            return []

    def analyze_episode(self, episode: Dict, messages: List[Dict]) -> Optional[EpisodeLesson]:
        """
        Use LLM to analyze an episode and extract lessons.

        Returns EpisodeLesson with:
        - satisfaction: 0.0-1.0 score based on conversation quality
        - successful_approach: What worked well
        - pitfalls_to_avoid: What should be avoided
        """
        if not messages or len(messages) < MIN_MESSAGES_FOR_ANALYSIS:
            return None

        try:
            from src.llm.provider_factory import generate_sync

            # Format conversation for analysis
            conversation = self._format_conversation(messages)
            topic = episode.get('topic', 'general conversation')
            emotional_state = episode.get('emotional_state', 'neutral')

            prompt = f"""Analyze this conversation between James (user) and the companion.

CONVERSATION TOPIC: {topic}
EMOTIONAL CONTEXT: {emotional_state}

CONVERSATION:
{conversation}

Analyze the quality of this interaction and extract learnings.

Respond in this EXACT format:
SATISFACTION: [0.0-1.0 score where 1.0 = excellent connection, 0.0 = conversation went poorly]
APPROACH_SUMMARY: [1-2 sentences describing what approach the companion used - e.g., "listened supportively without offering unsolicited advice", "matched his playful energy with teasing"]
WHAT_WORKED: [1-2 sentences about what made the conversation go well, or "Nothing notable" if it didn't]
PITFALLS: [1-2 sentences about what to avoid in similar situations, or "None identified" if the conversation went well]
BRIEF_SUMMARY: [1 sentence summary of what the conversation was about]

Consider:
- Did James seem satisfied/engaged? (short responses = disengaged, emojis/affection = engaged)
- Did the companion match his emotional needs? (support when stressed, playfulness when playful)
- Were there any awkward moments or mismatches?
- What made the connection strong or weak?"""

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )

            if not response:
                logger.warning(f"No response from LLM for episode {episode['episode_id']}")
                return None

            # Parse response
            return self._parse_analysis(episode, response)

        except Exception as e:
            logger.error(f"Episode analysis failed: {e}")
            return None

    def _format_conversation(self, messages: List[Dict], max_messages: int = 20) -> str:
        """Format messages for LLM analysis."""
        # Take last N messages if too long
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        lines = []
        for msg in messages:
            sender = msg.get('sender_name', 'Unknown')
            text = msg.get('message_text', '')[:500]  # Truncate long messages
            lines.append(f"{sender}: {text}")

        return "\n".join(lines)

    def _parse_analysis(self, episode: Dict, response: str) -> Optional[EpisodeLesson]:
        """Parse structured LLM response (SATISFACTION/WHAT_WORKED/etc.) into EpisodeLesson."""
        try:
            lines = response.strip().split('\n')

            # Defaults if a field is missing from the response
            satisfaction = 0.5
            approach = ""
            what_worked = ""
            pitfalls = ""
            summary = ""

            for line in lines:
                line = line.strip()
                if line.startswith('SATISFACTION:'):
                    try:
                        score_str = line.replace('SATISFACTION:', '').strip()
                        # LLMs sometimes return "0.8/1.0" or "80%" instead of "0.8"
                        if '/' in score_str:
                            score_str = score_str.split('/')[0]
                        if '%' in score_str:
                            score_str = str(float(score_str.replace('%', '')) / 100)
                        satisfaction = max(0.0, min(1.0, float(score_str)))
                    except (ValueError, TypeError):
                        satisfaction = 0.5
                elif line.startswith('APPROACH_SUMMARY:'):
                    approach = line.replace('APPROACH_SUMMARY:', '').strip()
                elif line.startswith('WHAT_WORKED:'):
                    what_worked = line.replace('WHAT_WORKED:', '').strip()
                elif line.startswith('PITFALLS:'):
                    pitfalls = line.replace('PITFALLS:', '').strip()
                elif line.startswith('BRIEF_SUMMARY:'):
                    summary = line.replace('BRIEF_SUMMARY:', '').strip()

            return EpisodeLesson(
                episode_id=str(episode['episode_id']),
                topic=episode.get('topic', 'general'),
                emotional_context=episode.get('emotional_state', 'neutral'),
                satisfaction=satisfaction,
                successful_approach=what_worked if what_worked else approach,
                pitfalls_to_avoid=pitfalls,
                summary=summary
            )

        except Exception as e:
            logger.error(f"Failed to parse analysis: {e}")
            return None

    def save_episode_lesson(self, lesson: EpisodeLesson) -> bool:
        """
        Save lesson to episode and update/create pattern.

        1. Updates episode with satisfaction and approach_summary
        2. Creates or updates episode_pattern for this topic/emotion combo
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # 1. Update the episode itself
                cursor.execute("""
                    UPDATE episodes
                    SET satisfaction = %s,
                        approach_summary = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE episode_id = %s::uuid
                """, (lesson.satisfaction, lesson.successful_approach, lesson.episode_id))

                # 2. Find or create pattern for this topic/emotion combination
                pattern = self._get_or_create_pattern(cursor, lesson)

                if pattern:
                    # 3. Update pattern with new learnings
                    self._update_pattern(cursor, pattern, lesson)

                conn.commit()
                logger.info(f"Saved lesson for episode {lesson.episode_id}: satisfaction={lesson.satisfaction:.2f}")
                return True

        except Exception as e:
            logger.error(f"Failed to save episode lesson: {e}")
            conn.rollback()
            return False

    def _get_or_create_pattern(self, cursor, lesson: EpisodeLesson) -> Optional[Dict]:
        """Get existing pattern or create new one for topic/emotion combo."""
        # Normalize topic and emotion for matching
        topic_category = self._normalize_topic(lesson.topic)
        emotional_context = lesson.emotional_context.lower() if lesson.emotional_context else 'neutral'

        # Try to find existing pattern
        cursor.execute("""
            SELECT * FROM episode_patterns
            WHERE topic_category = %s AND emotional_context = %s
        """, (topic_category, emotional_context))

        existing = cursor.fetchone()

        if existing:
            return dict(existing)

        # Create new pattern
        pattern_id = str(uuid.uuid4())
        pattern_name = f"{topic_category} ({emotional_context})"

        cursor.execute("""
            INSERT INTO episode_patterns
            (pattern_id, pattern_name, topic_category, emotional_context, episode_count, avg_satisfaction)
            VALUES (%s, %s, %s, %s, 0, 0.0)
            RETURNING *
        """, (pattern_id, pattern_name, topic_category, emotional_context))

        new_pattern = cursor.fetchone()
        return dict(new_pattern) if new_pattern else None

    def _update_pattern(self, cursor, pattern: Dict, lesson: EpisodeLesson):
        """Update pattern with new episode's learnings using running averages."""
        current_count = pattern.get('episode_count', 0)
        current_avg = pattern.get('avg_satisfaction', 0.0) or 0.0

        # Running average: ((old_avg * old_count) + new_value) / new_count
        new_count = current_count + 1
        new_avg = ((current_avg * current_count) + lesson.satisfaction) / new_count

        current_approach = pattern.get('successful_approach', '') or ''
        current_pitfalls = pattern.get('pitfalls_to_avoid', '') or ''

        new_approach = current_approach
        new_pitfalls = current_pitfalls

        # Only record approaches from good conversations (>= 0.6 satisfaction)
        # and skip duplicates via case-insensitive substring check
        if lesson.satisfaction >= 0.6 and lesson.successful_approach:
            if lesson.successful_approach.lower() not in current_approach.lower():
                new_approach = f"{new_approach}\n- {lesson.successful_approach}" if new_approach else f"- {lesson.successful_approach}"

        # Only record pitfalls from poor conversations (<= 0.5 satisfaction)
        # and skip the LLM's "None identified" placeholder
        if lesson.satisfaction <= 0.5 and lesson.pitfalls_to_avoid:
            is_duplicate = lesson.pitfalls_to_avoid.lower() in current_pitfalls.lower()
            is_placeholder = 'none' in lesson.pitfalls_to_avoid.lower()
            if not is_duplicate and not is_placeholder:
                new_pitfalls = f"{new_pitfalls}\n- {lesson.pitfalls_to_avoid}" if new_pitfalls else f"- {lesson.pitfalls_to_avoid}"

        # Ensure pitfalls_to_avoid column exists (migration safety net)
        try:
            cursor.execute("""
                ALTER TABLE episode_patterns
                ADD COLUMN IF NOT EXISTS pitfalls_to_avoid TEXT
            """)
        except Exception:
            pass  # Column already exists

        cursor.execute("""
            UPDATE episode_patterns
            SET episode_count = %s,
                avg_satisfaction = %s,
                successful_approach = %s,
                pitfalls_to_avoid = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE pattern_id = %s::uuid
        """, (new_count, new_avg, new_approach, new_pitfalls, pattern['pattern_id']))

    def _normalize_topic(self, topic: str) -> str:
        """
        Normalize free-text topic into a fixed category for pattern matching.

        This groups related topics so patterns accumulate faster --
        e.g., both "Jesse's school trouble" and "Kyler's birthday" map to
        'family_children' and share the same learned approaches.
        """
        if not topic:
            return 'general'

        topic_lower = topic.lower()

        # Map to categories via keyword matching
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


def analyze_episode_on_close(episode_id: str) -> Optional[EpisodeLesson]:
    """
    Analyze a single episode immediately when it closes.

    This can be called synchronously when detecting episode boundaries,
    or asynchronously as part of the daily batch.

    Returns:
        EpisodeLesson if analysis succeeded, None otherwise
    """
    extractor = EpisodeLearningExtractor()

    try:
        conn = extractor._get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM episodes WHERE episode_id = %s::uuid", (episode_id,))
            episode = cursor.fetchone()

        if not episode:
            logger.warning(f"Episode {episode_id} not found")
            return None

        messages = extractor.get_episode_messages(episode_id)

        if len(messages) < MIN_MESSAGES_FOR_ANALYSIS:
            logger.debug(f"Episode {episode_id} has too few messages ({len(messages)})")
            return None

        lesson = extractor.analyze_episode(dict(episode), messages)

        if lesson:
            extractor.save_episode_lesson(lesson)
            logger.info(f"Analyzed episode {episode_id} on close: satisfaction={lesson.satisfaction:.2f}")
            return lesson

        return None

    except Exception as e:
        logger.error(f"Failed to analyze episode on close: {e}")
        return None


def run_episode_learning():
    """
    Main entry point for the episode learning task.

    Called by Celery beat scheduler daily.
    """
    logger.info("Starting episode learning task...")

    extractor = EpisodeLearningExtractor()

    # Get unanalyzed episodes
    episodes = extractor.get_unanalyzed_episodes()

    if not episodes:
        logger.info("No unanalyzed episodes found")
        return {"analyzed": 0, "message": "No episodes to analyze"}

    logger.info(f"Found {len(episodes)} episodes to analyze")

    analyzed = 0
    failed = 0

    for episode in episodes:
        episode_id = str(episode['episode_id'])

        # Get messages for this episode
        messages = extractor.get_episode_messages(episode_id)

        if len(messages) < MIN_MESSAGES_FOR_ANALYSIS:
            logger.debug(f"Skipping episode {episode_id}: only {len(messages)} messages")
            continue

        # Analyze episode
        lesson = extractor.analyze_episode(episode, messages)

        if lesson:
            # Save lesson
            if extractor.save_episode_lesson(lesson):
                analyzed += 1
                logger.info(f"Analyzed episode {episode_id}: {lesson.topic} (satisfaction={lesson.satisfaction:.2f})")
            else:
                failed += 1
        else:
            failed += 1
            logger.warning(f"Failed to analyze episode {episode_id}")

    result = {
        "analyzed": analyzed,
        "failed": failed,
        "total": len(episodes)
    }

    logger.info(f"Episode learning complete: {result}")
    return result


# Celery task wrapper
try:
    from src.celery_app import celery_app

    @celery_app.task(name='tasks.learn_from_episodes', bind=True, max_retries=2)
    def learn_from_episodes_task(self):
        """
        Celery task to extract lessons from completed episodes.

        Scheduled to run daily at 4:00 AM Pacific.
        """
        try:
            result = run_episode_learning()
            return result
        except Exception as e:
            logger.error(f"Episode learning task failed: {e}")
            raise self.retry(exc=e, countdown=300)  # Retry after 5 minutes

except ImportError:
    # Celery not available (running standalone)
    pass


# For manual testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    logging.basicConfig(level=logging.INFO)
    result = run_episode_learning()
    print(f"Result: {result}")
