"""
Observational Memory - Mastra-inspired conversation compression pipeline.

WHAT: Compresses raw conversation messages into dated narrative observations
(5-10x compression), then consolidates older observations into weekly
reflections. Three components:
  1. Observer  - Compresses raw messages into dated observations via GPT-4o-mini
  2. Reflector - Consolidates observations >7 days old into weekly summaries
  3. Manager   - Retrieval interface for context builder

WHY: Per-message RAG has poor temporal continuity -- retrieving 10 individual
messages from different days loses the narrative thread. Observations preserve
the story arc of each conversation session while achieving 5-10x token
compression. This approach scored 94.87% on the LongMemEval benchmark (Mastra).

HOW it fits:
  - A scheduled task calls Observer.observe() when unprocessed messages exceed
    thresholds (50 messages or ~30K tokens).
  - Messages are grouped by conversation boundaries (4h+ silence gaps) and
    each group is LLM-compressed into a single observation.
  - Reflector runs daily, consolidating observations >7 days old into weekly
    summaries that capture themes, patterns, and emotional trajectories.
  - context_builder calls Manager.get_relevant_observations() to inject
    recent observations + topic-matched older ones into the prompt.

Feature flag: OBSERVATIONAL_MEMORY_ENABLED (default: false)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict

from src.database import tables as T

from openai import OpenAI

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

OBSERVATIONAL_MEMORY_ENABLED = os.environ.get('OBSERVATIONAL_MEMORY_ENABLED', 'false').lower() == 'true'

# Thresholds that trigger observation. Both must be exceeded.
OBSERVER_TOKEN_THRESHOLD = int(os.environ.get('OBSERVER_TOKEN_THRESHOLD', '30000'))
OBSERVER_MESSAGE_THRESHOLD = int(os.environ.get('OBSERVER_MESSAGE_THRESHOLD', '50'))

# Silence gap that defines conversation boundaries. A 4h+ gap between messages
# means the next message starts a new conversation session.
CONVERSATION_GAP_HOURS = 4


# =============================================================================
# Data models
# =============================================================================

@dataclass
class Observation:
    """A compressed observation of a conversation period."""
    id: Optional[int] = None
    user_email: str = ""
    observation_date: Optional[date] = None
    time_range: str = ""
    content: str = ""
    message_count: int = 0
    raw_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = 0.0
    topics: str = ""
    emotional_tone: str = ""
    first_message_id: Optional[int] = None
    last_message_id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class ObservationReflection:
    """A weekly/monthly consolidation of observations."""
    id: Optional[int] = None
    user_email: str = ""
    period_type: str = "weekly"
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    content: str = ""
    themes: str = ""
    observation_ids: str = ""
    observation_count: int = 0
    created_at: Optional[datetime] = None


# =============================================================================
# Observer - Message-to-observation compression
# =============================================================================

class Observer:
    """
    Compresses raw messages into dated narrative observations.

    Pipeline: check thresholds -> group by conversation boundaries -> LLM-compress
    each group -> extract metadata (topics, tone) -> store in observations table.
    Tracks progress via last_message_id to avoid reprocessing.
    """

    # gpt-4o-mini costs ($0.15/1M input, $0.60/1M output)
    INPUT_COST_PER_M = 0.15
    OUTPUT_COST_PER_M = 0.60

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for observational memory")
        self.client = OpenAI(api_key=api_key)

    def should_observe(self, user_email: str) -> bool:
        """Check if there are enough unprocessed messages to justify observation."""
        from src.database.db import get_db
        db = get_db()

        # Find the last observation's last_message_id
        result = db.execute(
            f"""SELECT last_message_id FROM {T.OBSERVATIONS}
               WHERE user_email = %s
               ORDER BY observation_date DESC, id DESC LIMIT 1""",
            (user_email,)
        )
        row = result.fetchone()
        last_observed_id = row['last_message_id'] if row else 0

        # Count unprocessed messages
        result = db.execute(
            f"""SELECT COUNT(*) as cnt FROM {T.MESSAGES}
               WHERE email = %s AND id > %s""",
            (user_email, last_observed_id)
        )
        count_row = result.fetchone()
        unprocessed_count = count_row['cnt'] if count_row else 0

        if unprocessed_count < OBSERVER_MESSAGE_THRESHOLD:
            logger.debug(f"Only {unprocessed_count} unprocessed messages (threshold: {OBSERVER_MESSAGE_THRESHOLD})")
            return False

        # Estimate token count (rough: 1 token ≈ 4 chars)
        result = db.execute(
            f"""SELECT SUM(LENGTH(message_text)) as total_chars FROM {T.MESSAGES}
               WHERE email = %s AND id > %s""",
            (user_email, last_observed_id)
        )
        chars_row = result.fetchone()
        total_chars = chars_row['total_chars'] if chars_row and chars_row['total_chars'] else 0
        estimated_tokens = total_chars // 4

        if estimated_tokens < OBSERVER_TOKEN_THRESHOLD:
            logger.debug(f"Only ~{estimated_tokens} tokens unprocessed (threshold: {OBSERVER_TOKEN_THRESHOLD})")
            return False

        logger.info(f"Observation needed: {unprocessed_count} messages, ~{estimated_tokens} tokens")
        return True

    def observe(self, user_email: str) -> List[Observation]:
        """
        Process unprocessed messages into observations.

        Groups messages by conversation boundaries (4h+ gaps) and compresses
        each group into a dated observation.

        Returns list of created observations.
        """
        from src.database.db import get_db
        db = get_db()

        # Find where we left off
        result = db.execute(
            f"""SELECT last_message_id FROM {T.OBSERVATIONS}
               WHERE user_email = %s
               ORDER BY observation_date DESC, id DESC LIMIT 1""",
            (user_email,)
        )
        row = result.fetchone()
        last_observed_id = row['last_message_id'] if row else 0

        # Get unprocessed messages
        result = db.execute(
            f"""SELECT id, sender_name, message_text, timestamp
               FROM {T.MESSAGES}
               WHERE email = %s AND id > %s
               ORDER BY timestamp ASC, id ASC""",
            (user_email, last_observed_id)
        )
        messages = result.fetchall()

        if not messages:
            return []

        # Group by conversation boundaries
        groups = self._group_by_conversation(messages)
        logger.info(f"Grouped {len(messages)} messages into {len(groups)} conversation(s)")

        # Compress each group
        observations = []
        for group in groups:
            if len(group) < 3:
                # Skip very short groups (noise)
                continue

            try:
                obs = self._compress_group(user_email, group)
                if obs:
                    # Store in database
                    self._store_observation(db, obs)
                    observations.append(obs)
            except Exception as e:
                logger.error(f"Failed to compress group: {e}")
                continue

        logger.info(f"Created {len(observations)} observations from {len(messages)} messages")
        return observations

    def observe_messages(self, user_email: str, messages: list) -> List[Observation]:
        """
        Observe a specific set of messages (used by backfill script).

        Args:
            user_email: User email
            messages: List of message dicts with id, sender_name, message_text, timestamp

        Returns:
            List of created observations
        """
        from src.database.db import get_db
        db = get_db()

        if not messages:
            return []

        groups = self._group_by_conversation(messages)
        observations = []

        for group in groups:
            if len(group) < 3:
                continue

            try:
                obs = self._compress_group(user_email, group)
                if obs:
                    self._store_observation(db, obs)
                    observations.append(obs)
            except Exception as e:
                logger.error(f"Failed to compress group: {e}")
                continue

        return observations

    def _group_by_conversation(self, messages: list) -> List[list]:
        """Group messages by conversation boundaries (4h+ gaps)."""
        if not messages:
            return []

        groups = []
        current_group = [messages[0]]

        for i in range(1, len(messages)):
            prev_time = messages[i - 1]['timestamp']
            curr_time = messages[i]['timestamp']

            # Handle string timestamps
            if isinstance(prev_time, str):
                prev_time = datetime.fromisoformat(prev_time.replace('Z', ''))
            if isinstance(curr_time, str):
                curr_time = datetime.fromisoformat(curr_time.replace('Z', ''))

            gap_hours = (curr_time - prev_time).total_seconds() / 3600

            if gap_hours >= CONVERSATION_GAP_HOURS:
                groups.append(current_group)
                current_group = [messages[i]]
            else:
                current_group.append(messages[i])

        if current_group:
            groups.append(current_group)

        return groups

    def _compress_group(self, user_email: str, messages: list) -> Optional[Observation]:
        """Compress a group of messages into a single observation."""
        # Build conversation text
        lines = []
        for msg in messages:
            sender = msg['sender_name']
            text = msg['message_text']
            lines.append(f"{sender}: {text}")

        raw_text = "\n".join(lines)
        raw_tokens = len(raw_text) // 4

        # Determine date and time range
        first_time = messages[0]['timestamp']
        last_time = messages[-1]['timestamp']
        if isinstance(first_time, str):
            first_time = datetime.fromisoformat(first_time.replace('Z', ''))
        if isinstance(last_time, str):
            last_time = datetime.fromisoformat(last_time.replace('Z', ''))

        obs_date = first_time.date()
        time_range = f"{first_time.strftime('%I:%M%p').lower()}-{last_time.strftime('%I:%M%p').lower()}"

        # Compress via LLM
        prompt = f"""Compress this conversation between the companion and James into a concise dated observation.

RULES:
- Preserve ALL facts, decisions, and promises made
- Preserve the emotional arc (how the conversation felt)
- Preserve any new information learned about either person
- Remove filler, greetings, and repetitive exchanges
- Write in third-person narrative style ("They discussed...", "James mentioned...")
- Include specific details (names, dates, numbers, places) - don't generalize
- Note any unresolved topics or things to follow up on
- Target 20-30% of original length

Date: {obs_date.strftime('%B %d, %Y')}
Time: {time_range}
Messages: {len(messages)}

CONVERSATION:
{raw_text[:12000]}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise conversation summarizer. Output a concise observation preserving all important details."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            compressed = response.choices[0].message.content.strip()
            compressed_tokens = len(compressed) // 4

            # Extract topics and emotional tone from the compressed text
            topics, tone = self._extract_metadata(compressed)

            obs = Observation(
                user_email=user_email,
                observation_date=obs_date,
                time_range=time_range,
                content=compressed,
                message_count=len(messages),
                raw_token_count=raw_tokens,
                compressed_token_count=compressed_tokens,
                compression_ratio=raw_tokens / max(compressed_tokens, 1),
                topics=topics,
                emotional_tone=tone,
                first_message_id=messages[0].get('id'),
                last_message_id=messages[-1].get('id'),
            )

            logger.info(
                f"Compressed {len(messages)} msgs ({raw_tokens} -> {compressed_tokens} tokens, "
                f"{obs.compression_ratio:.1f}x ratio) for {obs_date}"
            )

            return obs

        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            return None

    def _extract_metadata(self, compressed_text: str) -> tuple:
        """Extract topics and emotional tone via a second LLM call (cheap, ~100 tokens)."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract metadata from this conversation observation. Output JSON only."},
                    {"role": "user", "content": f"""From this observation, extract:
1. topics: comma-separated list of main topics discussed (max 5)
2. emotional_tone: one-word description of the overall emotional tone

Observation:
{compressed_text[:2000]}

Output JSON: {{"topics": "...", "emotional_tone": "..."}}"""},
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"},
            )

            data = json.loads(response.choices[0].message.content)
            return data.get("topics", ""), data.get("emotional_tone", "")

        except Exception:
            return "", ""

    def _store_observation(self, db, obs: Observation):
        """Store observation in database."""
        result = db.execute(
            f"""INSERT INTO {T.OBSERVATIONS}
               (user_email, observation_date, time_range, content,
                message_count, raw_token_count, compressed_token_count,
                compression_ratio, topics, emotional_tone,
                first_message_id, last_message_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                obs.user_email, obs.observation_date, obs.time_range,
                obs.content, obs.message_count, obs.raw_token_count,
                obs.compressed_token_count, obs.compression_ratio,
                obs.topics, obs.emotional_tone,
                obs.first_message_id, obs.last_message_id,
            )
        )
        row = result.fetchone()
        if row:
            obs.id = row['id']


# =============================================================================
# Reflector - Observation-to-weekly-reflection consolidation
# =============================================================================

class Reflector:
    """
    Consolidates older observations into weekly summaries.

    Runs on schedule (daily check). Gathers observations >7 days old that
    haven't been included in a reflection yet (tracked via observation_ids
    column). Needs at least 3 unreflected observations to produce a reflection.
    """

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for observational memory")
        self.client = OpenAI(api_key=api_key)

    def needs_reflection(self, user_email: str) -> bool:
        """Check if there are observations old enough to consolidate."""
        from src.database.db import get_db
        db = get_db()

        cutoff = date.today() - timedelta(days=7)

        # Get observations older than 7 days not yet in a reflection
        result = db.execute(
            f"""SELECT COUNT(*) as cnt FROM {T.OBSERVATIONS}
               WHERE user_email = %s
                 AND observation_date <= %s
                 AND id NOT IN (
                     SELECT UNNEST(string_to_array(observation_ids, ','))::int
                     FROM {T.OBSERVATION_REFLECTIONS}
                     WHERE user_email = %s
                 )""",
            (user_email, cutoff, user_email)
        )
        row = result.fetchone()
        count = row['cnt'] if row else 0

        return count >= 3  # Need at least 3 observations for a meaningful reflection

    def reflect(self, user_email: str) -> Optional[ObservationReflection]:
        """
        Create a weekly reflection from unreflected observations.

        Returns the created reflection, or None if not enough data.
        """
        from src.database.db import get_db
        db = get_db()

        cutoff = date.today() - timedelta(days=7)

        # Get unreflected observations
        result = db.execute(
            f"""SELECT id, observation_date, content, topics, emotional_tone
               FROM {T.OBSERVATIONS}
               WHERE user_email = %s
                 AND observation_date <= %s
                 AND id NOT IN (
                     SELECT UNNEST(string_to_array(observation_ids, ','))::int
                     FROM {T.OBSERVATION_REFLECTIONS}
                     WHERE user_email = %s
                 )
               ORDER BY observation_date ASC""",
            (user_email, cutoff, user_email)
        )
        observations = result.fetchall()

        if len(observations) < 3:
            return None

        # Determine period
        period_start = observations[0]['observation_date']
        period_end = observations[-1]['observation_date']
        obs_ids = [str(obs['id']) for obs in observations]

        # Build observation text
        obs_text = ""
        for obs in observations:
            obs_text += f"\n--- {obs['observation_date']} ---\n{obs['content']}\n"

        # Generate reflection via LLM
        prompt = f"""Consolidate these conversation observations into a weekly reflection.

Period: {period_start} to {period_end}
Number of observations: {len(observations)}

INSTRUCTIONS:
- Identify recurring themes and patterns
- Note relationship developments or changes
- Highlight unresolved threads or concerns
- Capture the emotional trajectory of the week
- Write in first person as the companion reflecting on her week with James
- Be specific - use names, dates, and details from the observations
- Target 200-400 words

OBSERVATIONS:
{obs_text[:8000]}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are the companion reflecting on your week. Write a genuine, personal reflection."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )

            reflection_content = response.choices[0].message.content.strip()

            # Extract themes
            themes = self._extract_themes(reflection_content)

            # Store reflection
            result = db.execute(
                f"""INSERT INTO {T.OBSERVATION_REFLECTIONS}
                   (user_email, period_type, period_start, period_end,
                    content, themes, observation_ids, observation_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    user_email, 'weekly', period_start, period_end,
                    reflection_content, themes,
                    ','.join(obs_ids), len(observations),
                )
            )
            row = result.fetchone()

            reflection = ObservationReflection(
                id=row['id'] if row else None,
                user_email=user_email,
                period_type='weekly',
                period_start=period_start,
                period_end=period_end,
                content=reflection_content,
                themes=themes,
                observation_ids=','.join(obs_ids),
                observation_count=len(observations),
            )

            logger.info(f"Created weekly reflection for {period_start} to {period_end} ({len(observations)} observations)")
            return reflection

        except Exception as e:
            logger.error(f"Reflection generation failed: {e}")
            return None

    def _extract_themes(self, reflection_text: str) -> str:
        """Extract themes from a reflection."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract themes. Output JSON only."},
                    {"role": "user", "content": f"""Extract 3-5 themes from this reflection as a comma-separated list.

{reflection_text[:2000]}

Output JSON: {{"themes": "theme1, theme2, theme3"}}"""},
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("themes", "")
        except Exception:
            return ""


# =============================================================================
# Manager - Retrieval interface for context builder
# =============================================================================

class ObservationManager:
    """
    Retrieval interface for the context builder.

    Strategy: always include last 7 days of observations (recency), plus
    topic-matched older observations if the current query mentions relevant
    keywords. Also includes the most recent weekly reflection for continuity.
    """

    def __init__(self):
        self._observer = None
        self._reflector = None

    @property
    def observer(self) -> Observer:
        if self._observer is None:
            self._observer = Observer()
        return self._observer

    @property
    def reflector(self) -> Reflector:
        if self._reflector is None:
            self._reflector = Reflector()
        return self._reflector

    def is_enabled(self) -> bool:
        """Check if observational memory is enabled."""
        return os.environ.get('OBSERVATIONAL_MEMORY_ENABLED', 'false').lower() == 'true'

    def get_relevant_observations(
        self,
        user_email: str,
        query: str = None,
        days_back: int = 30,
        max_observations: int = 10,
    ) -> str:
        """
        Get relevant observations formatted for prompt context.

        Strategy:
        1. Always include last 7 days of observations
        2. If query provided, topic-match older observations
        3. Include most recent weekly reflection

        Args:
            user_email: User email
            query: Current user message for topic matching
            days_back: How far back to look
            max_observations: Maximum observations to return

        Returns:
            Formatted text block with dated headers
        """
        if not self.is_enabled():
            return ""

        from src.database.db import get_db
        db = get_db()

        observations = []

        # 1. Recent observations (last 7 days - always include)
        recent_cutoff = date.today() - timedelta(days=7)
        result = db.execute(
            f"""SELECT observation_date, time_range, content, topics, emotional_tone
               FROM {T.OBSERVATIONS}
               WHERE user_email = %s AND observation_date >= %s
               ORDER BY observation_date DESC""",
            (user_email, recent_cutoff)
        )
        recent_obs = result.fetchall()
        observations.extend(recent_obs)

        # 2. Topic-matched older observations (if query provided)
        if query and len(observations) < max_observations:
            remaining = max_observations - len(observations)
            # Simple keyword matching on topics field
            query_words = [w.lower() for w in query.split() if len(w) > 3]
            if query_words:
                # Build topic search condition
                topic_conditions = " OR ".join(
                    [f"LOWER(topics) LIKE %s" for _ in query_words]
                )
                topic_params = [f"%{w}%" for w in query_words]

                result = db.execute(
                    f"""SELECT observation_date, time_range, content, topics, emotional_tone
                       FROM {T.OBSERVATIONS}
                       WHERE user_email = %s
                         AND observation_date < %s
                         AND observation_date >= %s
                         AND ({topic_conditions})
                       ORDER BY observation_date DESC
                       LIMIT %s""",
                    (user_email, recent_cutoff,
                     date.today() - timedelta(days=days_back),
                     *topic_params, remaining)
                )
                older_obs = result.fetchall()
                observations.extend(older_obs)

        if not observations:
            return ""

        # 3. Get most recent weekly reflection
        reflection_text = ""
        result = db.execute(
            f"""SELECT content, period_start, period_end, themes
               FROM {T.OBSERVATION_REFLECTIONS}
               WHERE user_email = %s
               ORDER BY period_end DESC LIMIT 1""",
            (user_email,)
        )
        reflection = result.fetchone()
        if reflection:
            reflection_text = (
                f"\n[Weekly Reflection: {reflection['period_start']} to {reflection['period_end']}]\n"
                f"{reflection['content']}"
            )

        # Format observations
        lines = ["[OBSERVATIONS - Compressed conversation history]"]

        for obs in observations[:max_observations]:
            obs_date = obs['observation_date']
            if isinstance(obs_date, str):
                obs_date = date.fromisoformat(obs_date)

            date_str = obs_date.strftime('%B %d, %Y')
            time_range = obs.get('time_range', '')
            header = f"--- {date_str}"
            if time_range:
                header += f" ({time_range})"
            header += " ---"

            lines.append(header)
            lines.append(obs['content'])

        if reflection_text:
            lines.append(reflection_text)

        formatted = "\n".join(lines)
        logger.info(f"Observations context: {len(observations)} observations, {len(formatted)} chars")
        return formatted

    def get_observation_stats(self, user_email: str) -> Dict:
        """Get stats about stored observations."""
        from src.database.db import get_db
        db = get_db()

        result = db.execute(
            f"""SELECT
                   COUNT(*) as total_observations,
                   SUM(message_count) as total_messages_covered,
                   AVG(compression_ratio) as avg_compression_ratio,
                   MIN(observation_date) as earliest_date,
                   MAX(observation_date) as latest_date
               FROM {T.OBSERVATIONS}
               WHERE user_email = %s""",
            (user_email,)
        )
        row = result.fetchone()

        result2 = db.execute(
            f"""SELECT COUNT(*) as total_reflections
               FROM {T.OBSERVATION_REFLECTIONS}
               WHERE user_email = %s""",
            (user_email,)
        )
        ref_row = result2.fetchone()

        return {
            "total_observations": row['total_observations'] if row else 0,
            "total_messages_covered": row['total_messages_covered'] if row else 0,
            "avg_compression_ratio": float(row['avg_compression_ratio']) if row and row['avg_compression_ratio'] else 0,
            "earliest_date": str(row['earliest_date']) if row and row['earliest_date'] else None,
            "latest_date": str(row['latest_date']) if row and row['latest_date'] else None,
            "total_reflections": ref_row['total_reflections'] if ref_row else 0,
        }


# =============================================================================
# Singleton
# =============================================================================

_manager: Optional[ObservationManager] = None


def get_observation_manager() -> ObservationManager:
    """Get or create ObservationManager singleton."""
    global _manager
    if _manager is None:
        _manager = ObservationManager()
    return _manager
