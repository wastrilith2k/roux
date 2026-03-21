"""
PostgreSQL Database Manager — Companion Framework

WHAT: Unified data access layer for all PostgreSQL operations — user profiles,
      relationship state, conversation history, episodes, benchmarks, and observations.
WHY:  Every subsystem (pipeline, handlers, Celery tasks, scripts) needs database
      access. This module centralizes connection management, schema initialization,
      and query methods so callers never touch raw SQL or connection pooling directly.
HOW:  CompanionDB creates per-request connections via a context manager (_get_connection).
      The _CursorWrapper eagerly fetches results before the connection closes, avoiding
      the common psycopg2 pitfall of reading from a closed cursor. Schema is created
      via _init_db() on first instantiation (idempotent CREATE IF NOT EXISTS).
      A module-level singleton (get_db) ensures one instance per process.

Environment safety: constructor validates that ENVIRONMENT and POSTGRES_DB match
(e.g., dev env must use a _dev database) to prevent accidental cross-environment writes.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import arrow
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from src.utils.timezone_utils import now_pacific_naive
from src.config.persona_config import get_persona_config
from src.database import tables as T


# ---------------------------------------------------------------------------
# Cursor wrapper — solves the "closed connection" problem
# ---------------------------------------------------------------------------

class _CursorWrapper:
    """Eagerly-fetching cursor wrapper for use outside connection context managers.

    Problem: CompanionDB.execute() uses a context manager that closes the connection
    on exit. If the caller tries cursor.fetchone() after execute() returns, the
    underlying connection is already closed and psycopg2 raises an error.

    Solution: This wrapper calls fetchall() immediately while the connection is
    still open, then serves rows from the cached list. The caller gets the same
    fetchone()/fetchall() interface without needing to worry about connection
    lifecycle.
    """

    def __init__(self, cursor):
        """Initialize with a psycopg2 cursor and fetch all results immediately."""
        try:
            self._data = cursor.fetchall()
            self._index = 0
        except Exception:
            # If fetchall fails (e.g., SELECT without result set), use empty list
            self._data = []
            self._index = 0

    def fetchone(self):
        """Return the next row from cached data, or None if no more rows."""
        if self._index < len(self._data):
            result = self._data[self._index]
            self._index += 1
            return result
        return None

    def fetchall(self):
        """Return all remaining cached rows."""
        result = self._data[self._index:]
        self._index = len(self._data)
        return result


# ---------------------------------------------------------------------------
# Main database manager
# ---------------------------------------------------------------------------

class CompanionDB:
    """Unified PostgreSQL database manager for the Companion Framework.

    Provides typed methods for user profiles, relationship state, conversation
    history, episodes, observations, and raw SQL execution. All methods manage
    their own connections via _get_connection() context manager.
    """

    def __init__(self):
        # Read PostgreSQL connection settings from environment
        self.pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.pg_port = os.getenv('POSTGRES_PORT', '5432')
        self.pg_db = os.getenv('POSTGRES_DB', 'companion')
        self.pg_user = os.getenv('POSTGRES_USER', 'companion')
        self.pg_pass = os.getenv('POSTGRES_PASSWORD', '')
        if not self.pg_pass:
            raise ValueError(
                "POSTGRES_PASSWORD environment variable is required. "
                "Set it in .env or your environment."
            )

        # Environment validation
        environment = os.getenv('ENVIRONMENT', 'production')
        db_suffix = self.pg_db.rsplit('_', 1)[-1] if '_' in self.pg_db else ''

        if environment == 'development' and not self.pg_db.endswith('_dev'):
            raise ValueError(
                f"ENVIRONMENT MISMATCH: ENVIRONMENT='{environment}' but "
                f"POSTGRES_DB='{self.pg_db}' does not end with '_dev'. "
                f"Development environment should use a '_dev' suffixed database."
            )

        if environment == 'production' and self.pg_db.endswith('_dev'):
            raise ValueError(
                f"ENVIRONMENT MISMATCH: ENVIRONMENT='{environment}' but "
                f"POSTGRES_DB='{self.pg_db}' looks like a dev database. "
                f"Production environment should not use a '_dev' database."
            )

        # Log which environment we're connecting to
        env_marker = "[DEV]" if environment == 'development' else "[PROD]"
        print(f"{env_marker} CompanionDB connecting to: {self.pg_db} @ {self.pg_host}:{self.pg_port}")

    @contextmanager
    def _get_connection(self, user_email: str = None):
        """Context manager for database connections.

        Args:
            user_email: If provided, sets the PostgreSQL search_path to the
                        user's schema so all queries resolve to that schema first.
        """
        conn = psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_db,
            user=self.pg_user,
            password=self.pg_pass
        )
        try:
            if user_email:
                from src.database.schema_manager import set_search_path
                set_search_path(conn, user_email)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = None, user_email: str = None):
        """
        Execute a raw SQL query and return a wrapper that holds the results.
        The connection is maintained within the context manager while data is fetched.
        Returns a cursor-like object that provides fetchone() and fetchall() methods.

        Args:
            query: SQL query string with %s placeholders.
            params: Optional tuple of query parameters.
            user_email: If provided, sets the search_path to the user's schema
                        before executing the query.
        """
        with self._get_connection(user_email=user_email) as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params or ())
            # CRITICAL: Fetch data while connection is still open
            # The connection closes when exiting this context, so cursor becomes invalid
            # We store the fetched data and return a wrapper object
            return _CursorWrapper(cursor)

    def commit(self):
        """No-op for compatibility with raw SQL code patterns"""
        pass

    # ==================== USER PROFILE METHODS ====================

    def get_profile(self, email: str) -> Dict:
        """Get user profile by email"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                f'SELECT * FROM {T.USER_PROFILES} WHERE email = %s',
                (email,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    'email': row['email'],
                    'display_name': row['display_name'],
                    'is_admin': bool(row['is_admin']),
                    'created_at': row['created_at'].isoformat() if row['created_at'] else '',
                    'last_seen': row['last_seen'].isoformat() if row['last_seen'] else ''
                }

            # Return default profile if not found
            return {
                'email': email,
                'display_name': email.split('@')[0],
                'is_admin': False,
                'created_at': '',
                'last_seen': ''
            }

    def get_user_id(self, email: str) -> Optional[int]:
        """Get user's integer ID from the users table by email"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT id FROM {T.USERS} WHERE email = %s',
                (email,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            return None

    def set_display_name(self, email: str, display_name: str):
        """Set or update user's display name"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            existing = self.get_profile(email)

            if existing['created_at']:
                # Update existing
                cursor.execute(f'''
                    UPDATE {T.USER_PROFILES}
                    SET display_name = %s, last_seen = %s
                    WHERE email = %s
                ''', (display_name, now_pacific_naive(), email))
            else:
                # Create new
                cursor.execute(f'''
                    INSERT INTO {T.USER_PROFILES} (email, display_name, is_admin, last_seen)
                    VALUES (%s, %s, FALSE, %s)
                ''', (email, display_name, now_pacific_naive()))

    def set_admin(self, email: str, is_admin: bool = True):
        """Set user as admin"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            existing = self.get_profile(email)

            if existing['created_at']:
                # Update existing
                cursor.execute(f'''
                    UPDATE {T.USER_PROFILES}
                    SET is_admin = %s, last_seen = %s
                    WHERE email = %s
                ''', (is_admin, now_pacific_naive(), email))
            else:
                # Create new
                cursor.execute(f'''
                    INSERT INTO {T.USER_PROFILES} (email, display_name, is_admin, last_seen)
                    VALUES (%s, %s, %s, %s)
                ''', (email, email.split('@')[0], is_admin, now_pacific_naive()))

    def is_admin(self, email: str) -> bool:
        """Check if user has admin privileges"""
        profile = self.get_profile(email)
        return profile['is_admin']

    def update_last_seen(self, email: str):
        """Update last seen timestamp"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure profile exists
            profile = self.get_profile(email)
            if not profile['created_at']:
                self.set_display_name(email, email.split('@')[0])

            cursor.execute(f'''
                UPDATE {T.USER_PROFILES}
                SET last_seen = %s
                WHERE email = %s
            ''', (now_pacific_naive(), email))

    # ==================== USER STATE METHODS ====================

    def get_state(self, email: str) -> Dict:
        """Get user state (closeness, romance, cooldowns)"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                f'SELECT * FROM {T.USER_STATE} WHERE email = %s',
                (email,)
            )
            row = cursor.fetchone()

            if row:
                # Convert timestamps
                last_negative = arrow.get(row['last_negative_event']) if row['last_negative_event'] else None
                last_message = arrow.get(row['last_message_time']) if row['last_message_time'] else None

                return {
                    'closeness_score': row['closeness_score'],
                    'romance_level': row['romance_level'],
                    'romance_enabled': bool(row['romance_enabled']),
                    'romance_decision': row['romance_decision'],
                    'emotion_profile': row['emotion_profile'],
                    'last_negative_event': last_negative,
                    'cooldown_active': bool(row['cooldown_active']),
                    'badgering_count': row['badgering_count'],
                    'last_message_time': last_message,
                    'attraction_cue_count': row.get('attraction_cue_count', 0) or 0,
                    'attraction_latent_state': bool(row.get('attraction_latent_state', False)),
                    'sms_preference': row.get('sms_preference', 'good_morning') or 'good_morning'
                }

            # Return default state
            return {
                'closeness_score': 15,
                'romance_level': 0,
                'romance_enabled': False,
                'romance_decision': None,
                'emotion_profile': 'Guarded',
                'last_negative_event': None,
                'cooldown_active': False,
                'badgering_count': 0,
                'last_message_time': None,
                'sms_preference': 'good_morning'
            }

    def save_state(self, email: str, state: Dict):
        """Save user state"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Convert arrow timestamps to datetime objects
            last_negative = state['last_negative_event'].datetime if state.get('last_negative_event') else None
            last_message = state['last_message_time'].datetime if state.get('last_message_time') else None

            # Check if state exists
            cursor.execute(f'SELECT email FROM {T.USER_STATE} WHERE email = %s', (email,))
            exists = cursor.fetchone() is not None

            if exists:
                # Update
                cursor.execute(f'''
                    UPDATE {T.USER_STATE}
                    SET closeness_score = %s,
                        romance_level = %s,
                        romance_enabled = %s,
                        romance_decision = %s,
                        emotion_profile = %s,
                        last_negative_event = %s,
                        cooldown_active = %s,
                        badgering_count = %s,
                        last_message_time = %s,
                        attraction_cue_count = %s,
                        attraction_latent_state = %s,
                        sms_preference = %s
                    WHERE email = %s
                ''', (
                    state['closeness_score'],
                    state['romance_level'],
                    state.get('romance_enabled', False),
                    state.get('romance_decision'),
                    state['emotion_profile'],
                    last_negative,
                    state.get('cooldown_active', False),
                    state.get('badgering_count', 0),
                    last_message,
                    state.get('attraction_cue_count', 0),
                    state.get('attraction_latent_state', False),
                    state.get('sms_preference', 'good_morning'),
                    email
                ))
            else:
                # Insert
                cursor.execute(f'''
                    INSERT INTO {T.USER_STATE} (
                        email, closeness_score, romance_level, romance_enabled,
                        romance_decision, emotion_profile, last_negative_event,
                        cooldown_active, badgering_count, last_message_time,
                        attraction_cue_count, attraction_latent_state, sms_preference
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    email,
                    state['closeness_score'],
                    state['romance_level'],
                    state.get('romance_enabled', False),
                    state.get('romance_decision'),
                    state['emotion_profile'],
                    last_negative,
                    state.get('cooldown_active', False),
                    state.get('badgering_count', 0),
                    last_message,
                    state.get('attraction_cue_count', 0),
                    state.get('attraction_latent_state', False),
                    state.get('sms_preference', 'good_morning')
                ))

    def update_closeness(self, email: str, score: int, profile: str):
        """Update closeness score and emotion profile"""
        state = self.get_state(email)
        state['closeness_score'] = score
        state['emotion_profile'] = profile
        self.save_state(email, state)

    def update_romance(self, email: str, level: float, decision: str = None):
        """Update romance level and decision"""
        state = self.get_state(email)
        state['romance_level'] = level
        if decision is not None:
            state['romance_decision'] = decision
        self.save_state(email, state)

    def set_romance_enabled(self, email: str, enabled: bool):
        """Enable or disable romance system for user"""
        state = self.get_state(email)
        state['romance_enabled'] = enabled
        self.save_state(email, state)

    def set_cooldown(self, email: str, active: bool, badgering_count: int = 0):
        """Set cooldown state"""
        state = self.get_state(email)
        state['cooldown_active'] = active
        state['badgering_count'] = badgering_count
        if active:
            state['last_negative_event'] = arrow.now('America/Los_Angeles')
        self.save_state(email, state)

    def set_sms_preference(self, email: str, preference: str):
        """Set SMS preference (off, good_morning, or full)"""
        state = self.get_state(email)
        state['sms_preference'] = preference
        self.save_state(email, state)

    # ==================== CONVERSATION METHODS ====================

    def store_message(self, email: str, sender_name: str, message_text: str,
                     sentiment: float = None, closeness: int = None, model: str = None,
                     romance_level: float = None, source: str = 'chat',
                     message_type: str = 'normal', conversation_id: int = None,
                     emotion_state: str = None, emotion_timestamp = None,
                     mood_intensity: float = None, mood_sources: str = None,
                     avatar_filename: str = None):
        """Store a single message (user or companion)

        Args:
            email: User's email
            sender_name: Who sent the message (e.g., "User", "Companion")
            message_text: The message content
            sentiment: Sentiment score (typically only for the companion's messages)
            closeness: Closeness score after this message (typically only for the companion's messages)
            model: Model used (typically only for the companion's messages)
            romance_level: Romance level (typically only for the companion's messages)
            source: Message source - 'chat', 'sms', 'voice', 'avatar', 'proactive', or 'test'
            message_type: 'normal', 'proactive', or 'tool_response'
            conversation_id: Links related messages (user message + companion response)
            emotion_state: Primary mood/emotion name (e.g., 'excited', 'worried')
            emotion_timestamp: When the emotion snapshot was captured
            mood_intensity: Mood strength (0.0-1.0)
            mood_sources: JSON string of mood determination sources
            avatar_filename: Avatar image filename displayed with message

        Returns:
            The ID of the inserted message
        """
        # Don't store test messages
        if source == 'test':
            print(f"⚠️  Test message - NOT storing to database")
            return None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {T.MESSAGES} (email, sender_name, message_text, timestamp,
                                     sentiment_score, closeness_after, model_used,
                                     romance_level, source, message_type, conversation_id,
                                     emotion_state, emotion_timestamp, mood_intensity,
                                     mood_sources, avatar_filename)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (email, sender_name, message_text, now_pacific_naive(),
                  sentiment, closeness, model, romance_level, source, message_type, conversation_id,
                  emotion_state, emotion_timestamp, mood_intensity, mood_sources, avatar_filename))

            result = cursor.fetchone()
            return result[0] if result else None

    def get_recent_messages(self, email: str, limit: int = 100, source: str = None) -> List[Dict]:
        """Get recent conversation history (individual messages)

        Args:
            email: User email
            limit: Maximum number of messages to retrieve
            source: Filter by message source ('chat', 'sms', 'voice', 'avatar', 'proactive', or None for all)

        Returns:
            List of individual messages in chronological order (oldest first)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if source:
                cursor.execute(f'''
                    SELECT id, sender_name, message_text, timestamp, sentiment_score,
                           closeness_after, model_used, romance_level, source,
                           message_type, conversation_id
                    FROM {T.MESSAGES}
                    WHERE email = %s AND source = %s
                    ORDER BY timestamp DESC, id DESC
                    LIMIT %s
                ''', (email, source, limit))
            else:
                cursor.execute(f'''
                    SELECT id, sender_name, message_text, timestamp, sentiment_score,
                           closeness_after, model_used, romance_level, source,
                           message_type, conversation_id
                    FROM {T.MESSAGES}
                    WHERE email = %s
                    ORDER BY timestamp DESC, id DESC
                    LIMIT %s
                ''', (email, limit))

            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'id': row['id'],
                    'sender_name': row['sender_name'],
                    'message_text': row['message_text'],
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else row['timestamp'],
                    'sentiment': row['sentiment_score'],
                    'closeness': row['closeness_after'],
                    'model': row['model_used'],
                    'romance_level': row['romance_level'],
                    'source': row['source'],
                    'message_type': row['message_type'],
                    'conversation_id': row['conversation_id']
                })

            return list(reversed(messages))  # Return in chronological order

    def get_message_count(self, email: str) -> int:
        """Get total message count for user"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT COUNT(*) FROM {T.MESSAGES} WHERE email = %s', (email,))
            return cursor.fetchone()[0]

    def get_last_message_id(self, email: str) -> Optional[int]:
        """Get the ID of the most recently created message for a user"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT id FROM {T.MESSAGES} WHERE email = %s ORDER BY id DESC LIMIT 1',
                (email,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_messages_paginated(self, email: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get paginated messages for user (for API history endpoint)

        Args:
            email: User email
            limit: Number of messages per page
            offset: Starting position

        Returns:
            List of message dicts with pagination
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'''
                SELECT id, sender_name, message_text, timestamp, sentiment_score,
                       closeness_after, model_used, romance_level, source,
                       message_type, conversation_id
                FROM {T.MESSAGES}
                WHERE email = %s
                ORDER BY timestamp DESC, id DESC
                LIMIT %s OFFSET %s
            ''', (email, limit, offset))

            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'id': row['id'],
                    'sender_name': row['sender_name'],
                    'message_text': row['message_text'],
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else row['timestamp'],
                    'sentiment': row['sentiment_score'],
                    'closeness': row['closeness_after'],
                    'model': row['model_used'],
                    'romance_level': row['romance_level'],
                    'source': row['source'],
                    'message_type': row['message_type'],
                    'conversation_id': row['conversation_id']
                })

            return messages

    def get_days_since_last_message(self, email: str) -> int:
        """
        Calculate days since the user's last message

        Args:
            email: User email

        Returns:
            Number of days since last message, or 0 if messages exist today
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            _pc = get_persona_config()
            cursor.execute(f'''
                SELECT timestamp FROM {T.MESSAGES}
                WHERE email = %s AND sender_name != %s AND message_text IS NOT NULL AND message_text != ''
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (email, _pc.companion_short_name))

            row = cursor.fetchone()
            if not row:
                return 0  # No previous messages

            from datetime import datetime
            last_timestamp = row['timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)

            now = now_pacific_naive()
            days_diff = (now - last_timestamp).days

            return max(0, days_diff)

    def get_minutes_since_last_message(self, email: str, from_user_only: bool = False) -> Optional[int]:
        """
        Calculate minutes since the last message.

        Used for conversation continuity detection - determines if this is
        a continuation of an active conversation or a new session.

        Args:
            email: User email
            from_user_only: If True, only count messages from the user (not the companion).
                          This gives accurate gap measurement - the companion's own messages
                          shouldn't reset the "time since the user spoke" clock.

        Returns:
            Minutes since last message, or None if no previous messages
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = f'''
                SELECT timestamp FROM {T.MESSAGES}
                WHERE email = %s AND message_text IS NOT NULL AND message_text != ''
            '''
            params = [email]
            if from_user_only:
                _pc = get_persona_config()
                query += " AND sender_name != %s"
                params.append(_pc.companion_short_name)
            query += '''
                ORDER BY timestamp DESC
                LIMIT 1
            '''
            cursor.execute(query, params)

            row = cursor.fetchone()
            if not row:
                return None  # No previous messages

            from datetime import datetime
            last_timestamp = row['timestamp']
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp)

            now = now_pacific_naive()
            minutes_diff = (now - last_timestamp).total_seconds() / 60

            return max(0, int(minutes_diff))

    def get_messages_since(self, email: str, cutoff_date: str) -> List[Dict]:
        """
        Get messages since a specific date.

        Args:
            email: User email
            cutoff_date: ISO timestamp cutoff (messages >= this date)

        Returns:
            List of individual message dicts in chronological order
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'''
                SELECT id, sender_name, message_text, timestamp, sentiment_score,
                       closeness_after, model_used, romance_level, source,
                       message_type, conversation_id
                FROM {T.MESSAGES}
                WHERE email = %s AND timestamp >= %s
                ORDER BY timestamp ASC, id ASC
            ''', (email, cutoff_date))

            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'id': row['id'],
                    'sender_name': row['sender_name'],
                    'message_text': row['message_text'],
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else row['timestamp'],
                    'sentiment': row['sentiment_score'],
                    'closeness': row['closeness_after'],
                    'model': row['model_used'],
                    'romance_level': row['romance_level'],
                    'source': row['source'],
                    'message_type': row['message_type'],
                    'conversation_id': row['conversation_id']
                })

            return messages

    def get_messages_between(self, email: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Get messages between two dates.

        Args:
            email: User email
            start_date: ISO timestamp start (inclusive)
            end_date: ISO timestamp end (exclusive)

        Returns:
            List of individual message dicts in chronological order
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'''
                SELECT id, sender_name, message_text, timestamp, sentiment_score,
                       closeness_after, model_used, romance_level, source,
                       message_type, conversation_id
                FROM {T.MESSAGES}
                WHERE email = %s AND timestamp >= %s AND timestamp < %s
                ORDER BY timestamp ASC, id ASC
            ''', (email, start_date, end_date))

            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'id': row['id'],
                    'sender_name': row['sender_name'],
                    'message_text': row['message_text'],
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else row['timestamp'],
                    'sentiment': row['sentiment_score'],
                    'closeness': row['closeness_after'],
                    'model': row['model_used'],
                    'romance_level': row['romance_level'],
                    'source': row['source'],
                    'message_type': row['message_type'],
                    'conversation_id': row['conversation_id']
                })

            return messages

    def get_all_conversations_formatted(self, email: str, limit: int = 50) -> str:
        """Get formatted conversation history for LLM context with emphasis on recent messages

        Args:
            email: User email
            limit: Number of individual messages to retrieve (not pairs)

        Returns:
            Formatted conversation string with timestamps and section markers
        """
        from datetime import datetime

        messages = self.get_recent_messages(email, limit)

        if not messages:
            return ""

        # Already in chronological order from get_recent_messages
        formatted = []
        total_messages = len(messages)

        # Add all messages with strong recency markers to help LLM attention
        for idx, msg in enumerate(messages):
            # Divide conversation into sections with clear markers
            # This helps the LLM pay attention to recent context
            if idx == 0:
                formatted.append("=== EARLIER CONVERSATION ===\n")
            elif idx == total_messages - 20:
                formatted.append("\n=== RECENT CONVERSATION ===\n")
            elif idx == total_messages - 10:
                formatted.append("\n=== MOST RECENT CONVERSATION (CRITICAL - READ CAREFULLY) ===\n")

            # Format timestamp for readability
            timestamp = msg.get('timestamp', '')
            time_str = ""
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = timestamp
                    time_str = f" [{dt.strftime('%I:%M%p %m/%d').lower()}]"
                except:
                    pass

            # Format based on message type
            sender = msg['sender_name']
            message_text = msg['message_text']
            message_type = msg.get('message_type', 'normal')

            _pc = get_persona_config()
            if message_type == 'proactive' and sender == _pc.companion_short_name:
                # Companion initiated - show as proactive
                formatted.append(f"{_pc.companion_short_name} (proactive){time_str}: {message_text}")
            else:
                # Normal message from either user or companion
                formatted.append(f"{sender}{time_str}: {message_text}")

        return "\n".join(formatted)

    # ==================== FEED/MICROBLOG METHODS ====================

    def create_feed_post(self, content: str, mood: str = None, tags: List[str] = None) -> int:
        """Create a new feed post from the companion"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            tags_str = ','.join(tags) if tags else None
            cursor.execute(f'''
                INSERT INTO {T.FEED_POSTS} (content, timestamp, mood, tags)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            ''', (content, now_pacific_naive(), mood, tags_str))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_feed_posts(self, limit: int = 50, include_private: bool = False) -> List[Dict]:
        """Get recent feed posts"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if include_private:
                cursor.execute(f'''
                    SELECT id, content, timestamp, mood, tags, is_public
                    FROM {T.FEED_POSTS}
                    ORDER BY timestamp DESC
                    LIMIT %s
                ''', (limit,))
            else:
                cursor.execute(f'''
                    SELECT id, content, timestamp, mood, tags, is_public
                    FROM {T.FEED_POSTS}
                    WHERE is_public = TRUE
                    ORDER BY timestamp DESC
                    LIMIT %s
                ''', (limit,))

            posts = []
            for row in cursor.fetchall():
                posts.append({
                    'id': row['id'],
                    'content': row['content'],
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else row['timestamp'],
                    'mood': row['mood'],
                    'tags': row['tags'].split(',') if row['tags'] else [],
                    'is_public': bool(row['is_public'])
                })

            return posts

    def delete_feed_post(self, post_id: int) -> bool:
        """Delete a feed post"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {T.FEED_POSTS} WHERE id = %s', (post_id,))
            return cursor.rowcount > 0

    # ==================== GENERIC STATE METHODS ====================

    def get_state_value(self, key: str) -> Optional[str]:
        """Get a state value by key (returns JSON string or None)"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'SELECT value FROM {T.STATE} WHERE key = %s', (key,))
            row = cursor.fetchone()
            return row['value'] if row else None

    def set_state_value(self, key: str, value: str):
        """Set a state value (value should be JSON string)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT key FROM {T.STATE} WHERE key = %s', (key,))
            exists = cursor.fetchone() is not None

            if exists:
                cursor.execute(f'''
                    UPDATE {T.STATE}
                    SET value = %s, updated_at = %s
                    WHERE key = %s
                ''', (value, now_pacific_naive(), key))
            else:
                cursor.execute(f'''
                    INSERT INTO {T.STATE} (key, value, updated_at)
                    VALUES (%s, %s, %s)
                ''', (key, value, now_pacific_naive()))

    def delete_state_value(self, key: str) -> bool:
        """Delete a state value by key"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {T.STATE} WHERE key = %s', (key,))
            return cursor.rowcount > 0

    # ==================== UPCOMING EVENTS METHODS ====================

    def create_event(self, user_email: str, event_type: str, description: str,
                    scheduled_time: Optional[str] = None, participants: Optional[List[str]] = None,
                    location: Optional[str] = None, notes: Optional[str] = None) -> int:
        """
        Create a new upcoming event/plan

        Args:
            user_email: User who made the plan
            event_type: Type of event (movie, dinner, meeting, activity, etc.)
            description: Description of the event
            scheduled_time: ISO timestamp when event is planned (optional)
            participants: List of participants (optional)
            location: Where the event takes place (optional)
            notes: Additional notes (optional)

        Returns:
            Event ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            participants_str = ','.join(participants) if participants else None
            cursor.execute(f'''
                INSERT INTO {T.UPCOMING_EVENTS}
                (user_email, event_type, description, scheduled_time, created_at, status, participants, location, notes)
                VALUES (%s, %s, %s, %s, %s, 'planned', %s, %s, %s)
                RETURNING id
            ''', (user_email, event_type, description, scheduled_time, now_pacific_naive(),
                  participants_str, location, notes))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_upcoming_events(self, user_email: str, status: str = 'planned', limit: int = 50) -> List[Dict]:
        """
        Get upcoming events for a user

        Args:
            user_email: User email
            status: Event status ('planned', 'completed', 'cancelled')
            limit: Max number of events to return

        Returns:
            List of event dicts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'''
                SELECT id, user_email, event_type, description, scheduled_time,
                       created_at, status, participants, location, notes
                FROM {T.UPCOMING_EVENTS}
                WHERE user_email = %s AND status = %s
                ORDER BY scheduled_time ASC NULLS LAST, created_at DESC
                LIMIT %s
            ''', (user_email, status, limit))

            events = []
            for row in cursor.fetchall():
                events.append({
                    'id': row['id'],
                    'user_email': row['user_email'],
                    'event_type': row['event_type'],
                    'description': row['description'],
                    'scheduled_time': row['scheduled_time'].isoformat() if row['scheduled_time'] and hasattr(row['scheduled_time'], 'isoformat') else row['scheduled_time'],
                    'created_at': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else row['created_at'],
                    'status': row['status'],
                    'participants': row['participants'].split(',') if row['participants'] else [],
                    'location': row['location'],
                    'notes': row['notes']
                })

            return events

    def get_past_due_events(self, cutoff_time: Optional[str] = None) -> List[Dict]:
        """
        Get events that are past their scheduled time and still marked as 'planned'
        These should be consolidated into Neo4j as past memories

        Args:
            cutoff_time: ISO timestamp cutoff (default: now)

        Returns:
            List of past-due event dicts
        """
        if not cutoff_time:
            cutoff_time = now_pacific_naive()

        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'''
                SELECT id, user_email, event_type, description, scheduled_time,
                       created_at, status, participants, location, notes
                FROM {T.UPCOMING_EVENTS}
                WHERE status = 'planned'
                  AND scheduled_time IS NOT NULL
                  AND scheduled_time < %s
                ORDER BY scheduled_time ASC
            ''', (cutoff_time,))

            events = []
            for row in cursor.fetchall():
                events.append({
                    'id': row['id'],
                    'user_email': row['user_email'],
                    'event_type': row['event_type'],
                    'description': row['description'],
                    'scheduled_time': row['scheduled_time'].isoformat() if hasattr(row['scheduled_time'], 'isoformat') else row['scheduled_time'],
                    'created_at': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else row['created_at'],
                    'status': row['status'],
                    'participants': row['participants'].split(',') if row['participants'] else [],
                    'location': row['location'],
                    'notes': row['notes']
                })

            return events

    def get_event_by_id(self, event_id: int) -> Optional[Dict]:
        """
        Get a single event by ID

        Args:
            event_id: Event ID

        Returns:
            Event dict or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'''
                SELECT id, user_email, event_type, description, scheduled_time,
                       created_at, status, participants, location, notes
                FROM {T.UPCOMING_EVENTS}
                WHERE id = %s
            ''', (event_id,))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'user_email': row['user_email'],
                    'event_type': row['event_type'],
                    'description': row['description'],
                    'scheduled_time': row['scheduled_time'].isoformat() if row['scheduled_time'] and hasattr(row['scheduled_time'], 'isoformat') else row['scheduled_time'],
                    'created_at': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else row['created_at'],
                    'status': row['status'],
                    'participants': row['participants'].split(',') if row['participants'] else [],
                    'location': row['location'],
                    'notes': row['notes']
                }
            return None

    def update_event_status(self, event_id: int, status: str) -> bool:
        """
        Update event status (planned -> completed or cancelled)

        Args:
            event_id: Event ID
            status: New status

        Returns:
            True if updated successfully
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE {T.UPCOMING_EVENTS}
                SET status = %s
                WHERE id = %s
            ''', (status, event_id))
            return cursor.rowcount > 0

    def delete_event(self, event_id: int) -> bool:
        """Delete an event"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {T.UPCOMING_EVENTS} WHERE id = %s', (event_id,))
            return cursor.rowcount > 0

    def get_events_for_prompt(self, user_email: str, limit: int = 5) -> str:
        """
        Format upcoming events for inclusion in system prompt

        Args:
            user_email: User email
            limit: Max number of events to include

        Returns:
            Formatted string for prompt
        """
        events = self.get_upcoming_events(user_email, status='planned', limit=limit)

        if not events:
            return ""

        lines = ["\n## UPCOMING PLANS & EVENTS"]
        lines.append("You have these upcoming plans (don't repeat them unless relevant):\n")

        for event in events:
            event_str = f"- {event['description']}"
            if event['scheduled_time']:
                try:
                    # Parse and format time nicely
                    from datetime import datetime
                    if isinstance(event['scheduled_time'], str):
                        event_time = datetime.fromisoformat(event['scheduled_time'])
                    else:
                        event_time = event['scheduled_time']
                    time_str = event_time.strftime('%A, %B %d at %I:%M %p').replace(' 0', ' ')
                    event_str += f" ({time_str})"
                except:
                    pass  # Skip time formatting if it fails

            if event['location']:
                event_str += f" at {event['location']}"

            lines.append(event_str)

        return "\n".join(lines)

    def get_autonomous_task(self, task_id: str) -> Optional[Dict]:
        """
        Retrieve an autonomous task by task_id.
        Properly handles cursor lifecycle within context manager.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                f"""
                SELECT task_id, user_id, task_name, status, narrative,
                       mood_delta, energy_delta, created_at, updated_at
                FROM {T.COMPANION_AUTONOMOUS_TASKS}
                WHERE task_id = %s
                """,
                (task_id,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    'task_id': row['task_id'],
                    'user_id': row['user_id'],
                    'task_name': row['task_name'],
                    'status': row['status'],
                    'narrative': row['narrative'],
                    'mood_delta': row['mood_delta'],
                    'energy_delta': row['energy_delta'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
                }

            return None


    # ==================== VECTOR SEARCH METHODS ====================

    def search_similar_messages(self, query_embedding: List[float], email: str = None,
                                limit: int = 10, min_similarity: float = 0.5) -> List[Dict]:
        """
        Find messages semantically similar to a query embedding using pgvector.

        Args:
            query_embedding: 1536-dimension embedding vector from OpenAI
            email: Optional - filter to specific user's messages
            limit: Max number of results
            min_similarity: Minimum cosine similarity threshold (0-1)

        Returns:
            List of messages with similarity scores, most similar first
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Convert embedding list to pgvector format
            embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

            if email:
                cursor.execute(f'''
                    SELECT
                        id, sender_name, message_text, timestamp, email,
                        1 - (embedding_vec <=> %s::vector) as similarity
                    FROM {T.MESSAGES}
                    WHERE embedding_vec IS NOT NULL
                      AND email = %s
                      AND 1 - (embedding_vec <=> %s::vector) >= %s
                    ORDER BY embedding_vec <=> %s::vector
                    LIMIT %s
                ''', (embedding_str, email, embedding_str, min_similarity, embedding_str, limit))
            else:
                cursor.execute(f'''
                    SELECT
                        id, sender_name, message_text, timestamp, email,
                        1 - (embedding_vec <=> %s::vector) as similarity
                    FROM {T.MESSAGES}
                    WHERE embedding_vec IS NOT NULL
                      AND 1 - (embedding_vec <=> %s::vector) >= %s
                    ORDER BY embedding_vec <=> %s::vector
                    LIMIT %s
                ''', (embedding_str, embedding_str, min_similarity, embedding_str, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'sender_name': row['sender_name'],
                    'message_text': row['message_text'],
                    'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else row['timestamp'],
                    'email': row['email'],
                    'similarity': float(row['similarity'])
                })

            return results

    def get_message_embedding(self, message_id: int) -> Optional[List[float]]:
        """Get the embedding vector for a specific message"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT embedding_vec::text FROM {T.MESSAGES} WHERE id = %s AND embedding_vec IS NOT NULL',
                (message_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                # Parse "[1,2,3,...]" format back to list
                vec_str = row[0].strip('[]')
                return [float(x) for x in vec_str.split(',')]
            return None

    def get_user_facts(self, email: str, limit: int = 50) -> List[Dict]:
        """
        Get facts for a user, ordered by recency.

        Args:
            email: User email to filter facts
            limit: Maximum facts to return

        Returns:
            List of fact dictionaries with temporal info
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT id, subject, predicate, object, confidence, importance,
                       temporal, context, source, created_at, last_mentioned
                FROM {T.FACTS}
                WHERE user_email = %s AND archived_at IS NULL
                ORDER BY last_mentioned DESC NULLS LAST, created_at DESC
                LIMIT %s
            ''', (email, limit))

            rows = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'subject': row[1],
                    'predicate': row[2],
                    'object': row[3],
                    'confidence': row[4],
                    'importance': row[5],
                    'temporal': row[6],
                    'context': row[7],
                    'source': row[8],
                    'created_at': row[9],
                    'last_mentioned': row[10]
                }
                for row in rows
            ]


# ---------------------------------------------------------------------------
# Module-level singleton — one CompanionDB per process
# ---------------------------------------------------------------------------

_db_instance = None


def get_db() -> CompanionDB:
    """Get the singleton CompanionDB instance (created on first call)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = CompanionDB()
    return _db_instance


# Backward compatibility alias (deprecated — use CompanionDB)
# Legacy alias removed - use CompanionDB directly
