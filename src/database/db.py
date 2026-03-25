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
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from src.utils.timezone_utils import now_pacific_naive
from src.config.persona_config import get_persona_config


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

        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_db,
            user=self.pg_user,
            password=self.pg_pass
        )

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = None):
        """
        Execute a raw SQL query and return a wrapper that holds the results.
        The connection is maintained within the context manager while data is fetched.
        Returns a cursor-like object that provides fetchone() and fetchall() methods.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params or ())
            # CRITICAL: Fetch data while connection is still open
            # The connection closes when exiting this context, so cursor becomes invalid
            # We store the fetched data and return a wrapper object
            return _CursorWrapper(cursor)

    def commit(self):
        """No-op for compatibility with raw SQL code patterns"""
        pass

    def _init_db(self):
        """Initialize database schema (tables already created by migration, but this ensures they exist)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # User profiles table (already exists from migration)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    email TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP
                )
            ''')

            # User state table (already exists from migration)
            # companion_id enables multi-agent: each companion has separate state per user
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_state (
                    email TEXT NOT NULL,
                    companion_id TEXT DEFAULT 'default',
                    closeness_score INTEGER DEFAULT 15,
                    romance_level REAL DEFAULT 0,
                    romance_enabled BOOLEAN DEFAULT FALSE,
                    romance_decision TEXT,
                    emotion_profile TEXT DEFAULT 'Guarded',
                    last_negative_event TIMESTAMP,
                    cooldown_active BOOLEAN DEFAULT FALSE,
                    badgering_count INTEGER DEFAULT 0,
                    last_message_time TIMESTAMP,
                    attraction_cue_count INTEGER DEFAULT 0,
                    attraction_latent_state BOOLEAN DEFAULT FALSE,
                    sms_preference TEXT DEFAULT 'good_morning',
                    PRIMARY KEY (email, companion_id),
                    FOREIGN KEY (email) REFERENCES user_profiles(email)
                )
            ''')

            # Messages table - stores all conversation messages
            # companion_id enables multi-agent: messages scoped per companion
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    companion_id TEXT DEFAULT 'default',
                    sender_name TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment_score FLOAT,
                    closeness_after INTEGER,
                    model_used TEXT,
                    romance_level FLOAT,
                    source TEXT DEFAULT 'chat',
                    message_type TEXT DEFAULT 'normal',
                    conversation_id INTEGER,
                    FOREIGN KEY (email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_email ON messages(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_companion_id ON messages(companion_id)')

            # Microblog/feed table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feed_posts (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    mood TEXT,
                    tags TEXT,
                    is_public BOOLEAN DEFAULT TRUE
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feed_posts_timestamp ON feed_posts(timestamp DESC)')

            # Generic state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_state_key ON state(key)')

            # Upcoming events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS upcoming_events (
                    id SERIAL PRIMARY KEY,
                    user_email TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    scheduled_time TIMESTAMP,
                    created_at TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'planned',
                    participants TEXT,
                    location TEXT,
                    notes TEXT,
                    FOREIGN KEY (user_email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_user_status ON upcoming_events(user_email, status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_scheduled_time ON upcoming_events(scheduled_time)')

            # Conversation patterns table - tracks user's conversation style
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_patterns (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    user_length INTEGER,
                    response_length INTEGER,
                    closeness INTEGER,
                    pattern_type TEXT DEFAULT 'exchange',
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_email ON conversation_patterns(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_recorded_at ON conversation_patterns(recorded_at DESC)')

            # Conversation patterns statistics table - aggregated statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_patterns_stats (
                    email TEXT PRIMARY KEY,
                    total_exchanges INTEGER DEFAULT 0,
                    avg_user_length FLOAT DEFAULT 0,
                    avg_response_length FLOAT DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (email) REFERENCES user_profiles(email)
                )
            ''')

            # Emotion analytics table - tracks emotional states over time
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotion_analytics (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    emotion TEXT,
                    confidence FLOAT,
                    intensity FLOAT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_email ON emotion_analytics(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_recorded_at ON emotion_analytics(recorded_at DESC)')

            # User preferences table - learned preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    preference_type TEXT,
                    preference_value TEXT,
                    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prefs_email ON user_preferences(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prefs_type ON user_preferences(preference_type)')

            # Episodes table - structured conversation episodes for enhanced episodic memory
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id SERIAL PRIMARY KEY,
                    episode_id UUID UNIQUE NOT NULL,
                    user_email TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    ended_at TIMESTAMP,
                    topic TEXT,
                    trigger TEXT,
                    emotional_state TEXT,
                    resolution TEXT,
                    user_satisfaction FLOAT,
                    companion_approach TEXT,
                    summary_embedding JSONB,
                    pattern_id UUID,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_user_email ON episodes(user_email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_started_at ON episodes(started_at DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_topic ON episodes(topic)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodes_pattern_id ON episodes(pattern_id)')

            # Episode messages - links messages to episodes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episode_messages (
                    id SERIAL PRIMARY KEY,
                    episode_id UUID NOT NULL,
                    message_id INTEGER NOT NULL,
                    turn_number INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode_messages_episode_id ON episode_messages(episode_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode_messages_message_id ON episode_messages(message_id)')

            # Episode patterns - consolidated learnings from similar episodes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episode_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_id UUID UNIQUE NOT NULL,
                    pattern_name TEXT NOT NULL,
                    description TEXT,
                    successful_approach TEXT,
                    pitfalls_to_avoid TEXT,
                    topic_category TEXT,
                    emotional_context TEXT,
                    episode_count INTEGER DEFAULT 0,
                    avg_satisfaction FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode_patterns_topic ON episode_patterns(topic_category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode_patterns_emotional ON episode_patterns(emotional_context)')

            # ==================== BENCHMARK TABLES ====================

            # Benchmark questions - curated test questions with expected answers
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS benchmark_questions (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    expected_answer TEXT NOT NULL,
                    expected_keywords TEXT,
                    negative_keywords TEXT,
                    category TEXT NOT NULL,
                    difficulty TEXT DEFAULT 'medium',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_questions_category ON benchmark_questions(category)')

            # Benchmark runs - metadata for each benchmark execution
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id SERIAL PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    config_snapshot JSONB,
                    overall_score FLOAT,
                    category_scores JSONB,
                    total_cost FLOAT DEFAULT 0,
                    total_time_seconds FLOAT DEFAULT 0,
                    question_count INTEGER DEFAULT 0,
                    observations_enabled BOOLEAN DEFAULT FALSE,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Benchmark results - per-question results for each run
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id SERIAL PRIMARY KEY,
                    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
                    question_id INTEGER NOT NULL REFERENCES benchmark_questions(id),
                    context_retrieved TEXT,
                    context_sources JSONB,
                    response TEXT,
                    accuracy_score FLOAT,
                    confabulation_score FLOAT,
                    completeness_score FLOAT,
                    context_utilization_score FLOAT,
                    keyword_hits INTEGER DEFAULT 0,
                    keyword_misses INTEGER DEFAULT 0,
                    negative_keyword_hits INTEGER DEFAULT 0,
                    context_build_time_ms INTEGER,
                    response_time_ms INTEGER,
                    judge_time_ms INTEGER,
                    cost_estimate FLOAT DEFAULT 0,
                    judge_reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_results_run ON benchmark_results(run_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_results_question ON benchmark_results(question_id)')

            # ==================== OBSERVATIONAL MEMORY TABLES ====================

            # Observations - compressed dated conversation summaries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS observations (
                    id SERIAL PRIMARY KEY,
                    user_email TEXT NOT NULL,
                    observation_date DATE NOT NULL,
                    time_range TEXT,
                    content TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    raw_token_count INTEGER DEFAULT 0,
                    compressed_token_count INTEGER DEFAULT 0,
                    compression_ratio FLOAT DEFAULT 0,
                    topics TEXT,
                    emotional_tone TEXT,
                    first_message_id INTEGER,
                    last_message_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_observations_user_date ON observations(user_email, observation_date DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(observation_date DESC)')

            # Observation reflections - weekly/monthly consolidations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS observation_reflections (
                    id SERIAL PRIMARY KEY,
                    user_email TEXT NOT NULL,
                    period_type TEXT NOT NULL DEFAULT 'weekly',
                    period_start DATE NOT NULL,
                    period_end DATE NOT NULL,
                    content TEXT NOT NULL,
                    themes TEXT,
                    observation_ids TEXT,
                    observation_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reflections_user_period ON observation_reflections(user_email, period_end DESC)')

            # ==================== MULTI-AGENT TABLES ====================

            # Companion reminders - scoped per companion instance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS companion_reminders (
                    id SERIAL PRIMARY KEY,
                    companion_id TEXT NOT NULL DEFAULT 'default',
                    user_email TEXT NOT NULL,
                    reminder_text TEXT NOT NULL,
                    remind_at TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (user_email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_companion_reminders_status ON companion_reminders(companion_id, status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_companion_reminders_remind_at ON companion_reminders(remind_at)')

            # Companion observations - scoped per companion instance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS companion_observations (
                    id SERIAL PRIMARY KEY,
                    companion_id TEXT NOT NULL DEFAULT 'default',
                    user_email TEXT NOT NULL,
                    observation_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_email) REFERENCES user_profiles(email)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_companion_observations_cid ON companion_observations(companion_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_companion_observations_user ON companion_observations(user_email, companion_id)')

    # ==================== USER PROFILE METHODS ====================

    def get_profile(self, email: str) -> Dict:
        """Get user profile by email"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                'SELECT * FROM user_profiles WHERE email = %s',
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
                'SELECT id FROM users WHERE email = %s',
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
                cursor.execute('''
                    UPDATE user_profiles
                    SET display_name = %s, last_seen = %s
                    WHERE email = %s
                ''', (display_name, now_pacific_naive(), email))
            else:
                # Create new
                cursor.execute('''
                    INSERT INTO user_profiles (email, display_name, is_admin, last_seen)
                    VALUES (%s, %s, FALSE, %s)
                ''', (email, display_name, now_pacific_naive()))

    def set_admin(self, email: str, is_admin: bool = True):
        """Set user as admin"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            existing = self.get_profile(email)

            if existing['created_at']:
                # Update existing
                cursor.execute('''
                    UPDATE user_profiles
                    SET is_admin = %s, last_seen = %s
                    WHERE email = %s
                ''', (is_admin, now_pacific_naive(), email))
            else:
                # Create new
                cursor.execute('''
                    INSERT INTO user_profiles (email, display_name, is_admin, last_seen)
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

            cursor.execute('''
                UPDATE user_profiles
                SET last_seen = %s
                WHERE email = %s
            ''', (now_pacific_naive(), email))

    # ==================== USER STATE METHODS ====================

    def get_state(self, email: str) -> Dict:
        """Get user state (closeness, romance, cooldowns)"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                'SELECT * FROM user_state WHERE email = %s',
                (email,)
            )
            row = cursor.fetchone()

            if row:
                # Convert timestamps
                import arrow
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
            cursor.execute('SELECT email FROM user_state WHERE email = %s', (email,))
            exists = cursor.fetchone() is not None

            if exists:
                # Update
                cursor.execute('''
                    UPDATE user_state
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
                cursor.execute('''
                    INSERT INTO user_state (
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
            import arrow
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
            cursor.execute('''
                INSERT INTO messages (email, sender_name, message_text, timestamp,
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
                cursor.execute('''
                    SELECT id, sender_name, message_text, timestamp, sentiment_score,
                           closeness_after, model_used, romance_level, source,
                           message_type, conversation_id
                    FROM messages
                    WHERE email = %s AND source = %s
                    ORDER BY timestamp DESC, id DESC
                    LIMIT %s
                ''', (email, source, limit))
            else:
                cursor.execute('''
                    SELECT id, sender_name, message_text, timestamp, sentiment_score,
                           closeness_after, model_used, romance_level, source,
                           message_type, conversation_id
                    FROM messages
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
            cursor.execute('SELECT COUNT(*) FROM messages WHERE email = %s', (email,))
            return cursor.fetchone()[0]

    def get_last_message_id(self, email: str) -> Optional[int]:
        """Get the ID of the most recently created message for a user"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id FROM messages WHERE email = %s ORDER BY id DESC LIMIT 1',
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
            cursor.execute('''
                SELECT id, sender_name, message_text, timestamp, sentiment_score,
                       closeness_after, model_used, romance_level, source,
                       message_type, conversation_id
                FROM messages
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
            cursor.execute('''
                SELECT timestamp FROM messages
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
            query = '''
                SELECT timestamp FROM messages
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
            cursor.execute('''
                SELECT id, sender_name, message_text, timestamp, sentiment_score,
                       closeness_after, model_used, romance_level, source,
                       message_type, conversation_id
                FROM messages
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
            cursor.execute('''
                SELECT id, sender_name, message_text, timestamp, sentiment_score,
                       closeness_after, model_used, romance_level, source,
                       message_type, conversation_id
                FROM messages
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
            cursor.execute('''
                INSERT INTO feed_posts (content, timestamp, mood, tags)
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
                cursor.execute('''
                    SELECT id, content, timestamp, mood, tags, is_public
                    FROM feed_posts
                    ORDER BY timestamp DESC
                    LIMIT %s
                ''', (limit,))
            else:
                cursor.execute('''
                    SELECT id, content, timestamp, mood, tags, is_public
                    FROM feed_posts
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
            cursor.execute('DELETE FROM feed_posts WHERE id = %s', (post_id,))
            return cursor.rowcount > 0

    # ==================== GENERIC STATE METHODS ====================

    def get_state_value(self, key: str) -> Optional[str]:
        """Get a state value by key (returns JSON string or None)"""
        with self._get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('SELECT value FROM state WHERE key = %s', (key,))
            row = cursor.fetchone()
            return row['value'] if row else None

    def set_state_value(self, key: str, value: str):
        """Set a state value (value should be JSON string)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key FROM state WHERE key = %s', (key,))
            exists = cursor.fetchone() is not None

            if exists:
                cursor.execute('''
                    UPDATE state
                    SET value = %s, updated_at = %s
                    WHERE key = %s
                ''', (value, now_pacific_naive(), key))
            else:
                cursor.execute('''
                    INSERT INTO state (key, value, updated_at)
                    VALUES (%s, %s, %s)
                ''', (key, value, now_pacific_naive()))

    def delete_state_value(self, key: str) -> bool:
        """Delete a state value by key"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM state WHERE key = %s', (key,))
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
            cursor.execute('''
                INSERT INTO upcoming_events
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
            cursor.execute('''
                SELECT id, user_email, event_type, description, scheduled_time,
                       created_at, status, participants, location, notes
                FROM upcoming_events
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
            cursor.execute('''
                SELECT id, user_email, event_type, description, scheduled_time,
                       created_at, status, participants, location, notes
                FROM upcoming_events
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
            cursor.execute('''
                SELECT id, user_email, event_type, description, scheduled_time,
                       created_at, status, participants, location, notes
                FROM upcoming_events
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
            cursor.execute('''
                UPDATE upcoming_events
                SET status = %s
                WHERE id = %s
            ''', (status, event_id))
            return cursor.rowcount > 0

    def delete_event(self, event_id: int) -> bool:
        """Delete an event"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM upcoming_events WHERE id = %s', (event_id,))
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
                """
                SELECT task_id, user_id, task_name, status, narrative,
                       mood_delta, energy_delta, created_at, updated_at
                FROM companion_autonomous_tasks
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
                cursor.execute('''
                    SELECT
                        id, sender_name, message_text, timestamp, email,
                        1 - (embedding_vec <=> %s::vector) as similarity
                    FROM messages
                    WHERE embedding_vec IS NOT NULL
                      AND email = %s
                      AND 1 - (embedding_vec <=> %s::vector) >= %s
                    ORDER BY embedding_vec <=> %s::vector
                    LIMIT %s
                ''', (embedding_str, email, embedding_str, min_similarity, embedding_str, limit))
            else:
                cursor.execute('''
                    SELECT
                        id, sender_name, message_text, timestamp, email,
                        1 - (embedding_vec <=> %s::vector) as similarity
                    FROM messages
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
                'SELECT embedding_vec::text FROM messages WHERE id = %s AND embedding_vec IS NOT NULL',
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
            cursor.execute('''
                SELECT id, subject, predicate, object, confidence, importance,
                       temporal, context, source, created_at, last_mentioned
                FROM facts
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
