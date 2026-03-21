"""
Companion Journal - The companion's private diary and reflection storage.

WHAT: PostgreSQL-backed storage for the companion's internal reflections,
private thoughts, and queued conversation threads. Entries are dated and
typed (daily_reflection, weekly_reflection, insight) with structured
insights stored as JSONB (emotional_arc, open_threads, queued_thoughts).

WHY: Facts and biographies describe the user and the world. The journal
describes the companion's own inner life -- how she felt about a
conversation, what she wants to bring up next time, unresolved threads
she is tracking. This gives her personality continuity and depth across
sessions.

HOW it fits:
  - The daily reflection task (cron) generates a journal entry summarizing
    the day's conversations, then stores it via store_reflection().
  - The proactive messaging system calls get_open_threads() and
    get_queued_thoughts() to find conversation hooks.
  - context_builder can inject recent reflections for emotional continuity.

Key distinction: facts are about James/the world; journal entries are about
the companion's perspective and inner state.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.database import tables as T

logger = logging.getLogger(__name__)


class CompanionJournal:
    """The companion's private journal storage."""

    def __init__(self, user_email: str = None):
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        self.user_email = user_email
        self._conn = None

    def _get_connection(self):
        """Get database connection."""
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

    def store_reflection(
        self,
        content: str,
        insights: Dict[str, Any] = None,
        entry_type: str = 'daily_reflection',
        entry_date: datetime = None
    ) -> Optional[int]:
        """
        Store a journal entry.

        Args:
            content: The main text content (private thought)
            insights: Structured insights (emotional_arc, open_threads, etc.)
            entry_type: Type of entry ('daily_reflection', 'weekly_reflection', 'insight')
            entry_date: Date for the entry (defaults to today)

        Returns:
            Entry ID if stored successfully, None otherwise
        """
        from src.utils.timezone_utils import now_pacific_naive

        if entry_date is None:
            entry_date = now_pacific_naive().date()
        elif hasattr(entry_date, 'date'):
            entry_date = entry_date.date()

        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.COMPANION_JOURNAL} (user_email, entry_date, entry_type, content, insights)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    self.user_email,
                    entry_date,
                    entry_type,
                    content,
                    json.dumps(insights) if insights else None
                ))

                entry_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"Stored journal entry {entry_id}: {entry_type} for {entry_date}")
                return entry_id

        except Exception as e:
            logger.error(f"Error storing journal entry: {e}")
            conn.rollback()
            return None

    def get_recent_reflections(
        self,
        days: int = 7,
        entry_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent journal entries.

        Args:
            days: How many days back to look
            entry_type: Optional filter by entry type

        Returns:
            List of journal entries
        """
        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                if entry_type:
                    cursor.execute(f"""
                        SELECT id, entry_date, entry_type, content, insights, created_at
                        FROM {T.COMPANION_JOURNAL}
                        WHERE user_email = %s
                        AND entry_type = %s
                        AND entry_date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY entry_date DESC
                    """, (self.user_email, entry_type, days))
                else:
                    cursor.execute(f"""
                        SELECT id, entry_date, entry_type, content, insights, created_at
                        FROM {T.COMPANION_JOURNAL}
                        WHERE user_email = %s
                        AND entry_date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY entry_date DESC
                    """, (self.user_email, days))

                return [
                    {
                        'id': row[0],
                        'date': row[1],
                        'type': row[2],
                        'content': row[3],
                        'insights': row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                        'created_at': row[5]
                    }
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.warning(f"Error getting journal entries: {e}")
            return []

    def get_entry_by_date(
        self,
        date: datetime,
        entry_type: str = 'daily_reflection'
    ) -> Optional[Dict[str, Any]]:
        """Get a specific journal entry by date."""
        if hasattr(date, 'date'):
            date = date.date()

        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, entry_date, entry_type, content, insights, created_at
                    FROM {T.COMPANION_JOURNAL}
                    WHERE user_email = %s
                    AND entry_date = %s
                    AND entry_type = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (self.user_email, date, entry_type))

                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'date': row[1],
                        'type': row[2],
                        'content': row[3],
                        'insights': row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                        'created_at': row[5]
                    }
                return None

        except Exception as e:
            logger.warning(f"Error getting journal entry: {e}")
            return None

    def search_insights(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search journal insights for matching content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching entries
        """
        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT id, entry_date, content, insights
                    FROM {T.COMPANION_JOURNAL}
                    WHERE user_email = %s
                    AND (
                        content ILIKE %s
                        OR insights::text ILIKE %s
                    )
                    ORDER BY entry_date DESC
                    LIMIT %s
                """, (self.user_email, f'%{query}%', f'%{query}%', limit))

                return [
                    {
                        'id': row[0],
                        'date': row[1],
                        'content': row[2],
                        'insights': row[3] if isinstance(row[3], dict) else json.loads(row[3]) if row[3] else {}
                    }
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.warning(f"Error searching journal: {e}")
            return []

    # =========================================================================
    # Derived queries -- extract structured data from reflection insights
    # =========================================================================

    def get_open_threads(self, max_age_days: int = 7) -> List[str]:
        """
        Get unresolved topics the companion is tracking.

        Scans the 'open_threads' key in the insights JSONB of recent daily
        reflections. Deduplicates by lowercase text, preserving most-recent
        first ordering.

        Args:
            max_age_days: How many days back to look

        Returns:
            Up to 10 unique open thread descriptions
        """
        reflections = self.get_recent_reflections(days=max_age_days, entry_type='daily_reflection')

        all_threads = []
        for r in reflections:
            threads = r.get('insights', {}).get('open_threads', [])
            all_threads.extend(threads)

        # Dedupe while preserving order (most recent first)
        seen: set = set()
        unique_threads = []
        for thread in all_threads:
            if thread.lower() not in seen:
                seen.add(thread.lower())
                unique_threads.append(thread)

        return unique_threads[:10]

    def get_queued_thoughts(self, max_age_days: int = 3) -> List[str]:
        """
        Get queued thoughts (things the companion wants to bring up).

        Args:
            max_age_days: How many days back to look

        Returns:
            List of queued thoughts
        """
        reflections = self.get_recent_reflections(days=max_age_days, entry_type='daily_reflection')

        all_thoughts = []
        for r in reflections:
            insights = r.get('insights', {})
            thoughts = insights.get('queued_thoughts', [])
            all_thoughts.extend(thoughts)

        # Dedupe
        seen = set()
        unique_thoughts = []
        for thought in all_thoughts:
            if thought.lower() not in seen:
                seen.add(thought.lower())
                unique_thoughts.append(thought)

        return unique_thoughts[:5]


# Singleton pattern
_journal_instance: Optional[CompanionJournal] = None


def get_companion_journal(user_email: str = None) -> CompanionJournal:
    """Get the journal instance."""
    global _journal_instance
    if _journal_instance is None or _journal_instance.user_email != user_email:
        _journal_instance = CompanionJournal(user_email)
    return _journal_instance
