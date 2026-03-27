"""Regression tests for issue #31: db.py methods must pass user_email to _get_connection.

Every CompanionDB method that accesses a user-schema table must call
_get_connection(user_email=...) so the search_path is set correctly.
Without this, all queries hit the public schema instead of the user's schema.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    """Create a CompanionDB instance without hitting a real database."""
    with patch.dict('os.environ', {
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'companion_dev',
        'POSTGRES_USER': 'companion',
        'POSTGRES_PASSWORD': 'test',
        'ENVIRONMENT': 'development',
    }):
        from src.database.db import CompanionDB
        return CompanionDB()


def _mock_conn():
    """Build a mock connection + cursor that behaves like psycopg2."""
    conn = MagicMock()
    cursor = MagicMock()
    # cursor.fetchone / fetchall defaults
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    # Make cursor_factory= kwarg work
    conn.cursor.return_value = cursor
    return conn


TEST_EMAIL = "alice@example.com"


# ---------------------------------------------------------------------------
# Group 1 — methods that already had email/user_email param
# ---------------------------------------------------------------------------

class TestGroup1_ExistingEmailParam:
    """Methods that had email param but were not passing it to _get_connection."""

    def _assert_user_email_passed(self, method_name, call_args, call_kwargs=None):
        """Call a method and verify _get_connection received user_email."""
        db = _make_db()
        mock_conn = _mock_conn()

        with patch.object(db, '_get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            method = getattr(db, method_name)
            try:
                method(*call_args, **(call_kwargs or {}))
            except Exception:
                pass  # We only care about the _get_connection call

            mock_get_conn.assert_called_once_with(user_email=TEST_EMAIL)

    def test_get_state_passes_email(self):
        self._assert_user_email_passed('get_state', (TEST_EMAIL,))

    def test_save_state_passes_email(self):
        state = {
            'closeness_score': 15, 'romance_level': 0,
            'romance_enabled': False, 'romance_decision': None,
            'emotion_profile': 'Guarded', 'last_negative_event': None,
            'cooldown_active': False, 'badgering_count': 0,
            'last_message_time': None, 'sms_preference': 'good_morning',
        }
        self._assert_user_email_passed('save_state', (TEST_EMAIL, state))

    def test_store_message_passes_email(self):
        self._assert_user_email_passed(
            'store_message',
            (TEST_EMAIL, 'Alice', 'Hello'),
        )

    def test_get_recent_messages_passes_email(self):
        self._assert_user_email_passed('get_recent_messages', (TEST_EMAIL,))

    def test_get_message_count_passes_email(self):
        self._assert_user_email_passed('get_message_count', (TEST_EMAIL,))

    def test_get_last_message_id_passes_email(self):
        self._assert_user_email_passed('get_last_message_id', (TEST_EMAIL,))

    def test_get_messages_paginated_passes_email(self):
        self._assert_user_email_passed('get_messages_paginated', (TEST_EMAIL,))

    def test_get_days_since_last_message_passes_email(self):
        self._assert_user_email_passed('get_days_since_last_message', (TEST_EMAIL,))

    def test_get_minutes_since_last_message_passes_email(self):
        self._assert_user_email_passed('get_minutes_since_last_message', (TEST_EMAIL,))

    def test_get_messages_since_passes_email(self):
        self._assert_user_email_passed(
            'get_messages_since',
            (TEST_EMAIL, '2026-01-01T00:00:00'),
        )

    def test_get_messages_between_passes_email(self):
        self._assert_user_email_passed(
            'get_messages_between',
            (TEST_EMAIL, '2026-01-01T00:00:00', '2026-01-02T00:00:00'),
        )

    def test_search_similar_messages_passes_email(self):
        embedding = [0.1] * 10
        self._assert_user_email_passed(
            'search_similar_messages',
            (embedding,),
            call_kwargs={'email': TEST_EMAIL},
        )

    def test_get_user_facts_passes_email(self):
        self._assert_user_email_passed('get_user_facts', (TEST_EMAIL,))

    def test_create_event_passes_user_email(self):
        self._assert_user_email_passed(
            'create_event',
            (TEST_EMAIL, 'meeting', 'Team standup'),
        )

    def test_get_upcoming_events_passes_user_email(self):
        self._assert_user_email_passed('get_upcoming_events', (TEST_EMAIL,))


# ---------------------------------------------------------------------------
# Group 2 — methods that gained a new user_email param
# ---------------------------------------------------------------------------

class TestGroup2_NewEmailParam:
    """Methods that previously had no email param but access user-schema tables."""

    def _assert_user_email_passed(self, method_name, call_args, call_kwargs=None):
        """Call a method with user_email kwarg and verify _get_connection received it."""
        db = _make_db()
        mock_conn = _mock_conn()

        with patch.object(db, '_get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            method = getattr(db, method_name)
            kwargs = call_kwargs or {}
            kwargs['user_email'] = TEST_EMAIL
            try:
                method(*call_args, **kwargs)
            except Exception:
                pass

            mock_get_conn.assert_called_once_with(user_email=TEST_EMAIL)

    def test_create_feed_post_passes_email(self):
        self._assert_user_email_passed('create_feed_post', ('Post content',))

    def test_get_feed_posts_passes_email(self):
        self._assert_user_email_passed('get_feed_posts', ())

    def test_delete_feed_post_passes_email(self):
        self._assert_user_email_passed('delete_feed_post', (1,))

    def test_get_state_value_passes_email(self):
        self._assert_user_email_passed('get_state_value', ('some_key',))

    def test_set_state_value_passes_email(self):
        self._assert_user_email_passed('set_state_value', ('key', 'value'))

    def test_delete_state_value_passes_email(self):
        self._assert_user_email_passed('delete_state_value', ('key',))

    def test_get_past_due_events_passes_email(self):
        self._assert_user_email_passed('get_past_due_events', ())

    def test_get_event_by_id_passes_email(self):
        self._assert_user_email_passed('get_event_by_id', (1,))

    def test_update_event_status_passes_email(self):
        self._assert_user_email_passed('update_event_status', (1, 'completed'))

    def test_delete_event_passes_email(self):
        self._assert_user_email_passed('delete_event', (1,))

    def test_get_autonomous_task_passes_email(self):
        self._assert_user_email_passed('get_autonomous_task', ('task-123',))

    def test_get_message_embedding_passes_email(self):
        self._assert_user_email_passed('get_message_embedding', (1,))


# ---------------------------------------------------------------------------
# Group 2 backward compatibility — omitting user_email still works (None)
# ---------------------------------------------------------------------------

class TestGroup2_BackwardCompatibility:
    """Methods with new user_email param still work when it's omitted."""

    def _assert_none_when_omitted(self, method_name, call_args):
        db = _make_db()
        mock_conn = _mock_conn()

        with patch.object(db, '_get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            method = getattr(db, method_name)
            try:
                method(*call_args)
            except Exception:
                pass

            mock_get_conn.assert_called_once_with(user_email=None)

    def test_get_state_value_defaults_to_none(self):
        self._assert_none_when_omitted('get_state_value', ('key',))

    def test_set_state_value_defaults_to_none(self):
        self._assert_none_when_omitted('set_state_value', ('key', 'val'))

    def test_delete_state_value_defaults_to_none(self):
        self._assert_none_when_omitted('delete_state_value', ('key',))

    def test_get_past_due_events_defaults_to_none(self):
        self._assert_none_when_omitted('get_past_due_events', ())

    def test_get_event_by_id_defaults_to_none(self):
        self._assert_none_when_omitted('get_event_by_id', (1,))

    def test_update_event_status_defaults_to_none(self):
        self._assert_none_when_omitted('update_event_status', (1, 'done'))

    def test_delete_event_defaults_to_none(self):
        self._assert_none_when_omitted('delete_event', (1,))

    def test_get_autonomous_task_defaults_to_none(self):
        self._assert_none_when_omitted('get_autonomous_task', ('t1',))

    def test_get_message_embedding_defaults_to_none(self):
        self._assert_none_when_omitted('get_message_embedding', (1,))
