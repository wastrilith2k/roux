"""Regression tests for issue #107: memory stores must route to per-user schema.

EpisodeStore and SynthesizedEventStore were already fixed before this issue.
This file verifies that FactStore, FactNetwork, SynthesizedBiographyStore,
and temporal_context functions all call set_search_path() when given a
user_email, so queries hit the correct per-user schema instead of public.
"""

import pytest
from unittest.mock import patch, MagicMock, call


TEST_EMAIL = "kai@example.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_psycopg2_connect():
    """Return a mock connection that acts like a psycopg2 connection."""
    conn = MagicMock()
    conn.closed = False
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    conn.cursor.return_value = cursor
    # Context manager support for cursor
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ---------------------------------------------------------------------------
# FactStore
# ---------------------------------------------------------------------------

class TestFactStoreSchemaRouting:
    """FactStore._get_connection must call set_search_path when user_email is provided."""

    def _make_store(self):
        from src.memory.fact_store import FactStore
        return FactStore()

    def test_get_connection_calls_set_search_path(self):
        """_get_connection(user_email) must call set_search_path."""
        store = self._make_store()
        mock_conn = _mock_psycopg2_connect()

        with patch('psycopg2.connect', return_value=mock_conn):
            with patch('src.database.schema_manager.set_search_path') as mock_ssp:
                store._get_connection(user_email=TEST_EMAIL)
                mock_ssp.assert_called_once_with(mock_conn, TEST_EMAIL)

    def test_get_connection_without_email_skips_set_search_path(self):
        """_get_connection() without user_email must NOT call set_search_path."""
        store = self._make_store()
        mock_conn = _mock_psycopg2_connect()

        with patch('psycopg2.connect', return_value=mock_conn):
            with patch('src.database.schema_manager.set_search_path') as mock_ssp:
                store._get_connection()
                mock_ssp.assert_not_called()

    def test_get_connection_reuses_conn_for_same_user(self):
        """Calling _get_connection twice with same email reuses the connection."""
        store = self._make_store()
        mock_conn = _mock_psycopg2_connect()

        with patch('psycopg2.connect', return_value=mock_conn) as mock_connect:
            with patch('src.database.schema_manager.set_search_path'):
                store._get_connection(user_email=TEST_EMAIL)
                store._get_connection(user_email=TEST_EMAIL)
                # Should only connect once
                assert mock_connect.call_count == 1

    def test_get_connection_reconnects_for_different_user(self):
        """Changing user_email must create a new connection with new schema."""
        store = self._make_store()
        mock_conn1 = _mock_psycopg2_connect()
        mock_conn2 = _mock_psycopg2_connect()

        with patch('psycopg2.connect', side_effect=[mock_conn1, mock_conn2]) as mock_connect:
            with patch('src.database.schema_manager.set_search_path') as mock_ssp:
                store._get_connection(user_email=TEST_EMAIL)
                store._get_connection(user_email="other@example.com")

                assert mock_connect.call_count == 2
                assert mock_ssp.call_count == 2
                mock_ssp.assert_any_call(mock_conn1, TEST_EMAIL)
                mock_ssp.assert_any_call(mock_conn2, "other@example.com")

    def test_store_fact_passes_user_email_to_get_connection(self):
        """store_fact(user_email=X) must pass X to _get_connection."""
        store = self._make_store()

        with patch.object(store, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            with patch.object(store, 'is_duplicate', return_value=False):
                with patch.object(store, 'handle_contradictions', return_value=(0, [])):
                    store.store_fact(
                        subject="James", predicate="works_at", obj="Cavallo",
                        user_email=TEST_EMAIL
                    )
                    mock_gc.assert_called_with(TEST_EMAIL)

    def test_search_facts_hybrid_passes_user_email(self):
        """search_facts_hybrid(user_email=X) must pass X to _get_connection."""
        store = self._make_store()

        with patch.object(store, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            with patch.object(store, '_get_embedding', return_value=None):
                store.search_facts_hybrid("test query", user_email=TEST_EMAIL)
                mock_gc.assert_called_with(TEST_EMAIL)

    def test_get_facts_for_subject_passes_user_email(self):
        """get_facts_for_subject(user_email=X) must pass X to _get_connection."""
        store = self._make_store()

        with patch.object(store, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            store.get_facts_for_subject("James", user_email=TEST_EMAIL)
            mock_gc.assert_called_with(TEST_EMAIL)

    def test_get_important_facts_passes_user_email(self):
        """get_important_facts(user_email=X) must pass X to _get_connection."""
        store = self._make_store()

        with patch.object(store, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            store.get_important_facts(user_email=TEST_EMAIL)
            mock_gc.assert_called_with(TEST_EMAIL)


# ---------------------------------------------------------------------------
# FactNetwork
# ---------------------------------------------------------------------------

class TestFactNetworkSchemaRouting:
    """FactNetwork._get_connection must call set_search_path when user_email is provided."""

    def _make_network(self):
        from src.memory.fact_network import FactNetwork
        return FactNetwork()

    def test_get_connection_calls_set_search_path(self):
        network = self._make_network()
        mock_conn = _mock_psycopg2_connect()

        with patch('psycopg2.connect', return_value=mock_conn):
            with patch('src.database.schema_manager.set_search_path') as mock_ssp:
                network._get_connection(user_email=TEST_EMAIL)
                mock_ssp.assert_called_once_with(mock_conn, TEST_EMAIL)

    def test_get_connection_without_email_skips_set_search_path(self):
        network = self._make_network()
        mock_conn = _mock_psycopg2_connect()

        with patch('psycopg2.connect', return_value=mock_conn):
            with patch('src.database.schema_manager.set_search_path') as mock_ssp:
                network._get_connection()
                mock_ssp.assert_not_called()

    def test_create_link_passes_user_email(self):
        network = self._make_network()

        with patch.object(network, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            network.create_link(1, 2, "similar", user_email=TEST_EMAIL)
            mock_gc.assert_called_with(TEST_EMAIL)

    def test_get_linked_facts_passes_user_email(self):
        network = self._make_network()

        with patch.object(network, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            network.get_linked_facts(1, user_email=TEST_EMAIL)
            mock_gc.assert_called_with(TEST_EMAIL)

    def test_detect_links_passes_user_email(self):
        network = self._make_network()

        with patch.object(network, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            network.detect_links_for_new_fact(
                new_fact_id=1, new_fact_subject="James",
                new_fact_object="works at Cavallo", user_email=TEST_EMAIL
            )
            mock_gc.assert_called_with(TEST_EMAIL)


# ---------------------------------------------------------------------------
# SynthesizedBiographyStore
# ---------------------------------------------------------------------------

class TestBiographyStoreSchemaRouting:
    """SynthesizedBiographyStore._get_connection must call set_search_path."""

    def _make_store(self):
        from src.memory.synthesized_biographies import SynthesizedBiographyStore
        return SynthesizedBiographyStore()

    def test_get_connection_calls_set_search_path(self):
        store = self._make_store()
        mock_conn = _mock_psycopg2_connect()

        with patch('psycopg2.connect', return_value=mock_conn):
            with patch('src.database.schema_manager.set_search_path') as mock_ssp:
                store._get_connection(user_email=TEST_EMAIL)
                mock_ssp.assert_called_once_with(mock_conn, TEST_EMAIL)

    def test_get_connection_without_email_skips_set_search_path(self):
        store = self._make_store()
        mock_conn = _mock_psycopg2_connect()

        with patch('psycopg2.connect', return_value=mock_conn):
            with patch('src.database.schema_manager.set_search_path') as mock_ssp:
                store._get_connection()
                mock_ssp.assert_not_called()

    def test_store_paragraph_passes_user_email(self):
        store = self._make_store()

        with patch.object(store, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            store.store_paragraph(
                theme="work", subject="James", content="Works at Cavallo.",
                fact_ids=[1, 2], base_importance=7.0, user_email=TEST_EMAIL
            )
            mock_gc.assert_called_with(TEST_EMAIL)

    def test_get_paragraphs_for_subject_passes_user_email(self):
        store = self._make_store()

        with patch.object(store, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            store.get_paragraphs_for_subject("James", user_email=TEST_EMAIL)
            mock_gc.assert_called_with(TEST_EMAIL)

    def test_get_all_paragraphs_passes_user_email(self):
        store = self._make_store()

        with patch.object(store, '_get_connection', return_value=_mock_psycopg2_connect()) as mock_gc:
            store.get_all_paragraphs(user_email=TEST_EMAIL)
            mock_gc.assert_called_with(TEST_EMAIL)


# ---------------------------------------------------------------------------
# temporal_context.py
# ---------------------------------------------------------------------------

class TestTemporalContextSchemaRouting:
    """temporal_context functions must pass user_email to db._get_connection."""

    def test_get_recent_significant_events_passes_user_email(self):
        """get_recent_significant_events must pass user_email to db._get_connection."""
        mock_conn = _mock_psycopg2_connect()
        mock_db = MagicMock()
        mock_db._get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db._get_connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch('src.database.db.get_db', return_value=mock_db):
            from src.memory.temporal_context import get_recent_significant_events
            get_recent_significant_events(user_email=TEST_EMAIL)

            mock_db._get_connection.assert_called_once_with(TEST_EMAIL)

    def test_get_recent_notable_messages_passes_user_email(self):
        """get_recent_notable_messages must pass user_email to db._get_connection."""
        mock_conn = _mock_psycopg2_connect()
        mock_db = MagicMock()
        mock_db._get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db._get_connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch('src.database.db.get_db', return_value=mock_db):
            from src.memory.temporal_context import get_recent_notable_messages
            get_recent_notable_messages(user_email=TEST_EMAIL)

            mock_db._get_connection.assert_called_once_with(TEST_EMAIL)
