"""
Tests for memory eviction — issue #20.

Covers:
- Recency weighting in semantic search
- Embedding pruning task safety rails and logic
- Graphiti pruning task safety rails and logic
- Storage metrics in build_with_diagnostics
- last_retrieved_at column in schema DDL
- Beat schedule registration for new tasks
- Convention compliance (get_db, parameterized queries)
"""

import os
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Phase 1: Recency weighting
# ---------------------------------------------------------------------------

class TestRecencyWeighting:
    """Test _apply_recency_weighting in semantic_search.py."""

    def test_recent_message_gets_full_weight(self):
        """Messages from today should get recency_factor ~1.0."""
        from src.memory.semantic_search import _apply_recency_weighting

        now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
        results = [{'similarity': 0.8, 'timestamp': now - timedelta(hours=1), 'id': 1}]
        weighted = _apply_recency_weighting(results, now=now)

        assert weighted[0]['recency_factor'] == 1.0
        assert weighted[0]['similarity'] == 0.8

    def test_old_message_gets_reduced_weight(self):
        """Messages older than the decay window get the floor value."""
        from src.memory.semantic_search import _apply_recency_weighting, RECENCY_DECAY_DAYS, RECENCY_FLOOR

        now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
        old_ts = now - timedelta(days=RECENCY_DECAY_DAYS + 30)
        results = [{'similarity': 0.8, 'timestamp': old_ts, 'id': 1}]
        weighted = _apply_recency_weighting(results, now=now)

        assert weighted[0]['recency_factor'] == RECENCY_FLOOR
        assert weighted[0]['similarity'] == round(0.8 * RECENCY_FLOOR, 4)

    def test_midpoint_message_gets_partial_weight(self):
        """Messages at half the decay window should get ~0.75 recency factor."""
        from src.memory.semantic_search import _apply_recency_weighting, RECENCY_DECAY_DAYS, RECENCY_FLOOR

        now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
        half_ts = now - timedelta(days=RECENCY_DECAY_DAYS // 2)
        results = [{'similarity': 0.8, 'timestamp': half_ts, 'id': 1}]
        weighted = _apply_recency_weighting(results, now=now)

        expected_factor = 1.0 - (1.0 - RECENCY_FLOOR) * 0.5
        assert abs(weighted[0]['recency_factor'] - expected_factor) < 0.02

    def test_recency_reorders_results(self):
        """Recent lower-similarity should outrank old higher-similarity after weighting."""
        from src.memory.semantic_search import _apply_recency_weighting, RECENCY_DECAY_DAYS

        now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
        results = [
            {'similarity': 0.9, 'timestamp': now - timedelta(days=RECENCY_DECAY_DAYS + 10), 'id': 1},
            {'similarity': 0.7, 'timestamp': now - timedelta(days=1), 'id': 2},
        ]
        weighted = _apply_recency_weighting(results, now=now)

        # The recent message (id=2) should now rank first despite lower raw similarity
        assert weighted[0]['id'] == 2

    def test_empty_results_returns_empty(self):
        """Empty input should return empty output."""
        from src.memory.semantic_search import _apply_recency_weighting

        assert _apply_recency_weighting([]) == []

    def test_missing_timestamp_gets_floor(self):
        """Messages with no timestamp get the floor recency factor."""
        from src.memory.semantic_search import _apply_recency_weighting, RECENCY_FLOOR

        now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
        results = [{'similarity': 0.8, 'timestamp': None, 'id': 1}]
        weighted = _apply_recency_weighting(results, now=now)

        assert weighted[0]['recency_factor'] == RECENCY_FLOOR

    def test_string_timestamp_is_handled(self):
        """ISO string timestamps should be parsed correctly."""
        from src.memory.semantic_search import _apply_recency_weighting

        now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
        results = [{'similarity': 0.8, 'timestamp': '2026-03-26T10:00:00+00:00', 'id': 1}]
        weighted = _apply_recency_weighting(results, now=now)

        assert weighted[0]['recency_factor'] == 1.0

    def test_naive_timestamp_assumed_pacific(self):
        """Naive timestamps (no tz) should be treated as Pacific time."""
        from src.memory.semantic_search import _apply_recency_weighting

        now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
        naive_ts = datetime(2026, 3, 25, 12, 0)
        results = [{'similarity': 0.8, 'timestamp': naive_ts, 'id': 1}]
        weighted = _apply_recency_weighting(results, now=now)

        assert weighted[0]['recency_factor'] > 0.99


class TestSearchMemoryRecencyIntegration:
    """Test that search_memory applies recency weighting and tracks retrieval."""

    @patch('src.memory.semantic_search._update_last_retrieved_at')
    @patch('src.memory.semantic_search.get_db')
    @patch('src.memory.semantic_search.generate_embedding')
    def test_search_memory_applies_recency_and_tracks(self, mock_embed, mock_get_db, mock_update):
        """search_memory should apply recency weighting and call _update_last_retrieved_at."""
        from src.memory.semantic_search import search_memory

        mock_embed.return_value = [0.1] * 1536
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.search_similar_messages.return_value = [
            {'id': 1, 'sender_name': 'User', 'message_text': 'Hello',
             'timestamp': datetime.now(timezone.utc), 'email': 'test@test.com',
             'similarity': 0.8},
        ]

        results = search_memory("test query", email="test@test.com", use_reranking=False)

        assert len(results) == 1
        assert 'recency_factor' in results[0]
        mock_update.assert_called_once_with([1])


class TestUpdateLastRetrievedAt:
    """Test _update_last_retrieved_at uses get_db() convention."""

    @patch('src.memory.semantic_search.get_db')
    def test_uses_get_db_not_raw_psycopg2(self, mock_get_db):
        """Must use get_db() singleton, not raw psycopg2.connect()."""
        from src.memory.semantic_search import _update_last_retrieved_at

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        _update_last_retrieved_at([1, 2, 3])

        mock_get_db.assert_called_once()
        mock_db.execute.assert_called_once()

    def test_empty_ids_is_noop(self):
        """Empty message IDs should not call the database."""
        from src.memory.semantic_search import _update_last_retrieved_at

        with patch('src.memory.semantic_search.get_db') as mock_get_db:
            _update_last_retrieved_at([])
            mock_get_db.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 1: Embedding pruning task
# ---------------------------------------------------------------------------

class TestEmbeddingPruningTask:
    """Test embedding_pruning_task.py safety rails and logic."""

    def test_embedding_pruning_task_exists(self):
        """The embedding pruning task file should exist."""
        task_path = os.path.join('src', 'tasks', 'embedding_pruning_task.py')
        assert os.path.exists(task_path), "embedding_pruning_task.py should exist"

    def test_embedding_pruning_has_safety_rails(self):
        """The task must have safety rails for high importance and fact references."""
        task_path = os.path.join('src', 'tasks', 'embedding_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        # Safety rail: never archive high importance
        assert 'EMBEDDING_IMPORTANCE_THRESHOLD' in source, \
            "Must have configurable importance threshold"
        assert 'importance >=' in source or 'high_importance' in source, \
            "Must check for high-importance messages"

        # Safety rail: check fact references
        assert 'archived_at IS NULL' in source, \
            "Must check active (non-archived) fact references"
        assert 'FACTS' in source or 'facts' in source, \
            "Must query facts table for references"

        # Age-based filtering
        assert 'EMBEDDING_MAX_AGE_DAYS' in source, \
            "Must have configurable max age for embeddings"

        # Retrieval recency
        assert 'last_retrieved_at' in source, \
            "Must use last_retrieved_at for retrieval frequency filtering"
        assert 'EMBEDDING_RETRIEVAL_RECENCY_DAYS' in source, \
            "Must have configurable retrieval recency window"

    def test_embedding_pruning_nullifies_not_deletes(self):
        """The task must SET embedding_vec = NULL, not DELETE rows."""
        task_path = os.path.join('src', 'tasks', 'embedding_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        assert 'embedding_vec = NULL' in source, \
            "Must nullify embedding_vec, not delete the message row"
        # Should not have DELETE FROM messages
        assert 'DELETE FROM' not in source or 'messages' not in source.split('DELETE FROM')[-1][:50], \
            "Must never DELETE message rows"

    def test_embedding_pruning_supports_dry_run(self):
        """The task must support dry_run mode."""
        task_path = os.path.join('src', 'tasks', 'embedding_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        assert 'dry_run' in source, "Must support dry_run parameter"
        assert 'not dry_run' in source or 'and not dry_run' in source, \
            "Must check dry_run before making changes"

    def test_embedding_pruning_uses_get_db(self):
        """The task must use get_db() singleton, not raw psycopg2.connect()."""
        task_path = os.path.join('src', 'tasks', 'embedding_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        assert 'get_db()' in source, "Must use get_db() for database access"
        assert 'psycopg2.connect(' not in source, \
            "Must NOT use raw psycopg2.connect() — use get_db() convention"

    def test_embedding_pruning_uses_parameterized_queries(self):
        """SQL interval values must use parameterized queries, not f-string interpolation."""
        task_path = os.path.join('src', 'tasks', 'embedding_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        # Should use INTERVAL '1 day' * %s pattern, not f-string interpolation
        assert "INTERVAL '1 day' * %s" in source, \
            "Must use parameterized INTERVAL queries (INTERVAL '1 day' * %s)"
        # Should NOT have f-string interpolated intervals
        assert "INTERVAL '{" not in source, \
            "Must NOT use f-string interpolation for SQL INTERVAL values"


class TestGraphitiPruningTask:
    """Test graphiti_pruning_task.py safety rails and logic."""

    def test_graphiti_pruning_task_exists(self):
        """The graphiti pruning task file should exist."""
        task_path = os.path.join('src', 'tasks', 'graphiti_pruning_task.py')
        assert os.path.exists(task_path), "graphiti_pruning_task.py should exist"

    def test_graphiti_pruning_has_safety_rails(self):
        """The task must preserve high-importance edges and entity nodes."""
        task_path = os.path.join('src', 'tasks', 'graphiti_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        # Configurable retention periods
        assert 'EPISODE_MAX_AGE_DAYS' in source, \
            "Must have configurable episode max age"
        assert 'EDGE_TTL_DAYS' in source, \
            "Must have configurable edge TTL"

        # Importance threshold for edge protection
        assert 'EDGE_IMPORTANCE_THRESHOLD' in source, \
            "Must have configurable edge importance threshold"

        # Episodic node type filtering
        assert 'Episodic' in source, \
            "Must target Episodic nodes for cleanup"

    def test_graphiti_pruning_supports_dry_run(self):
        """The task must support dry_run mode."""
        task_path = os.path.join('src', 'tasks', 'graphiti_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        assert 'dry_run' in source, "Must support dry_run parameter"
        assert 'not dry_run' in source, "Must check dry_run before deleting"

    def test_graphiti_pruning_uses_actual_delete_counts(self):
        """Deletion counts must come from result.consume().counters, not estimated counts (issue #59)."""
        task_path = os.path.join('src', 'tasks', 'graphiti_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        # After DETACH DELETE, the code must read from consume().counters
        assert 'consume().counters.nodes_deleted' in source, \
            "Episode deletion count must come from result.consume().counters.nodes_deleted"
        assert 'consume().counters.relationships_deleted' in source, \
            "Edge deletion count must come from result.consume().counters.relationships_deleted"

        # Must NOT assign the estimated count variables to the removed counters
        lines = source.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            # The old bug: episodes_removed = old_episodes or edges_removed = expired_edges
            if 'episodes_removed = old_episodes' in stripped:
                pytest.fail(
                    "episodes_removed must not be set from old_episodes estimate — "
                    "use result.consume().counters.nodes_deleted"
                )
            if 'edges_removed = expired_edges' in stripped:
                pytest.fail(
                    "edges_removed must not be set from expired_edges estimate — "
                    "use result.consume().counters.relationships_deleted"
                )

    def test_graphiti_pruning_returns_actual_counts_from_driver(self):
        """The pruning task must return actual deletion counts from Neo4j, not estimates (issue #59)."""
        from src.tasks.graphiti_pruning_task import prune_old_episodes

        # Build mock Neo4j counters that differ from the COUNT estimates
        mock_episode_counters = MagicMock()
        mock_episode_counters.nodes_deleted = 3  # actual: 3 deleted

        mock_edge_counters = MagicMock()
        mock_edge_counters.relationships_deleted = 7  # actual: 7 deleted

        mock_episode_summary = MagicMock()
        mock_episode_summary.counters = mock_episode_counters

        mock_edge_summary = MagicMock()
        mock_edge_summary.counters = mock_edge_counters

        mock_episode_result = MagicMock()
        mock_episode_result.consume.return_value = mock_episode_summary

        mock_edge_result = MagicMock()
        mock_edge_result.consume.return_value = mock_edge_summary

        # Track which query is being run to return appropriate mocks
        call_count = {'n': 0}

        def mock_session_run(query, **kwargs):
            call_count['n'] += 1
            n = call_count['n']
            result = MagicMock()

            if n == 1:
                # Total episode count
                record = MagicMock()
                record.__getitem__ = lambda self, key: 100
                result.single.return_value = record
            elif n == 2:
                # Total edge count
                record = MagicMock()
                record.__getitem__ = lambda self, key: 200
                result.single.return_value = record
            elif n == 3:
                # Old episode count estimate (5, but only 3 will actually be deleted)
                record = MagicMock()
                record.__getitem__ = lambda self, key: 5
                result.single.return_value = record
            elif n == 4:
                # Expired edge count estimate (10, but only 7 will actually be deleted)
                record = MagicMock()
                record.__getitem__ = lambda self, key: 10
                result.single.return_value = record
            elif n == 5:
                # Step 4a: remove low-importance edges from old episodes
                pass
            elif n == 6:
                # Step 4b: delete orphaned episode nodes — return mock with counters
                return mock_episode_result
            elif n == 7:
                # Step 5: DELETE expired low-importance edges — return mock with counters
                return mock_edge_result
            return result

        mock_session = MagicMock()
        mock_session.run = mock_session_run
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        with patch('src.tasks.graphiti_pruning_task._get_neo4j_driver', return_value=mock_driver):
            # __wrapped__ gives us the raw function without Celery's bound-task self injection
            result = prune_old_episodes.__wrapped__(dry_run=False)

        # The key assertion: returned counts must be from the driver, not estimates
        assert result['episodes_removed'] == 3, \
            f"Expected 3 (actual), got {result['episodes_removed']} — must use counters.nodes_deleted, not estimate"
        assert result['edges_removed'] == 7, \
            f"Expected 7 (actual), got {result['edges_removed']} — must use counters.relationships_deleted, not estimate"
        # Estimates should still be reported separately
        assert result['old_episodes'] == 5
        assert result['expired_low_importance_edges'] == 10

    def test_graphiti_pruning_only_deletes_episodes_and_edges(self):
        """The task must not delete Entity nodes — only episodes and edges."""
        task_path = os.path.join('src', 'tasks', 'graphiti_pruning_task.py')
        with open(task_path) as f:
            source = f.read()

        # Should target Episodic nodes and RELATES_TO edges, not Entity nodes
        assert 'DETACH DELETE e' in source or 'DELETE e' in source, \
            "Must delete episode nodes"
        assert 'RELATES_TO' in source, \
            "Must target RELATES_TO edges for cleanup"
        # Should NOT have "DELETE" on Entity nodes
        lines = source.split('\n')
        for line in lines:
            if 'DELETE' in line and 'Entity' in line and not line.strip().startswith('#'):
                pytest.fail("Must never delete Entity nodes directly")


# ---------------------------------------------------------------------------
# Phase 3: Storage metrics
# ---------------------------------------------------------------------------

class TestStorageMetrics:
    """Test _get_storage_metrics in context_builder.py."""

    def test_storage_metrics_function_exists(self):
        """The _get_storage_metrics function should exist in context_builder."""
        from src.core.conversation.context_builder import _get_storage_metrics
        assert callable(_get_storage_metrics)

    def test_storage_metrics_returns_dict_on_db_failure(self):
        """When DB is unavailable, should return empty dict (not raise)."""
        from src.core.conversation.context_builder import _get_storage_metrics

        # In test env with no real DB, this should still return gracefully
        metrics = _get_storage_metrics()
        assert isinstance(metrics, dict)

    def test_storage_metrics_queries_embeddings(self):
        """The function should query embedding_vec stats from messages table."""
        import inspect
        from src.core.conversation.context_builder import _get_storage_metrics

        source = inspect.getsource(_get_storage_metrics)
        assert 'embedding_vec' in source, "Must query embedding_vec column"
        assert 'embedding_count' in source, "Must return embedding_count"
        assert 'total_messages' in source, "Must return total_messages"
        assert 'oldest_embedding' in source, "Must return oldest_embedding"
        assert 'avg_embedding_age_days' in source, "Must return avg_embedding_age_days"

    def test_storage_metrics_uses_get_db(self):
        """The function must use get_db() singleton, not raw psycopg2.connect()."""
        import inspect
        from src.core.conversation.context_builder import _get_storage_metrics

        source = inspect.getsource(_get_storage_metrics)
        assert 'get_db()' in source, "Must use get_db() for database access"
        assert 'psycopg2.connect(' not in source, \
            "Must NOT use raw psycopg2.connect() — use get_db() convention"

    def test_build_with_diagnostics_includes_storage_metrics(self):
        """build_with_diagnostics should include storage_metrics in output."""
        import inspect
        from src.core.conversation.context_builder import ContextBuilder

        source = inspect.getsource(ContextBuilder.build_with_diagnostics)
        assert 'storage_metrics' in source, \
            "build_with_diagnostics must include storage_metrics in diagnostics"
        assert '_get_storage_metrics' in source, \
            "build_with_diagnostics must call _get_storage_metrics()"


# ---------------------------------------------------------------------------
# Schema and Celery registration
# ---------------------------------------------------------------------------

class TestSchemaDDL:
    """Test that schema DDL includes last_retrieved_at column."""

    def test_messages_table_has_last_retrieved_at(self):
        """The messages DDL must include the last_retrieved_at column."""
        from src.database.schema_ddl import get_user_schema_ddl

        ddl = get_user_schema_ddl('test_schema')
        assert 'last_retrieved_at' in ddl


class TestCeleryRegistration:
    """Test that new tasks are registered in celery_app.py via source inspection."""

    def test_embedding_pruning_task_registered(self):
        """Embedding pruning task should be in the include list."""
        with open('src/celery_app.py') as f:
            source = f.read()
        assert 'src.tasks.embedding_pruning_task' in source

    def test_graphiti_pruning_task_registered(self):
        """Graphiti pruning task should be in the include list."""
        with open('src/celery_app.py') as f:
            source = f.read()
        assert 'src.tasks.graphiti_pruning_task' in source

    def test_embedding_pruning_beat_schedule(self):
        """Embedding pruning should be in the beat schedule."""
        with open('src/celery_app.py') as f:
            source = f.read()
        assert 'weekly-embedding-pruning' in source
        assert 'tasks.embedding_pruning.prune_stale_embeddings' in source

    def test_graphiti_pruning_beat_schedule(self):
        """Graphiti pruning should be in the beat schedule."""
        with open('src/celery_app.py') as f:
            source = f.read()
        assert 'monthly-graphiti-pruning' in source
        assert 'tasks.graphiti_pruning.prune_old_episodes' in source
