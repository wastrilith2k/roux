"""Regression test for issue #66 — DETACH DELETE must not cascade past high-importance edges.

The old Step 4 used DETACH DELETE on Episodic nodes, which unconditionally
removed all relationships including those with importance >= 7.  The fix
splits this into two sub-steps: remove only low-importance edges first, then
delete only orphaned episode nodes.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.tasks.graphiti_pruning_task import (
    prune_old_episodes,
    HIGH_IMPORTANCE_EDGE_FLOOR,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight fake Neo4j session that records executed Cypher
# ---------------------------------------------------------------------------

class FakeResult:
    """Mimics a neo4j.Result with single() and consume()."""

    def __init__(self, record: dict | None = None, counters: dict | None = None):
        self._record = record or {}
        self._counters = counters or {}

    def single(self):
        return self._record

    def consume(self):
        summary = MagicMock()
        summary.counters.nodes_deleted = self._counters.get('nodes_deleted', 0)
        summary.counters.relationships_deleted = self._counters.get('relationships_deleted', 0)
        return summary


class RecordingSession:
    """Captures every Cypher statement so tests can assert on query text."""

    def __init__(self, run_results: list[FakeResult] | None = None):
        self.queries: list[str] = []
        self._results = list(run_results or [])
        self._call_idx = 0

    def run(self, query: str, **kwargs):
        self.queries.append(query)
        if self._call_idx < len(self._results):
            result = self._results[self._call_idx]
        else:
            result = FakeResult()
        self._call_idx += 1
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeDriver:
    """Fake Neo4j driver that yields RecordingSessions."""

    def __init__(self, sessions: list[RecordingSession]):
        self._sessions = list(sessions)
        self._idx = 0
        self.closed = False

    def session(self):
        s = self._sessions[self._idx]
        self._idx += 1
        return s

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEpisodePruningDoesNotCascadeHighImportanceEdges:
    """Verify Step 4 never issues DETACH DELETE on episode nodes."""

    def _build_driver_for_wet_run(self):
        """Build a FakeDriver that simulates a wet run with old episodes present.

        Session sequence matches the code's session usage:
          session 0: Step 1 — two queries in one session (episode count, edge count)
          session 1: Step 2 — count old episodes -> returns 5
          session 2: Step 3 — count expired edges -> returns 2
          session 3: Step 4 — two queries (4a remove edges, 4b delete orphaned nodes)
          session 4: Step 5 — remove expired low-importance edges
        """
        s0 = RecordingSession([
            FakeResult({'episode_count': 100}),
            FakeResult({'edge_count': 500}),
        ])
        s1 = RecordingSession([FakeResult({'old_episode_count': 5})])
        s2 = RecordingSession([FakeResult({'expired_edge_count': 2})])
        # Step 4 session: two queries (4a remove edges, 4b delete nodes)
        s3 = RecordingSession([
            FakeResult(),                                        # 4a
            FakeResult(counters={'nodes_deleted': 3}),           # 4b
        ])
        # Step 5 session: one query
        s4 = RecordingSession([
            FakeResult(counters={'relationships_deleted': 2}),
        ])
        return FakeDriver([s0, s1, s2, s3, s4])

    @patch('src.tasks.graphiti_pruning_task._get_neo4j_driver')
    def test_no_detach_delete_on_episodes(self, mock_get_driver):
        """DETACH DELETE must never appear in any query targeting Episodic nodes."""
        driver = self._build_driver_for_wet_run()
        mock_get_driver.return_value = driver

        # Call the underlying function directly (bypass Celery decoration)
        result = prune_old_episodes.apply(args=[], kwargs={'dry_run': False}).get()

        assert result['status'] == 'success'

        # Collect ALL queries across all sessions
        all_queries = []
        for s in driver._sessions:
            all_queries.extend(s.queries)

        for query in all_queries:
            q_upper = query.upper()
            if 'EPISODIC' in q_upper or '(E:' in q_upper.replace(' ', ''):
                assert 'DETACH DELETE' not in q_upper, (
                    f"DETACH DELETE found in episode query — this would cascade "
                    f"past the high-importance edge guard:\n{query}"
                )

    @patch('src.tasks.graphiti_pruning_task._get_neo4j_driver')
    def test_step4_filters_edges_by_importance_floor(self, mock_get_driver):
        """Step 4a must reference the importance floor when removing edges."""
        driver = self._build_driver_for_wet_run()
        mock_get_driver.return_value = driver

        prune_old_episodes.apply(args=[], kwargs={'dry_run': False}).get()

        # Step 4 is in session index 3
        step4_queries = driver._sessions[3].queries
        assert len(step4_queries) >= 2, "Expected at least two sub-steps in Step 4"

        # 4a: the edge-removal query must guard on importance
        edge_removal_query = step4_queries[0].upper()
        assert 'IMPORTANCE' in edge_removal_query, (
            "Step 4a edge removal must filter by importance to preserve high-importance edges"
        )
        assert 'DELETE' in edge_removal_query

    @patch('src.tasks.graphiti_pruning_task._get_neo4j_driver')
    def test_step4b_only_deletes_orphaned_episodes(self, mock_get_driver):
        """Step 4b must only delete episodes with no remaining relationships."""
        driver = self._build_driver_for_wet_run()
        mock_get_driver.return_value = driver

        prune_old_episodes.apply(args=[], kwargs={'dry_run': False}).get()

        step4_queries = driver._sessions[3].queries
        node_deletion_query = step4_queries[1].upper()

        # Must check for absence of relationships before deleting
        assert 'NOT' in node_deletion_query, (
            "Step 4b must check that episode has no remaining edges before deletion"
        )
        assert 'DELETE' in node_deletion_query
        assert 'DETACH' not in node_deletion_query, (
            "Step 4b must use plain DELETE, not DETACH DELETE"
        )

    @patch('src.tasks.graphiti_pruning_task._get_neo4j_driver')
    def test_dry_run_does_not_execute_delete(self, mock_get_driver):
        """Dry run must not execute any DELETE queries."""
        s0 = RecordingSession([
            FakeResult({'episode_count': 50}),
            FakeResult({'edge_count': 200}),
        ])
        s1 = RecordingSession([FakeResult({'old_episode_count': 10})])
        s2 = RecordingSession([FakeResult({'expired_edge_count': 5})])
        driver = FakeDriver([s0, s1, s2])
        mock_get_driver.return_value = driver

        result = prune_old_episodes.apply(args=[], kwargs={'dry_run': True}).get()

        assert result['status'] == 'success'
        assert result['dry_run'] is True
        assert result['episodes_removed'] == 0
        assert result['edges_removed'] == 0

        all_queries = []
        for s in driver._sessions:
            all_queries.extend(s.queries)

        for query in all_queries:
            assert 'DELETE' not in query.upper(), (
                f"Dry run must not execute DELETE queries: {query}"
            )
