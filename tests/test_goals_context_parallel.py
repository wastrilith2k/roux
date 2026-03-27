"""
Regression test for issue #63: goals_context never fetched in parallel context build path.

_build_parallel() was missing the goals_context future, so context.goals_context
was always an empty string on the primary hot path.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')


def _stub_sources(builder, overrides=None):
    """Patch all context sources on builder to return empty defaults.

    Returns a dict of {source_name: mock} so callers can assert on specific ones.
    """
    sources = {
        '_get_scene_state': "",
        '_get_presence_mode': "",
        '_get_internal_state': "",
        '_get_fertility_context': "",
        '_get_user_context': "",
        '_get_values_context': "",
        '_get_activities_context': "",
        '_get_recent_events_context': "",
        '_get_synthesized_biographies': "",
        '_get_relationship_insights': "",
        '_get_relationship_dynamics_context': "",
        '_get_relationship_evaluation_context': "",
        '_get_entity_profiles': "",
        '_get_personality': "",
        '_get_memories': ("", ""),
        '_get_graphiti_context': "",
        '_get_synthesized_events': "",
        '_get_episode_context': "",
        '_get_core_memory': "",
        '_get_observations_context': "",
        '_get_reflections_context': "",
        '_get_opinions_context': "",
        '_get_curiosity_context': "",
        '_get_goals_context': "",
        '_get_conversation_history_structured': ([], "", ""),
        '_get_time_awareness_context': "",
        '_get_derived_scene_context': "",
    }
    if overrides:
        sources.update(overrides)

    mocks = {}
    for attr, ret_val in sources.items():
        m = MagicMock(return_value=ret_val)
        setattr(builder, attr, m)
        mocks[attr] = m
    return mocks


@patch('src.database.db.get_db')
class TestGoalsContextParallel:
    """Verify _build_parallel submits _get_goals_context as a future."""

    def _make_builder(self, mock_get_db):
        mock_get_db.return_value = MagicMock()
        from src.core.conversation.context_builder import ContextBuilder
        return ContextBuilder()

    def test_parallel_build_populates_goals_context(self, mock_get_db):
        """goals_context must be non-empty when _get_goals_context returns data."""
        builder = self._make_builder(mock_get_db)
        sentinel = "- Learn to play guitar\n- Read more books"

        mocks = _stub_sources(builder, {'_get_goals_context': sentinel})

        context = builder._build_parallel("test@example.com", "hello", 50)

        mocks['_get_goals_context'].assert_called_once_with("test@example.com")
        assert context.goals_context == sentinel

    def test_parallel_build_goals_context_defaults_empty_on_none(self, mock_get_db):
        """goals_context should gracefully handle None return."""
        builder = self._make_builder(mock_get_db)

        mocks = _stub_sources(builder, {'_get_goals_context': None})

        context = builder._build_parallel("test@example.com", "hello", 50)

        mocks['_get_goals_context'].assert_called_once_with("test@example.com")
        # When result is None, the parallel collector sets "" via `result or ""`
        assert context.goals_context == ""
