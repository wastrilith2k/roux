"""Regression tests for issue #100: simulation fails — Postgres tables not
initialized and generate_daily_schedule rejects companion_id.

Verifies that:
1. SimulationRunner.__init__ calls _ensure_schema before _ensure_user_profiles
2. _ensure_schema creates public tables and per-companion user schemas
3. _run_schedule_generation does NOT pass companion_id to generate_daily_schedule
4. _run_schedule_generation passes the simulated clock time as target_date
"""
import os
import sys
from unittest.mock import patch, MagicMock, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(companions=None):
    """Create a minimal SimulationRunner with mocked dependencies."""
    companions = companions or ['alice', 'bob']

    with patch('scripts.simulate_relationship.SimulationRunner._load_simulation_config'), \
         patch('scripts.simulate_relationship.SimulationRunner._ensure_schema') as mock_schema, \
         patch('scripts.simulate_relationship.SimulationRunner._ensure_user_profiles') as mock_profiles, \
         patch('scripts.simulate_relationship.SimulationRunner._save_all_states'), \
         patch('scripts.simulate_relationship.SimulationEventEmitter'), \
         patch('src.core.clock.SimulationClock') as MockClock, \
         patch('src.core.clock.set_clock'):
        mock_clock = MockClock.return_value
        mock_clock.now.return_value = MagicMock(
            strftime=MagicMock(return_value='Monday 09:00 AM'),
            isoformat=MagicMock(return_value='2026-01-01T09:00:00-08:00'),
            replace=MagicMock(return_value=MagicMock()),
            hour=9,
        )

        from scripts.simulate_relationship import SimulationRunner
        runner = SimulationRunner(companions=companions, start_day=0)

        # Set defaults that _load_simulation_config would normally set
        runner.MODEL_ROTATION = ['deepseek/deepseek-chat']
        runner.LLM_MAX_TOKENS = 200
        runner.LLM_TEMPERATURE = 0.9
        runner.SUMMARY_MAX_TOKENS = 150
        runner.SUMMARY_TEMPERATURE = 0.5
        runner.MOODS = ['neutral']
        runner.ACTIVITIES = ['reading']
        runner.OPENER_SEEDS = ['Hello']
        runner.TONE_MODIFIERS = ['']
        runner.LIFE_EVENT_PROB = 0.0
        runner.setting = {}

    return runner, mock_schema, mock_profiles


# ---------------------------------------------------------------------------
# Tests — schema initialization
# ---------------------------------------------------------------------------

class TestSimulationSchemaInit:
    """_ensure_schema must be called during __init__ before _ensure_user_profiles."""

    def test_ensure_schema_called_during_init(self):
        """__init__ must call _ensure_schema so Postgres tables exist."""
        _runner, mock_schema, _mock_profiles = _make_runner()
        mock_schema.assert_called_once()

    def test_ensure_schema_called_before_user_profiles(self):
        """_ensure_schema must run before _ensure_user_profiles (tables must
        exist before inserting rows)."""
        call_order = []

        def record_schema(self_arg=None):
            call_order.append('schema')

        def record_profiles(self_arg=None):
            call_order.append('profiles')

        with patch('scripts.simulate_relationship.SimulationRunner._load_simulation_config'), \
             patch('scripts.simulate_relationship.SimulationRunner._ensure_schema', record_schema), \
             patch('scripts.simulate_relationship.SimulationRunner._ensure_user_profiles', record_profiles), \
             patch('scripts.simulate_relationship.SimulationRunner._save_all_states'), \
             patch('scripts.simulate_relationship.SimulationEventEmitter'), \
             patch('src.core.clock.SimulationClock') as MockClock, \
             patch('src.core.clock.set_clock'):
            mock_clock = MockClock.return_value
            mock_clock.now.return_value = MagicMock(
                isoformat=MagicMock(return_value='2026-01-01T09:00:00-08:00'),
            )

            from scripts.simulate_relationship import SimulationRunner
            SimulationRunner(companions=['alice', 'bob'], start_day=0)

        assert call_order == ['schema', 'profiles'], (
            f"Expected _ensure_schema before _ensure_user_profiles, got: {call_order}"
        )

    def test_ensure_schema_creates_public_and_user_schemas(self):
        """_ensure_schema must execute public DDL and call ensure_user_schema
        for each companion email."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch('scripts.simulate_relationship.SimulationRunner._load_simulation_config'), \
             patch('scripts.simulate_relationship.SimulationRunner._ensure_user_profiles'), \
             patch('scripts.simulate_relationship.SimulationRunner._save_all_states'), \
             patch('scripts.simulate_relationship.SimulationEventEmitter'), \
             patch('src.core.clock.SimulationClock') as MockClock, \
             patch('src.core.clock.set_clock'), \
             patch('src.database.connection.get_connection', return_value=mock_conn), \
             patch('src.database.schema_ddl.get_public_schema_ddl', return_value='CREATE TABLE ...') as mock_ddl, \
             patch('src.database.schema_manager.ensure_user_schema') as mock_ensure:

            mock_clock = MockClock.return_value
            mock_clock.now.return_value = MagicMock(
                isoformat=MagicMock(return_value='2026-01-01T09:00:00-08:00'),
            )

            from scripts.simulate_relationship import SimulationRunner
            SimulationRunner(companions=['kai', 'mira'], start_day=0)

            # Public DDL was executed
            mock_ddl.assert_called_once()
            mock_cursor.execute.assert_called_with('CREATE TABLE ...')

            # ensure_user_schema called for each companion
            assert mock_ensure.call_count == 2
            emails = [c.args[1] for c in mock_ensure.call_args_list]
            assert 'kai@companion.local' in emails
            assert 'mira@companion.local' in emails

    def test_ensure_schema_uses_connection_module(self):
        """_ensure_schema must use get_connection() from src.database.connection,
        not duplicate connection logic with raw psycopg2.connect()."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch('scripts.simulate_relationship.SimulationRunner._load_simulation_config'), \
             patch('scripts.simulate_relationship.SimulationRunner._ensure_user_profiles'), \
             patch('scripts.simulate_relationship.SimulationRunner._save_all_states'), \
             patch('scripts.simulate_relationship.SimulationEventEmitter'), \
             patch('src.core.clock.SimulationClock') as MockClock, \
             patch('src.core.clock.set_clock'), \
             patch('src.database.connection.get_connection', return_value=mock_conn) as mock_get_conn, \
             patch('src.database.schema_ddl.get_public_schema_ddl', return_value='CREATE TABLE ...'), \
             patch('src.database.schema_manager.ensure_user_schema'):

            mock_clock = MockClock.return_value
            mock_clock.now.return_value = MagicMock(
                isoformat=MagicMock(return_value='2026-01-01T09:00:00-08:00'),
            )

            from scripts.simulate_relationship import SimulationRunner
            SimulationRunner(companions=['kai'], start_day=0)

            # get_connection() from connection module must be called
            mock_get_conn.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — schedule generation
# ---------------------------------------------------------------------------

class TestScheduleGenerationNoCompanionId:
    """_run_schedule_generation must NOT pass companion_id to generate_daily_schedule."""

    def test_does_not_pass_companion_id(self):
        """generate_daily_schedule should be called without companion_id kwarg."""
        runner, _, _ = _make_runner()

        with patch('src.scheduling.calendar_schedule_generator.generate_daily_schedule') as mock_gen:
            mock_gen.return_value = {'events': []}
            runner._run_schedule_generation('alice')

            mock_gen.assert_called_once()
            _args, kwargs = mock_gen.call_args
            assert 'companion_id' not in kwargs, (
                "generate_daily_schedule should not receive companion_id"
            )

    def test_passes_simulated_clock_time(self):
        """generate_daily_schedule should receive the simulation clock's
        current time as target_date."""
        runner, _, _ = _make_runner()
        clock_time = runner.clock.now()

        with patch('src.scheduling.calendar_schedule_generator.generate_daily_schedule') as mock_gen:
            mock_gen.return_value = {'events': []}
            runner._run_schedule_generation('alice')

            mock_gen.assert_called_once_with(target_date=clock_time)

    def test_schedule_generation_does_not_raise_type_error(self):
        """Calling _run_schedule_generation must not raise TypeError for
        unexpected keyword argument 'companion_id'."""
        runner, _, _ = _make_runner()

        with patch('src.scheduling.calendar_schedule_generator.generate_daily_schedule') as mock_gen:
            # If companion_id were passed, this spec would reject it
            def strict_generate(target_date=None, force=False):
                return {'events': []}
            mock_gen.side_effect = strict_generate

            # Should not raise
            runner._run_schedule_generation('alice')
