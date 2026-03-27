"""Regression tests for issue #97: call context columns on cost tracking tables.

Verifies that:
1. Fresh-install schema creates tables with context columns.
2. track_*_call methods store call_purpose, conversation_id, message_id, companion_id.
3. Migration script adds columns to existing databases idempotently.
4. Pipeline _track_llm_cost passes context through to tracker.
5. Simulation _track_simulation_cost passes call_purpose and companion_id.
6. get_cost_breakdown_by_purpose returns grouped results.
7. Tool-detection calls are tagged with call_purpose='tool_detection'.
"""
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path):
    """Create a CostTracker with an isolated SQLite database."""
    db_path = str(tmp_path / "test_costs.db")
    from src.services.cost_tracker import CostTracker
    return CostTracker(db_path=db_path)


def _query_row(db_path: str, table: str):
    """Fetch a single row from a table as a dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(f"SELECT * FROM {table}").fetchone()
    conn.close()
    return row


def _query_all(db_path: str, table: str):
    """Fetch all rows from a table as dicts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Schema: fresh install has context columns
# ---------------------------------------------------------------------------

class TestSchemaHasContextColumns:
    """Fresh-install schema must include the four new columns."""

    def test_openrouter_usage_has_context_columns(self, tracker):
        conn = sqlite3.connect(tracker.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(openrouter_usage)")
        col_names = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert 'call_purpose' in col_names
        assert 'conversation_id' in col_names
        assert 'message_id' in col_names
        assert 'companion_id' in col_names

    def test_fireworks_usage_has_context_columns(self, tracker):
        conn = sqlite3.connect(tracker.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(fireworks_usage)")
        col_names = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert 'call_purpose' in col_names
        assert 'conversation_id' in col_names
        assert 'message_id' in col_names
        assert 'companion_id' in col_names

    def test_openai_usage_has_context_columns(self, tracker):
        conn = sqlite3.connect(tracker.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(openai_usage)")
        col_names = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert 'call_purpose' in col_names
        assert 'conversation_id' in col_names
        assert 'message_id' in col_names
        assert 'companion_id' in col_names


# ---------------------------------------------------------------------------
# Tracking methods store context columns
# ---------------------------------------------------------------------------

class TestContextColumnsPopulated:
    """track_* methods must write context columns to usage tables."""

    def test_track_openrouter_stores_context(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000,
            completion_tokens=500,
            model="deepseek/deepseek-chat",
            call_purpose='conversation',
            conversation_id='conv-123',
            message_id='msg-456',
            companion_id='roux',
        )

        row = _query_row(tracker.db_path, "openrouter_usage")
        assert row['call_purpose'] == 'conversation'
        assert row['conversation_id'] == 'conv-123'
        assert row['message_id'] == 'msg-456'
        assert row['companion_id'] == 'roux'

    def test_track_fireworks_stores_context(self, tracker):
        tracker.track_fireworks_call(
            user_id="alice@example.com",
            prompt_tokens=1000,
            completion_tokens=500,
            call_purpose='classification',
            companion_id='companion-a',
        )

        row = _query_row(tracker.db_path, "fireworks_usage")
        assert row['call_purpose'] == 'classification'
        assert row['companion_id'] == 'companion-a'
        assert row['conversation_id'] is None
        assert row['message_id'] is None

    def test_track_openai_stores_context(self, tracker):
        tracker.track_openai_call(
            user_id="alice@example.com",
            prompt_tokens=500,
            completion_tokens=100,
            service_type='tool_detection',
            call_purpose='tool_detection',
            companion_id='roux',
        )

        row = _query_row(tracker.db_path, "openai_usage")
        assert row['call_purpose'] == 'tool_detection'
        assert row['companion_id'] == 'roux'

    def test_default_call_purpose_is_conversation(self, tracker):
        """When call_purpose is not provided, it defaults to 'conversation'."""
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=100,
            completion_tokens=50,
        )

        row = _query_row(tracker.db_path, "openrouter_usage")
        assert row['call_purpose'] == 'conversation'

    def test_simulation_purpose_stored(self, tracker):
        """Simulation calls should be tagged with call_purpose='simulation'."""
        tracker.track_openrouter_call(
            user_id="roux@simulation",
            prompt_tokens=1000,
            completion_tokens=500,
            call_purpose='simulation',
            companion_id='roux',
        )

        row = _query_row(tracker.db_path, "openrouter_usage")
        assert row['call_purpose'] == 'simulation'
        assert row['companion_id'] == 'roux'


# ---------------------------------------------------------------------------
# Migration script: adds columns to existing databases
# ---------------------------------------------------------------------------

class TestMigrationScript:
    """Migration must add columns idempotently to existing databases."""

    def _create_old_schema_db(self, db_path: str):
        """Create a database with the old schema (no context columns)."""
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE openrouter_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                model TEXT,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                response_time_ms INTEGER,
                error BOOLEAN DEFAULT 0
            );
            CREATE TABLE fireworks_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                model TEXT,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                endpoint TEXT,
                response_time_ms INTEGER,
                error BOOLEAN DEFAULT 0
            );
            CREATE TABLE openai_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                model TEXT,
                service_type TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                audio_seconds REAL DEFAULT 0,
                characters INTEGER DEFAULT 0,
                cost_usd REAL NOT NULL,
                error BOOLEAN DEFAULT 0
            );
        """)
        # Insert a row to verify data is preserved
        conn.execute(
            "INSERT INTO openrouter_usage "
            "(user_id, model, prompt_tokens, completion_tokens, total_tokens, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("alice@example.com", "deepseek/deepseek-chat", 100, 50, 150, 0.001),
        )
        conn.commit()
        conn.close()

    def test_migration_adds_columns(self, tmp_path):
        db_path = str(tmp_path / "old_costs.db")
        self._create_old_schema_db(db_path)

        from migrations.add_cost_context_columns import migrate
        migrate(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for table in ("openrouter_usage", "fireworks_usage", "openai_usage"):
            cursor.execute(f"PRAGMA table_info({table})")
            col_names = {row[1] for row in cursor.fetchall()}
            assert 'call_purpose' in col_names, f"{table} missing call_purpose"
            assert 'conversation_id' in col_names, f"{table} missing conversation_id"
            assert 'message_id' in col_names, f"{table} missing message_id"
            assert 'companion_id' in col_names, f"{table} missing companion_id"

        conn.close()

    def test_migration_preserves_existing_data(self, tmp_path):
        db_path = str(tmp_path / "old_costs.db")
        self._create_old_schema_db(db_path)

        from migrations.add_cost_context_columns import migrate
        migrate(db_path)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM openrouter_usage").fetchone()
        conn.close()

        assert row['user_id'] == 'alice@example.com'
        assert row['prompt_tokens'] == 100
        assert row['call_purpose'] == 'conversation'  # DEFAULT value

    def test_migration_is_idempotent(self, tmp_path):
        db_path = str(tmp_path / "old_costs.db")
        self._create_old_schema_db(db_path)

        from migrations.add_cost_context_columns import migrate
        migrate(db_path)
        migrate(db_path)  # Second run should not raise

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(openrouter_usage)")
        col_names = [row[1] for row in cursor.fetchall()]
        conn.close()

        # No duplicate columns
        assert col_names.count('call_purpose') == 1


# ---------------------------------------------------------------------------
# Pipeline integration: _track_llm_cost passes context
# ---------------------------------------------------------------------------

class TestPipelinePassesContext:
    """Pipeline._track_llm_cost must forward context kwargs to tracker."""

    def _make_pipeline(self):
        with patch.dict('os.environ', {
            'FIREWORKS_API_KEY': 'test',
            'ENVIRONMENT': 'test',
        }):
            with patch('src.core.conversation.context_builder.get_context_builder', return_value=MagicMock()), \
                 patch('src.core.memory_validation_agent.get_memory_validation_agent', return_value=MagicMock()), \
                 patch('src.core.message_validator_agent.get_message_validator_agent', return_value=MagicMock()):
                from src.core.conversation.pipeline import ConversationPipeline
                return ConversationPipeline()

    def test_track_llm_cost_passes_conversation_context(self):
        pipeline = self._make_pipeline()
        pipeline._last_chain_usage = {'input_tokens': 1000, 'output_tokens': 500}
        pipeline._last_chain_provider_type = 'openrouter'

        with patch.dict('os.environ', {'COMPANION_ID': 'roux'}):
            with patch('src.services.cost_tracker.get_cost_tracker') as mock_get:
                mock_tracker = MagicMock()
                mock_get.return_value = mock_tracker

                pipeline._track_llm_cost('alice@example.com', 'deepseek/deepseek-chat')

                mock_tracker.track_openrouter_call.assert_called_once_with(
                    user_id='alice@example.com',
                    prompt_tokens=1000,
                    completion_tokens=500,
                    model='deepseek/deepseek-chat',
                    call_purpose='conversation',
                    conversation_id=None,
                    message_id=None,
                    companion_id='roux',
                )

    def test_track_llm_cost_passes_custom_purpose(self):
        pipeline = self._make_pipeline()
        pipeline._last_chain_usage = {'input_tokens': 500, 'output_tokens': 200}
        pipeline._last_chain_provider_type = 'fireworks'

        with patch.dict('os.environ', {'COMPANION_ID': 'test-companion'}):
            with patch('src.services.cost_tracker.get_cost_tracker') as mock_get:
                mock_tracker = MagicMock()
                mock_get.return_value = mock_tracker

                pipeline._track_llm_cost(
                    'alice@example.com', 'test-model',
                    call_purpose='summary',
                    conversation_id='conv-abc',
                )

                mock_tracker.track_fireworks_call.assert_called_once()
                call_kwargs = mock_tracker.track_fireworks_call.call_args
                assert call_kwargs.kwargs.get('call_purpose') or call_kwargs[1].get('call_purpose') == 'summary'

    def test_track_llm_cost_openai_passes_context(self):
        pipeline = self._make_pipeline()
        pipeline._last_chain_usage = {'input_tokens': 300, 'output_tokens': 100}
        pipeline._last_chain_provider_type = 'openai'

        with patch.dict('os.environ', {'COMPANION_ID': 'roux'}):
            with patch('src.services.cost_tracker.get_cost_tracker') as mock_get:
                mock_tracker = MagicMock()
                mock_get.return_value = mock_tracker

                pipeline._track_llm_cost('alice@example.com', 'gpt-4o-mini')

                mock_tracker.track_openai_call.assert_called_once_with(
                    user_id='alice@example.com',
                    prompt_tokens=300,
                    completion_tokens=100,
                    service_type='chat',
                    model='gpt-4o-mini',
                    call_purpose='conversation',
                    conversation_id=None,
                    message_id=None,
                    companion_id='roux',
                )


# ---------------------------------------------------------------------------
# Simulation integration: passes call_purpose and companion_id
# ---------------------------------------------------------------------------

class TestSimulationPassesContext:
    """Simulation _track_simulation_cost must tag with purpose and companion."""

    def test_simulation_tags_openrouter_call(self):
        with patch('scripts.simulate_relationship.SimulationRunner._load_simulation_config'), \
             patch('scripts.simulate_relationship.SimulationRunner._ensure_user_profiles'), \
             patch('scripts.simulate_relationship.SimulationRunner._save_all_states'), \
             patch('scripts.simulate_relationship.SimulationEventEmitter'), \
             patch('src.core.clock.SimulationClock') as MockClock, \
             patch('src.core.clock.set_clock'):
            mock_clock = MockClock.return_value
            mock_clock.now.return_value = MagicMock(
                strftime=MagicMock(return_value='Monday 09:00 AM'),
                isoformat=MagicMock(return_value='2026-01-01T09:00:00-08:00'),
                replace=MagicMock(return_value=MagicMock()),
            )
            from scripts.simulate_relationship import SimulationRunner
            runner = SimulationRunner(companions=['alice', 'bob'], start_day=0)

        mock_provider = MagicMock()
        mock_provider.get_last_usage.return_value = {
            'input_tokens': 1000, 'output_tokens': 200,
        }

        mock_tracker = MagicMock()
        mock_tracker.track_openrouter_call.return_value = 0.001

        with patch('src.services.cost_tracker.get_cost_tracker', return_value=mock_tracker):
            runner._track_simulation_cost(mock_provider, 'deepseek/deepseek-chat', 'alice')

        mock_tracker.track_openrouter_call.assert_called_once_with(
            user_id='alice@simulation',
            prompt_tokens=1000,
            completion_tokens=200,
            model='deepseek/deepseek-chat',
            call_purpose='simulation',
            companion_id='alice',
        )


# ---------------------------------------------------------------------------
# Cost breakdown by purpose
# ---------------------------------------------------------------------------

class TestCostBreakdownByPurpose:
    """get_cost_breakdown_by_purpose must group costs by call_purpose."""

    def test_breakdown_groups_by_purpose(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='conversation',
        )
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=2000, completion_tokens=1000,
            call_purpose='simulation',
        )
        tracker.track_openai_call(
            user_id="alice@example.com",
            prompt_tokens=500, completion_tokens=100,
            service_type='tool_detection',
            call_purpose='tool_detection',
        )

        breakdown = tracker.get_cost_breakdown_by_purpose("alice@example.com")

        purposes = {entry['call_purpose'] for entry in breakdown}
        assert 'conversation' in purposes
        assert 'simulation' in purposes
        assert 'tool_detection' in purposes

        for entry in breakdown:
            assert entry['total_cost'] > 0
            assert entry['call_count'] >= 1

    def test_breakdown_filters_by_companion(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='conversation', companion_id='roux',
        )
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='conversation', companion_id='other',
        )

        breakdown = tracker.get_cost_breakdown_by_purpose(
            "alice@example.com", companion_id='roux',
        )

        assert len(breakdown) == 1
        assert breakdown[0]['call_count'] == 1


# ---------------------------------------------------------------------------
# get_cost_summary filters by call_purpose and companion_id
# ---------------------------------------------------------------------------

class TestGetCostSummaryFilters:
    """get_cost_summary must actually filter when call_purpose/companion_id are given."""

    def test_filters_by_call_purpose(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='conversation',
        )
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=2000, completion_tokens=1000,
            call_purpose='simulation',
        )

        # Unfiltered should include both
        summary_all = tracker.get_cost_summary("alice@example.com")
        assert summary_all.total_today > 0

        # Filter to simulation only
        summary_sim = tracker.get_cost_summary(
            "alice@example.com", call_purpose='simulation',
        )
        summary_conv = tracker.get_cost_summary(
            "alice@example.com", call_purpose='conversation',
        )

        # Filtered totals must be less than unfiltered
        assert summary_sim.total_today < summary_all.total_today
        assert summary_conv.total_today < summary_all.total_today

        # Sum of filtered parts should equal the whole
        assert abs(
            summary_sim.total_today + summary_conv.total_today
            - summary_all.total_today
        ) < 1e-9

    def test_filters_by_companion_id(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            companion_id='roux',
        )
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            companion_id='other',
        )

        summary_roux = tracker.get_cost_summary(
            "alice@example.com", companion_id='roux',
        )
        summary_other = tracker.get_cost_summary(
            "alice@example.com", companion_id='other',
        )
        summary_all = tracker.get_cost_summary("alice@example.com")

        # Each filtered summary should be roughly half
        assert summary_roux.total_today > 0
        assert summary_other.total_today > 0
        assert abs(
            summary_roux.total_today + summary_other.total_today
            - summary_all.total_today
        ) < 1e-9

    def test_filters_by_purpose_and_companion(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='conversation', companion_id='roux',
        )
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='simulation', companion_id='roux',
        )
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='conversation', companion_id='other',
        )

        summary = tracker.get_cost_summary(
            "alice@example.com",
            call_purpose='conversation',
            companion_id='roux',
        )

        # Should only match one of the three calls
        summary_all = tracker.get_cost_summary("alice@example.com")
        assert summary.total_today > 0
        assert summary.total_today < summary_all.total_today

    def test_filtered_month_costs_match_today(self, tracker):
        """Filtered month costs should include today's filtered data."""
        tracker.track_fireworks_call(
            user_id="alice@example.com",
            prompt_tokens=500, completion_tokens=200,
            call_purpose='tool_detection',
        )

        summary = tracker.get_cost_summary(
            "alice@example.com", call_purpose='tool_detection',
        )

        # Today's cost and month's cost should be equal (only one day of data)
        assert summary.total_today > 0
        assert abs(summary.total_today - summary.total_month) < 1e-9

    def test_nonexistent_filter_returns_zero(self, tracker):
        tracker.track_openrouter_call(
            user_id="alice@example.com",
            prompt_tokens=1000, completion_tokens=500,
            call_purpose='conversation',
        )

        summary = tracker.get_cost_summary(
            "alice@example.com", call_purpose='nonexistent',
        )

        assert summary.total_today == 0.0
        assert summary.total_month == 0.0
