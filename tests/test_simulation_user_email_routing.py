"""Regression tests for issue #102: simulation queries public schema instead
of per-user schema for messages and user_state.

Verifies that:
1. _store_message passes user_email to db.execute for schema routing
2. _save_all_states passes user_email to db.execute for schema routing
3. _process_conversation passes user_email to db.execute for schema routing
4. _get_recent_messages passes user_email to db.execute for schema routing
5. _get_all_recent_for_speaker passes user_email to db.execute
6. print_week_summary passes user_email to db.execute
7. Schedule generator uses path_utils.get_data_dir() instead of /app/data
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
    companions = companions or ['kai', 'mira']

    with patch('scripts.simulate_relationship.SimulationRunner._load_simulation_config'), \
         patch('scripts.simulate_relationship.SimulationRunner._ensure_schema'), \
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
        runner.LIFE_EVENTS = {'kai': [], 'mira': []}

    return runner


# ---------------------------------------------------------------------------
# Tests — _store_message passes user_email
# ---------------------------------------------------------------------------

class TestStoreMessageSchema:
    """_store_message must pass user_email to db.execute so the message
    lands in the correct per-user schema, not public."""

    def test_store_message_passes_user_email(self):
        """db.execute must receive user_email kwarg when storing a message."""
        runner = _make_runner()
        runner._current_conv_email = 'mira@companion.local'

        mock_db = MagicMock()
        with patch('src.database.db.get_db', return_value=mock_db):
            runner._store_message('kai', 'mira', 'hello there', conv_id=1)

        mock_db.execute.assert_called_once()
        _, kwargs = mock_db.execute.call_args
        assert 'user_email' in kwargs, (
            "_store_message must pass user_email to db.execute"
        )
        assert kwargs['user_email'] == 'mira@companion.local'

    def test_store_message_uses_conversation_email(self):
        """All messages in a conversation should use the same schema
        (the responder's email set by _run_conversation_between)."""
        runner = _make_runner()
        runner._current_conv_email = 'mira@companion.local'

        mock_db = MagicMock()
        with patch('src.database.db.get_db', return_value=mock_db):
            # Initiator sends to responder
            runner._store_message('kai', 'mira', 'hello', conv_id=1)
            # Responder replies to initiator
            runner._store_message('mira', 'kai', 'hi back', conv_id=1)

        # Both calls should use the same conversation email
        for c in mock_db.execute.call_args_list:
            _, kwargs = c
            assert kwargs['user_email'] == 'mira@companion.local'


# ---------------------------------------------------------------------------
# Tests — _save_all_states passes user_email
# ---------------------------------------------------------------------------

class TestSaveAllStatesSchema:
    """_save_all_states must pass user_email to db.execute so user_state
    updates go to the correct per-user schema."""

    def test_save_all_states_passes_user_email(self):
        """Each companion's state update must include user_email."""
        runner = _make_runner()

        mock_db = MagicMock()
        with patch('src.database.db.get_db', return_value=mock_db):
            runner._save_all_states()

        assert mock_db.execute.call_count == 2
        emails_used = [c[1]['user_email'] for c in mock_db.execute.call_args_list]
        # For kai's state, the "other" is mira, so user_email=mira@companion.local
        # For mira's state, the "other" is kai, so user_email=kai@companion.local
        assert 'mira@companion.local' in emails_used
        assert 'kai@companion.local' in emails_used


# ---------------------------------------------------------------------------
# Tests — _process_conversation passes user_email
# ---------------------------------------------------------------------------

class TestProcessConversationSchema:
    """_process_conversation must pass user_email when querying messages
    so it reads from the correct per-user schema."""

    def test_process_conversation_passes_user_email(self):
        """db.execute for reading messages must include user_email."""
        runner = _make_runner()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []  # No rows — just verify the call
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_result

        with patch('src.database.db.get_db', return_value=mock_db):
            runner._process_conversation('kai', 'mira', 1, 'mira@companion.local')

        mock_db.execute.assert_called_once()
        _, kwargs = mock_db.execute.call_args
        assert kwargs['user_email'] == 'mira@companion.local'

    def test_process_conversation_defaults_to_responder_email(self):
        """When user_email is not provided, defaults to responder's email."""
        runner = _make_runner()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_result

        with patch('src.database.db.get_db', return_value=mock_db):
            runner._process_conversation('kai', 'mira', 1)

        _, kwargs = mock_db.execute.call_args
        assert kwargs['user_email'] == 'mira@companion.local'


# ---------------------------------------------------------------------------
# Tests — _get_recent_messages passes user_email
# ---------------------------------------------------------------------------

class TestGetRecentMessagesSchema:
    """_get_recent_messages must pass user_email to db.execute."""

    def test_get_recent_messages_with_conversation_id(self):
        """When scoped to a conversation, passes user_email."""
        runner = _make_runner()
        runner._current_conv_email = 'mira@companion.local'

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_result

        with patch('src.database.db.get_db', return_value=mock_db):
            runner._get_recent_messages('kai', 'mira', conversation_id=1)

        _, kwargs = mock_db.execute.call_args
        assert kwargs['user_email'] == 'mira@companion.local'

    def test_get_recent_messages_without_conversation_id(self):
        """Cross-conversation query also passes user_email."""
        runner = _make_runner()
        runner._current_conv_email = 'mira@companion.local'

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_result

        with patch('src.database.db.get_db', return_value=mock_db):
            runner._get_recent_messages('kai', 'mira')

        _, kwargs = mock_db.execute.call_args
        assert kwargs['user_email'] == 'mira@companion.local'


# ---------------------------------------------------------------------------
# Tests — _get_all_recent_for_speaker passes user_email
# ---------------------------------------------------------------------------

class TestGetAllRecentForSpeakerSchema:
    """_get_all_recent_for_speaker must pass user_email to db.execute."""

    def test_passes_user_email(self):
        runner = _make_runner()
        runner._current_conv_email = 'mira@companion.local'

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_result

        with patch('src.database.db.get_db', return_value=mock_db):
            runner._get_all_recent_for_speaker('kai')

        _, kwargs = mock_db.execute.call_args
        assert 'user_email' in kwargs


# ---------------------------------------------------------------------------
# Tests — _run_conversation_between sets _current_conv_email
# ---------------------------------------------------------------------------

class TestConversationEmailRouting:
    """_run_conversation_between must set _current_conv_email to the
    responder's email so all operations route to the same schema."""

    def test_sets_current_conv_email_to_responder(self):
        """_current_conv_email must be set to responder@companion.local."""
        runner = _make_runner()

        with patch.object(runner, '_generate_message', return_value='hi'), \
             patch.object(runner, '_store_message'), \
             patch.object(runner, '_emit_message'), \
             patch.object(runner, '_process_conversation'), \
             patch('random.randint', return_value=1):
            runner._run_conversation_between('kai', 'mira')

        assert runner._current_conv_email == 'mira@companion.local'

    def test_process_conversation_receives_conv_email(self):
        """_process_conversation must be called with the conversation email."""
        runner = _make_runner()

        with patch.object(runner, '_generate_message', return_value='hi'), \
             patch.object(runner, '_store_message'), \
             patch.object(runner, '_emit_message'), \
             patch.object(runner, '_process_conversation') as mock_process, \
             patch('random.randint', return_value=1):
            runner._run_conversation_between('kai', 'mira')

        mock_process.assert_called_once()
        args = mock_process.call_args[0]
        assert args[3] == 'mira@companion.local', (
            "_process_conversation must receive the conversation email"
        )


# ---------------------------------------------------------------------------
# Tests — print_week_summary passes user_email
# ---------------------------------------------------------------------------

class TestPrintWeekSummarySchema:
    """print_week_summary must pass user_email to db.execute calls."""

    def test_passes_user_email_for_message_count(self):
        """Message count query must include user_email."""
        runner = _make_runner()
        runner._day_costs = []

        mock_result = MagicMock()
        mock_result.fetchone.return_value = {'cnt': 0}
        mock_result.fetchall.return_value = []
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_result

        with patch('src.database.db.get_db', return_value=mock_db):
            runner.print_week_summary(0)

        # Should have calls for both companions, all with user_email
        for c in mock_db.execute.call_args_list:
            _, kwargs = c
            assert 'user_email' in kwargs, (
                "All db.execute calls in print_week_summary must include user_email"
            )


# ---------------------------------------------------------------------------
# Tests — schedule generator path resolution
# ---------------------------------------------------------------------------

class TestScheduleGeneratorPaths:
    """Schedule generator must use path_utils.get_data_dir() instead of
    hardcoded /app/data when DATA_DIR env var is not set."""

    def test_daily_plans_dir_not_hardcoded_to_app(self):
        """DAILY_PLANS_DIR must not default to /app/data when outside Docker."""
        # Remove DATA_DIR env var if set, to test default behavior
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('DATA_DIR', None)

            # Re-import to pick up the new default
            import importlib
            from src.utils import path_utils
            data_dir = path_utils.get_data_dir()

            from src.scheduling import calendar_schedule_generator as csg
            importlib.reload(csg)

            # If /app/data doesn't exist, DAILY_PLANS_DIR should NOT start with /app
            if not os.path.exists('/app/data'):
                assert not csg.DAILY_PLANS_DIR.startswith('/app'), (
                    f"DAILY_PLANS_DIR should not use /app/data outside Docker, "
                    f"got: {csg.DAILY_PLANS_DIR}"
                )
                assert data_dir in csg.DAILY_PLANS_DIR

    def test_calendar_id_file_not_hardcoded_to_app(self):
        """CALENDAR_ID_FILE must not default to /app/data when outside Docker."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('DATA_DIR', None)

            import importlib
            from src.utils import path_utils
            data_dir = path_utils.get_data_dir()

            from src.scheduling import calendar_schedule_generator as csg
            importlib.reload(csg)

            if not os.path.exists('/app/data'):
                assert not csg.CALENDAR_ID_FILE.startswith('/app'), (
                    f"CALENDAR_ID_FILE should not use /app/data outside Docker, "
                    f"got: {csg.CALENDAR_ID_FILE}"
                )
                assert data_dir in csg.CALENDAR_ID_FILE

    def test_data_dir_env_var_overrides(self):
        """DATA_DIR env var should still take priority when set."""
        with patch.dict(os.environ, {'DATA_DIR': '/custom/data'}):
            import importlib
            from src.scheduling import calendar_schedule_generator as csg
            importlib.reload(csg)

            assert csg.DAILY_PLANS_DIR.startswith('/custom/data')
            assert csg.CALENDAR_ID_FILE.startswith('/custom/data')
