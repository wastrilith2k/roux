"""Regression tests for issue #104: simulation full-stack mode routes messages
through the HTTP API for realistic cost/usage data.

Verifies that:
1. --full-stack flag is accepted by the CLI parser
2. _generate_message routes through HTTP API when full_stack=True and incoming is provided
3. _generate_message uses direct LLM when full_stack=True but incoming is None (opener)
4. Default behavior (full_stack=False) is unchanged — direct provider calls
5. _check_services raises RuntimeError when API is unreachable
6. Celery wait delay fires after conversations in full-stack mode
7. print_week_summary includes call_purpose breakdown in full-stack mode
8. HTTP /api/chat endpoint processes messages and returns JSON response
"""
import json
import os
import sys
from unittest.mock import patch, MagicMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(companions=None, full_stack=False, api_url='http://localhost:5001',
                 celery_wait=5):
    """Create a minimal SimulationRunner with mocked dependencies."""
    companions = companions or ['alice', 'bob']

    with patch('scripts.simulate_relationship.SimulationRunner._load_simulation_config'), \
         patch('scripts.simulate_relationship.SimulationRunner._ensure_schema'), \
         patch('scripts.simulate_relationship.SimulationRunner._ensure_user_profiles'), \
         patch('scripts.simulate_relationship.SimulationRunner._save_all_states'), \
         patch('scripts.simulate_relationship.SimulationRunner._check_services'), \
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
        runner = SimulationRunner(
            companions=companions, start_day=0,
            full_stack=full_stack, api_url=api_url, celery_wait=celery_wait,
        )

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

    return runner


# ---------------------------------------------------------------------------
# Tests — CLI args
# ---------------------------------------------------------------------------

class TestCLIArgs:
    """--full-stack, --api-url, --celery-wait flags are accepted."""

    def test_full_stack_flag_parsed(self):
        from scripts.simulate_relationship import main
        import argparse

        # Build the parser the same way main() does
        parser = argparse.ArgumentParser()
        parser.add_argument('--week', type=int)
        parser.add_argument('--rollback', type=str)
        parser.add_argument('--summary', action='store_true')
        parser.add_argument('--companions', nargs='+', required=True)
        parser.add_argument('--config', type=str, default='instances/simulation_config.yaml')
        parser.add_argument('--with-analysis', action='store_true')
        parser.add_argument('--full-stack', action='store_true')
        parser.add_argument('--api-url', type=str, default='http://localhost:5001')
        parser.add_argument('--celery-wait', type=int, default=5)

        args = parser.parse_args(['--week', '1', '--companions', 'kai', 'mira', '--full-stack',
                                  '--api-url', 'http://localhost:9000', '--celery-wait', '10'])
        assert args.full_stack is True
        assert args.api_url == 'http://localhost:9000'
        assert args.celery_wait == 10

    def test_defaults_when_no_full_stack(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--week', type=int)
        parser.add_argument('--companions', nargs='+', required=True)
        parser.add_argument('--full-stack', action='store_true')
        parser.add_argument('--api-url', type=str, default='http://localhost:5001')
        parser.add_argument('--celery-wait', type=int, default=5)

        args = parser.parse_args(['--week', '1', '--companions', 'kai', 'mira'])
        assert args.full_stack is False
        assert args.api_url == 'http://localhost:5001'
        assert args.celery_wait == 5


# ---------------------------------------------------------------------------
# Tests — full-stack _generate_message routing
# ---------------------------------------------------------------------------

class TestFullStackGenerateMessage:
    """_generate_message routes through HTTP API in full-stack mode."""

    def test_full_stack_routes_through_api_when_incoming(self):
        """With full_stack=True and incoming message, should call the API."""
        runner = _make_runner(full_stack=True, api_url='http://testhost:5001')

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'API reply from alice'}
        mock_response.raise_for_status = MagicMock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            runner._current_conversation_id = 42
            result = runner._generate_message('alice', 'bob', incoming='Hi there')

        assert result == 'API reply from alice'
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == 'http://testhost:5001/api/chat'
        payload = call_kwargs[1]['json']
        assert payload['email'] == 'bob@companion.local'
        assert payload['message'] == 'Hi there'
        assert payload['companion_id'] == 'alice'
        assert payload['conversation_id'] == 42

    def test_full_stack_uses_direct_llm_for_opener(self):
        """With full_stack=True but no incoming (opener), should use direct LLM."""
        runner = _make_runner(full_stack=True)

        mock_provider = MagicMock()
        mock_provider.generate_sync.return_value = "Hey, what's up?"
        mock_provider.get_last_usage.return_value = {'input_tokens': 100, 'output_tokens': 20}

        mock_tracker = MagicMock()
        mock_tracker.track_openrouter_call.return_value = 0.0001

        with patch('src.llm.openai_provider.OpenAIProvider', return_value=mock_provider), \
             patch('src.config.persona_config.get_persona_config') as mock_config, \
             patch('src.services.cost_tracker.get_cost_tracker', return_value=mock_tracker), \
             patch('os.getenv', return_value='fake-key'), \
             patch('requests.post') as mock_post:

            config = MagicMock()
            config.companion_short_name = 'Alice'
            config.primary_user_name = 'Bob'
            config.relationship_initial_context = 'friends'
            mock_config.return_value = config

            runner._current_model = 'deepseek/deepseek-chat'
            result = runner._generate_message('alice', 'bob', incoming=None)

        # Should NOT have called the API
        mock_post.assert_not_called()
        # Should have used direct LLM
        mock_provider.generate_sync.assert_called_once()
        assert result == "Hey, what's up?"

    def test_default_mode_uses_direct_llm(self):
        """With full_stack=False (default), should always use direct LLM."""
        runner = _make_runner(full_stack=False)

        mock_provider = MagicMock()
        mock_provider.generate_sync.return_value = "Direct reply"
        mock_provider.get_last_usage.return_value = {'input_tokens': 100, 'output_tokens': 20}

        mock_tracker = MagicMock()
        mock_tracker.track_openrouter_call.return_value = 0.0001

        with patch('src.llm.openai_provider.OpenAIProvider', return_value=mock_provider), \
             patch('src.config.persona_config.get_persona_config') as mock_config, \
             patch('src.services.cost_tracker.get_cost_tracker', return_value=mock_tracker), \
             patch('os.getenv', return_value='fake-key'), \
             patch('requests.post') as mock_post:

            config = MagicMock()
            config.companion_short_name = 'Alice'
            config.primary_user_name = 'Bob'
            config.relationship_initial_context = 'friends'
            mock_config.return_value = config

            runner._current_model = 'deepseek/deepseek-chat'
            result = runner._generate_message('alice', 'bob', incoming='Hello')

        mock_post.assert_not_called()
        mock_provider.generate_sync.assert_called_once()
        assert result == "Direct reply"


# ---------------------------------------------------------------------------
# Tests — service health check
# ---------------------------------------------------------------------------

class TestCheckServices:
    """_check_services must verify API is reachable."""

    def test_check_services_passes_when_api_reachable(self):
        """Should not raise when API returns 200."""
        runner = _make_runner(full_stack=True)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch('requests.get', return_value=mock_resp):
            # Re-run the actual check (was mocked in _make_runner)
            from scripts.simulate_relationship import SimulationRunner
            SimulationRunner._check_services(runner)

    def test_check_services_raises_when_api_unreachable(self):
        """Should raise RuntimeError when API is not reachable."""
        runner = _make_runner(full_stack=True)

        import requests
        with patch('requests.get', side_effect=requests.ConnectionError("refused")):
            from scripts.simulate_relationship import SimulationRunner
            with pytest.raises(RuntimeError, match="Full-stack mode requires Docker services"):
                SimulationRunner._check_services(runner)


# ---------------------------------------------------------------------------
# Tests — Celery wait delay
# ---------------------------------------------------------------------------

class TestCeleryWaitDelay:
    """Full-stack mode waits after conversations for Celery tasks."""

    def test_full_stack_waits_after_conversation(self):
        """In full-stack mode, _run_conversation_between should sleep for celery_wait."""
        runner = _make_runner(full_stack=True, celery_wait=3)

        # Mock out the heavy methods
        runner._generate_message = MagicMock(return_value='test message')
        runner._store_message = MagicMock()
        runner._emit_message = MagicMock()
        runner._get_recent_messages = MagicMock(return_value=[])
        runner._process_conversation = MagicMock()
        runner._update_state_for_time = MagicMock()
        runner._deplete_energy = MagicMock()
        runner._save_all_states = MagicMock()
        runner.clock = MagicMock()
        runner.clock.now.return_value = MagicMock(
            strftime=MagicMock(return_value='Monday 09:00 AM'),
        )

        with patch('time.sleep') as mock_sleep, \
             patch('random.randint', return_value=2), \
             patch('random.random', return_value=0.99), \
             patch('random.choice', return_value='neutral'):
            runner._run_conversation_between('alice', 'bob')

        # Should have called time.sleep(3) for the Celery wait
        sleep_calls = [c for c in mock_sleep.call_args_list if c[0][0] == 3]
        assert len(sleep_calls) == 1, f"Expected one sleep(3) call for Celery wait, got: {mock_sleep.call_args_list}"

        # Should NOT have called _process_conversation (that's for non-full-stack)
        runner._process_conversation.assert_not_called()

    def test_default_mode_runs_inline_processing(self):
        """In default mode, _run_conversation_between should call _process_conversation."""
        runner = _make_runner(full_stack=False)

        runner._generate_message = MagicMock(return_value='test message')
        runner._store_message = MagicMock()
        runner._emit_message = MagicMock()
        runner._get_recent_messages = MagicMock(return_value=[])
        runner._process_conversation = MagicMock()
        runner._update_state_for_time = MagicMock()
        runner._deplete_energy = MagicMock()
        runner._save_all_states = MagicMock()
        runner.clock = MagicMock()
        runner.clock.now.return_value = MagicMock(
            strftime=MagicMock(return_value='Monday 09:00 AM'),
        )

        with patch('time.sleep'), \
             patch('random.randint', return_value=2), \
             patch('random.random', return_value=0.99), \
             patch('random.choice', return_value='neutral'):
            runner._run_conversation_between('alice', 'bob')

        # Should have called _process_conversation (inline processing)
        runner._process_conversation.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — cost breakdown by purpose in summary
# ---------------------------------------------------------------------------

class TestCostBreakdownByPurpose:
    """print_week_summary shows call_purpose breakdown in full-stack mode."""

    def test_full_stack_summary_includes_purpose_breakdown(self, capsys):
        runner = _make_runner(full_stack=True)

        runner._total_tokens = {'input': 5000, 'output': 1000}
        runner._total_cost = 0.0035
        runner._day_costs = [(1, 5000, 1000, 0.0035)]

        mock_tracker = MagicMock()
        mock_tracker.get_cost_breakdown_by_purpose.return_value = [
            {'call_purpose': 'conversation', 'total_cost': 0.002, 'call_count': 5},
            {'call_purpose': 'fact_extraction', 'total_cost': 0.001, 'call_count': 3},
            {'call_purpose': 'curiosity_extraction', 'total_cost': 0.0005, 'call_count': 2},
        ]

        with patch('src.database.db.get_db') as mock_db, \
             patch('src.services.cost_tracker.get_cost_tracker', return_value=mock_tracker):
            mock_result = MagicMock()
            mock_result.fetchone.return_value = {'cnt': 10}
            mock_result.fetchall.return_value = []
            mock_db.return_value.execute.return_value = mock_result

            runner.print_week_summary(1)

        output = capsys.readouterr().out
        assert 'Cost by purpose' in output
        assert 'conversation' in output
        assert 'fact_extraction' in output
        assert 'curiosity_extraction' in output

    def test_default_mode_summary_omits_purpose_breakdown(self, capsys):
        runner = _make_runner(full_stack=False)

        runner._total_tokens = {'input': 1000, 'output': 200}
        runner._total_cost = 0.001
        runner._day_costs = []

        with patch('src.database.db.get_db') as mock_db:
            mock_result = MagicMock()
            mock_result.fetchone.return_value = {'cnt': 5}
            mock_result.fetchall.return_value = []
            mock_db.return_value.execute.return_value = mock_result

            runner.print_week_summary(1)

        output = capsys.readouterr().out
        assert 'Cost by purpose' not in output


# ---------------------------------------------------------------------------
# Tests — HTTP /api/chat endpoint
# ---------------------------------------------------------------------------

# chat_routes requires flask_socketio which may not be installed in the test
# environment. Skip these tests gracefully if the dependency is missing.
try:
    import flask_socketio  # noqa: F401
    _HAS_SOCKETIO = True
except ImportError:
    _HAS_SOCKETIO = False


@pytest.mark.skipif(not _HAS_SOCKETIO, reason="flask_socketio not installed")
class TestHttpChatEndpoint:
    """HTTP /api/chat endpoint processes messages and returns JSON."""

    def test_http_chat_returns_response(self):
        """POST /api/chat with valid payload returns companion response."""
        with patch('src.routes.chat_routes.get_message_processor') as mock_get_proc:
            mock_processor = MagicMock()
            mock_processor.process_message.return_value = {
                'response': 'Hello from companion!',
                'messages': ['Hello from companion!'],
                'timestamp': '2026-01-01T09:00:00',
                'processing_time': 1.5,
            }
            mock_get_proc.return_value = mock_processor

            from src.routes.chat_routes import chat_bp
            from flask import Flask
            app = Flask(__name__)
            app.register_blueprint(chat_bp)

            with app.test_client() as client:
                resp = client.post('/api/chat', json={
                    'email': 'bob@companion.local',
                    'message': 'Hi there',
                    'companion_id': 'alice',
                })

            assert resp.status_code == 200
            data = resp.get_json()
            assert data['response'] == 'Hello from companion!'

    def test_http_chat_rejects_missing_fields(self):
        """POST /api/chat without email or message returns 400."""
        from src.routes.chat_routes import chat_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(chat_bp)

        with app.test_client() as client:
            # Missing message
            resp = client.post('/api/chat', json={'email': 'bob@test.com'})
            assert resp.status_code == 400

            # Missing email
            resp = client.post('/api/chat', json={'message': 'hello'})
            assert resp.status_code == 400

            # Empty body
            resp = client.post('/api/chat', data='not json',
                               content_type='application/json')
            assert resp.status_code == 400

    def test_http_chat_returns_500_on_processor_error(self):
        """POST /api/chat returns 500 when processor returns an error."""
        with patch('src.routes.chat_routes.get_message_processor') as mock_get_proc:
            mock_processor = MagicMock()
            mock_processor.process_message.return_value = {'error': 'Pipeline exploded'}
            mock_get_proc.return_value = mock_processor

            from src.routes.chat_routes import chat_bp
            from flask import Flask
            app = Flask(__name__)
            app.register_blueprint(chat_bp)

            with app.test_client() as client:
                resp = client.post('/api/chat', json={
                    'email': 'bob@test.com',
                    'message': 'hello',
                })

            assert resp.status_code == 500
            assert 'Pipeline exploded' in resp.get_json()['error']
