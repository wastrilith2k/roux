"""Regression tests for issue #94: simulation bypasses pipeline — no cost or token tracking.

Verifies that:
1. _generate_message calls get_last_usage after generate_sync and records to cost tracker
2. _get_day_summary calls get_last_usage after generate_sync and records to cost tracker
3. Per-day and total accumulators are updated correctly
4. print_week_summary includes token and cost totals
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(companions=None):
    """Create a minimal SimulationRunner with mocked dependencies."""
    companions = companions or ['alice', 'bob']

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

    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateMessageCostTracking:
    """_generate_message must call get_last_usage and record to cost tracker."""

    def test_generate_message_tracks_cost(self):
        runner = _make_runner()

        mock_provider = MagicMock()
        mock_provider.generate_sync.return_value = "Hey there!"
        mock_provider.get_last_usage.return_value = {
            'input_tokens': 500,
            'output_tokens': 80,
        }

        mock_tracker = MagicMock()
        mock_tracker.track_openrouter_call.return_value = 0.00027

        with patch('src.llm.openai_provider.OpenAIProvider', return_value=mock_provider), \
             patch('src.config.persona_config.get_persona_config') as mock_config, \
             patch('src.services.cost_tracker.get_cost_tracker', return_value=mock_tracker), \
             patch('os.getenv', return_value='fake-key'):

            config = MagicMock()
            config.companion_short_name = 'Alice'
            config.primary_user_name = 'Bob'
            mock_config.return_value = config

            runner._current_model = 'deepseek/deepseek-chat'
            result = runner._generate_message('alice', 'bob', incoming='Hi')

        # Verify get_last_usage was called
        mock_provider.get_last_usage.assert_called_once()

        # Verify cost tracker was called with correct args
        mock_tracker.track_openrouter_call.assert_called_once_with(
            user_id='alice@simulation',
            prompt_tokens=500,
            completion_tokens=80,
            model='deepseek/deepseek-chat',
        )

        # Verify accumulators were updated
        assert runner._total_tokens['input'] == 500
        assert runner._total_tokens['output'] == 80
        assert runner._total_cost == 0.00027


class TestGetDaySummaryCostTracking:
    """_get_day_summary must call get_last_usage and record to cost tracker."""

    def test_get_day_summary_tracks_cost(self):
        runner = _make_runner()

        mock_provider = MagicMock()
        mock_provider.generate_sync.return_value = "- Alice talked about cats"
        mock_provider.get_last_usage.return_value = {
            'input_tokens': 300,
            'output_tokens': 50,
        }

        mock_tracker = MagicMock()
        mock_tracker.track_openrouter_call.return_value = 0.00017

        with patch('src.llm.openai_provider.OpenAIProvider', return_value=mock_provider), \
             patch('src.config.persona_config.get_persona_config') as mock_config, \
             patch('src.services.cost_tracker.get_cost_tracker', return_value=mock_tracker), \
             patch('time.sleep'):

            config = MagicMock()
            config.primary_user_name = 'Bob'
            mock_config.return_value = config

            # Provide recent messages so summarization is attempted
            runner._get_all_recent_for_speaker = MagicMock(return_value=[
                {'sender': 'alice', 'content': 'Hello Bob'},
                {'sender': 'bob', 'content': 'Hi Alice'},
            ])

            result = runner._get_day_summary('alice', 'bob')

        mock_provider.get_last_usage.assert_called_once()
        mock_tracker.track_openrouter_call.assert_called_once_with(
            user_id='alice@simulation',
            prompt_tokens=300,
            completion_tokens=50,
            model='deepseek/deepseek-chat',
        )

        assert runner._total_tokens['input'] == 300
        assert runner._total_tokens['output'] == 50


class TestDayCostAccumulation:
    """Per-day cost breakdown must be recorded via _reset_day_cost."""

    def test_reset_day_cost_records_entry(self):
        runner = _make_runner()

        runner._current_day = 3
        runner._day_tokens = {'input': 1000, 'output': 200}
        runner._day_cost = 0.0015

        runner._reset_day_cost()

        assert len(runner._day_costs) == 1
        day_num, t_in, t_out, cost = runner._day_costs[0]
        assert day_num == 3
        assert t_in == 1000
        assert t_out == 200
        assert cost == 0.0015

        # Accumulators should be reset
        assert runner._day_tokens == {'input': 0, 'output': 0}
        assert runner._day_cost == 0.0

    def test_reset_day_cost_skips_empty_days(self):
        runner = _make_runner()
        runner._reset_day_cost()
        assert len(runner._day_costs) == 0


class TestWeekSummaryCostOutput:
    """print_week_summary must include token and cost totals."""

    def test_summary_includes_cost_data(self, capsys):
        runner = _make_runner()

        runner._total_tokens = {'input': 5000, 'output': 1000}
        runner._total_cost = 0.0035
        runner._day_costs = [
            (1, 2500, 500, 0.0017),
            (2, 2500, 500, 0.0018),
        ]

        with patch('src.database.db.get_db') as mock_db:
            mock_result = MagicMock()
            mock_result.fetchone.return_value = {'cnt': 10}
            mock_result.fetchall.return_value = []
            mock_db.return_value.execute.return_value = mock_result

            runner.print_week_summary(1)

        output = capsys.readouterr().out

        assert 'COST SUMMARY' in output
        assert '5000 in / 1000 out' in output
        assert 'Total tokens: 6000' in output
        assert '$0.0035' in output
        assert 'Day 1:' in output
        assert 'Day 2:' in output
