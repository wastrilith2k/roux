"""Regression tests for issue #106: Batch conversation-level tasks per-conversation.

Verifies that:
1. schedule_conversation_batch debounces correctly via Redis timestamps.
2. run_conversation_batch skips when conversation is still active.
3. run_conversation_batch runs all three sub-tasks when conversation is idle.
4. message_handler dispatches the batch scheduler instead of per-message tasks.
5. always_on_service dispatches the batch scheduler instead of per-message tasks.
"""
import time
import pytest
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Debounce mechanism
# ---------------------------------------------------------------------------

class TestScheduleConversationBatch:
    """schedule_conversation_batch must store timestamp in Redis and schedule task."""

    def test_stores_timestamp_and_schedules_task(self):
        """Each call should set the Redis key and call apply_async with countdown."""
        mock_redis = MagicMock()
        mock_apply = MagicMock()

        with patch('src.tasks.conversation_batch_task._get_redis', return_value=mock_redis), \
             patch('src.tasks.conversation_batch_task.run_conversation_batch') as mock_task, \
             patch('src.tasks.conversation_batch_task.time') as mock_time:

            mock_time.time.return_value = 1234567890.0
            mock_task.apply_async = mock_apply

            from src.tasks.conversation_batch_task import schedule_conversation_batch
            schedule_conversation_batch('test@example.com')

            # Verify Redis key was set
            mock_redis.set.assert_called_once_with(
                'conv_batch:last_msg:test@example.com',
                '1234567890.0'
            )

            # Verify task was scheduled with countdown
            mock_apply.assert_called_once()
            call_kwargs = mock_apply.call_args.kwargs
            assert call_kwargs['countdown'] == 300  # default IDLE_SECONDS
            assert call_kwargs['args'] == ['test@example.com', '1234567890.0']

    def test_handles_redis_failure_gracefully(self):
        """Should log warning but not raise if Redis is unavailable."""
        with patch('src.tasks.conversation_batch_task._get_redis',
                   side_effect=Exception("Redis down")):
            from src.tasks.conversation_batch_task import schedule_conversation_batch
            # Should not raise
            schedule_conversation_batch('test@example.com')


class TestRunConversationBatchDebounce:
    """run_conversation_batch must skip if conversation is still active."""

    def test_skips_when_newer_message_exists(self):
        """If Redis timestamp differs from scheduled_at, skip processing."""
        mock_redis = MagicMock()
        # Simulate a newer message having arrived
        mock_redis.get.return_value = b'9999999999.0'

        with patch('src.tasks.conversation_batch_task._get_redis', return_value=mock_redis):
            from src.tasks.conversation_batch_task import run_conversation_batch

            # Call the underlying function directly (bypass Celery)
            result = run_conversation_batch.run('test@example.com', '1234567890.0')

            assert result['status'] == 'skipped'
            assert result['reason'] == 'conversation_still_active'

    def test_runs_when_timestamp_matches(self):
        """If Redis timestamp matches scheduled_at, run batch processing."""
        mock_redis = MagicMock()
        mock_redis.get.side_effect = [
            b'1234567890.0',  # last_msg check - matches
            None,             # last_completed check - no previous batch
        ]

        with patch('src.tasks.conversation_batch_task._get_redis', return_value=mock_redis), \
             patch('src.tasks.conversation_batch_task._get_conversation_messages', return_value=[]), \
             patch('src.tasks.conversation_batch_task.logger'):

            from src.tasks.conversation_batch_task import run_conversation_batch

            result = run_conversation_batch.run('test@example.com', '1234567890.0')

            assert result['status'] == 'no_messages'

    def test_runs_all_three_tasks_on_idle(self):
        """When idle, should run episode tracking, event synthesis, and outcome."""
        mock_redis = MagicMock()
        mock_redis.get.side_effect = [
            b'1234567890.0',  # last_msg check - matches
            b'1234567800.0',  # last_completed - previous batch
        ]

        mock_messages = [
            {'id': 1, 'sender_name': 'TestUser', 'message_text': 'Hello there', 'timestamp': None},
            {'id': 2, 'sender_name': 'TestCompanion', 'message_text': 'Hi! How are you?', 'timestamp': None},
            {'id': 3, 'sender_name': 'TestUser', 'message_text': 'I am good', 'timestamp': None},
        ]

        mock_pc = MagicMock()
        mock_pc.primary_user_name = 'TestUser'
        mock_pc.companion_short_name = 'TestCompanion'

        with patch('src.tasks.conversation_batch_task._get_redis', return_value=mock_redis), \
             patch('src.tasks.conversation_batch_task._get_conversation_messages', return_value=mock_messages), \
             patch('src.tasks.conversation_batch_task._batch_episode_tracking', return_value={'status': 'success'}) as mock_ep, \
             patch('src.tasks.conversation_batch_task._batch_event_synthesis', return_value={'status': 'no_event'}) as mock_ev, \
             patch('src.tasks.conversation_batch_task._batch_interaction_outcome', return_value={'status': 'success'}) as mock_io, \
             patch('src.config.persona_config.get_persona_config', return_value=mock_pc):

            from src.tasks.conversation_batch_task import run_conversation_batch

            result = run_conversation_batch.run('test@example.com', '1234567890.0')

            assert result['status'] == 'success'

            # Verify all three sub-tasks were called
            mock_ep.assert_called_once()
            mock_ev.assert_called_once()
            mock_io.assert_called_once()

            # Verify event synthesis got the full transcript
            ev_call_args = mock_ev.call_args
            transcript = ev_call_args.args[1]
            assert 'TestUser: Hello there' in transcript
            assert 'TestCompanion: Hi! How are you?' in transcript

            # Verify last_completed was updated
            mock_redis.set.assert_called()


# ---------------------------------------------------------------------------
# Message handler integration
# ---------------------------------------------------------------------------

class TestMessageHandlerBatchIntegration:
    """message_handler must call schedule_conversation_batch instead of per-message tasks."""

    def test_handler_dispatches_batch_not_per_message(self):
        """Verify the three per-message task imports are gone and batch is used."""
        with open('src/handlers/message_handler.py', 'r') as f:
            source = f.read()

        # These per-message task dispatches should NOT be present
        assert 'detect_and_synthesize_events.delay' not in source, \
            "event_synthesis is still dispatched per-message in message_handler"
        assert 'process_message_episode.delay' not in source, \
            "episode_tracking is still dispatched per-message in message_handler"
        assert 'analyze_interaction_outcome.delay' not in source, \
            "interaction_outcome is still dispatched per-message in message_handler"

        # The batch scheduler SHOULD be present
        assert 'schedule_conversation_batch' in source, \
            "schedule_conversation_batch not found in message_handler"


# ---------------------------------------------------------------------------
# Always-on service integration
# ---------------------------------------------------------------------------

class TestAlwaysOnServiceBatchIntegration:
    """always_on_service must call schedule_conversation_batch instead of per-message tasks."""

    def test_service_dispatches_batch_not_per_message(self):
        """Verify the per-message task dispatches are replaced with batch scheduler."""
        with open('src/autonomy/always_on_service.py', 'r') as f:
            source = f.read()

        # These per-message task dispatches should NOT be present
        assert 'detect_and_synthesize_events.delay' not in source, \
            "event_synthesis is still dispatched per-message in always_on_service"
        assert 'process_message_episode.delay' not in source, \
            "episode_tracking is still dispatched per-message in always_on_service"

        # The batch scheduler SHOULD be present
        assert 'schedule_conversation_batch' in source, \
            "schedule_conversation_batch not found in always_on_service"


# ---------------------------------------------------------------------------
# Batch interaction outcome
# ---------------------------------------------------------------------------

class TestBatchInteractionOutcome:
    """_batch_interaction_outcome should analyze the full conversation at once."""

    def test_disabled_when_feature_flag_off(self):
        """Should return disabled status when COMPANION_OUTCOME_TRACKING_ENABLED=false."""
        with patch.dict('os.environ', {'COMPANION_OUTCOME_TRACKING_ENABLED': 'false'}):
            from src.tasks.conversation_batch_task import _batch_interaction_outcome
            result = _batch_interaction_outcome('test@example.com', [], [])
            assert result['status'] == 'disabled'

    def test_skips_without_messages(self):
        """Should skip when there are no user or companion messages."""
        with patch.dict('os.environ', {'COMPANION_OUTCOME_TRACKING_ENABLED': 'true'}):
            from src.tasks.conversation_batch_task import _batch_interaction_outcome
            result = _batch_interaction_outcome('test@example.com', [], [])
            assert result['status'] == 'skipped'

    def test_makes_single_llm_call(self):
        """Should make exactly ONE LLM call for the whole conversation."""
        mock_pc = MagicMock()
        mock_pc.primary_user_name = 'James'
        mock_pc.companion_short_name = 'Roux'

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db._get_connection.return_value = mock_conn

        with patch.dict('os.environ', {'COMPANION_OUTCOME_TRACKING_ENABLED': 'true'}), \
             patch('src.config.persona_config.get_persona_config', return_value=mock_pc), \
             patch('src.llm.provider_factory.generate_sync', return_value="ACTION_TYPE: general\nTOPIC: chat\nENGAGEMENT: engaged\nCONTINUED: yes\nRESONANCE: 0.5\nNOTES: Good conversation") as mock_gen, \
             patch('src.llm.provider_factory.get_resilient_provider_chain', return_value=MagicMock()), \
             patch('src.services.cost_tracker.track_llm_call'), \
             patch('src.database.db.get_db', return_value=mock_db):

            from src.tasks.conversation_batch_task import _batch_interaction_outcome

            user_msgs = [(1, "Hello"), (3, "I'm good"), (5, "Tell me more")]
            companion_msgs = [(2, "Hi!"), (4, "Great to hear")]

            result = _batch_interaction_outcome('test@example.com', user_msgs, companion_msgs)

            assert result['status'] == 'success'
            # Exactly one LLM call for the entire conversation
            assert mock_gen.call_count == 1

    def test_uses_schema_aware_connection(self):
        """Regression: _batch_interaction_outcome must use get_db()._get_connection(user_email=...)
        for per-user schema routing, NOT raw psycopg2.connect(). See issue #102."""
        mock_pc = MagicMock()
        mock_pc.primary_user_name = 'James'
        mock_pc.companion_short_name = 'Roux'

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db._get_connection.return_value = mock_conn

        with patch.dict('os.environ', {'COMPANION_OUTCOME_TRACKING_ENABLED': 'true'}), \
             patch('src.config.persona_config.get_persona_config', return_value=mock_pc), \
             patch('src.llm.provider_factory.generate_sync', return_value="ACTION_TYPE: general\nTOPIC: chat\nENGAGEMENT: engaged\nCONTINUED: yes\nRESONANCE: 0.5\nNOTES: Good conversation"), \
             patch('src.llm.provider_factory.get_resilient_provider_chain', return_value=MagicMock()), \
             patch('src.services.cost_tracker.track_llm_call'), \
             patch('src.database.db.get_db', return_value=mock_db):

            from src.tasks.conversation_batch_task import _batch_interaction_outcome

            user_msgs = [(1, "Hello"), (3, "I'm good")]
            companion_msgs = [(2, "Hi!"), (4, "Great to hear")]

            result = _batch_interaction_outcome('user@example.com', user_msgs, companion_msgs)

            assert result['status'] == 'success'

            # CRITICAL: Must use get_db()._get_connection with user_email for schema routing
            mock_db._get_connection.assert_called_once_with(user_email='user@example.com')

            # Verify the INSERT was executed via the schema-aware connection
            mock_cursor.execute.assert_called_once()
            sql = mock_cursor.execute.call_args.args[0]
            assert 'INSERT INTO' in sql
            assert 'interaction_outcomes' in sql.lower()

    def test_no_raw_psycopg2_in_batch_outcome(self):
        """Source-level check: _batch_interaction_outcome must NOT use psycopg2.connect()."""
        import inspect
        from src.tasks.conversation_batch_task import _batch_interaction_outcome
        source = inspect.getsource(_batch_interaction_outcome)
        assert 'psycopg2.connect' not in source, \
            "_batch_interaction_outcome must use get_db()._get_connection(), not raw psycopg2.connect()"


# ---------------------------------------------------------------------------
# Batch event synthesis
# ---------------------------------------------------------------------------

class TestBatchEventSynthesis:
    """_batch_event_synthesis should scan the full transcript once."""

    def test_no_event_returns_early(self):
        """Should return no_event when no triggers found in transcript."""
        with patch('src.memory.synthesized_events.detect_event_type', return_value=None):
            from src.tasks.conversation_batch_task import _batch_event_synthesis
            result = _batch_event_synthesis('test@example.com', 'Just chatting about the weather', [1, 2])
            assert result['status'] == 'no_event'

    def test_skip_subject_returns_skipped(self):
        """Should skip when subject extraction returns SKIP."""
        with patch('src.memory.synthesized_events.detect_event_type', return_value=('crisis', ['emergency'])), \
             patch('src.memory.synthesized_events.extract_event_subject', return_value='SKIP'):
            from src.tasks.conversation_batch_task import _batch_event_synthesis
            result = _batch_event_synthesis('test@example.com', 'The cat ran away', [1])
            assert result['status'] == 'skipped'


# ---------------------------------------------------------------------------
# Batch episode tracking
# ---------------------------------------------------------------------------

class TestBatchEpisodeTracking:
    """_batch_episode_tracking should create/link episodes for the full conversation."""

    def test_links_all_messages_to_episode(self):
        """Should link every message in the conversation to the episode."""
        mock_store = MagicMock()
        mock_episode = MagicMock()
        mock_episode.episode_id = 'test-uuid'
        mock_episode.topic = 'general chat'
        mock_episode.message_count = 0
        mock_episode.emotional_state = 'neutral'
        mock_store.get_current_episode.return_value = mock_episode

        mock_pc = MagicMock()
        mock_pc.primary_user_name = 'TestUser'
        mock_pc.companion_short_name = 'TestCompanion'

        messages = [
            {'id': 1, 'sender_name': 'TestUser', 'message_text': 'Hello'},
            {'id': 2, 'sender_name': 'TestCompanion', 'message_text': 'Hi!'},
            {'id': 3, 'sender_name': 'TestUser', 'message_text': 'How are you?'},
            {'id': 4, 'sender_name': 'TestCompanion', 'message_text': 'Great!'},
        ]
        user_msgs = [(1, 'Hello'), (3, 'How are you?')]
        companion_msgs = [(2, 'Hi!'), (4, 'Great!')]

        with patch('src.memory.episodic_episodes.get_episode_store', return_value=mock_store), \
             patch('src.memory.episodic_episodes.detect_episode_boundary', return_value=(False, 'continuation', 'general chat')), \
             patch('src.memory.episodic_episodes.detect_topic'), \
             patch('src.memory.episodic_episodes.detect_emotional_state'):

            from src.tasks.conversation_batch_task import _batch_episode_tracking

            result = _batch_episode_tracking(
                'test@example.com', messages, user_msgs, companion_msgs, mock_pc
            )

            assert result['status'] == 'success'
            assert result['messages_linked'] == 4

            # Verify all 4 messages were linked
            assert mock_store.add_message_to_episode.call_count == 4
            linked_ids = [c.args[1] for c in mock_store.add_message_to_episode.call_args_list]
            assert linked_ids == [1, 2, 3, 4]
