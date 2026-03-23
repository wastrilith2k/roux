"""Tests for message coalescing / bundling in the chat route layer.

Regression tests for GitHub issue #5: messages queued while the pipeline was
past its cancel checkpoints were silently dropped, leaving the user with no
response for those messages.

Strategy: we test _process_with_bundling in isolation.  Since chat_routes.py
pulls in heavy transitive dependencies (database, Firebase, etc.) that aren't
installed in the test environment, we temporarily inject stubs into sys.modules
before importing the module, and clean everything up afterward so other tests
are unaffected.
"""

import importlib
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import isolation — stub out heavy deps, import chat_routes, then restore.
# ---------------------------------------------------------------------------

# Modules we need to stub for the chat_routes import chain.
_STUBS_NEEDED = [
    'arrow', 'psycopg2', 'psycopg2.extras', 'psycopg2.pool',
    'pgvector', 'pgvector.psycopg2', 'neo4j', 'redis',
    'flask_socketio',
    'src.database.db', 'src.utils.timezone_utils',
    'src.core.conversation.pipeline', 'src.config.persona_config',
    'src.core.user_context',
]

# Save originals so we can restore later
_originals = {}
_was_absent = []

for _mod_name in _STUBS_NEEDED:
    if _mod_name in sys.modules:
        _originals[_mod_name] = sys.modules[_mod_name]
    else:
        _was_absent.append(_mod_name)
        if _mod_name == 'src.database.db':
            _m = MagicMock()
            _m.get_db.return_value = MagicMock()
            sys.modules[_mod_name] = _m
        elif _mod_name == 'src.utils.timezone_utils':
            _m = MagicMock()
            _m.now_pacific_naive.return_value = MagicMock(
                isoformat=lambda: '2026-03-23T12:00:00'
            )
            sys.modules[_mod_name] = _m
        elif _mod_name == 'src.core.conversation.pipeline':
            _m = MagicMock()

            class _FakePipelineCancelled(Exception):
                pass

            _m.PipelineCancelled = _FakePipelineCancelled
            _m.get_conversation_handler.return_value = MagicMock()
            sys.modules[_mod_name] = _m
        else:
            sys.modules[_mod_name] = MagicMock()

# Now import what we need
from src.routes.chat_routes import (  # noqa: E402
    _process_with_bundling,
    _user_processing,
    _user_message_queue,
)
from src.utils.message_bundler import bundle_messages  # noqa: E402

# Restore sys.modules so other tests aren't affected
for _mod_name in _was_absent:
    sys.modules.pop(_mod_name, None)
for _mod_name, _orig in _originals.items():
    sys.modules[_mod_name] = _orig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_processor(responses, cancel_delay=None):
    """
    Build a fake MessageProcessor whose process_message() returns *responses*
    in order.  If *cancel_delay* is set, the first call blocks for that many
    seconds (simulating an LLM call) before returning.
    """
    call_count = 0
    calls = []

    class FakeProcessor:
        def process_message(self, data, email, sid, cancel_check=None):
            nonlocal call_count
            idx = call_count
            call_count += 1
            calls.append(dict(data=dict(data), email=email, sid=sid))

            if idx == 0 and cancel_delay:
                time.sleep(cancel_delay)

            if idx < len(responses):
                return responses[idx]
            return {'response': 'fallback', 'messages': ['fallback']}

    return FakeProcessor(), calls


def _make_response(text):
    return {
        'response': text,
        'messages': [text],
        'avatar_url': None,
        'timestamp': '2026-03-23T12:00:00',
        'processing_time': 0.1,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMessageCoalescingRetry:
    """Verify that queued messages are never silently dropped."""

    def test_queued_message_processed_after_successful_completion(self):
        """
        Regression for #5: message B arrives while the pipeline is past its
        cancel checkpoints processing message A.  The pipeline finishes
        normally for A.  B must still be processed afterward.
        """
        processor, calls = _make_fake_processor(
            [_make_response('reply to A'), _make_response('reply to B')],
            cancel_delay=0.3,
        )

        socketio = MagicMock()
        email = 'user@test.com'
        sid = 'sid-1'

        with patch(
            'src.routes.chat_routes.get_message_processor',
            return_value=processor,
        ):

            def queue_message_b():
                for _ in range(50):
                    if email in _user_processing:
                        break
                    time.sleep(0.01)
                _user_message_queue.setdefault(email, []).append('message B')
                _user_processing[email].set()

            t = threading.Thread(target=queue_message_b)
            t.start()
            _process_with_bundling(socketio, {'message': 'message A'}, email, sid)
            t.join(timeout=5)

        # Both messages must have been processed
        assert len(calls) == 2, (
            f"Expected 2 process_message calls, got {len(calls)}. "
            "Queued message was likely dropped."
        )
        assert calls[0]['data']['message'] == 'message A'
        assert 'message B' in calls[1]['data']['message']

        emitted = [
            c for c in socketio.emit.call_args_list if c[0][0] == 'message'
        ]
        assert len(emitted) == 2, (
            f"Expected 2 message emissions, got {len(emitted)}."
        )

    def test_cancelled_pipeline_bundles_messages(self):
        """
        When the pipeline IS cancelled (hits a cancel checkpoint), both
        messages should be bundled into one retry.
        """
        processor, calls = _make_fake_processor(
            [{'cancelled': True}, _make_response('reply to A+B')],
        )

        socketio = MagicMock()
        email = 'user2@test.com'
        sid = 'sid-2'

        # Pre-queue message B
        _user_message_queue[email] = ['message B']

        with patch(
            'src.routes.chat_routes.get_message_processor',
            return_value=processor,
        ):
            _process_with_bundling(socketio, {'message': 'message A'}, email, sid)

        assert len(calls) == 2
        bundled_msg = calls[1]['data']['message']
        assert 'message A' in bundled_msg
        assert 'message B' in bundled_msg

        emitted = [
            c for c in socketio.emit.call_args_list if c[0][0] == 'message'
        ]
        assert len(emitted) == 1

    def test_no_queued_messages_exits_cleanly(self):
        """When no messages are queued, the loop exits after one pass."""
        processor, calls = _make_fake_processor([_make_response('hello')])
        socketio = MagicMock()

        with patch(
            'src.routes.chat_routes.get_message_processor',
            return_value=processor,
        ):
            _process_with_bundling(socketio, {'message': 'hi'}, 'u@t.com', 's')

        assert len(calls) == 1
        emitted = [
            c for c in socketio.emit.call_args_list if c[0][0] == 'message'
        ]
        assert len(emitted) == 1

    def test_rapid_fire_three_messages(self):
        """
        Rapid-fire A, B, C: pipeline finishes A normally, then B+C should
        be bundled together in a single follow-up call.
        """
        processor, calls = _make_fake_processor(
            [_make_response('reply A'), _make_response('reply B+C')],
            cancel_delay=0.3,
        )

        socketio = MagicMock()
        email = 'user3@test.com'
        sid = 'sid-3'

        with patch(
            'src.routes.chat_routes.get_message_processor',
            return_value=processor,
        ):

            def queue_bc():
                for _ in range(50):
                    if email in _user_processing:
                        break
                    time.sleep(0.01)
                _user_message_queue.setdefault(email, []).extend(
                    ['message B', 'message C']
                )
                _user_processing[email].set()

            t = threading.Thread(target=queue_bc)
            t.start()
            _process_with_bundling(socketio, {'message': 'message A'}, email, sid)
            t.join(timeout=5)

        assert len(calls) == 2
        bundled = calls[1]['data']['message']
        assert 'message B' in bundled
        assert 'message C' in bundled

    def test_cleanup_after_processing(self):
        """After all processing, user state dicts should be cleaned up."""
        processor, _ = _make_fake_processor([_make_response('done')])
        socketio = MagicMock()
        email = 'cleanup@test.com'

        with patch(
            'src.routes.chat_routes.get_message_processor',
            return_value=processor,
        ):
            _process_with_bundling(socketio, {'message': 'hi'}, email, 's')

        assert email not in _user_processing
        assert email not in _user_message_queue
