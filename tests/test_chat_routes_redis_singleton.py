"""Regression test for issue #58: per-call Redis connections in chat_routes.py.

The bug: ``chat_routes.py`` created a **new** ``redis.from_url()`` connection on
every WebSocket connect and disconnect event, which is wasteful and can exhaust
Redis connection limits under load.

The fix: A module-level ``_get_redis()`` singleton (matching the pattern in
``conversation_compressor.py``) so all presence-key operations share one client.
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import isolation — stub heavy deps so chat_routes can be imported in the
# test environment (same approach as test_message_coalescing.py).
# ---------------------------------------------------------------------------

_STUBS_NEEDED = [
    'arrow', 'psycopg2', 'psycopg2.extras', 'psycopg2.pool',
    'pgvector', 'pgvector.psycopg2', 'neo4j', 'redis',
    'flask_socketio',
    'src.database.db', 'src.utils.timezone_utils',
    'src.core.conversation.pipeline', 'src.config.persona_config',
    'src.core.user_context',
]

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

import src.routes.chat_routes as chat_routes_mod  # noqa: E402

# Restore sys.modules
for _mod_name in _was_absent:
    sys.modules.pop(_mod_name, None)
for _mod_name, _orig in _originals.items():
    sys.modules[_mod_name] = _orig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRedisSingleton:
    """_get_redis() must return the same client instance every time."""

    def setup_method(self):
        # Reset singleton before each test
        chat_routes_mod._redis_client = None

    def test_returns_same_instance(self):
        """Calling _get_redis() twice must return the exact same object."""
        fake_client = MagicMock(name="redis_client")

        with patch.object(
            chat_routes_mod._redis_mod, 'from_url', return_value=fake_client
        ) as mock_from_url:
            first = chat_routes_mod._get_redis()
            second = chat_routes_mod._get_redis()

            assert first is second, (
                "_get_redis() must return the same Redis client instance"
            )
            # from_url should only be called once (singleton)
            mock_from_url.assert_called_once()

    def test_thread_safe_singleton(self):
        """_get_redis() must be thread-safe — concurrent calls produce one client."""
        fake_client = MagicMock(name="redis_client")
        barrier = threading.Barrier(4)
        results = [None] * 4

        def call_get_redis(idx):
            barrier.wait()
            results[idx] = chat_routes_mod._get_redis()

        with patch.object(
            chat_routes_mod._redis_mod, 'from_url', return_value=fake_client
        ) as mock_from_url:
            threads = [threading.Thread(target=call_get_redis, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert all(r is fake_client for r in results), (
                "All threads must receive the same Redis client instance"
            )
            mock_from_url.assert_called_once()

    def test_no_inline_redis_from_url_in_handlers(self):
        """Connect/disconnect handlers must NOT call redis.from_url() inline.

        We inspect the source to ensure there are no remaining inline
        ``redis.from_url(...)`` calls outside the singleton helper and the
        pub/sub listener (which legitimately needs its own connection).
        """
        source_path = (
            Path(__file__).parent.parent / 'src' / 'routes' / 'chat_routes.py'
        )
        source = source_path.read_text()

        lines = source.splitlines()
        violations = []
        in_get_redis = False
        in_approval_listener = False

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Track when we enter/leave the _get_redis function
            if 'def _get_redis' in line:
                in_get_redis = True
                continue
            if in_get_redis:
                if stripped and not line[0].isspace():
                    in_get_redis = False
                else:
                    continue

            # Track when we enter/leave start_approval_listener
            if 'def start_approval_listener' in line:
                in_approval_listener = True
                continue
            if in_approval_listener:
                if stripped and not line[0].isspace():
                    in_approval_listener = False
                else:
                    continue

            # Check for inline redis.from_url calls
            if 'from_url' in line and 'redis' in line.lower():
                violations.append((i, line.strip()))

        assert not violations, (
            f"Found inline redis.from_url() outside _get_redis() and "
            f"start_approval_listener():\n"
            + "\n".join(f"  line {n}: {l}" for n, l in violations)
        )
