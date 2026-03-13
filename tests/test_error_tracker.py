"""Tests for the error tracker utility."""

import pytest
from unittest.mock import patch, MagicMock
from src.utils.error_tracker import record_error, _error_buffer


class TestRecordError:
    def setup_method(self):
        """Clear buffer before each test."""
        global _error_buffer
        import src.utils.error_tracker as et
        et._error_buffer = []

    def test_never_raises(self):
        """record_error must NEVER raise, even with broken DB."""
        with patch('src.utils.error_tracker._get_connection', side_effect=Exception("boom")):
            # Should not raise
            record_error(ValueError("test error"), module='test')

    def test_buffers_when_db_unavailable(self):
        """When DB is down, errors should be buffered in memory."""
        import src.utils.error_tracker as et
        with patch('src.utils.error_tracker._get_connection', return_value=None):
            record_error(ValueError("test error"), module='test_mod', companion_id='kai')
        assert len(et._error_buffer) == 1
        assert et._error_buffer[0]['module'] == 'test_mod'
        assert et._error_buffer[0]['companion_id'] == 'kai'
        assert et._error_buffer[0]['error_type'] == 'ValueError'

    def test_buffer_limit(self):
        """Buffer should not grow beyond _MAX_BUFFER."""
        import src.utils.error_tracker as et
        with patch('src.utils.error_tracker._get_connection', return_value=None):
            for i in range(150):
                record_error(ValueError(f"error {i}"), module='test')
        assert len(et._error_buffer) <= 100

    def test_captures_exception_type(self):
        import src.utils.error_tracker as et
        with patch('src.utils.error_tracker._get_connection', return_value=None):
            record_error(ConnectionError("network down"), module='db')
        assert et._error_buffer[0]['error_type'] == 'ConnectionError'
        assert 'network down' in et._error_buffer[0]['error_message']
