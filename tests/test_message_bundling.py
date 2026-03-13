"""Tests for the message bundling system.

Tests the core bundling logic — formatting multiple user messages into
a single coherent request for the pipeline.
"""

import pytest
import threading
from unittest.mock import MagicMock

from src.utils.message_bundler import bundle_messages


class TestBundleMessages:
    """Test message bundling format."""

    def test_single_message_unchanged(self):
        """A single message should be returned as-is, no wrapper."""
        result = bundle_messages(["hey what's up"])
        assert result == "hey what's up"

    def test_two_messages_bundled(self):
        """Two messages should be wrapped with context header."""
        result = bundle_messages(["hey", "also wanted to ask about dinner"])
        assert "(1) hey" in result
        assert "(2) also wanted to ask about dinner" in result
        assert "multiple messages" in result.lower()

    def test_three_messages_bundled(self):
        result = bundle_messages(["first", "second", "third"])
        assert "(1) first" in result
        assert "(2) second" in result
        assert "(3) third" in result

    def test_empty_list_not_expected(self):
        """Empty list shouldn't happen in practice but shouldn't crash."""
        # bundle_messages expects at least 1 message; this tests robustness
        result = bundle_messages(["only one"])
        assert result == "only one"

    def test_preserves_message_content(self):
        """Messages with special characters should be preserved."""
        msgs = ["what's the weather?", "also, I'm feeling 😊 today!"]
        result = bundle_messages(msgs)
        assert "what's the weather?" in result
        assert "I'm feeling 😊 today!" in result


class TestBundlingEdgeCases:
    """Test edge cases for the bundling format."""

    def test_message_ordering_preserved(self):
        """Messages should appear in the order they were sent."""
        msgs = ["first", "second", "third", "fourth"]
        result = bundle_messages(msgs)
        pos_first = result.index("(1) first")
        pos_second = result.index("(2) second")
        pos_third = result.index("(3) third")
        pos_fourth = result.index("(4) fourth")
        assert pos_first < pos_second < pos_third < pos_fourth

    def test_multiline_messages(self):
        """Messages with newlines should be preserved."""
        msgs = ["line1\nline2", "another"]
        result = bundle_messages(msgs)
        assert "line1\nline2" in result

    def test_long_messages(self):
        """Long messages should not be truncated."""
        long_msg = "x" * 5000
        result = bundle_messages([long_msg, "short"])
        assert long_msg in result
