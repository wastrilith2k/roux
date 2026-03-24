"""
Tests for tool reliability improvements (Issue #10, Task 1).

Covers:
- Structured tool reasoning produces correct decisions
- Verification guardrail fires for mutating external actions
- Code executor availability TTL prevents permanent disabling
- End-to-end pipeline tool path with reasoning
"""

import json
import os
import time
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')


# =========================================================================
# Tool Reasoning Unit Tests
# =========================================================================

class TestToolDecisionParsing:
    """Test that _parse_decision correctly handles LLM output."""

    def test_valid_json_parsed(self):
        from src.core.conversation.tool_reasoning import _parse_decision

        response = json.dumps({
            "reasoning": "User asked about weather",
            "needs_tool": True,
            "tool_action": "weather",
            "tool_parameters": {"location": "Portland"},
            "verification_needed": False,
            "verification_query": None,
            "confidence": 0.95,
        })

        decision = _parse_decision(response)
        assert decision.needs_tool is True
        assert decision.tool_action == "weather"
        assert decision.tool_parameters == {"location": "Portland"}
        assert decision.verification_needed is False
        assert decision.confidence == 0.95

    def test_markdown_fenced_json_parsed(self):
        from src.core.conversation.tool_reasoning import _parse_decision

        response = '```json\n{"reasoning": "casual chat", "needs_tool": false, "tool_action": null, "tool_parameters": {}, "verification_needed": false, "verification_query": null, "confidence": 0.9}\n```'

        decision = _parse_decision(response)
        assert decision.needs_tool is False
        assert decision.tool_action is None

    def test_invalid_json_returns_no_tool(self):
        from src.core.conversation.tool_reasoning import _parse_decision

        decision = _parse_decision("SKIP")
        assert decision.needs_tool is False
        assert "Could not parse" in decision.reasoning

    def test_empty_response_returns_no_tool(self):
        from src.core.conversation.tool_reasoning import _parse_decision

        decision = _parse_decision("")
        assert decision.needs_tool is False


class TestToolDecisionLogic:
    """Test that make_tool_decision produces correct decisions via mocked LLM."""

    @patch('src.llm.openai_provider.get_openai_tool_provider')
    def test_weather_query_needs_tool(self, mock_provider_fn):
        from src.core.conversation.tool_reasoning import make_tool_decision

        provider = MagicMock()
        provider.generate_sync.return_value = json.dumps({
            "reasoning": "User is asking about the weather",
            "needs_tool": True,
            "tool_action": "weather",
            "tool_parameters": {"location": "Portland"},
            "verification_needed": False,
            "verification_query": None,
            "confidence": 0.95,
        })
        mock_provider_fn.return_value = provider

        decision = make_tool_decision("What's the weather like?")
        assert decision.needs_tool is True
        assert decision.tool_action == "weather"

    @patch('src.llm.openai_provider.get_openai_tool_provider')
    def test_casual_message_skips_tool(self, mock_provider_fn):
        from src.core.conversation.tool_reasoning import make_tool_decision

        provider = MagicMock()
        provider.generate_sync.return_value = json.dumps({
            "reasoning": "This is casual conversation",
            "needs_tool": False,
            "tool_action": None,
            "tool_parameters": {},
            "verification_needed": False,
            "verification_query": None,
            "confidence": 0.9,
        })
        mock_provider_fn.return_value = provider

        decision = make_tool_decision("Hey, how are you?")
        assert decision.needs_tool is False

    @patch('src.llm.openai_provider.get_openai_tool_provider')
    def test_no_provider_defaults_to_no_tool(self, mock_provider_fn):
        from src.core.conversation.tool_reasoning import make_tool_decision

        mock_provider_fn.return_value = None

        decision = make_tool_decision("What's the weather?")
        assert decision.needs_tool is False
        assert "No OpenAI provider" in decision.reasoning


class TestVerificationGuardrail:
    """Test that mutating actions enforce verification."""

    @patch('src.llm.openai_provider.get_openai_tool_provider')
    def test_send_email_forces_verification(self, mock_provider_fn):
        from src.core.conversation.tool_reasoning import make_tool_decision

        provider = MagicMock()
        provider.generate_sync.return_value = json.dumps({
            "reasoning": "User wants to send an email",
            "needs_tool": True,
            "tool_action": "send_email",
            "tool_parameters": {"to": "test@example.com", "subject": "Hi", "body": "Hello"},
            "verification_needed": False,  # LLM says no, but we override
            "verification_query": None,
            "confidence": 0.9,
        })
        mock_provider_fn.return_value = provider

        decision = make_tool_decision("Send an email to test@example.com saying hi")
        assert decision.needs_tool is True
        assert decision.verification_needed is True  # Enforced by MUTATING_ACTIONS
        assert decision.verification_query is not None

    @patch('src.llm.openai_provider.get_openai_tool_provider')
    def test_calendar_create_forces_verification(self, mock_provider_fn):
        from src.core.conversation.tool_reasoning import make_tool_decision

        provider = MagicMock()
        provider.generate_sync.return_value = json.dumps({
            "reasoning": "User wants to add a calendar event",
            "needs_tool": True,
            "tool_action": "create_calendar_event",
            "tool_parameters": {"summary": "Meeting"},
            "verification_needed": False,
            "verification_query": None,
            "confidence": 0.85,
        })
        mock_provider_fn.return_value = provider

        decision = make_tool_decision("Add a meeting to my calendar tomorrow at 3pm")
        assert decision.verification_needed is True

    @patch('src.llm.openai_provider.get_openai_tool_provider')
    def test_web_search_no_verification(self, mock_provider_fn):
        from src.core.conversation.tool_reasoning import make_tool_decision

        provider = MagicMock()
        provider.generate_sync.return_value = json.dumps({
            "reasoning": "User wants to search for something",
            "needs_tool": True,
            "tool_action": "web_search",
            "tool_parameters": {"query": "best pizza portland"},
            "verification_needed": False,
            "verification_query": None,
            "confidence": 0.9,
        })
        mock_provider_fn.return_value = provider

        decision = make_tool_decision("Search for the best pizza in Portland")
        assert decision.verification_needed is False  # Read-only, no override


class TestBuildVerificationCode:
    """Test verification code generation."""

    def test_verification_code_generated(self):
        from src.core.conversation.tool_reasoning import ToolDecision, build_verification_code

        decision = ToolDecision(
            needs_tool=True,
            reasoning="Sending email",
            tool_action="send_email",
            verification_needed=True,
            verification_query="Check meeting time for tomorrow",
        )

        code = build_verification_code(decision)
        assert code is not None
        assert "search.web_search" in code
        assert "Check meeting time" in code

    def test_no_verification_returns_none(self):
        from src.core.conversation.tool_reasoning import ToolDecision, build_verification_code

        decision = ToolDecision(
            needs_tool=True,
            reasoning="Weather check",
            tool_action="weather",
            verification_needed=False,
        )

        assert build_verification_code(decision) is None


class TestBuildToolCode:
    """Test tool code generation from decisions."""

    def test_weather_code(self):
        from src.core.conversation.tool_reasoning import ToolDecision, build_tool_code

        decision = ToolDecision(
            needs_tool=True,
            reasoning="weather",
            tool_action="weather",
            tool_parameters={"location": "Portland"},
        )

        code = build_tool_code(decision)
        assert code is not None
        assert "weather.get_current_weather" in code
        assert "Portland" in code

    def test_search_code(self):
        from src.core.conversation.tool_reasoning import ToolDecision, build_tool_code

        decision = ToolDecision(
            needs_tool=True,
            reasoning="search",
            tool_action="web_search",
            tool_parameters={"query": "python tutorials"},
        )

        code = build_tool_code(decision)
        assert code is not None
        assert "search.web_search" in code

    def test_no_tool_returns_none(self):
        from src.core.conversation.tool_reasoning import ToolDecision, build_tool_code

        decision = ToolDecision(needs_tool=False, reasoning="casual chat")
        assert build_tool_code(decision) is None

    def test_unknown_action_returns_none(self):
        from src.core.conversation.tool_reasoning import ToolDecision, build_tool_code

        decision = ToolDecision(
            needs_tool=True,
            reasoning="unknown",
            tool_action="teleport",
            tool_parameters={},
        )

        assert build_tool_code(decision) is None


# =========================================================================
# Code Executor Availability TTL Tests
# =========================================================================

class TestCodeExecutorAvailabilityTTL:
    """Test that code executor doesn't permanently cache negative availability."""

    def test_negative_cache_expires(self):
        from src.core.code_executor import CodeExecutor

        executor = CodeExecutor(base_url="http://fake:9999")
        executor._available = False
        executor._available_checked_at = time.time() - 120  # 2 minutes ago (> 60s TTL)

        # Mock the health check to succeed this time
        with patch('src.core.code_executor.requests.get') as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert executor.is_available() is True

    def test_negative_cache_within_ttl(self):
        from src.core.code_executor import CodeExecutor

        executor = CodeExecutor(base_url="http://fake:9999")
        executor._available = False
        executor._available_checked_at = time.time() - 10  # 10 seconds ago (< 60s TTL)

        # Should return False without making any HTTP request
        with patch('src.core.code_executor.requests.get') as mock_get:
            assert executor.is_available() is False
            mock_get.assert_not_called()

    def test_positive_cache_persists(self):
        from src.core.code_executor import CodeExecutor

        executor = CodeExecutor(base_url="http://fake:9999")
        executor._available = True

        # Should return True without making any HTTP request
        with patch('src.core.code_executor.requests.get') as mock_get:
            assert executor.is_available() is True
            mock_get.assert_not_called()

    def test_first_check_makes_request(self):
        from src.core.code_executor import CodeExecutor

        executor = CodeExecutor(base_url="http://fake:9999")

        with patch('src.core.code_executor.requests.get') as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert executor.is_available() is True
            mock_get.assert_called_once()


# =========================================================================
# Feature Flag Tests
# =========================================================================

class TestToolReasoningFeatureFlag:
    """Test that tool reasoning can be toggled via env var."""

    def test_disabled_returns_no_tool(self):
        from src.core.conversation.tool_reasoning import ToolDecision

        with patch('src.core.conversation.tool_reasoning.TOOL_REASONING_ENABLED', False):
            from src.core.conversation.tool_reasoning import make_tool_decision
            decision = make_tool_decision("What's the weather?")
            assert decision.needs_tool is False
            assert "disabled" in decision.reasoning
