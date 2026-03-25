"""Regression tests for issue #12: NameError from undefined `name` variable.

The bug: _assemble_prompt() and _check_image_intent() used `{name}` in f-strings
but `name` was only defined in _default_cognitive_prompt(), a separate method.
The correct variable is `_companion` (set from _pc.companion_short_name).
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


@dataclass
class FakePersonaConfig:
    companion_short_name: str = "Kai"
    companion_name: str = "Kai Tanaka"
    primary_user_name: str = "James"
    c_possessive: str = "her"
    c_pronoun_subject: str = "she"
    c_pronoun_object: str = "her"
    u_possessive: str = "his"
    u_pronoun_subject: str = "he"
    u_pronoun_object: str = "him"


def _get_fake_persona_config():
    return FakePersonaConfig()


class TestAssemblePromptNameResolution:
    """Verify _assemble_prompt uses _companion, not undefined `name`."""

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_final_reminder_uses_companion_variable(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Section 4 FINAL REMINDER must not raise NameError on `name`."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()

            # Should not raise NameError
            prompt = pipeline._assemble_prompt(context, "hello")

        # The companion name should appear in the final reminder section
        assert "Kai" in prompt
        # The old undefined `name` variable would have raised NameError,
        # so reaching this point means the bug is fixed.

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_final_reminder_contains_companion_not_literal_name(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Verify the prompt contains the actual companion name, not a Python variable repr."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()
            prompt = pipeline._assemble_prompt(context, "hey there")

        assert "You are Kai." in prompt


class TestCheckImageIntentNameResolution:
    """Verify _check_image_intent uses _companion, not undefined `name`."""

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_selfie_fallback_uses_companion_variable(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """When intimate intent is downgraded and prompt is empty, must not raise NameError."""
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()

        # Create a fake intent that will trigger the downgrade path:
        # intimate type + no intimate keywords in text + prompt that gets stripped to empty
        fake_intent = MagicMock()
        fake_intent.confidence = 0.96
        fake_intent.prompt_suggestion = "nude"  # Will be stripped by regex, leaving empty
        fake_intent.reason = "test"

        from src.core.image_intent_detector import ImageIntentType
        fake_intent.intent_type = ImageIntentType.INTIMATE

        fake_detector = MagicMock()
        fake_detector.detect.return_value = fake_intent

        with patch(
            'src.core.image_intent_detector.get_image_intent_detector',
            return_value=fake_detector
        ), patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            # This should not raise NameError
            # The intent will be downgraded to SELFIE because there are no
            # intimate keywords in the user/companion text
            result = pipeline._check_image_intent(
                user_message="send me a pic",
                companion_response="sure here you go",
                user_email="test@test.com"
            )

        # After downgrade, prompt_suggestion should use companion name
        assert "Kai" in fake_intent.prompt_suggestion
        assert "casual selfie" in fake_intent.prompt_suggestion
