"""Regression tests for issue #50: hardcoded pronouns and user name in LLM-facing prompts.

The bug: Multiple files contained hardcoded "His", "James", "HIS", "his", "him"
in strings sent to the LLM, violating the CLAUDE.md convention that pronouns
and user names must come from get_persona_config().
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class FakePersonaConfig:
    """Persona config with non-default pronouns to detect hardcoding."""
    companion_short_name: str = "Zara"
    companion_name: str = "Zara Chen"
    companion_email: str = ""
    companion_entity_profile: str = "companion"
    companion_pronoun_subject: str = "she"
    companion_pronoun_object: str = "her"
    companion_pronoun_possessive: str = "her"
    companion_pronoun_reflexive: str = "herself"
    primary_user_name: str = "Mika"
    primary_user_email: str = ""
    primary_user_entity_profile: str = "user"
    primary_user_timezone: str = "America/Los_Angeles"
    user_pronoun_subject: str = "she"
    user_pronoun_object: str = "her"
    user_pronoun_possessive: str = "her"
    user_pronoun_reflexive: str = "herself"
    relationship_initial_context: str = "They recently started talking."
    personality_prompt_path: str = "prompts/core/personality.md"
    cognitive_flow_prompt_path: str = "prompts/core/cognitive_flow.md"
    image_base_prompt_path: str = "src/image/prompts/base_prompt.txt"
    image_intimate_prompt_path: str = "src/image/prompts/intimate_prompt.txt"
    calendar_name: str = "Companion Schedule"
    calendar_id_file: str = "data/companion_calendar_id.txt"
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = ""
    edge_tts_fallback: str = ""
    known_entity_entries: List[Dict[str, str]] = field(default_factory=list)
    pet_names: List[str] = field(default_factory=list)
    user_children: List[str] = field(default_factory=list)
    family_names: List[str] = field(default_factory=list)
    work_names: List[str] = field(default_factory=list)
    semantic_patterns: Dict[str, List[str]] = field(default_factory=dict)
    invalid_fact_patterns: List[List[str]] = field(default_factory=list)

    @property
    def companion_pronouns(self):
        return {
            "subject": self.companion_pronoun_subject,
            "object": self.companion_pronoun_object,
            "possessive": self.companion_pronoun_possessive,
            "reflexive": self.companion_pronoun_reflexive,
        }

    @property
    def user_pronouns(self):
        return {
            "subject": self.user_pronoun_subject,
            "object": self.user_pronoun_object,
            "possessive": self.user_pronoun_possessive,
            "reflexive": self.user_pronoun_reflexive,
        }

    def get_known_entities_dict(self):
        return {}

    def get_resolved_invalid_fact_patterns(self):
        return []


def _get_fake_persona_config(**kwargs):
    return FakePersonaConfig(**kwargs)


class TestTimeAwarenessNoHardcodedPronouns:
    """Issue #50: time_awareness.py used hardcoded 'His' in routine section."""

    def test_routine_section_uses_persona_pronoun(self):
        """_get_routine_section must use user_pronoun_possessive, not hardcoded 'His'."""
        from src.core.time_awareness import TimeAwareness

        fake_activity = {
            'description': 'Working from home',
            'location': 'home office',
        }

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ), patch(
            'src.core.user_context.get_user_probable_activity',
            return_value=fake_activity
        ):
            ta = TimeAwareness()
            result = ta._get_routine_section(datetime.now())

        assert "His routine" not in result, "Hardcoded 'His' found in routine section"
        assert "Her routine" in result, "User possessive pronoun not used"
        assert "Working from home" in result
        assert "(at home office)" in result


class TestChatRoutesNoHardcodedUserName:
    """Issue #50: chat_routes.py used hardcoded 'James' in photo pipeline text."""

    def test_photo_pipeline_text_source_has_no_hardcoded_james(self):
        """The handle_image function in chat_routes must not hardcode 'James'."""
        import inspect
        # Read the source file directly to verify no hardcoded "James" in the
        # photo pipeline text construction
        from pathlib import Path
        source = Path(__file__).parent.parent / 'src' / 'routes' / 'chat_routes.py'
        content = source.read_text()

        # Find the handle_image function's pipeline_text lines
        # They should use a variable from get_persona_config(), not "James"
        assert "James sent a photo" not in content, \
            "Hardcoded 'James' found in chat_routes.py photo pipeline text"
        assert "get_persona_config" in content, \
            "chat_routes.py should use get_persona_config for user name"


class TestPipelineNoHardcodedNamesOrPronouns:
    """Issue #50: pipeline.py used hardcoded 'James', 'HIS', 'his', 'him' in prompts."""

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_assembled_prompt_uses_persona_names(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """_assemble_prompt must use persona config values, not hardcoded names/pronouns."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()
            prompt = pipeline._assemble_prompt(context, "hello")

        # User name must come from config
        assert "Mika" in prompt, "Persona user name not found in prompt"

        # Hardcoded names/pronouns must not appear in LLM-facing text
        # Check identity section specifically
        assert "= HIS," not in prompt, "Hardcoded 'HIS' found in identity section"
        assert "not his." not in prompt, "Hardcoded 'his' found in identity section"

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_proactive_prompt_uses_persona_name(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Proactive mode instructions must use persona user name, not hardcoded 'James'."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()
            extra = {'is_proactive_message': True}
            prompt = pipeline._assemble_prompt(context, "hey", extra_context=extra)

        # Proactive section must use config name and pronouns
        assert "Mika" in prompt
        assert "she didn't message you first" in prompt, \
            "User subject pronoun not used in proactive instructions"

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_voice_mode_uses_persona_name(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Voice mode instructions must use persona user name, not hardcoded 'James'."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()
            extra = {'source': 'telegram-voice'}
            prompt = pipeline._assemble_prompt(context, "hey", extra_context=extra)

        assert "James sent a voice note" not in prompt, \
            "Hardcoded 'James' found in voice mode instructions"
        assert "Mika sent a voice note" in prompt

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_closing_lines_use_persona_pronouns(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Closing lines must use persona pronouns, not hardcoded 'his'."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()
            prompt = pipeline._assemble_prompt(context, "hello")

        # Closing reminder should use persona pronouns
        assert "her facts are her" in prompt, \
            "User possessive pronoun not used in closing lines"
        assert "his facts are his" not in prompt, \
            "Hardcoded 'his' found in closing lines"
        assert "Answer her question" in prompt, \
            "User possessive pronoun not used in 'Answer ... question' line"
