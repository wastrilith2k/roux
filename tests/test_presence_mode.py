"""
Tests for presence mode system (Issue #26).

Covers:
- PresenceMode enum values and defaults
- PresenceModeManager get/set with mocked DB
- Prompt formatting for in_person and texting modes
- Context builder integration (presence_mode field populated)
- Pipeline prompt assembly: texting mode injects behavioral constraints
- Pipeline prompt assembly: in_person mode does NOT inject texting constraints
- Scene tracker respects texting mode (clears physical_presence)
"""

import os
import json
import pytest

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

from unittest.mock import patch, MagicMock
from src.core.presence_mode import (
    PresenceMode,
    PresenceModeManager,
    DEFAULT_PRESENCE_MODE,
    get_presence_mode_manager,
)


# =========================================================================
# PresenceMode Enum Tests
# =========================================================================

class TestPresenceModeEnum:
    """Test PresenceMode enum values and behavior."""

    def test_in_person_value(self):
        assert PresenceMode.IN_PERSON.value == "in_person"

    def test_texting_value(self):
        assert PresenceMode.TEXTING.value == "texting"

    def test_default_is_in_person(self):
        assert DEFAULT_PRESENCE_MODE == PresenceMode.IN_PERSON

    def test_from_string_in_person(self):
        assert PresenceMode("in_person") == PresenceMode.IN_PERSON

    def test_from_string_texting(self):
        assert PresenceMode("texting") == PresenceMode.TEXTING

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            PresenceMode("flying")


# =========================================================================
# PresenceModeManager Tests
# =========================================================================

class TestPresenceModeManager:
    """Test PresenceModeManager get/set/format with mocked DB."""

    def _make_manager_with_mock_db(self, scene_state=None):
        """Create a manager with a mocked database returning the given scene_state."""
        manager = PresenceModeManager()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        if scene_state is not None:
            mock_cursor.fetchone.return_value = {'scene_state': scene_state}
        else:
            mock_cursor.fetchone.return_value = None

        mock_db = MagicMock()
        mock_db._get_connection.return_value = mock_conn
        manager._db = mock_db

        return manager, mock_cursor

    def test_get_returns_default_when_no_row(self):
        manager, _ = self._make_manager_with_mock_db(scene_state=None)
        mode = manager.get_presence_mode("test@example.com")
        assert mode == DEFAULT_PRESENCE_MODE

    def test_get_returns_default_when_no_presence_key(self):
        manager, _ = self._make_manager_with_mock_db(
            scene_state={'location': 'bedroom'}
        )
        mode = manager.get_presence_mode("test@example.com")
        assert mode == DEFAULT_PRESENCE_MODE

    def test_get_returns_texting_when_set(self):
        manager, _ = self._make_manager_with_mock_db(
            scene_state={'presence_mode': 'texting'}
        )
        mode = manager.get_presence_mode("test@example.com")
        assert mode == PresenceMode.TEXTING

    def test_get_returns_in_person_when_set(self):
        manager, _ = self._make_manager_with_mock_db(
            scene_state={'presence_mode': 'in_person'}
        )
        mode = manager.get_presence_mode("test@example.com")
        assert mode == PresenceMode.IN_PERSON

    def test_get_returns_default_for_invalid_mode_in_db(self):
        manager, _ = self._make_manager_with_mock_db(
            scene_state={'presence_mode': 'invalid_mode'}
        )
        mode = manager.get_presence_mode("test@example.com")
        assert mode == DEFAULT_PRESENCE_MODE

    def test_get_handles_json_string_scene_state(self):
        """scene_state may come back as a JSON string from the DB."""
        manager, _ = self._make_manager_with_mock_db(
            scene_state=json.dumps({'presence_mode': 'texting'})
        )
        mode = manager.get_presence_mode("test@example.com")
        assert mode == PresenceMode.TEXTING

    def test_set_persists_mode(self):
        manager, mock_cursor = self._make_manager_with_mock_db(
            scene_state={'location': 'kitchen'}
        )
        mock_cursor.rowcount = 1  # Simulate successful update

        result = manager.set_presence_mode("test@example.com", PresenceMode.TEXTING)
        assert result is True

        # Verify the UPDATE was called with presence_mode merged in
        update_call = mock_cursor.execute.call_args_list[-1]
        saved_json = json.loads(update_call[0][1][0])
        assert saved_json['presence_mode'] == 'texting'
        assert saved_json['location'] == 'kitchen'  # Preserved existing field

    def test_set_returns_false_when_no_row(self):
        manager, mock_cursor = self._make_manager_with_mock_db(scene_state=None)
        # fetchone returns None -> no row
        result = manager.set_presence_mode("test@example.com", PresenceMode.TEXTING)
        assert result is False


# =========================================================================
# Prompt Formatting Tests
# =========================================================================

class TestPresenceModePromptFormatting:
    """Test that format_for_prompt returns correct context strings."""

    def _make_manager_with_mode(self, mode_value):
        manager = PresenceModeManager()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {
            'scene_state': {'presence_mode': mode_value}
        }
        mock_db = MagicMock()
        mock_db._get_connection.return_value = mock_conn
        manager._db = mock_db
        return manager

    def test_texting_format_contains_texting_keyword(self):
        manager = self._make_manager_with_mode('texting')
        prompt = manager.format_for_prompt("test@example.com")
        assert "TEXTING" in prompt

    def test_texting_format_forbids_physical_actions(self):
        manager = self._make_manager_with_mode('texting')
        prompt = manager.format_for_prompt("test@example.com")
        assert "Do NOT describe physical actions" in prompt

    def test_texting_format_says_not_same_space(self):
        manager = self._make_manager_with_mode('texting')
        prompt = manager.format_for_prompt("test@example.com")
        assert "NOT in the same physical space" in prompt

    def test_in_person_format_allows_physical(self):
        manager = self._make_manager_with_mode('in_person')
        prompt = manager.format_for_prompt("test@example.com")
        assert "IN PERSON" in prompt
        assert "Physical actions" in prompt

    def test_in_person_format_does_not_forbid_actions(self):
        manager = self._make_manager_with_mode('in_person')
        prompt = manager.format_for_prompt("test@example.com")
        assert "Do NOT describe physical actions" not in prompt


# =========================================================================
# ConversationContext Integration Tests
# =========================================================================

class TestPresenceModeInContext:
    """Test that presence_mode is wired into ConversationContext."""

    def test_context_has_presence_mode_field(self):
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext()
        assert hasattr(ctx, 'presence_mode')
        assert ctx.presence_mode == ""

    def test_context_presence_mode_in_prompt_sections(self):
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext(presence_mode="Communication mode: TEXTING")
        sections = ctx.to_prompt_sections()
        assert 'presence_mode' in sections
        assert "TEXTING" in sections['presence_mode']

    def test_empty_presence_mode_not_in_sections(self):
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext(presence_mode="")
        sections = ctx.to_prompt_sections()
        assert 'presence_mode' not in sections


# =========================================================================
# Pipeline Prompt Assembly Tests
# =========================================================================

class TestPresenceModeInPipeline:
    """Test that the pipeline injects texting constraints based on presence mode."""

    def _make_pipeline(self):
        """Create a pipeline instance with minimal mocking."""
        from src.core.conversation.pipeline import ConversationPipeline

        with patch('src.core.conversation.pipeline.get_context_builder'):
            with patch('src.core.conversation.pipeline.get_memory_validation_agent'):
                with patch('src.core.conversation.pipeline.get_message_validator_agent'):
                    pipeline = ConversationPipeline.__new__(ConversationPipeline)
                    pipeline.context_builder = MagicMock()
                    pipeline.memory_agent = MagicMock()
                    pipeline.validator_agent = MagicMock()
                    pipeline.cognitive_prompt = "Be yourself."
                    pipeline._source_timings = {}
        return pipeline

    def _build_prompt(self, pipeline, presence_mode="", extra_context=None):
        """Call _assemble_prompt with a context that has the given presence_mode."""
        from src.core.conversation.context_builder import ConversationContext

        ctx = ConversationContext(
            user_email="test@example.com",
            user_message="hey",
            presence_mode=presence_mode,
            entity_profiles="Name: Test Companion",
        )

        with patch('src.core.entity_profile_loader.get_current_time_context', return_value="Time: noon"):
            with patch('src.core.conversation.checkpoint_detector.get_checkpoint_detector') as mock_cp:
                mock_cp.return_value.should_checkpoint.return_value = False
                prompt = pipeline._assemble_prompt(
                    ctx, "hey", None,
                    extra_context=extra_context
                )
        return prompt

    def test_texting_presence_injects_texting_mode(self):
        """Bug reproduction: texting mode should inject constraints even without telegram source."""
        pipeline = self._make_pipeline()
        prompt = self._build_prompt(
            pipeline,
            presence_mode="Communication mode: TEXTING (SMS/messaging)\nYou are NOT in the same physical space."
        )
        assert "<texting_mode>" in prompt
        assert "No physical actions" in prompt

    def test_in_person_does_not_inject_texting_mode(self):
        pipeline = self._make_pipeline()
        prompt = self._build_prompt(
            pipeline,
            presence_mode="Communication mode: IN PERSON\nYou are physically together."
        )
        assert "<texting_mode>" not in prompt

    def test_empty_presence_does_not_inject_texting_mode(self):
        pipeline = self._make_pipeline()
        prompt = self._build_prompt(pipeline, presence_mode="")
        assert "<texting_mode>" not in prompt

    def test_telegram_text_still_triggers_texting_mode(self):
        """Backward compatibility: telegram-text source should still work."""
        pipeline = self._make_pipeline()
        prompt = self._build_prompt(
            pipeline,
            presence_mode="",
            extra_context={'source': 'telegram-text'}
        )
        assert "<texting_mode>" in prompt

    def test_texting_mode_final_reminder_present(self):
        """Texting mode adds a reminder in the final section."""
        pipeline = self._make_pipeline()
        prompt = self._build_prompt(
            pipeline,
            presence_mode="Communication mode: TEXTING (SMS/messaging)\nDo NOT physical."
        )
        assert "TEXTING MODE: No physical actions" in prompt

    def test_in_person_no_texting_final_reminder(self):
        pipeline = self._make_pipeline()
        prompt = self._build_prompt(
            pipeline,
            presence_mode="Communication mode: IN PERSON\nPhysical actions appropriate."
        )
        assert "TEXTING MODE" not in prompt

    def test_presence_mode_in_reference_data(self):
        """Presence mode should appear in the reference_data section."""
        pipeline = self._make_pipeline()
        prompt = self._build_prompt(
            pipeline,
            presence_mode="Communication mode: TEXTING"
        )
        assert "<presence_mode>" in prompt
        assert "</presence_mode>" in prompt


# =========================================================================
# Token Budget Tests
# =========================================================================

class TestPresenceModeTokenBudget:
    """Test that presence_mode has proper token budget and tier."""

    def test_presence_mode_has_budget(self):
        from src.core.conversation.token_budget import SOURCE_TOKEN_BUDGETS
        assert 'presence_mode' in SOURCE_TOKEN_BUDGETS
        assert SOURCE_TOKEN_BUDGETS['presence_mode'] > 0

    def test_presence_mode_in_tier_1(self):
        from src.core.conversation.token_budget import TIER_1
        assert 'presence_mode' in TIER_1
