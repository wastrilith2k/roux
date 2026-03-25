"""Regression tests for voice_service lazy config loading (issue #13)."""

import importlib
import sys
import types
from unittest.mock import patch, MagicMock

import pytest


class TestVoiceServiceImportDoesNotCallConfig:
    """Importing voice_service must NOT call get_persona_config() at module level."""

    def test_import_does_not_call_get_persona_config(self):
        """Regression: importing voice_service should not trigger _get_voice_config()
        at module level, which previously caused import-time crashes when persona
        config wasn't initialised."""
        # Remove cached module so we get a fresh import
        mod_name = 'src.voice.voice_service'
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch('src.config.persona_config.get_persona_config') as mock_cfg:
                mock_cfg.side_effect = RuntimeError(
                    "get_persona_config called at import time"
                )
                # Re-import — should succeed without calling get_persona_config
                mod = importlib.import_module(mod_name)
                mock_cfg.assert_not_called()
        finally:
            # Restore original module to avoid polluting other tests
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_config_read_at_init_not_import(self):
        """Config values are read when VoiceService is instantiated, not on import."""
        mod_name = 'src.voice.voice_service'
        saved = sys.modules.pop(mod_name, None)
        try:
            mod = importlib.import_module(mod_name)

            mock_persona = MagicMock()
            mock_persona.edge_tts_fallback = 'en-US-AriaNeural'
            mock_persona.elevenlabs_voice_id = 'test-voice-id'
            mock_persona.elevenlabs_model = 'eleven_multilingual_v2'

            with patch.object(mod, '_get_voice_config', return_value=mock_persona):
                svc = mod.VoiceService()
                assert svc._edge_tts_voice == 'en-US-AriaNeural'
                assert svc._elevenlabs_voice_id == 'test-voice-id'
                assert svc._elevenlabs_model == 'eleven_multilingual_v2'
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved


class TestVoiceServiceConfigRefresh:
    """Voice config should reflect runtime changes when a new service is created."""

    def test_new_instance_picks_up_config_changes(self):
        """Regression: previously config was frozen at import time, so runtime
        config changes (e.g. hot-reload of persona.yaml) were never reflected."""
        from src.voice import voice_service as mod

        persona_v1 = MagicMock()
        persona_v1.edge_tts_fallback = 'voice-v1'
        persona_v1.elevenlabs_voice_id = 'id-v1'
        persona_v1.elevenlabs_model = 'model-v1'

        persona_v2 = MagicMock()
        persona_v2.edge_tts_fallback = 'voice-v2'
        persona_v2.elevenlabs_voice_id = 'id-v2'
        persona_v2.elevenlabs_model = 'model-v2'

        with patch.object(mod, '_get_voice_config', side_effect=[persona_v1, persona_v2]):
            svc1 = mod.VoiceService()
            assert svc1._edge_tts_voice == 'voice-v1'

            svc2 = mod.VoiceService()
            assert svc2._edge_tts_voice == 'voice-v2'


class TestVoiceEnabledEnvVar:
    """VOICE_ENABLED env var should not have redundant double-expansion."""

    def test_voice_enabled_defaults_true(self):
        """VOICE_ENABLED defaults to True when env var is not set."""
        import src.voice.voice_service as mod
        # The module-level VOICE_ENABLED is already evaluated, so test via reimport
        mod_name = 'src.voice.voice_service'
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch.dict('os.environ', {}, clear=False):
                # Remove the key if present so default kicks in
                import os
                os.environ.pop('COMPANION_VOICE_ENABLED', None)
                mod = importlib.import_module(mod_name)
                assert mod.VOICE_ENABLED is True
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_voice_enabled_false(self):
        """VOICE_ENABLED is False when env var is 'false'."""
        mod_name = 'src.voice.voice_service'
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch.dict('os.environ', {'COMPANION_VOICE_ENABLED': 'false'}):
                mod = importlib.import_module(mod_name)
                assert mod.VOICE_ENABLED is False
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
