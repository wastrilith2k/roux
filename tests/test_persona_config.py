"""Tests for persona configuration loading."""
import pytest
import os
from pathlib import Path

from src.config.persona_config import get_persona_config, reset_persona_config, PersonaConfig


class TestPersonaConfig:
    def setup_method(self):
        reset_persona_config()

    def test_loads_default_config(self):
        config = get_persona_config()
        assert isinstance(config, PersonaConfig)
        assert config.companion_short_name  # Not empty
        assert config.primary_user_name  # Not empty

    def test_singleton_pattern(self):
        config1 = get_persona_config()
        config2 = get_persona_config()
        assert config1 is config2

    def test_companion_id_loads_instance(self):
        """When companion_id is provided, should load from instances/ directory."""
        # This test works if instances/kai/persona.yaml exists
        kai_config = get_persona_config(companion_id='kai')
        assert kai_config.companion_short_name == 'Kai'
        assert kai_config.primary_user_name == 'Mira'

    def test_different_companions_different_configs(self):
        kai = get_persona_config(companion_id='kai')
        mira = get_persona_config(companion_id='mira')
        assert kai.companion_short_name != mira.companion_short_name
        assert kai.primary_user_name == mira.companion_short_name

    def test_companion_id_bypasses_singleton(self):
        default = get_persona_config()
        kai = get_persona_config(companion_id='kai')
        assert default is not kai

    def test_path_traversal_in_companion_id_rejected(self):
        """companion_id with path traversal characters must be rejected."""
        malicious_ids = [
            '../etc',
            '../../secrets',
            'foo/../../bar',
            'kai/../../../etc',
            '..',
            '.',
            'foo/bar',
        ]
        for malicious_id in malicious_ids:
            with pytest.raises(ValueError, match="Invalid companion_id"):
                get_persona_config(companion_id=malicious_id)

    def test_valid_companion_id_formats_accepted(self):
        """Legitimate companion_id values should not be rejected by validation."""
        # These should not raise ValueError (they may fail to find a file, but
        # that's fine — the validation step should pass)
        valid_ids = ['kai', 'mira', 'companion-1', 'test_bot', 'Agent007']
        for valid_id in valid_ids:
            try:
                get_persona_config(companion_id=valid_id)
            except ValueError:
                pytest.fail(f"Valid companion_id '{valid_id}' was incorrectly rejected")

    def test_env_var_overrides(self):
        reset_persona_config()
        os.environ['COMPANION_NAME'] = 'TestBot'
        try:
            config = get_persona_config()
            assert config.companion_short_name == 'TestBot'
        finally:
            del os.environ['COMPANION_NAME']
            reset_persona_config()
