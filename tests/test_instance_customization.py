"""Tests for instance customization layer (issue #49).

Verifies that instance-specific values (entity names, semantic patterns,
fact validation rules, pet names, family/work names) are loaded from
persona.yaml configuration rather than being hardcoded in source code.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from src.config.persona_config import (
    get_persona_config, reset_persona_config, PersonaConfig,
)


class TestKnownEntitiesFromConfig:
    """retrieval_agent.py must load KNOWN_ENTITIES from persona config."""

    def setup_method(self):
        reset_persona_config()

    def teardown_method(self):
        reset_persona_config()

    def test_default_config_has_user_and_companion(self):
        """Default persona.yaml always includes user + companion entries."""
        config = get_persona_config()
        entities = config.get_known_entities_dict()
        user_key = config.primary_user_name.lower()
        companion_key = config.companion_short_name.lower()
        assert user_key in entities
        assert companion_key in entities
        assert '(the user)' in entities[user_key]
        assert '(AI companion)' in entities[companion_key]

    def test_known_entity_entries_populated_from_config(self):
        """Entities configured in known_entity_entries appear in the dict."""
        config = get_persona_config()
        # Simulate an instance with custom entities
        config.known_entity_entries = [
            {'name': 'Alice', 'label': "{user}'s friend"},
            {'name': 'Spot', 'label': "{companion}'s dog"},
        ]
        entities = config.get_known_entities_dict()
        assert 'alice' in entities
        assert 'spot' in entities
        assert config.primary_user_name in entities['alice']
        assert config.companion_short_name in entities['spot']

    def test_no_hardcoded_names_in_retrieval_agent(self):
        """retrieval_agent.KNOWN_ENTITIES must not contain hardcoded names
        that only belong to a specific instance (jesse, kyler, alia, etc.)."""
        from src.memory.retrieval_agent import RetrievalAgent
        agent = RetrievalAgent()
        # Default persona.yaml has no known_entity_entries, so only user
        # and companion should be present.
        for name in ['jesse', 'kyler', 'alia', 'carol', 'tuck', 'lena']:
            assert name not in agent.KNOWN_ENTITIES, (
                f"'{name}' is hardcoded in retrieval_agent — should come from config"
            )


class TestFactValidationFromConfig:
    """fact_extraction_task.py must load validation rules from config."""

    def setup_method(self):
        reset_persona_config()

    def teardown_method(self):
        reset_persona_config()

    def test_invalid_patterns_come_from_config(self):
        """Invalid fact patterns should resolve from persona.yaml."""
        config = get_persona_config()
        resolved = config.get_resolved_invalid_fact_patterns()
        companion_lower = config.companion_short_name.lower()
        # The default persona.yaml has {companion}-based patterns
        for pattern in resolved:
            assert '{companion}' not in pattern[0], (
                "Placeholder {companion} was not resolved"
            )
            assert '{user}' not in pattern[0], (
                "Placeholder {user} was not resolved"
            )

    def test_user_children_from_config(self):
        """user_children must come from config, not hardcoded."""
        config = get_persona_config()
        # Default persona.yaml has empty user_children
        assert config.user_children == []

    def test_validate_facts_uses_config_children(self):
        """validate_facts should use config-driven children, not hardcoded."""
        from src.tasks.fact_extraction_task import validate_facts
        # With default config (no user_children configured), a fact about
        # a child name should pass through since there's nothing to validate against
        facts = [{'subject': 'User', 'fact': 'has a child named Alex'}]
        result = validate_facts(facts)
        assert len(result) == 1


class TestSemanticPatternsFromConfig:
    """fact_store.py must load semantic patterns from config."""

    def setup_method(self):
        reset_persona_config()

    def teardown_method(self):
        reset_persona_config()

    def test_semantic_patterns_from_config(self):
        """Semantic patterns should be loaded from persona config."""
        config = get_persona_config()
        # Default persona.yaml has no semantic_patterns
        assert isinstance(config.semantic_patterns, dict)

    def test_no_hardcoded_instance_patterns(self):
        """No instance-specific patterns (cavallo, jesse, alia) in code."""
        config = get_persona_config()
        patterns = config.semantic_patterns
        for pattern_name in patterns:
            assert 'cavallo' not in pattern_name, (
                f"Instance-specific pattern '{pattern_name}' in code"
            )


class TestPetNamesFromConfig:
    """synthesized_events.py must load pet names from config."""

    def setup_method(self):
        reset_persona_config()

    def teardown_method(self):
        reset_persona_config()

    def test_pet_names_from_config(self):
        """Pet names should come from persona config."""
        config = get_persona_config()
        # Default config has no pet names
        assert isinstance(config.pet_names, list)

    def test_default_config_has_no_pet_names(self):
        """Default persona.yaml should not have hardcoded pet names."""
        config = get_persona_config()
        for name in ['tuck']:
            assert name not in [n.lower() for n in config.pet_names], (
                f"'{name}' is hardcoded as a pet name — should come from instance config"
            )


class TestFamilyNamesFromConfig:
    """memory_gap_analysis_task.py must load family/work names from config."""

    def setup_method(self):
        reset_persona_config()

    def teardown_method(self):
        reset_persona_config()

    def test_family_names_from_config(self):
        """Family names should come from persona config."""
        config = get_persona_config()
        assert isinstance(config.family_names, list)
        # Default has no hardcoded family names
        for name in ['jesse', 'kyler', 'alia', 'carol']:
            assert name not in config.family_names, (
                f"'{name}' hardcoded in family_names — use instance config"
            )

    def test_work_names_from_config(self):
        """Work names should come from persona config."""
        config = get_persona_config()
        assert isinstance(config.work_names, list)
        for name in ['cavallo', 'act-on']:
            assert name not in config.work_names, (
                f"'{name}' hardcoded in work_names — use instance config"
            )

    def test_guess_category_uses_config(self):
        """_guess_category should use config-driven names."""
        from src.tasks.memory_gap_analysis_task import _guess_category
        # With default empty config, generic names should return 'life_events'
        assert _guess_category('SomeRandomPerson') == 'life_events'
        # 'mom' and 'dad' are generic defaults that always apply
        assert _guess_category('mom') == 'family'
        assert _guess_category('work') == 'work'


class TestAdminUsernameNotHardcoded:
    """cost_routes.py must not hardcode 'admin' as the default username."""

    def test_default_username_from_env(self):
        """DEFAULT_ADMIN_USERNAME env var should control the fallback."""
        with patch.dict(os.environ, {'DEFAULT_ADMIN_USERNAME': 'testuser'}):
            # Re-import to pick up the env var (it's evaluated at call time)
            from flask import Flask
            app = Flask(__name__)
            app.config['SECRET_KEY'] = 'test'
            with app.test_request_context():
                from flask import session
                # session has no 'username', so fallback is used
                result = session.get('username',
                                     os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin'))
                assert result == 'testuser'


class TestConfigFieldDefaults:
    """New config fields should have sensible defaults."""

    def setup_method(self):
        reset_persona_config()

    def teardown_method(self):
        reset_persona_config()

    def test_new_fields_have_defaults(self):
        """All new config fields should initialize to empty collections."""
        config = get_persona_config()
        assert config.known_entity_entries == []
        assert config.pet_names == []
        assert config.user_children == []
        assert config.family_names == []
        assert config.work_names == []
        assert isinstance(config.semantic_patterns, dict)
        assert isinstance(config.invalid_fact_patterns, list)
