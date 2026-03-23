"""
Persona Configuration — Companion Framework

WHAT: Loads the companion's identity configuration from a YAML file and exposes
      it as a typed PersonaConfig dataclass.
WHY:  Every module that needs "who am I?" or "who is the user?" imports from here
      instead of hardcoding names, emails, or file paths. This makes the framework
      persona-agnostic — swap the YAML and you get a different companion.
HOW:  On first access, _load_config() reads data/persona.yaml (or an instance-
      specific file for multi-agent), applies env-var overrides, and caches the
      result as a module-level singleton. Multi-agent mode (companion_id != None)
      loads from instances/<id>/persona.yaml without caching (each agent is unique).

Priority chain: env vars > YAML values > hardcoded defaults.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PersonaConfig dataclass — typed identity for every subsystem
# ---------------------------------------------------------------------------

@dataclass
class PersonaConfig:
    """All identity wiring for the companion engine.

    Grouped by concern: companion identity, primary user, relationship seed,
    file paths, and voice synthesis settings.
    """

    # --- Companion identity ---
    companion_name: str           # Full name, e.g. "Your Companion Name"
    companion_short_name: str     # First name / nickname used in prompts
    companion_email: str
    companion_entity_profile: str # Key into entity_profiles YAML ("companion")
    companion_pronoun_subject: str
    companion_pronoun_object: str
    companion_pronoun_possessive: str
    companion_pronoun_reflexive: str

    # --- Primary user identity ---
    primary_user_name: str
    primary_user_email: str
    primary_user_entity_profile: str
    primary_user_timezone: str
    user_pronoun_subject: str
    user_pronoun_object: str
    user_pronoun_possessive: str
    user_pronoun_reflexive: str

    # --- Relationship seed (initial context before any conversation) ---
    relationship_initial_context: str

    # --- File paths (relative to project root) ---
    personality_prompt_path: str
    cognitive_flow_prompt_path: str
    image_base_prompt_path: str
    image_intimate_prompt_path: str
    calendar_name: str
    calendar_id_file: str

    # --- Voice synthesis identity ---
    elevenlabs_voice_id: str
    elevenlabs_model: str
    edge_tts_fallback: str

    @property
    def companion_pronouns(self) -> Dict[str, str]:
        return {
            "subject": self.companion_pronoun_subject,
            "object": self.companion_pronoun_object,
            "possessive": self.companion_pronoun_possessive,
            "reflexive": self.companion_pronoun_reflexive,
        }

    @property
    def user_pronouns(self) -> Dict[str, str]:
        return {
            "subject": self.user_pronoun_subject,
            "object": self.user_pronoun_object,
            "possessive": self.user_pronoun_possessive,
            "reflexive": self.user_pronoun_reflexive,
        }


# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Locate the project root by looking for a data/ directory.

    Checks Docker container path first (/app), then the path relative to
    this source file. Falls back to /app if nothing matches.
    """
    candidates = [
        Path('/app'),                                    # Docker container
        Path(__file__).parent.parent.parent,             # Relative to source
    ]
    for path in candidates:
        try:
            if (path / 'data').exists():
                return path
        except PermissionError:
            continue
    return Path('/app')


# ---------------------------------------------------------------------------
# YAML loading with env-var overrides
# ---------------------------------------------------------------------------

def _load_config(companion_id: str = None) -> PersonaConfig:
    """Load persona config from YAML, then apply env-var overrides.

    Args:
        companion_id: If provided, loads instance-specific persona from
                      instances/<companion_id>/persona.yaml (multi-agent mode).
                      Otherwise loads the default data/persona.yaml.
    """
    project_root = _find_project_root()

    # Determine which YAML to load
    if companion_id:
        yaml_path = project_root / 'instances' / companion_id / 'persona.yaml'
    else:
        yaml_path = project_root / 'data' / 'persona.yaml'

    # Read YAML (graceful fallback to empty dict)
    data = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            logger.info(f"Loaded persona config from {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to load {yaml_path}: {e}. Using defaults.")
    else:
        logger.info("No persona.yaml found, using hardcoded defaults.")

    # Destructure YAML sections
    companion = data.get('companion', {})
    user = data.get('primary_user', {})
    relationship = data.get('relationship', {})
    prompts = data.get('prompts', {})
    image = data.get('image', {})
    calendar = data.get('calendar', {})
    voice = data.get('voice', {})

    # Build config with YAML values (or sensible defaults)
    config = PersonaConfig(
        companion_name=companion.get('name', 'Companion'),
        companion_short_name=companion.get('short_name', 'Companion'),
        companion_email=companion.get('email', ''),
        companion_entity_profile=companion.get('entity_profile', 'companion'),
        companion_pronoun_subject=companion.get('pronouns', {}).get('subject', 'they'),
        companion_pronoun_object=companion.get('pronouns', {}).get('object', 'them'),
        companion_pronoun_possessive=companion.get('pronouns', {}).get('possessive', 'their'),
        companion_pronoun_reflexive=companion.get('pronouns', {}).get('reflexive', 'themselves'),

        primary_user_name=user.get('name', 'User'),
        primary_user_email=user.get('email', ''),
        primary_user_entity_profile=user.get('entity_profile', 'user'),
        primary_user_timezone=user.get('timezone', 'America/Los_Angeles'),
        user_pronoun_subject=user.get('pronouns', {}).get('subject', 'they'),
        user_pronoun_object=user.get('pronouns', {}).get('object', 'them'),
        user_pronoun_possessive=user.get('pronouns', {}).get('possessive', 'their'),
        user_pronoun_reflexive=user.get('pronouns', {}).get('reflexive', 'themselves'),

        relationship_initial_context=relationship.get(
            'initial_context',
            'They recently started talking.'
        ),

        personality_prompt_path=prompts.get('personality', 'prompts/core/personality.md'),
        cognitive_flow_prompt_path=prompts.get('cognitive_flow', 'prompts/core/cognitive_flow.md'),
        image_base_prompt_path=image.get('base_prompt', 'src/image/prompts/base_prompt.txt'),
        image_intimate_prompt_path=image.get('intimate_prompt', 'src/image/prompts/intimate_prompt.txt'),
        calendar_name=calendar.get('name', "Companion Schedule"),
        calendar_id_file=calendar.get('id_file', 'data/companion_calendar_id.txt'),

        elevenlabs_voice_id=voice.get('elevenlabs_voice_id', 'gJx1vCzNCD1EQHT212Ls'),
        elevenlabs_model=voice.get('elevenlabs_model', 'eleven_v3'),
        edge_tts_fallback=voice.get('edge_tts_fallback', 'en-US-AvaNeural'),
    )

    # --- Env-var overrides (highest priority) ---
    # Operators can override any identity field without touching the YAML.
    env_overrides = {
        'COMPANION_NAME': 'companion_short_name',
        'COMPANION_FULL_NAME': 'companion_name',
        'PRIMARY_USER_NAME': 'primary_user_name',
        'PRIMARY_USER_EMAIL': 'primary_user_email',
        'COMPANION_PRIMARY_USER_EMAIL': 'primary_user_email',  # Legacy alias
        'PRIMARY_USER_TIMEZONE': 'primary_user_timezone',
        'ELEVENLABS_VOICE_ID': 'elevenlabs_voice_id',
        'ELEVENLABS_MODEL': 'elevenlabs_model',
    }
    for env_key, attr_name in env_overrides.items():
        env_val = os.environ.get(env_key)
        if env_val:
            setattr(config, attr_name, env_val)
            logger.debug(f"Persona config override: {env_key} -> {attr_name}")

    return config


# ---------------------------------------------------------------------------
# Module-level singleton and public API
# ---------------------------------------------------------------------------

_config: Optional[PersonaConfig] = None


def get_persona_config(companion_id: str = None) -> PersonaConfig:
    """Get persona configuration (singleton for default, fresh load per companion_id).

    Args:
        companion_id: If provided, loads instance-specific config (not cached).
                      Without it, returns the cached default config.
    """
    if companion_id:
        # Multi-agent: always load fresh (each companion has its own identity)
        return _load_config(companion_id=companion_id)

    global _config
    if _config is None:
        _config = _load_config()
    return _config


def reset_persona_config():
    """Reset the singleton (for testing or hot-reload)."""
    global _config
    _config = None
