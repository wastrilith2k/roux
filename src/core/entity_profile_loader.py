"""
Entity Profile Loader -- YAML-based identity and fact storage for all entities.

WHAT: Loads, caches, and formats entity profiles (companion, user, secondary
      characters) from data/entity_profiles/*.yaml. Provides context-filtered
      profile formatting for prompt injection and a basic contradiction checker.

WHY:  Entity profiles are the ground truth for who the companion is, who the
      user is, and key relationship facts. They replaced Neo4j with simple,
      human-editable YAML files following the DeepAgents pattern. Every LLM
      call includes relevant profile sections so the companion doesn't forget
      core identity facts.

HOW:  On init, all YAML files in the profiles directory are loaded into a dict.
      `get_relevant_profiles()` picks profiles mentioned in the conversation
      context plus the companion and primary user (in that order -- companion
      identity first). `format_profiles_for_prompt()` optionally applies the
      EntityProfileContextFilter to reduce token count. Dynamic age calculation
      from birthdate ensures age stays current without manual updates.

Singleton: `get_entity_profile_loader()` at module bottom.
"""

import os
import yaml
import logging
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

class EntityProfileLoader:
    """Load and cache entity profiles from YAML files."""

    def __init__(self):
        """Initialize loader and load all profiles."""
        self.profiles_dir = Path(os.getenv('ENTITY_PROFILES_DIR', self._find_profiles_dir()))
        self.profiles = {}
        self.load_all_profiles()

    def _find_profiles_dir(self) -> str:
        """Find entity profiles directory, checking multiple locations."""
        candidates = [
            '/app/data/entity_profiles',                    # Docker container
            '/root/companion/data/entity_profiles',              # Production server
            Path(__file__).parent.parent.parent / 'data' / 'entity_profiles',  # Relative to source
        ]
        for path in candidates:
            try:
                if Path(path).exists():
                    logger.info(f"Found entity profiles at: {path}")
                    return str(path)
            except PermissionError:
                # Skip paths we can't access (e.g., /root when not root)
                continue
        # Default fallback
        return '/app/data/entity_profiles'

    def load_all_profiles(self):
        """Load all YAML profiles from the profiles directory."""
        if not self.profiles_dir.exists():
            logger.warning(f"Entity profiles directory not found: {self.profiles_dir}")
            return

        yaml_files = self.profiles_dir.glob('*.yaml')
        for yaml_file in yaml_files:
            entity_name = yaml_file.stem  # filename without .yaml
            try:
                with open(yaml_file, 'r') as f:
                    profile = yaml.safe_load(f)
                    self.profiles[entity_name] = profile
                    logger.info(f"✅ Loaded entity profile: {entity_name}")
            except Exception as e:
                logger.error(f"Failed to load profile {yaml_file}: {e}")

    def get_profile(self, entity_name: str) -> Optional[Dict]:
        """Get profile for a specific entity."""
        return self.profiles.get(entity_name.lower())

    def get_profiles(self, entity_names: List[str]) -> Dict[str, Dict]:
        """Get profiles for multiple entities."""
        result = {}
        for name in entity_names:
            profile = self.get_profile(name)
            if profile:
                result[name.lower()] = profile
        return result

    def get_relevant_profiles(self, context: str) -> Dict[str, Dict]:
        """
        Extract relevant entity names from context and return their profiles.
        Used to inject only relevant profiles into prompts.
        """
        relevant = {}

        # Check which entities are mentioned in context
        context_lower = context.lower()

        for entity_name in self.profiles.keys():
            if entity_name in context_lower:
                relevant[entity_name] = self.profiles[entity_name]

        # Always include companion and primary user (companion first - needs identity context)
        # Use ordered insertion to ensure companion appears before user in prompts
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        ordered_relevant = {}

        # Companion first (their identity is primary)
        companion_key = _pc.companion_entity_profile
        if companion_key in self.profiles:
            ordered_relevant[companion_key] = self.profiles[companion_key]

        # Primary user second
        user_key = _pc.primary_user_entity_profile
        if user_key in self.profiles:
            ordered_relevant[user_key] = self.profiles[user_key]

        # Then any other mentioned entities
        for entity_name, profile in relevant.items():
            if entity_name not in ordered_relevant:
                ordered_relevant[entity_name] = profile

        return ordered_relevant

    def format_profiles_for_prompt(
        self,
        profiles: Dict[str, Dict],
        user_message: str = "",
        recent_messages: List[str] = None
    ) -> str:
        """
        Format profiles as human-readable text for injection into prompts.

        If user_message is provided, applies context filtering to reduce
        profile size while ensuring relevant sections are included.

        Args:
            profiles: Dict of entity profiles
            user_message: Current message for context filtering (optional)
            recent_messages: Recent messages for context (optional)

        Returns:
            Formatted profiles string for prompt inclusion.
        """
        formatted = "# CRITICAL ENTITY PROFILES (Source of Truth)\n\n"

        # Apply context filtering if message provided
        if user_message:
            try:
                from src.core.entity_profile_context_filter import get_entity_profile_context_filter
                context_filter = get_entity_profile_context_filter()

                for entity_name, profile in profiles.items():
                    filtered_profile = context_filter.get_filtered_profile(
                        entity_name=entity_name,
                        user_message=user_message,
                        recent_messages=recent_messages or [],
                        full_profile=profile
                    )
                    formatted += f"## {entity_name.upper()}\n"
                    formatted += self._format_profile_section(filtered_profile)
                    formatted += "\n"

                return formatted

            except Exception as e:
                logger.warning(f"Context filtering failed, using full profiles: {e}")
                # Fall through to unfiltered formatting

        # Unfiltered formatting (original behavior)
        for entity_name, profile in profiles.items():
            formatted += f"## {entity_name.upper()}\n"
            formatted += self._format_profile_section(profile)
            formatted += "\n"

        return formatted

    def _format_profile_section(self, profile: Dict, indent: int = 0) -> str:
        """Recursively format a profile section."""
        lines = []
        prefix = "  " * indent

        for key, value in profile.items():
            # Calculate age dynamically from birthdate
            if key == 'birthdate' and isinstance(value, str):
                try:
                    birthdate = datetime.fromisoformat(value)
                    now = datetime.now(ZoneInfo('America/Los_Angeles'))
                    age = (now - birthdate).days // 365
                    lines.append(f"{prefix}birthdate: {value}")
                    lines.append(f"{prefix}age: {age} years old (calculated from birthdate)")
                except Exception as e:
                    lines.append(f"{prefix}{key}: {value}")
            elif isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_profile_section(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  - ")
                        lines.append(self._format_profile_section(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return "\n".join(lines)

    def validate_against_profile(self, entity_name: str, statement: str) -> List[str]:
        """
        Check if a statement contradicts a known profile.
        Returns list of contradiction issues found.
        """
        issues = []
        profile = self.get_profile(entity_name)

        if not profile:
            return issues

        statement_lower = statement.lower()

        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        _user = _pc.primary_user_entity_profile
        _companion = _pc.companion_entity_profile

        # Check employment (critical)
        if entity_name.lower() == _user:
            employment = profile.get('employment', {})
            current_employer = employment.get('current_employer', '').lower()

            # Check for wrong employer mentions
            if 'leantaas' in statement_lower and any(x in statement_lower for x in ['work', 'job', 'standup', 'ping', 'email']):
                issues.append(f"PROFILE CONTRADICTION: {_pc.primary_user_name} works for {current_employer}, not LeanTaaS")

        # Check parenting roles (critical for companion)
        if entity_name.lower() == _companion:
            role_clarification = profile.get('role_clarification', {})
            is_not = role_clarification.get('is_not', [])

            if any('mother' in x.lower() for x in is_not):
                parenting_patterns = [
                    r"i\s+(?:tuck|put|help)\s+(?:the\s+)?kids?\s+(?:to\s+)?bed",
                    r"i\s+(?:get|send)\s+(?:the\s+)?kids?\s+(?:to\s+)?school",
                    r"my\s+(?:son|daughter|children?|kids?)",
                    r"i'm\s+(?:their\s+)?(?:mom|mother|step-?mom)",
                ]
                if any(re.search(p, statement_lower) for p in parenting_patterns):
                    issues.append(f"PROFILE CONTRADICTION: {_pc.companion_short_name} is not a parent to {_pc.primary_user_name}'s kids - she has no parenting role with them")

        return issues


def get_entity_profile_loader() -> EntityProfileLoader:
    """Get or create EntityProfileLoader singleton."""
    if not hasattr(get_entity_profile_loader, '_instance'):
        get_entity_profile_loader._instance = EntityProfileLoader()
    return get_entity_profile_loader._instance


def get_current_time_context() -> str:
    """
    Get current date/time in PST for inclusion in LLM prompts.
    This gives the companion their perception of "now".

    Returns:
        Formatted string like: "Current time: Monday, November 25, 2025 at 2:30 PM PST"
    """
    now = datetime.now(ZoneInfo('America/Los_Angeles'))
    return now.strftime("Current time: %A, %B %d, %Y at %-I:%M %p PST")


# Import re at module level for validate_against_profile
import re
