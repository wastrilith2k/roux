"""
Entity Profile Manager - YAML-based entity profiles with git audit trail.

WHAT: CRUD operations for entity profiles -- the biographical ground truth
about every person (and pet) the companion knows. Profiles are plain YAML
files stored under data/entity_profiles/ and version-controlled via git.

WHY: The companion needs a stable, human-editable source of truth for facts
that should never drift or contradict (e.g., "James has two sons: Jesse and
Kyler"). Entity profiles serve as the immutable foundation that the LLM is
instructed not to contradict, even if conversation-extracted facts disagree.

HOW it fits:
  - entity_profile_loader.py reads these YAML files at startup and caches
    them for fast lookup.
  - get_grounding_context() generates the "KNOWN ENTITIES" block injected
    into every prompt to prevent hallucination about core biographical facts.
  - add_hard_fact() and add_alias() are called when the fact approval
    pipeline promotes a conversation-extracted fact to ground truth.
  - Every write is auto-committed to git so changes have a full audit trail.

Design principles:
  1. YAML files in data/entity_profiles/ -- human-readable and editable.
  2. Git-versioned: every mutation is auto-committed with a descriptive message.
  3. Aliases: canonical, learned (from conversation), pending (needs approval).
  4. Hard facts: high-confidence facts promoted from extraction or confirmed
     by the user, stored directly in the profile YAML.

Usage:
    manager = EntityProfileManager()
    profile = manager.get_profile('james')
    manager.add_alias('james', 'jimmy', status='learned')
    manager.add_hard_fact('james', 'preferences', 'James prefers peppermint tea')
"""

import os
import yaml
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import frontmatter as _fm
    _FRONTMATTER_AVAILABLE = True
except ImportError:
    _FRONTMATTER_AVAILABLE = False

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

PROFILES_DIR = Path('/app/data/entity_profiles')


class EntityProfileManager:
    """Manages entity profiles with YAML storage and git versioning."""

    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.profiles_dir = profiles_dir
        self._cache: Dict[str, Dict] = {}

    def _get_companion_name(self) -> str:
        """Get the companion's short name from persona config."""
        from src.config.persona_config import get_persona_config
        return get_persona_config().companion_short_name

    def _get_companion_entity_id(self) -> str:
        """Get the companion's entity ID (lowercase short name)."""
        return self._get_companion_name().lower()

    def get_profile(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Load an entity profile from .md (front-matter) or .yaml (fallback)."""
        if entity_id in self._cache:
            return self._cache[entity_id]

        entity_id_lower = entity_id.lower()

        # Try .md first (front-matter dict)
        if _FRONTMATTER_AVAILABLE:
            md_path = self.profiles_dir / f"{entity_id_lower}.md"
            if md_path.exists():
                try:
                    post = _fm.load(str(md_path))
                    profile = dict(post.metadata)
                    self._cache[entity_id] = profile
                    return profile
                except Exception as e:
                    logger.warning(f"Failed to parse markdown profile {md_path}: {e}")

        # Fallback to .yaml
        yaml_path = self.profiles_dir / f"{entity_id_lower}.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    profile = yaml.safe_load(f)
                    self._cache[entity_id] = profile
                    return profile
            except Exception as e:
                logger.warning(f"Failed to load YAML profile {yaml_path}: {e}")

        logger.warning(f"Profile not found: {entity_id}")
        return None

    def list_profiles(self) -> List[str]:
        """List all available entity profile IDs."""
        profiles = []
        for file in self.profiles_dir.glob("*.yaml"):
            if file.name != "README.md":
                profiles.append(file.stem)
        return profiles

    def add_alias(self, entity_id: str, alias: str,
                  status: str = 'learned',
                  note: str = None,
                  auto_commit: bool = True) -> bool:
        """
        Add an alias for an entity.

        Args:
            entity_id: The entity to add alias for (e.g., 'james')
            alias: The alias to add (e.g., 'jimmy')
            status: 'canonical' (official), 'learned' (from conversation), 'pending' (needs approval)
            note: Optional note about where/how this alias was learned
            auto_commit: Whether to git commit the change
        """
        profile = self.get_profile(entity_id)
        if not profile:
            logger.error(f"Cannot add alias: profile {entity_id} not found")
            return False

        # Initialize aliases section if needed
        if 'aliases' not in profile:
            profile['aliases'] = []

        # Check if alias already exists
        existing_aliases = [a.get('name', a) if isinstance(a, dict) else a
                          for a in profile['aliases']]
        if alias.lower() in [a.lower() for a in existing_aliases]:
            logger.info(f"Alias '{alias}' already exists for {entity_id}")
            return False

        # Add the new alias
        alias_entry = {
            'name': alias,
            'status': status,
            'added': clock_now().strftime('%Y-%m-%d'),
        }
        if note:
            alias_entry['note'] = note

        profile['aliases'].append(alias_entry)

        # Write and commit
        commit_msg = f"Add alias '{alias}' for {entity_id} (status: {status})"
        return self._write_profile(entity_id, profile, commit_msg, auto_commit)

    def add_hard_fact(self, entity_id: str, category: str, fact_text: str,
                      source: str = 'promoted',
                      confidence: float = 1.0,
                      auto_commit: bool = True) -> bool:
        """
        Add a hard fact to an entity profile.

        Hard facts are:
        - Stored directly in the YAML profile
        - Versioned with git
        - Used for context grounding

        Args:
            entity_id: The entity (e.g., 'james')
            category: Category like 'preferences', 'relationships', 'work'
            fact_text: The fact statement
            source: Where this came from ('promoted', 'user_confirmed', 'manual')
            confidence: 0.0-1.0 (hard facts should be high confidence)
            auto_commit: Whether to git commit
        """
        profile = self.get_profile(entity_id)
        if not profile:
            logger.error(f"Cannot add fact: profile {entity_id} not found")
            return False

        # Initialize hard_facts section if needed
        if 'hard_facts' not in profile:
            profile['hard_facts'] = {}
        if category not in profile['hard_facts']:
            profile['hard_facts'][category] = []

        # Check for duplicates
        existing = profile['hard_facts'][category]
        for f in existing:
            existing_text = f.get('fact', f) if isinstance(f, dict) else f
            if existing_text.lower().strip() == fact_text.lower().strip():
                logger.info(f"Fact already exists in {entity_id}.{category}")
                return False

        # Add the fact
        fact_entry = {
            'fact': fact_text,
            'source': source,
            'confidence': confidence,
            'added': clock_now().strftime('%Y-%m-%d'),
        }
        profile['hard_facts'][category].append(fact_entry)

        # Write and commit
        short_fact = fact_text[:50] + '...' if len(fact_text) > 50 else fact_text
        commit_msg = f"Add hard fact for {entity_id}.{category}: {short_fact}"
        return self._write_profile(entity_id, profile, commit_msg, auto_commit)

    def update_field(self, entity_id: str, field_path: str, value: Any,
                     auto_commit: bool = True) -> bool:
        """
        Update a specific field in the profile using dot notation.

        Args:
            entity_id: The entity (e.g., 'james')
            field_path: Dot-separated path (e.g., 'employment.current_employer')
            value: New value to set
            auto_commit: Whether to git commit
        """
        profile = self.get_profile(entity_id)
        if not profile:
            return False

        # Navigate to the field
        parts = field_path.split('.')
        current = profile
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        old_value = current.get(parts[-1])
        current[parts[-1]] = value

        commit_msg = f"Update {entity_id}.{field_path}: {old_value} -> {value}"
        return self._write_profile(entity_id, profile, commit_msg, auto_commit)

    def get_all_aliases(self, entity_id: str) -> List[str]:
        """Get all aliases for an entity (including the name itself)."""
        profile = self.get_profile(entity_id)
        if not profile:
            return []

        aliases = [profile.get('name', entity_id)]

        if 'aliases' in profile:
            for a in profile['aliases']:
                if isinstance(a, dict):
                    aliases.append(a.get('name', ''))
                else:
                    aliases.append(a)

        return [a for a in aliases if a]  # Filter empty

    def resolve_entity(self, name: str) -> Optional[str]:
        """
        Resolve a name or alias to its canonical entity ID.

        Checks in order: exact entity ID match, exact alias match, then
        first-name-of-alias match (e.g., "James" matches alias "James Smith").

        Args:
            name: A name or alias to look up

        Returns:
            Entity ID (filename stem) if found, None otherwise
        """
        name_lower = name.lower()

        for entity_id in self.list_profiles():
            if entity_id.lower() == name_lower:
                return entity_id

            for alias in self.get_all_aliases(entity_id):
                if alias.lower() == name_lower:
                    return entity_id
                # First-name match: "James" matches full-name alias "James Smith"
                if ' ' in alias and alias.lower().split()[0] == name_lower:
                    return entity_id

        return None

    # =========================================================================
    # Grounding context generation (injected into every LLM prompt)
    # =========================================================================

    def get_grounding_context(self) -> str:
        """
        Generate the "KNOWN ENTITIES" block for LLM prompts.

        Prefers Markdown body from .md profiles (richer narrative format).
        Falls back to structured YAML-based summary generation.
        """
        # Try .md files first — return their Markdown bodies
        if _FRONTMATTER_AVAILABLE:
            md_bodies = []
            for md_file in sorted(self.profiles_dir.glob("*.md")):
                try:
                    post = _fm.load(str(md_file))
                    if post.content.strip():
                        md_bodies.append(post.content.strip())
                except Exception:
                    pass
            if md_bodies:
                return '\n\n---\n\n'.join(md_bodies)
        # Fall through to YAML-based implementation
        return self._get_grounding_context_legacy()

    def _get_grounding_context_legacy(self) -> str:
        """
        Generate the "KNOWN ENTITIES" block for LLM prompts.

        This is the single most important anti-hallucination mechanism: it
        tells the LLM exactly who each person is, their relationships, and
        critical rules (e.g., "the companion has NO children"). The LLM is
        instructed never to contradict this block.

        Returns a multi-line formatted string with entity summaries and rules.
        """
        lines = ["KNOWN ENTITIES (ground truth - DO NOT contradict):"]

        # Track key facts for critical rules
        james_children = []
        companion_has_children = False

        for entity_id in self.list_profiles():
            profile = self.get_profile(entity_id)
            if not profile:
                continue

            name = profile.get('name', entity_id.title())
            first_name = name.split()[0] if ' ' in name else name
            role = profile.get('role', '')

            # Build a summary line
            summary_parts = []

            # Role description
            if role == 'primary_user':
                summary_parts.append("User, primary person the companion talks to")
            elif role == 'ai_companion':
                pass  # Let entity profile speak for itself

            # Partner info
            if 'romantic_relationship' in profile:
                partner = profile['romantic_relationship'].get('partner')
                status = profile['romantic_relationship'].get('status', '')
                if partner:
                    summary_parts.append(f"{partner}'s partner")

            # Family info
            if 'family' in profile:
                fam = profile['family']
                if 'children' in fam:
                    kids = [c.get('name', c) if isinstance(c, dict) else c
                           for c in fam['children']]
                    if kids:
                        summary_parts.append(f"has children: {', '.join(kids)}")
                        if entity_id == 'james':
                            james_children = kids
                        if entity_id == self._get_companion_entity_id():
                            companion_has_children = True

                if 'parents' in fam:
                    parents = fam['parents']
                    if 'mother' in parents:
                        mom = parents['mother']
                        mom_name = mom.get('name') if isinstance(mom, dict) else mom
                        if mom_name:
                            summary_parts.append(f"mother is {mom_name}")

                if 'spouse' in fam:
                    summary_parts.append(f"legally married to {fam['spouse']}")

            # Work info
            if 'employment' in profile:
                emp = profile['employment']
                if emp.get('job_title'):
                    summary_parts.append(f"works as {emp['job_title']}")
                if emp.get('work_status'):
                    summary_parts.append(emp['work_status'])

            # Key preferences
            if 'preferences' in profile:
                prefs = profile['preferences']
                if 'drinks' in prefs:
                    drinks = prefs['drinks']
                    if drinks.get('coffee') and 'not' in drinks['coffee'].lower():
                        summary_parts.append("does NOT drink coffee")
                    if drinks.get('tea'):
                        summary_parts.append("prefers tea")

            # Cat for the companion
            if 'romantic_relationship' in profile:
                cats = profile['romantic_relationship'].get('cats', [])
                for cat in cats:
                    if isinstance(cat, dict):
                        summary_parts.append(f"has cat named {cat.get('name', 'unknown')}")

            if summary_parts:
                lines.append(f"- {first_name}: {'. '.join(summary_parts)}.")
            else:
                lines.append(f"- {first_name}")

        # Add critical rules
        lines.append("")
        lines.append("CRITICAL RULES:")

        # Companion's children rule
        _companion_name = self._get_companion_name()
        if not companion_has_children:
            lines.append(f"- {_companion_name} has NO children. NEVER extract facts saying she has children.")

        # James's children rule
        if james_children:
            lines.append(f"- James's children are {' and '.join(james_children)}, NOT Nicholas or any other names.")

        # Standard rules
        lines.append(f"- {_companion_name} is not a parent to James's children.")
        lines.append("- Facts about what JAMES likes/does should have James as subject.")
        lines.append(f"- Facts about what {_companion_name.upper()} likes/does should have {_companion_name} as subject.")

        return '\n'.join(lines)

    def _write_profile(self, entity_id: str, profile: Dict,
                       commit_message: str, auto_commit: bool = True) -> bool:
        """Write profile to YAML and optionally git commit."""
        file_path = self.profiles_dir / f"{entity_id.lower()}.yaml"

        try:
            # Update cache
            self._cache[entity_id] = profile

            # Write YAML
            with open(file_path, 'w') as f:
                yaml.dump(profile, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            logger.info(f"Wrote profile: {file_path}")

            # Git commit if requested
            if auto_commit:
                self._git_commit(file_path, commit_message)

            return True

        except Exception as e:
            logger.error(f"Error writing profile {entity_id}: {e}")
            return False

    def _git_commit(self, file_path: Path, message: str) -> bool:
        """Commit a file change to git."""
        try:
            # Get the repo root (parent of data/)
            repo_root = self.profiles_dir.parent.parent
            rel_path = file_path.relative_to(repo_root)

            # Add and commit
            subprocess.run(
                ['git', 'add', str(rel_path)],
                cwd=repo_root,
                capture_output=True,
                check=True
            )

            subprocess.run(
                ['git', 'commit', '-m', f"entity: {message}"],
                cwd=repo_root,
                capture_output=True,
                check=True
            )

            logger.info(f"Git commit: {message}")
            return True

        except subprocess.CalledProcessError as e:
            # Commit might fail if nothing changed
            stderr = e.stderr.decode() if e.stderr else ''
            if 'nothing to commit' in stderr:
                logger.debug("No changes to commit")
                return True
            logger.error(f"Git commit failed: {stderr}")
            return False
        except Exception as e:
            logger.error(f"Git error: {e}")
            return False


# Singleton instance
_manager: Optional[EntityProfileManager] = None

def get_entity_manager() -> EntityProfileManager:
    """Get the singleton EntityProfileManager instance."""
    global _manager
    if _manager is None:
        _manager = EntityProfileManager()
    return _manager
