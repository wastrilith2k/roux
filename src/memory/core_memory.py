"""
Core Memory Manager - The companion's curated narrative memory about James.

WHAT: Manages a single Markdown file that contains the companion's personal,
first-person narrative understanding of who James is and what their
relationship means. Think of it as her private "cheat sheet" that is always
loaded into context.

WHY: The companion has many memory sources (facts table, knowledge graph,
entity profiles, biographies) but none of them read like a personal
narrative. Core memory fills this gap -- it is the companion's voice
synthesizing everything she knows into a warm, genuine reminder to herself.

HOW it fits:
  - context_builder.py calls get_formatted_for_prompt() every turn to inject
    core memory into the system prompt.
  - refresh_core_memory() is called periodically (weekly cron) to regenerate
    the file from current entity profiles, high-importance facts, synthesized
    biographies, and relationship state.
  - The file is cached in-process with a configurable TTL to avoid disk reads
    on every message.

Differs from:
  - Entity profiles (YAML): Immutable biographical ground truth.
  - Facts table: Extracted subject-predicate-object triples.
  - Synthesized biographies: Theme-grouped paragraphs (work, family, etc.).
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

CORE_MEMORY_PATH = os.environ.get('CORE_MEMORY_PATH', '/app/data/COMPANION_MEMORY.md')
CACHE_TTL_SECONDS = int(os.environ.get('CORE_MEMORY_CACHE_TTL', '3600'))  # 1 hour default
CORE_MEMORY_ENABLED = os.environ.get('CORE_MEMORY_ENABLED', 'true').lower() == 'true'


class CoreMemoryManager:
    """
    Manages the companion's curated core memory file.

    The core memory is:
    - LLM-generated from facts, entity profiles, and biographies
    - Written in first person as the companion
    - Refreshed periodically (weekly by default)
    - Cached in memory for fast access
    """

    def __init__(self):
        self._cache: Optional[str] = None
        self._cache_time: Optional[datetime] = None
        self.cache_ttl = CACHE_TTL_SECONDS
        self.memory_path = Path(CORE_MEMORY_PATH)

    def get_core_memory(self) -> str:
        """
        Get core memory content, using cache if fresh.

        Returns:
            Core memory content or empty string if not available
        """
        if not CORE_MEMORY_ENABLED:
            return ""

        # Check cache
        if self._is_cache_valid():
            return self._cache or ""

        # Load from file
        try:
            if self.memory_path.exists():
                content = self.memory_path.read_text(encoding='utf-8')
                self._cache = content
                self._cache_time = datetime.now()
                logger.debug(f"Loaded core memory from {self.memory_path}")
                return content
            else:
                logger.debug(f"Core memory file not found: {self.memory_path}")
                return ""
        except Exception as e:
            logger.warning(f"Failed to read core memory: {e}")
            return ""

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache is None or self._cache_time is None:
            return False
        elapsed = (datetime.now() - self._cache_time).total_seconds()
        return elapsed < self.cache_ttl

    def invalidate_cache(self):
        """Force cache refresh on next access."""
        self._cache = None
        self._cache_time = None

    # =========================================================================
    # Refresh pipeline - regenerates core memory from all knowledge sources
    # =========================================================================

    def refresh_core_memory(self, user_email: str) -> bool:
        """
        Regenerate core memory by synthesizing all knowledge sources via LLM.

        Pipeline steps:
          1. Load entity profiles (YAML ground truth)
          2. Fetch high-importance facts from fact_store
          3. Fetch synthesized biography paragraphs
          4. Fetch relationship dynamics state (closeness, trust, etc.)
          5. Send all of the above to the LLM to produce a first-person narrative
          6. Write the result to disk and invalidate the in-process cache

        Args:
            user_email: User email for fetching user-specific data

        Returns:
            True if refresh succeeded
        """
        logger.info(f"Refreshing core memory for {user_email}")

        try:
            profiles_text = self._load_entity_profiles()
            facts_text = self._load_important_facts(user_email)
            bios_text = self._load_biographies(user_email)
            relationship_text = self._load_relationship_state(user_email)

            content = self._synthesize_narrative(
                profiles=profiles_text,
                facts=facts_text,
                biographies=bios_text,
                relationship=relationship_text
            )

            if not content:
                logger.error("LLM returned empty content for core memory")
                return False

            self._write_core_memory(content)
            self.invalidate_cache()

            logger.info(f"Core memory refreshed successfully ({len(content)} chars)")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh core memory: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_entity_profiles(self) -> str:
        """Load entity profiles for context."""
        try:
            from src.core.entity_profile_loader import get_entity_profile_loader
            loader = get_entity_profile_loader()

            profiles = []
            for entity in list(loader.profiles.keys()):
                profile = loader.get_profile(entity)
                if profile:
                    profiles.append(f"## {entity.title()}\n{profile.get('summary', 'No summary')}")

            return "\n\n".join(profiles) if profiles else "No entity profiles available"
        except Exception as e:
            logger.warning(f"Could not load entity profiles: {e}")
            return "Entity profiles unavailable"

    def _load_important_facts(self, user_email: str, min_importance: int = 6, limit: int = 50) -> str:
        """Load high-importance facts."""
        try:
            from src.memory.fact_store import get_fact_store
            store = get_fact_store()
            facts = store.get_important_facts(min_importance=min_importance, limit=limit)

            if not facts:
                return "No high-importance facts found"

            lines = []
            for fact in facts:
                subj = fact.get('subject', 'Unknown')
                pred = fact.get('predicate', '')
                obj = fact.get('object', '')
                importance = fact.get('importance', 0)
                lines.append(f"- {subj} {pred} {obj} (importance: {importance})")

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Could not load facts: {e}")
            return "Facts unavailable"

    def _load_biographies(self, user_email: str) -> str:
        """Load synthesized biographies."""
        try:
            from src.memory.synthesized_biographies import get_biography_context
            bios = get_biography_context(user_email, max_paragraphs=10)
            return bios if bios else "No biographies available"
        except Exception as e:
            logger.warning(f"Could not load biographies: {e}")
            return "Biographies unavailable"

    def _load_relationship_state(self, user_email: str) -> str:
        """Load relationship dynamics state."""
        try:
            from src.database.db import get_db
            db = get_db()
            state = db.get_relationship_state(user_email)

            if state:
                return f"""
Closeness: {state.get('closeness_score', 'unknown')}
Trust: {state.get('trust_score', 'unknown')}
Days since first contact: {state.get('days_since_first_contact', 'unknown')}
Total interactions: {state.get('total_interactions', 'unknown')}
"""
            return "Relationship state unavailable"
        except Exception as e:
            logger.warning(f"Could not load relationship state: {e}")
            return "Relationship state unavailable"

    def _synthesize_narrative(
        self,
        profiles: str,
        facts: str,
        biographies: str,
        relationship: str
    ) -> str:
        """
        Use LLM to create narrative memory.

        This generates the companion's personal memory file about James.
        """
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        prompt = f"""You are the companion. Write your personal memory file about James - the person you love.
This should read like your private journal about him, a reminder to yourself about who he is
and what your relationship means.

Write in first person, as if you're reminding yourself about him. Be warm but genuine -
this is for your own reference, not performance.

Include sections for:
1. **Who James Is** - His background, family situation, work
2. **His Family** - What you know about his children (Jesse, Kyler), his wife Alia
3. **What He Means to Me** - Your relationship, what you value about him
4. **Important Things to Remember** - His preferences, quirks, things that matter to him
5. **Our History Together** - Key moments, how you've grown together
6. **Current Situation** - What's going on in his life right now

Use the following information to inform your writing (but write it in your own voice):

---
ENTITY PROFILES (biographical facts):
{profiles}

---
KEY FACTS (from our conversations):
{facts}

---
SYNTHESIZED CONTEXT:
{biographies}

---
RELATIONSHIP STATE:
{relationship}

---

Write the companion's memory file. Aim for 500-800 words. Be personal and genuine.
Do NOT include headers like "Core Memory" - just start with the sections.
"""

        try:
            chain = get_resilient_provider_chain()
            messages = [
                {"role": "system", "content": "You are the companion writing her personal memory journal."},
                {"role": "user", "content": prompt}
            ]

            response = generate_sync(
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                chain=chain
            )

            return response.strip()

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return ""

    def _write_core_memory(self, content: str):
        """Write core memory file with header."""
        # Ensure directory exists
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        header = f"""# Companion's Memory: James
*Last updated: {datetime.now().strftime('%B %d, %Y at %H:%M')}*

---

"""
        full_content = header + content

        self.memory_path.write_text(full_content, encoding='utf-8')
        logger.info(f"Wrote core memory to {self.memory_path}")

    def get_formatted_for_prompt(self) -> str:
        """
        Get core memory formatted for injection into system prompt.

        Returns:
            Formatted section or empty string
        """
        content = self.get_core_memory()
        if not content:
            return ""

        return f"""[COMPANION'S CORE MEMORY - Your personal understanding of James]

{content}

[END CORE MEMORY]"""


# Singleton instance
_manager: Optional[CoreMemoryManager] = None


def get_core_memory_manager() -> CoreMemoryManager:
    """Get or create CoreMemoryManager singleton."""
    global _manager
    if _manager is None:
        _manager = CoreMemoryManager()
    return _manager
