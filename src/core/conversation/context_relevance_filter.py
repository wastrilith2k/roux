"""
Context Relevance Filter - Scores and filters context sections by relevance.

Evaluates which context sections are relevant to the current message and
suppresses irrelevant ones to reduce prompt noise and token usage.

Protected sections (entity_profiles, personality, continuity_context) are
never excluded.

Feature flag: COMPANION_CONTEXT_FILTER_ENABLED (default: false — needs testing before enabling)
"""

import os
import logging
import time
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

COMPANION_CONTEXT_FILTER_ENABLED = os.environ.get('COMPANION_CONTEXT_FILTER_ENABLED', 'false').lower() == 'true'

# Sections that are ALWAYS included regardless of relevance score
PROTECTED_SECTIONS = {
    'entity_profiles',
    'personality',
    'continuity_context',
    'identity_anchor',
    'current_time',
    'core_memory',
    'cognitive_instructions',
    'final_reminder',
}


class ContextRelevanceFilter:
    """Filters context sections based on relevance to current message."""

    def filter_sections(
        self,
        user_message: str,
        sections: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """
        Score and filter context sections by relevance.

        Args:
            user_message: The current user message
            sections: List of (section_name, section_content) tuples

        Returns:
            Filtered list of (section_name, section_content) tuples
        """
        if not COMPANION_CONTEXT_FILTER_ENABLED:
            return sections

        if not user_message or not sections:
            return sections

        start = time.time()

        # Separate protected and filterable sections
        protected = []
        filterable = []
        for name, content in sections:
            if name in PROTECTED_SECTIONS:
                protected.append((name, content))
            else:
                filterable.append((name, content))

        if not filterable:
            return sections

        try:
            from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

            # Build section list for scoring
            section_list = "\n".join(
                f"{i+1}. [{name}]: {content[:150]}..."
                for i, (name, content) in enumerate(filterable)
            )

            prompt = f"""Given this message from James, score each context section's relevance (1-5).

James's message: {user_message[:300]}

Context sections:
{section_list}

For each section number, respond with ONLY the number and score, one per line.
Score 1 = irrelevant, 3 = somewhat relevant, 5 = highly relevant.
Example:
1: 4
2: 1
3: 5"""

            chain = get_resilient_provider_chain()
            response = generate_sync(
                messages=[
                    {"role": "system", "content": "Score context relevance 1-5. Reply with ONLY number:score pairs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                chain=chain,
                timeout=3
            )

            # Parse scores
            scores = {}
            for line in response.strip().split('\n'):
                line = line.strip()
                if ':' in line:
                    try:
                        idx_str, score_str = line.split(':', 1)
                        idx = int(idx_str.strip()) - 1  # 0-indexed
                        score = int(score_str.strip())
                        scores[idx] = min(5, max(1, score))
                    except (ValueError, IndexError):
                        continue

            # Filter: keep sections with score >= 2
            kept = []
            removed = []
            for i, (name, content) in enumerate(filterable):
                score = scores.get(i, 3)  # Default to 3 (keep) if not scored
                if score >= 2:
                    kept.append((name, content))
                else:
                    removed.append(name)

            processing_time = int((time.time() - start) * 1000)

            if removed:
                logger.info(
                    f"Context filter: removed {len(removed)} sections "
                    f"({', '.join(removed)}), kept {len(kept)} ({processing_time}ms)"
                )
            else:
                logger.debug(f"Context filter: all sections kept ({processing_time}ms)")

            return protected + kept

        except Exception as e:
            processing_time = int((time.time() - start) * 1000)
            logger.warning(f"Context relevance filter failed ({processing_time}ms): {e}")
            return sections  # Fail open — return all sections


# Singleton
_filter: Optional[ContextRelevanceFilter] = None


def get_context_relevance_filter() -> ContextRelevanceFilter:
    global _filter
    if _filter is None:
        _filter = ContextRelevanceFilter()
    return _filter
