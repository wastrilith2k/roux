"""
Schedule Context — learned user routine from conversation.

WHAT: Retrieves schedule-related facts that the companion has learned from
      conversations (e.g., "I work 9-5", "I pick up the kids at 3:30",
      "I'm usually free on weekends") and formats them as time-relevant
      context for the system prompt.

WHY:  The hardcoded schedule in user_context.py is static and can't adapt
      when the user's routine changes. The companion should learn routine
      information organically from conversation and use it to reason about
      what the user is likely doing at any given time.

HOW:  Queries the fact store for facts with schedule-related predicates
      (e.g., "has_routine", "works_at", "schedule") and filters them based
      on the current time of day and day of week. The result supplements
      the existing time awareness context.

Singleton: get_schedule_context_provider() at module bottom.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

# Predicates that indicate schedule/routine knowledge in the fact store.
# These are matched case-insensitively against fact predicates.
SCHEDULE_PREDICATES = [
    'has_routine',
    'has_schedule',
    'works',
    'works_at',
    'commutes',
    'picks_up',
    'drops_off',
    'attends',
    'exercises',
    'goes_to',
    'usually_does',
    'free_on',
    'busy_on',
    'available',
    'unavailable',
    'wakes_up',
    'sleeps_at',
    'schedule',
    'routine',
]

# Keywords that suggest a fact contains schedule/routine information,
# even if the predicate doesn't match SCHEDULE_PREDICATES exactly.
SCHEDULE_KEYWORDS = [
    'morning', 'afternoon', 'evening', 'night',
    'weekday', 'weekend', 'monday', 'tuesday', 'wednesday',
    'thursday', 'friday', 'saturday', 'sunday',
    'work', 'job', 'office', 'commute',
    'school', 'class', 'pick up', 'drop off',
    'gym', 'exercise', 'workout',
    'lunch', 'dinner', 'breakfast',
    'routine', 'schedule', 'usually', 'typically',
    'every day', 'every week', 'always',
    'free', 'busy', 'available',
    'a.m.', 'p.m.', 'am', 'pm',
]

# Time-of-day labels to fact content keyword associations.
# Used to boost relevance of facts that mention the current time period.
TIME_OF_DAY_KEYWORDS = {
    'morning': ['morning', 'breakfast', 'wake', 'commute', 'gym', 'exercise', 'school', 'drop off'],
    'afternoon': ['afternoon', 'lunch', 'work', 'office', 'school', 'pick up'],
    'evening': ['evening', 'dinner', 'kids', 'family', 'home', 'pick up', 'commute'],
    'night': ['night', 'bed', 'sleep', 'relax', 'wind down', 'late'],
}

# Day-type labels for relevance boosting.
DAY_TYPE_KEYWORDS = {
    'weekday': ['work', 'job', 'office', 'commute', 'school', 'weekday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
    'weekend': ['weekend', 'saturday', 'sunday', 'kids', 'relax', 'free', 'day off', 'off work'],
}

# Module-level singleton
_provider: Optional['ScheduleContextProvider'] = None


def get_schedule_context_provider() -> 'ScheduleContextProvider':
    """Get or create the singleton ScheduleContextProvider."""
    global _provider
    if _provider is None:
        _provider = ScheduleContextProvider()
    return _provider


class ScheduleContextProvider:
    """
    Provides learned schedule context from the fact store.

    Queries for schedule-related facts and filters them by relevance
    to the current time of day and day of week.
    """

    def __init__(self):
        self._fact_store = None

    @property
    def fact_store(self):
        """Lazy-load fact store."""
        if self._fact_store is None:
            from src.memory.fact_store import FactStore
            self._fact_store = FactStore()
        return self._fact_store

    def get_learned_schedule_facts(self, user_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve facts from the fact store that represent schedule/routine knowledge.

        Searches for facts whose predicate matches known schedule predicates,
        or whose content contains schedule-related keywords.

        Args:
            user_name: The user's name to search as fact subject

        Returns:
            List of fact dicts with schedule/routine information
        """
        try:
            # Get all facts about the user — the fact store handles
            # connection management and error recovery.
            all_facts = self.fact_store.get_facts_for_subject(
                user_name, limit=100
            )

            schedule_facts = []
            for fact in all_facts:
                if self._is_schedule_fact(fact):
                    schedule_facts.append(fact)

            return schedule_facts

        except Exception as e:
            logger.warning(f"Failed to retrieve schedule facts: {e}")
            return []

    def _is_schedule_fact(self, fact: Dict[str, Any]) -> bool:
        """
        Determine if a fact represents schedule/routine knowledge.

        Checks both the predicate and the object (content) for schedule indicators.
        """
        predicate = (fact.get('predicate') or '').lower().strip()
        obj = (fact.get('object') or '').lower()

        # Check predicate against known schedule predicates
        for sched_pred in SCHEDULE_PREDICATES:
            if sched_pred in predicate:
                return True

        # Check object content for schedule keywords
        for keyword in SCHEDULE_KEYWORDS:
            if keyword in obj:
                return True

        return False

    def compute_time_relevance(
        self,
        fact: Dict[str, Any],
        time_of_day: str,
        day_type: str
    ) -> float:
        """
        Score how relevant a schedule fact is to the current time context.

        Args:
            fact: The fact dict
            time_of_day: One of 'morning', 'afternoon', 'evening', 'night'
            day_type: One of 'weekday', 'weekend'

        Returns:
            Relevance score from 0.0 to 1.0 (higher = more relevant now)
        """
        obj = (fact.get('object') or '').lower()
        predicate = (fact.get('predicate') or '').lower()
        content = f"{predicate} {obj}"

        score = 0.0

        # Base relevance: all schedule facts have some value
        score += 0.3

        # Time-of-day boost: facts mentioning current time period are more relevant
        tod_keywords = TIME_OF_DAY_KEYWORDS.get(time_of_day, [])
        for keyword in tod_keywords:
            if keyword in content:
                score += 0.2
                break  # Only count once

        # Day-type boost: weekday/weekend match
        day_keywords = DAY_TYPE_KEYWORDS.get(day_type, [])
        for keyword in day_keywords:
            if keyword in content:
                score += 0.2
                break

        # Importance boost from fact store (0-10 scale, normalized)
        importance = fact.get('importance', 5)
        score += (importance / 10) * 0.2

        # Confidence from fact store
        confidence = fact.get('effective_confidence') or fact.get('confidence', 0.7)
        score *= confidence

        return min(score, 1.0)

    def get_relevant_schedule_context(
        self,
        user_name: str,
        current_time: Optional[datetime] = None,
        max_facts: int = 8
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get schedule facts ranked by relevance to the current time.

        Args:
            user_name: User's name for fact lookup
            current_time: Current datetime (defaults to clock_now())
            max_facts: Maximum number of facts to return

        Returns:
            List of (fact, relevance_score) tuples, sorted by relevance descending
        """
        if current_time is None:
            current_time = clock_now()

        # Determine time context
        hour = current_time.hour
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'

        day_type = 'weekend' if current_time.weekday() >= 5 else 'weekday'

        # Retrieve all schedule facts
        facts = self.get_learned_schedule_facts(user_name)

        # Score and rank by relevance
        scored = []
        for fact in facts:
            relevance = self.compute_time_relevance(fact, time_of_day, day_type)
            scored.append((fact, relevance))

        # Sort by relevance (descending), then importance (descending)
        scored.sort(key=lambda x: (x[1], x[0].get('importance', 0)), reverse=True)

        return scored[:max_facts]

    def format_for_prompt(
        self,
        user_name: str,
        current_time: Optional[datetime] = None
    ) -> str:
        """
        Format learned schedule context for injection into the system prompt.

        Returns an empty string if no schedule facts are available.

        Args:
            user_name: User's name for fact lookup
            current_time: Current datetime (defaults to clock_now())

        Returns:
            Formatted schedule context string, or empty string
        """
        ranked_facts = self.get_relevant_schedule_context(
            user_name, current_time
        )

        if not ranked_facts:
            return ""

        lines = [f"[{user_name}'s LEARNED ROUTINE — from past conversations]"]

        for fact, relevance in ranked_facts:
            predicate = fact.get('predicate', '')
            obj = fact.get('object', '')
            # Format as a natural sentence
            line = f"- {predicate}: {obj}"
            lines.append(line)

        from src.config.persona_config import get_persona_config
        u_subject = get_persona_config().user_pronoun_subject

        lines.append(
            f"\nUse this to understand what {user_name} is likely doing "
            f"right now. These are things {u_subject}'s told you — reference them "
            "naturally, not as a list."
        )

        return "\n".join(lines)
