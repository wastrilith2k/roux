"""
Goal Formation -- LLM-driven goal creation from various signal sources.

WHAT: Two formation pathways that analyze the companion's internal state and
      produce new goals:
      1. form_goals_from_reflections() -- weekly, analyzes journal entries
      2. form_goals_from_signals()     -- every 12 h, analyzes curiosities /
                                          opinions / research findings

WHY:  Goals shouldn't be hand-configured.  They emerge naturally from what the
      companion thinks about, worries about, and is curious about -- just like
      a real person's goals come from reflection and experience.

HOW IT FITS:
  - Both functions call GoalManager.create_goal() which handles deduplication
    and the active-goal cap (7).
  - Scheduled by the AlwaysOnService (reflections weekly, signals every 12 h).
  - Created goals are later decomposed into steps by GoalPlanner.
"""

import os
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# Shared LLM response helpers
# =============================================================================

def _parse_llm_json(response: str) -> List[dict]:
    """Parse an LLM response as a JSON array, stripping markdown fences."""
    text = response.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        json_lines = [l for l in lines if not l.startswith('```')]
        text = '\n'.join(json_lines)
    return json.loads(text)


def _create_goals_from_parsed(manager, goals_data: List[dict]) -> int:
    """Iterate parsed LLM output and call create_goal for each. Returns count created."""
    count = 0
    for g in goals_data:
        if isinstance(g, dict) and 'goal' in g:
            result = manager.create_goal(
                goal=g['goal'],
                motivation=g.get('motivation', ''),
                category=g.get('category', 'general'),
                priority=g.get('priority', 0.5),
                goal_mode=g.get('goal_mode', 'doing'),
                energy_cost=g.get('energy_cost', 'medium')
            )
            if result:  # None means duplicate or limit hit
                count += 1
    return count


# =============================================================================
# Pathway 1: Goals from journal reflections (weekly)
# =============================================================================

def form_goals_from_reflections(user_email: str = None) -> int:
    """
    Form goals from recent daily reflections in the companion journal.

    Looks for patterns like:
      - "I want to..." statements
      - Concerns about the user that suggest supportive goals
      - Curiosity patterns that suggest exploration goals
      - Personal hobby excitement -> personal goals
      - Desire for downtime -> being goals

    Returns the number of goals formed.
    """
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email

    try:
        from src.memory.companion_journal import get_companion_journal
        from src.llm.provider_factory import generate_sync
        from src.autonomy.goals import get_goal_manager

        journal = get_companion_journal(user_email)
        manager = get_goal_manager(user_email)

        # Gather inputs: recent reflections + existing goals (for dedup in prompt)
        reflections = journal.get_recent_reflections(days=7, entry_type='daily_reflection')
        if not reflections:
            return 0

        existing_goals = manager.get_active_goals(limit=10)
        existing_goal_text = (
            "\n".join([f"- {g.goal}" for g in existing_goals])
            if existing_goals else "None"
        )

        reflection_text = "\n\n".join([
            f"**{r['date']}:**\n{r['content'][:500]}"
            for r in reflections[:5]
        ])

        prompt = f"""Analyze the companion's recent reflections and identify any personal goals she seems to be forming.

RECENT REFLECTIONS:
{reflection_text}

EXISTING GOALS (don't duplicate):
{existing_goal_text}

Look for:
- "I want to understand X" → understanding goal
- Concerns about James's wellbeing → supportive goal
- Interest patterns → exploration goal
- Relationship desires → relationship goal
- Self-improvement desires → self goal
- Personal hobbies she's excited about → personal goal (swimming, cosplay, photography, cooking, etc.)
- Desire for downtime or rest → being goal
- Something she wants to discuss with James → relating goal

IMPORTANT: The companion is a whole person. She should have personal goals about her own hobbies
and interests (swimming, cosplay, photography, cooking, reading) — not just relationship goals.
Being goals ("have some downtime", "take it easy today") are just as valuable as doing goals.

Return JSON array of goals (max 2, empty if none new):
[
  {{
    "goal": "short goal statement",
    "motivation": "why she wants this",
    "category": "relationship|self|understanding|support|personal",
    "priority": 0.1-0.9,
    "goal_mode": "doing|being|relating",
    "energy_cost": "none|low|medium|high"
  }}
]

goal_mode guide:
- "doing" = requires active steps (research, create, prepare)
- "being" = a state to maintain, not a task (rest, relax, enjoy)
- "relating" = about the relationship, best done in conversation

energy_cost guide:
- "none" = just exist in this state
- "low" = quick lookup or small action
- "medium" = focused work
- "high" = multi-step sustained effort

Only include genuinely new goals with evidence in reflections.
Return ONLY valid JSON array:"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        if not response:
            return 0

        goals = _parse_llm_json(response)
        count = _create_goals_from_parsed(manager, goals)

        logger.info(f"Formed {count} goals from reflections")
        return count

    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse goal response: {e}")
        return 0
    except Exception as e:
        logger.warning(f"Goal formation failed: {e}")
        return 0


# =============================================================================
# Pathway 2: Goals from multi-signal sources (every 12 h)
# =============================================================================

def form_goals_from_signals(user_email: str = None) -> int:
    """
    Form goals from multiple signal sources beyond journal reflections.

    Signal sources:
      - High-urgency curiosity threads (urgency >= 0.7)
      - Strong opinions (confidence >= 0.7)
      - Unshared research findings

    Returns the number of goals formed.
    """
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email

    try:
        from src.llm.provider_factory import generate_sync
        from src.autonomy.goals import get_goal_manager

        manager = get_goal_manager(user_email)

        # ---- 1. Gather signals from various subsystems ----
        signals = []

        # High-urgency curiosities
        try:
            from src.core.proactive_curiosity import get_proactive_curiosity
            curiosity = get_proactive_curiosity()
            urgent = curiosity.get_high_urgency_topics(threshold=0.7)
            for t in urgent[:3]:
                signals.append(f"High-urgency curiosity ({t.urgency:.0%}): {t.topic}")
        except Exception as e:
            logger.debug(f"Could not get curiosities for goal signals: {e}")

        # Strong opinions
        try:
            from src.autonomy.opinion_store import get_opinion_store
            store = get_opinion_store()
            strong = store.get_strong_opinions(min_confidence=0.7)
            for op in strong[:3]:
                signals.append(f"Strong opinion ({op.confidence:.0%}): {op.opinion}")
        except Exception as e:
            logger.debug(f"Could not get opinions for goal signals: {e}")

        # Unshared research findings
        try:
            data_dir = os.environ.get('DATA_DIR', '/app/data')
            findings_file = os.path.join(data_dir, 'autonomous_findings.json')
            if os.path.exists(findings_file):
                with open(findings_file, 'r') as f:
                    findings = json.load(f)
                unshared = [item for item in findings if not item.get('shared', False)]
                for item in unshared[:2]:
                    signals.append(
                        f"Unshared finding: {item.get('topic', 'unknown')} - "
                        f"{(item.get('finding', '') or '')[:100]}"
                    )
        except Exception as e:
            logger.debug(f"Could not get findings for goal signals: {e}")

        if not signals:
            logger.debug("No signals for goal formation")
            return 0

        # ---- 2. Get existing goals for dedup context ----
        existing_goals = manager.get_active_goals(limit=10)
        existing_text = (
            "\n".join([f"- {g.goal}" for g in existing_goals])
            if existing_goals else "None"
        )
        signals_text = "\n".join([f"- {s}" for s in signals])

        # ---- 3. LLM decides what goals to form ----
        prompt = f"""Given these signals about what the companion cares about, what 1-2 goals
would naturally emerge? Goals should be things SHE wants to do,
not things to help James with.

SIGNALS:
{signals_text}

EXISTING GOALS (don't duplicate):
{existing_text}

IMPORTANT:
- Only create goals that are genuinely supported by the signals
- The companion is a whole person with her own interests
- Being goals are valid (e.g. "take time to process these feelings")
- Don't force goals from weak signals

Return JSON array (max 2, empty [] if nothing warrants a new goal):
[
  {{
    "goal": "short goal statement",
    "motivation": "why, based on which signal(s)",
    "category": "relationship|self|understanding|support|personal",
    "priority": 0.1-0.9,
    "goal_mode": "doing|being|relating",
    "energy_cost": "none|low|medium|high"
  }}
]

Return ONLY valid JSON array:"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        if not response:
            return 0

        goals = _parse_llm_json(response)
        count = _create_goals_from_parsed(manager, goals)

        logger.info(f"Formed {count} goals from {len(signals)} signals")
        return count

    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse goal signal response: {e}")
        return 0
    except Exception as e:
        logger.warning(f"Signal-based goal formation failed: {e}")
        return 0
