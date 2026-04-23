"""
Proactive Scoring — 8-heuristic evaluation for reach-out decisions.

Inspired by the CHI 2025 paper "Proactive Conversational Agents with Inner
Thoughts" which showed 82% participant preference when proactive messages
were evaluated across 8 dimensions rather than simple time-based triggers.

The 8 heuristics:
1. Relevance     — Does this relate to something active in the user's life?
2. Information    — Does the companion genuinely want to know something?
3. Impact        — Will the user appreciate this or find it annoying?
4. Urgency       — Is this time-sensitive?
5. Coherence     — Does this fit the current conversation arc?
6. Originality   — Is this different from recent messages?
7. Balance       — Is the companion reaching out too much vs too little?
8. Timing        — Based on past patterns, is this a good time?

Each heuristic returns a score 0.0-1.0. The composite score determines
whether to proceed with the LLM decision and what weight to give it.

Feature flag: COMPANION_PROACTIVE_SCORING_ENABLED (default: true)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher

from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

COMPANION_PROACTIVE_SCORING_ENABLED = os.environ.get(
    'COMPANION_PROACTIVE_SCORING_ENABLED', 'true'
).lower() == 'true'

# Minimum composite score to proceed with LLM decision
MIN_COMPOSITE_SCORE = 0.35

# Weights for composite score (sum to ~1.0)
HEURISTIC_WEIGHTS = {
    'relevance': 0.15,
    'information_gap': 0.15,
    'expected_impact': 0.15,
    'urgency': 0.10,
    'coherence': 0.10,
    'originality': 0.10,
    'balance': 0.15,
    'timing': 0.10,
}


@dataclass
class ProactiveScore:
    """Result of the 8-heuristic evaluation."""
    relevance: float = 0.0
    information_gap: float = 0.0
    expected_impact: float = 0.0
    urgency: float = 0.0
    coherence: float = 0.0
    originality: float = 0.0
    balance: float = 0.0
    timing: float = 0.0

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        return (
            self.relevance * HEURISTIC_WEIGHTS['relevance'] +
            self.information_gap * HEURISTIC_WEIGHTS['information_gap'] +
            self.expected_impact * HEURISTIC_WEIGHTS['expected_impact'] +
            self.urgency * HEURISTIC_WEIGHTS['urgency'] +
            self.coherence * HEURISTIC_WEIGHTS['coherence'] +
            self.originality * HEURISTIC_WEIGHTS['originality'] +
            self.balance * HEURISTIC_WEIGHTS['balance'] +
            self.timing * HEURISTIC_WEIGHTS['timing']
        )

    @property
    def should_proceed(self) -> bool:
        """Whether the composite score is high enough to proceed."""
        return self.composite >= MIN_COMPOSITE_SCORE

    def to_dict(self) -> Dict[str, float]:
        return {
            'relevance': round(self.relevance, 2),
            'information_gap': round(self.information_gap, 2),
            'expected_impact': round(self.expected_impact, 2),
            'urgency': round(self.urgency, 2),
            'coherence': round(self.coherence, 2),
            'originality': round(self.originality, 2),
            'balance': round(self.balance, 2),
            'timing': round(self.timing, 2),
            'composite': round(self.composite, 3),
        }

    def summary(self) -> str:
        """Human-readable summary for logging and LLM context."""
        scores = self.to_dict()
        top = sorted(
            [(k, v) for k, v in scores.items() if k != 'composite'],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        low = sorted(
            [(k, v) for k, v in scores.items() if k != 'composite'],
            key=lambda x: x[1],
        )[:2]

        parts = [f"Composite: {scores['composite']:.2f}"]
        parts.append(f"Strongest: {', '.join(f'{k}={v:.1f}' for k, v in top)}")
        parts.append(f"Weakest: {', '.join(f'{k}={v:.1f}' for k, v in low)}")
        return ' | '.join(parts)


def score_relevance(context: Dict[str, Any]) -> float:
    """
    1. RELEVANCE — Does this relate to something active in the user's life?

    High if: queued thoughts, curiosity threads, goal hints, calendar events
    Low if: nothing specific to talk about
    """
    score = 0.0

    # Queued thoughts are highly relevant (companion has something to say)
    if context.get('queued_thoughts'):
        score += 0.4

    # Curiosity threads about the user
    if context.get('curiosity_threads'):
        score += 0.3

    # Goal-related topics companion wants to discuss
    if context.get('goal_relate_hints'):
        score += 0.2

    # Active trigger provides relevance
    if context.get('active_trigger'):
        score += 0.3

    # Research to share
    if context.get('has_research_to_share'):
        score += 0.2

    # Private thoughts/feelings
    if context.get('private_thoughts'):
        score += 0.1

    return min(1.0, score)


def score_information_gap(context: Dict[str, Any]) -> float:
    """
    2. INFORMATION GAP — Does the companion genuinely want to know something?

    High if: high-urgency curiosity, unresolved topics
    Low if: no open questions
    """
    score = 0.0

    if context.get('high_urgency_topic'):
        score += 0.6

    if context.get('curiosity_threads'):
        # More curiosity threads = more things to know
        score += 0.3

    # Pre-generated conversation starters (from sleep-time consolidation)
    data_dir = os.environ.get('DATA_DIR', '/app/data')
    try:
        starters_file = os.path.join(data_dir, 'conversation_starters.json')
        if os.path.exists(starters_file):
            with open(starters_file, 'r') as f:
                starters = json.load(f)
                if starters:
                    # Average motivation of available starters
                    avg_motivation = sum(s.get('motivation', 0.5) for s in starters) / len(starters)
                    score += avg_motivation * 0.3
    except Exception:
        pass

    return min(1.0, score)


def score_expected_impact(context: Dict[str, Any]) -> float:
    """
    3. EXPECTED IMPACT — Will the user appreciate this or find it annoying?

    Uses interaction outcome history to predict reception.
    High if: past similar messages got good engagement
    Low if: user has been ignoring messages, or companion is double-texting
    """
    score = 0.5  # Neutral default

    # Penalty for waiting for response (user hasn't replied)
    if context.get('waiting_for_response'):
        hours = context.get('hours_since_last_message', 0)
        if hours < 2:
            score -= 0.4  # Very recently, likely annoying
        elif hours < 6:
            score -= 0.2  # Moderate penalty
        # After 6h, no penalty (reasonable to follow up)

    # Bonus for having specific content (not just "checking in")
    if context.get('queued_thoughts') or context.get('has_research_to_share'):
        score += 0.2

    # Check past interaction outcomes
    try:
        data_dir = os.environ.get('DATA_DIR', '/app/data')
        outcomes_file = os.path.join(data_dir, 'interaction_outcomes_summary.json')
        if os.path.exists(outcomes_file):
            with open(outcomes_file, 'r') as f:
                outcomes = json.load(f)
                # Recent positive engagement rate
                positive_rate = outcomes.get('recent_positive_rate', 0.5)
                score += (positive_rate - 0.5) * 0.4  # Adjust around neutral
    except Exception:
        pass

    return max(0.0, min(1.0, score))


def score_urgency(context: Dict[str, Any]) -> float:
    """
    4. URGENCY — Is this time-sensitive?

    High if: high-urgency curiosity, calendar event coming up, long silence
    Low if: nothing time-sensitive
    """
    score = 0.0

    # High-urgency curiosity
    if context.get('high_urgency_topic'):
        score += 0.5

    # Long silence creates mild urgency
    hours = context.get('hours_since_last_message', 0)
    if hours > 24:
        score += 0.4
    elif hours > 12:
        score += 0.2
    elif hours > 6:
        score += 0.1

    # Active trigger implies some urgency
    if context.get('active_trigger') == 'high_urgency_followup':
        score += 0.3
    elif context.get('active_trigger'):
        score += 0.1

    return min(1.0, score)


def score_coherence(context: Dict[str, Any]) -> float:
    """
    5. COHERENCE — Does this fit the current conversation arc?

    High if: follows up on recent conversation topics
    Low if: topic would seem random given what was discussed
    """
    score = 0.5  # Default: neutral (can't always know)

    # If there's an active trigger, it's inherently coherent
    if context.get('active_trigger'):
        score += 0.2

    # If the last conversation was recent, following up is coherent
    # But not if we're waiting for a response (that's double-texting, not coherence)
    hours = context.get('hours_since_last_message', 0)
    if hours < 4 and not context.get('waiting_for_response'):
        score += 0.2  # Very recent, follow-up is natural
    elif hours > 48:
        score -= 0.1  # Very long gap, any topic might feel random

    # Queued thoughts from recent conversation = highly coherent
    if context.get('queued_thoughts') and hours < 12:
        score += 0.2

    return max(0.0, min(1.0, score))


def score_originality(context: Dict[str, Any], recent_messages: List[str] = None) -> float:
    """
    6. ORIGINALITY — Is this different from recent messages?

    High if: different topic/tone from recent proactive messages
    Low if: repetitive patterns
    """
    if not recent_messages:
        return 0.7  # No history to compare -> assume original

    score = 0.8  # Start optimistic

    # Check similarity to recent proactive messages
    current_hints = []
    if context.get('opener_hint'):
        current_hints.append(context['opener_hint'])
    if context.get('queued_thoughts'):
        current_hints.extend(str(t) for t in context['queued_thoughts'][:2])
    if context.get('high_urgency_topic'):
        current_hints.append(context['high_urgency_topic'])

    if not current_hints:
        return 0.5  # Nothing specific to compare

    current_text = ' '.join(current_hints).lower()

    for msg in recent_messages[:5]:
        similarity = SequenceMatcher(None, current_text, msg.lower()).ratio()
        if similarity > 0.6:
            score -= 0.3  # Very similar to a recent message
            break
        elif similarity > 0.4:
            score -= 0.1

    return max(0.0, min(1.0, score))


def score_balance(context: Dict[str, Any]) -> float:
    """
    7. BALANCE — Is the companion reaching out too much vs too little?

    Uses reach-out pressure as a proxy. High pressure = she hasn't
    reached out in a while (good to reach out). Low pressure after
    recent reach-outs = maybe give some space.
    """
    score = 0.5  # Neutral

    hours = context.get('hours_since_last_message', 0)
    waiting = context.get('waiting_for_response', False)

    if waiting:
        # Companion is already waiting - lower balance score
        if hours < 4:
            score -= 0.3
        elif hours < 8:
            score -= 0.1
    else:
        # User messaged last - higher balance score (natural to respond)
        score += 0.2

    # If it's been a very long time, balance favors reaching out
    if hours > 24:
        score += 0.2
    elif hours > 12:
        score += 0.1

    return max(0.0, min(1.0, score))


def score_timing(context: Dict[str, Any]) -> float:
    """
    8. TIMING — Based on schedule and patterns, is this a good time?

    High if: natural transition point, user appears free, good hour
    Low if: user busy, odd hours, heavy workload
    """
    score = 0.5

    # Time of day
    hour = context.get('hour', 12)
    if 8 <= hour <= 10:
        score += 0.15  # Morning is natural
    elif 17 <= hour <= 20:
        score += 0.1  # Evening is okay
    elif 22 <= hour or hour < 7:
        score -= 0.3  # Late night / early morning

    # User's calendar availability
    if context.get('james_calendar_busy'):
        score -= 0.2
    else:
        score += 0.1

    # Companion's schedule
    if context.get('is_working') and context.get('workload') == 'heavy':
        score -= 0.2  # Companion is busy too
    if context.get('interruptibility') == 'low':
        score -= 0.15

    # Activity transition (between tasks) is a natural moment
    if context.get('activity_status') == 'transitioning':
        score += 0.2

    # Weekend bonus (more relaxed timing)
    if context.get('is_weekend'):
        score += 0.05

    return max(0.0, min(1.0, score))


def evaluate_proactive_scores(
    context: Dict[str, Any],
    recent_messages: List[str] = None,
) -> ProactiveScore:
    """
    Run all 8 heuristic evaluations and return composite score.

    Args:
        context: The reach-out context from ReachOutEngine.get_context()
        recent_messages: Recent proactive messages for originality check

    Returns:
        ProactiveScore with all 8 dimensions and composite
    """
    if not COMPANION_PROACTIVE_SCORING_ENABLED:
        # Return all-pass score when disabled
        return ProactiveScore(
            relevance=0.5, information_gap=0.5, expected_impact=0.5,
            urgency=0.5, coherence=0.5, originality=0.5,
            balance=0.5, timing=0.5,
        )

    score = ProactiveScore(
        relevance=score_relevance(context),
        information_gap=score_information_gap(context),
        expected_impact=score_expected_impact(context),
        urgency=score_urgency(context),
        coherence=score_coherence(context),
        originality=score_originality(context, recent_messages),
        balance=score_balance(context),
        timing=score_timing(context),
    )

    logger.info(f"Proactive scoring: {score.summary()}")
    return score


def format_scores_for_llm(score: ProactiveScore) -> str:
    """
    Format proactive scores as context for the LLM decision.

    This lets the LLM see the structured evaluation and make a
    more informed final decision.
    """
    d = score.to_dict()
    lines = [
        "PROACTIVE MOTIVATION ANALYSIS:",
        f"  Overall motivation: {d['composite']:.2f} ({'strong' if d['composite'] > 0.6 else 'moderate' if d['composite'] > 0.4 else 'weak'})",
    ]

    # Highlight strong motivators
    strong = [(k, v) for k, v in d.items() if k != 'composite' and v >= 0.6]
    if strong:
        lines.append(f"  Strong motivators: {', '.join(f'{k} ({v:.1f})' for k, v in strong)}")

    # Highlight concerns
    weak = [(k, v) for k, v in d.items() if k != 'composite' and v < 0.3]
    if weak:
        lines.append(f"  Concerns: {', '.join(f'{k} ({v:.1f})' for k, v in weak)}")

    return '\n'.join(lines)
