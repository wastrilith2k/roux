"""
Token Budget — Per-source token budgets and truncation for context assembly.

WHAT: Defines per-source token budgets, priority tiers, and sentence-aware
      truncation so that no single context source can dominate the prompt.
WHY:  Without per-source caps, a verbose biography or memory dump can consume
      thousands of tokens while leaving insufficient room for reasoning.
      Count-based limits (e.g. 15 memories, 12 paragraphs) don't map to
      consistent token usage. Token budgets enforce predictable allocation.
HOW:  Each source has a budget in estimated tokens. truncate_to_budget()
      trims text at a sentence boundary. Priority tiers control which
      sources lose budget first when the total cap is exceeded.

See issue #21 for design rationale.
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Fast token estimate: ~4 characters per token.

    This is a standard heuristic that's accurate enough for budget
    enforcement without requiring a tokenizer dependency.
    """
    return len(text) // 4


# ---------------------------------------------------------------------------
# Per-source token budgets
# ---------------------------------------------------------------------------

SOURCE_TOKEN_BUDGETS = {
    'entity_profiles': 1500,
    'personality': 800,
    'memories': 1200,
    'conversation_history': 2000,
    'biographies': 1000,
    'graphiti_context': 800,
    'core_memory': 600,
    'scene_state': 600,
    'internal_state': 400,
    'relationship_dynamics': 600,
    'relationship_insights': 500,
    'relationship_evaluation': 500,
    'episode_context': 600,
    'observations_context': 600,
    'session_summary': 600,
    'reflections_context': 500,
    'opinions_context': 500,
    'curiosity_context': 400,
    'goals_context': 400,
    'values_context': 400,
    'activities_context': 400,
    'temporal_context': 400,
    'synthesized_events': 500,
    'fertility_context': 300,
    'user_context': 300,
    'presence_mode': 200,
    'continuity_context': 400,
    'memory_validation': 800,
    'derived_scene_context': 500,
    'schedule': 600,
}

# Hard cap for all context sources combined (excludes fixed identity/instructions)
TOTAL_CAP = 12000

# Minimum tokens reserved for generation (reasoning headroom)
REASONING_RESERVE = 4000


# ---------------------------------------------------------------------------
# Priority tiers for dynamic reallocation
# ---------------------------------------------------------------------------
# Tier 1: Always full budget — never trimmed
# Tier 2: Proportional trim when over total cap
# Tier 3: Best-effort — first to trim

TIER_1 = frozenset({
    'entity_profiles', 'personality', 'conversation_history', 'core_memory',
    'presence_mode',
})

TIER_2 = frozenset({
    'memories', 'graphiti_context', 'biographies', 'episode_context',
    'memory_validation', 'relationship_dynamics', 'relationship_insights',
    'relationship_evaluation', 'scene_state', 'internal_state',
    'session_summary', 'derived_scene_context', 'schedule',
})

TIER_3 = frozenset({
    'opinions_context', 'curiosity_context', 'reflections_context',
    'activities_context', 'observations_context', 'values_context',
    'goals_context', 'temporal_context', 'synthesized_events',
    'fertility_context', 'user_context', 'continuity_context',
})


def get_tier(source_name: str) -> int:
    """Return the priority tier (1, 2, or 3) for a source."""
    if source_name in TIER_1:
        return 1
    if source_name in TIER_2:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Sentence-boundary-aware truncation
# ---------------------------------------------------------------------------

def _is_sentence_boundary(text: str, i: int) -> bool:
    """Check whether position *i* in *text* is a likely sentence boundary.

    Requires the punctuation to be followed by whitespace (or end-of-string)
    and then an uppercase letter (or end-of-string).  Additionally rejects
    common false positives for periods:

    * Ellipsis — period preceded by another period (``"..."``)
    * Abbreviations — preceding word is 1-2 characters (``"Dr."``, ``"U.S."``)
    """
    ch = text[i]

    # Must be followed by whitespace or end of string
    if i + 1 < len(text) and text[i + 1] not in ' \n\t':
        return False

    # Period-specific guards
    if ch == '.':
        # Reject ellipsis: period preceded by another period
        if i > 0 and text[i - 1] == '.':
            return False

        # Reject short-word abbreviations (Dr., Mr., U., e.g., etc.)
        # Walk back to find the start of the word before the period.
        word_start = i - 1
        while word_start >= 0 and text[word_start] not in ' \n\t':
            if text[word_start] == '.':
                # Embedded period (e.g. "U.S.") — also not a sentence end
                return False
            word_start -= 1
        word_len = i - (word_start + 1)
        if word_len <= 2:
            return False

    # The next non-whitespace character must be uppercase or end-of-string
    j = i + 1
    while j < len(text) and text[j] in ' \n\t':
        j += 1
    if j < len(text) and not text[j].isupper():
        return False

    return True


def truncate_to_budget(text: str, budget_tokens: int) -> str:
    """Truncate text to fit within a token budget at a sentence boundary.

    Tries to break at the last sentence-ending punctuation (.!?) that
    fits within the budget. Falls back to a word boundary if no sentence
    boundary is found.

    Returns the original text if it already fits.
    """
    if not text or estimate_tokens(text) <= budget_tokens:
        return text

    # Convert budget to approximate character limit
    char_limit = budget_tokens * 4

    if char_limit <= 0:
        return ""

    truncated = text[:char_limit]

    # Try to find the last sentence boundary within the limit
    best_break = -1
    for i in range(len(truncated) - 1, -1, -1):
        if truncated[i] in '.!?' and _is_sentence_boundary(truncated, i):
            best_break = i + 1
            break

    if best_break > 0:
        return truncated[:best_break].rstrip()

    # No sentence boundary — fall back to last word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space].rstrip()

    # No word boundary either — hard truncate
    return truncated.rstrip()


# ---------------------------------------------------------------------------
# Apply budgets to a dict of source sections
# ---------------------------------------------------------------------------

def apply_source_budgets(sections: dict) -> dict:
    """Apply per-source token budgets to a sections dict.

    Args:
        sections: Dict of {source_name: content_string} from
                  ConversationContext.to_prompt_sections().

    Returns:
        Dict with the same keys, values truncated to their budgets.
        Logs when truncation occurs.
    """
    result = {}
    truncation_events = []

    for name, content in sections.items():
        budget = SOURCE_TOKEN_BUDGETS.get(name)
        if budget is None:
            # Unknown source — pass through unmodified
            result[name] = content
            continue

        before_tokens = estimate_tokens(content)
        if before_tokens <= budget:
            result[name] = content
            continue

        result[name] = truncate_to_budget(content, budget)
        after_tokens = estimate_tokens(result[name])
        truncation_events.append((name, before_tokens, after_tokens, budget))

    if truncation_events:
        details = ", ".join(
            f"{name}: {before}->{after} (budget {budget})"
            for name, before, after, budget in truncation_events
        )
        logger.info(f"Per-source truncation: {details}")

    return result


# ---------------------------------------------------------------------------
# Enforce total cap with tier-based reallocation
# ---------------------------------------------------------------------------

def enforce_total_cap(sections: dict, total_cap: int = TOTAL_CAP) -> dict:
    """Trim sections by tier to fit within the total token cap.

    Tier 3 sources are trimmed first (proportionally), then Tier 2.
    Tier 1 is never trimmed. If the total still exceeds the cap after
    trimming Tier 2 and 3 to zero, Tier 1 passes through unchanged
    (the caller handles that via the existing _enforce_context_budget).

    Args:
        sections: Dict of {source_name: content_string}, already
                  per-source-budget-truncated.
        total_cap: Maximum total tokens for all sources combined.

    Returns:
        Dict with the same keys, potentially further truncated.
    """
    total_tokens = sum(estimate_tokens(v) for v in sections.values())
    if total_tokens <= total_cap:
        return dict(sections)

    excess = total_tokens - total_cap
    result = dict(sections)

    # Phase 1: Trim Tier 3 proportionally
    excess = _trim_tier(result, 3, excess)
    if excess <= 0:
        return result

    # Phase 2: Trim Tier 2 proportionally
    excess = _trim_tier(result, 2, excess)

    if excess > 0:
        logger.warning(
            f"Total cap still exceeded by ~{excess} tokens after trimming "
            f"Tier 2 and 3. Tier 1 sources untouched."
        )

    return result


def _trim_tier(sections: dict, tier: int, excess: int) -> int:
    """Proportionally trim all sources in a given tier to recover excess tokens.

    Modifies sections in-place. Returns remaining excess (0 or negative
    means we recovered enough).
    """
    if excess <= 0:
        return excess

    tier_sources = {
        name: estimate_tokens(content)
        for name, content in sections.items()
        if get_tier(name) == tier and estimate_tokens(content) > 0
    }
    if not tier_sources:
        return excess

    tier_total = sum(tier_sources.values())
    if tier_total <= 0:
        return excess

    # How much of this tier do we need to cut? Cap at full tier.
    to_cut = min(excess, tier_total)
    cut_fraction = to_cut / tier_total

    trimmed_details = []
    tokens_recovered = 0
    for name, current_tokens in tier_sources.items():
        new_budget = int(current_tokens * (1 - cut_fraction))
        if new_budget < current_tokens:
            before = current_tokens
            sections[name] = truncate_to_budget(sections[name], new_budget)
            after = estimate_tokens(sections[name])
            tokens_recovered += before - after
            trimmed_details.append(f"{name}: {before}->{after}")

    if trimmed_details:
        logger.info(
            f"Tier {tier} reallocation: recovered ~{tokens_recovered} tokens "
            f"({', '.join(trimmed_details)})"
        )

    return excess - tokens_recovered


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def get_budget_diagnostics(sections: dict, provider_limit: int = None) -> dict:
    """Generate per-source token budget diagnostics.

    Returns a dict with per-source usage, truncation info, totals,
    and reasoning headroom.
    """
    per_source = {}
    for name, content in sections.items():
        tokens = estimate_tokens(content)
        budget = SOURCE_TOKEN_BUDGETS.get(name)
        per_source[name] = {
            'tokens': tokens,
            'budget': budget,
            'utilization': round(tokens / budget, 2) if budget and budget > 0 else None,
            'tier': get_tier(name),
        }

    total_tokens = sum(estimate_tokens(v) for v in sections.values())
    reasoning_headroom = None
    if provider_limit:
        reasoning_headroom = provider_limit - total_tokens

    return {
        'per_source': per_source,
        'total_tokens': total_tokens,
        'total_cap': TOTAL_CAP,
        'total_utilization': round(total_tokens / TOTAL_CAP, 2) if TOTAL_CAP > 0 else None,
        'reasoning_reserve': REASONING_RESERVE,
        'reasoning_headroom': reasoning_headroom,
        'provider_limit': provider_limit,
    }
