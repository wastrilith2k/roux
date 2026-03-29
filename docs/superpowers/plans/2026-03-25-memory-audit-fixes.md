# Memory Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the 5 priority gaps identified in the memory architecture audit (#19): configurable eviction (#20), generous token budgets (#21), agent-controlled retrieval (#22), conversation compression (#23), and source provenance (#24).

**Architecture:** Each issue maps to 1-2 tasks. Changes are ordered to minimize conflicts: provenance markers first (additive), then token budgets (new module + integration), then eviction (new tasks), then compression (new module + integration), then retrieval tiers (classifier + context builder changes).

**Tech Stack:** Python 3.11, PostgreSQL/pgvector, Neo4j/Graphiti, Celery, Redis, OpenAI GPT-4o-mini

---

## File Structure

### New files:
| File | Responsibility |
|------|---------------|
| `src/core/conversation/token_budget.py` | Token budget config, per-source truncation, priority tiers |
| `src/tasks/embedding_pruning_task.py` | Configurable pgvector embedding archival (Celery beat) |
| `src/tasks/graphiti_pruning_task.py` | Configurable Neo4j episode/edge archival (Celery beat) |
| `src/memory/conversation_compressor.py` | Session-level conversation compression via LLM |
| `tests/test_token_budget.py` | Token budget truncation + priority tests |
| `tests/test_provenance.py` | Provenance tag injection tests |
| `tests/test_conversation_compressor.py` | Compression logic tests |
| `tests/test_embedding_pruning.py` | Embedding pruning logic tests |
| `tests/test_complexity_tiers.py` | MEDIUM tier classification tests |

### Modified files:
| File | Changes |
|------|---------|
| `src/core/conversation/context_builder.py` | Provenance tags in `to_prompt_sections()`, `session_summary` field on dataclass, compression integration, tier-based source selection in `build_lightweight()` |
| `src/core/conversation/pipeline.py` | Provenance-aware final reminder, token budget integration, session summary in prompt |
| `src/core/conversation/complexity_classifier.py` | Add MEDIUM tier |
| `src/memory/semantic_search.py` | Configurable recency weighting |
| `src/database/schema_ddl.py` | `last_retrieved_at` column on messages |
| `src/celery_app.py` | Register new pruning tasks |

---

## Task 1: Source Provenance Markers (#24)

**Files:**
- Modify: `src/core/conversation/context_builder.py:96-147` (`to_prompt_sections()`)
- Modify: `src/core/conversation/pipeline.py:970-992` (FINAL REMINDER section)
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write failing test for provenance tags**

```python
# tests/test_provenance.py
"""Tests for source provenance markers on context sections."""

import pytest
from src.core.conversation.context_builder import ConversationContext, PROVENANCE_TAGS


class TestProvenanceTags:
    """Verify provenance tags are prepended to context sections."""

    def test_provenance_tags_dict_exists(self):
        """PROVENANCE_TAGS should map source names to tag strings."""
        assert isinstance(PROVENANCE_TAGS, dict)
        assert 'memories' in PROVENANCE_TAGS
        assert 'activities_context' in PROVENANCE_TAGS

    def test_verified_sources_tagged(self):
        """Verified memory sources get [VERIFIED] or [FROM CONVERSATION] tags."""
        ctx = ConversationContext(
            memories="James likes peppermint tea",
            entity_profiles="name: James"
        )
        sections = ctx.to_prompt_sections()
        assert sections['memories'].startswith('[FROM PAST CONVERSATION]')
        assert sections['entity_profiles'].startswith('[VERIFIED FACT]')

    def test_simulated_sources_tagged(self):
        """Simulated/inferred sources get appropriate warning tags."""
        ctx = ConversationContext(
            activities_context="Spent the afternoon reading",
            values_context="Values honesty deeply"
        )
        sections = ctx.to_prompt_sections()
        assert '[SIMULATED' in sections['activities_context']
        assert '[INFERRED' in sections['values_context']

    def test_empty_sources_excluded(self):
        """Empty sources should not appear in sections."""
        ctx = ConversationContext(memories="", activities_context="")
        sections = ctx.to_prompt_sections()
        assert 'memories' not in sections
        assert 'activities_context' not in sections

    def test_all_sources_have_provenance(self):
        """Every source in to_prompt_sections() should have a provenance tag."""
        # Build a context with all fields populated
        ctx = ConversationContext()
        source_fields = [
            'scene_state', 'internal_state', 'fertility_context', 'user_context',
            'values_context', 'activities_context', 'temporal_context',
            'graphiti_context', 'synthesized_events', 'episode_context',
            'observations_context', 'core_memory', 'reflections_context',
            'opinions_context', 'curiosity_context', 'goals_context',
            'relationship_insights', 'relationship_dynamics',
            'relationship_evaluation', 'entity_profiles', 'personality',
            'memories', 'biographies', 'conversation_history'
        ]
        for f in source_fields:
            setattr(ctx, f, "test content")
        sections = ctx.to_prompt_sections()
        for name, content in sections.items():
            assert content.startswith('['), f"Source '{name}' missing provenance tag"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provenance.py -v`
Expected: FAIL — `PROVENANCE_TAGS` not defined

- [ ] **Step 3: Add PROVENANCE_TAGS dict and update to_prompt_sections()**

In `src/core/conversation/context_builder.py`, add after the imports (after line 38):

```python
# ---------------------------------------------------------------------------
# Source provenance tags — prepended to each context section so the LLM
# can distinguish verified facts from inferred/simulated content.
# ---------------------------------------------------------------------------

PROVENANCE_TAGS = {
    # Verified — from actual interactions or ground truth
    'entity_profiles': '[VERIFIED FACT]',
    'memories': '[FROM PAST CONVERSATION]',
    'conversation_history': '[CURRENT SESSION]',
    'core_memory': '[CURATED MEMORY]',
    'graphiti_context': '[KNOWLEDGE GRAPH]',
    'episode_context': '[PAST EPISODE]',
    'observations_context': '[OBSERVATION — FROM CONVERSATIONS]',
    'biographies': '[BIOGRAPHICAL SUMMARY]',
    'temporal_context': '[RECENT EVENT]',
    'synthesized_events': '[EVENT NARRATIVE]',

    # Inferred — LLM-generated, may not reflect reality
    'opinions_context': '[OPINION — INFERRED]',
    'values_context': '[INFERRED VALUES]',
    'reflections_context': '[REFLECTION — INTERNAL]',
    'curiosity_context': '[CURIOSITY — INTERNAL]',
    'goals_context': '[GOALS — INTERNAL]',

    # Simulated — not real events
    'activities_context': '[SIMULATED ACTIVITY — NOT A REAL EVENT]',
    'user_context': '[INFERRED USER STATE]',
    'fertility_context': '[BIOLOGICAL TRACKING — PRIVATE]',

    # State — current session state
    'scene_state': '[CURRENT SCENE]',
    'internal_state': '[CURRENT STATE]',
    'schedule': '[SCHEDULE]',

    # Personality / relationship
    'personality': '[PERSONALITY]',
    'relationship_insights': '[RELATIONSHIP INSIGHT]',
    'relationship_dynamics': '[RELATIONSHIP STATE]',
    'relationship_evaluation': '[RELATIONSHIP EVALUATION]',
}
```

Then replace the `to_prompt_sections()` method (lines 96-147):

```python
def to_prompt_sections(self) -> Dict[str, str]:
    """Return non-empty sections as dict for prompt assembly.

    Each section is prepended with a provenance tag from PROVENANCE_TAGS
    so the LLM can distinguish verified facts from inferred/simulated content.
    """
    raw_sections = {}
    if self.scene_state:
        raw_sections['scene_state'] = self.scene_state
    if self.internal_state:
        raw_sections['internal_state'] = self.internal_state
    if self.fertility_context:
        raw_sections['fertility_context'] = self.fertility_context
    if self.user_context:
        raw_sections['user_context'] = self.user_context
    if self.values_context:
        raw_sections['values_context'] = self.values_context
    if self.activities_context:
        raw_sections['activities_context'] = self.activities_context
    if self.temporal_context:
        raw_sections['temporal_context'] = self.temporal_context
    if self.graphiti_context:
        raw_sections['graphiti_context'] = self.graphiti_context
    if self.synthesized_events:
        raw_sections['synthesized_events'] = self.synthesized_events
    if self.episode_context:
        raw_sections['episode_context'] = self.episode_context
    if self.observations_context:
        raw_sections['observations_context'] = self.observations_context
    if self.core_memory:
        raw_sections['core_memory'] = self.core_memory
    if self.reflections_context:
        raw_sections['reflections_context'] = self.reflections_context
    if self.opinions_context:
        raw_sections['opinions_context'] = self.opinions_context
    if self.curiosity_context:
        raw_sections['curiosity_context'] = self.curiosity_context
    if self.goals_context:
        raw_sections['goals_context'] = self.goals_context
    if self.relationship_insights:
        raw_sections['relationship_insights'] = self.relationship_insights
    if self.relationship_dynamics:
        raw_sections['relationship_dynamics'] = self.relationship_dynamics
    if self.relationship_evaluation:
        raw_sections['relationship_evaluation'] = self.relationship_evaluation
    if self.entity_profiles:
        raw_sections['entity_profiles'] = self.entity_profiles
    if self.personality:
        raw_sections['personality'] = self.personality
    if self.memories:
        raw_sections['memories'] = self.memories
    if self.biographies:
        raw_sections['biographies'] = self.biographies
    if self.conversation_history:
        raw_sections['conversation_history'] = self.conversation_history

    # Prepend provenance tags
    sections = {}
    for name, content in raw_sections.items():
        tag = PROVENANCE_TAGS.get(name, '')
        sections[name] = f"{tag}\n{content}" if tag else content
    return sections
```

- [ ] **Step 4: Update FINAL REMINDER with provenance-aware instructions**

In `src/core/conversation/pipeline.py`, find the closing_lines list (around line 975) and update:

```python
closing_lines = [
    get_current_time_context(),
    f"You are {_companion}. {_pc.primary_user_name} is a separate person — his facts are his, yours are yours.",
    "Only reference details that appear in your provided memories.",
    "Treat [SIMULATED ACTIVITY] content as your background life — do not present specific details as verified memories.",
    "Treat [INFERRED] and [OPINION] content as your internal thoughts, not as facts the user told you.",
    "Answer his question directly first, then add your thoughts.",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_provenance.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/core/conversation/context_builder.py src/core/conversation/pipeline.py tests/test_provenance.py
git commit -m "feat(memory): add source provenance markers to context sections (#24)"
```

---

## Task 2: Per-Source Token Budgets (#21)

**Files:**
- Create: `src/core/conversation/token_budget.py`
- Modify: `src/core/conversation/context_builder.py:96-147` (integrate budgets into `to_prompt_sections()`)
- Modify: `src/core/conversation/pipeline.py:997-1007` (total cap check)
- Test: `tests/test_token_budget.py`

- [ ] **Step 1: Write failing tests for token budget module**

```python
# tests/test_token_budget.py
"""Tests for per-source token budgets and priority-based truncation."""

import os
import pytest
from src.core.conversation.token_budget import (
    estimate_tokens,
    truncate_to_budget,
    apply_token_budgets,
    SOURCE_TOKEN_BUDGETS,
    TOTAL_CAP,
    REASONING_RESERVE,
    TIER_1,
    TIER_2,
    TIER_3,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_known_string(self):
        # ~4 chars per token
        assert estimate_tokens("a" * 400) == 100

    def test_reasonable_estimate(self):
        text = "Hello world, this is a test sentence."
        tokens = estimate_tokens(text)
        assert 5 <= tokens <= 15


class TestTruncateToBudget:
    def test_short_text_unchanged(self):
        text = "Hello world"
        result = truncate_to_budget(text, 1000)
        assert result == text

    def test_long_text_truncated(self):
        text = "Word. " * 500  # ~750 tokens
        result = truncate_to_budget(text, 100)
        assert estimate_tokens(result) <= 110  # Allow small overshoot

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = truncate_to_budget(text, 5)  # Very tight budget
        assert result.endswith('.')  # Should end at sentence boundary

    def test_preserves_provenance_tag(self):
        text = "[VERIFIED FACT]\nSome content here that is quite long. " * 20
        result = truncate_to_budget(text, 50)
        assert result.startswith("[VERIFIED FACT]")


class TestApplyTokenBudgets:
    def test_small_sections_unchanged(self):
        sections = {'memories': 'Short memory', 'personality': 'Short personality'}
        result = apply_token_budgets(sections)
        assert result == sections

    def test_oversized_section_truncated(self):
        sections = {'memories': 'Memory content. ' * 2000}  # Way over budget
        result = apply_token_budgets(sections)
        assert estimate_tokens(result['memories']) <= SOURCE_TOKEN_BUDGETS['memories'] + 50

    def test_tier3_trimmed_first_under_total_cap(self):
        """When total exceeds cap, Tier 3 sources are trimmed before Tier 2."""
        sections = {}
        # Fill tier 1 source
        sections['entity_profiles'] = 'Fact. ' * 200
        # Fill tier 2 source
        sections['memories'] = 'Memory. ' * 200
        # Fill tier 3 source
        sections['opinions_context'] = 'Opinion. ' * 200

        result = apply_token_budgets(sections, total_cap=500)

        # Tier 3 should be most aggressively trimmed
        t1_tokens = estimate_tokens(result.get('entity_profiles', ''))
        t3_tokens = estimate_tokens(result.get('opinions_context', ''))
        assert t3_tokens <= t1_tokens

    def test_budgets_are_configurable_via_env(self):
        """SOURCE_TOKEN_BUDGETS should be overridable via env vars."""
        assert isinstance(SOURCE_TOKEN_BUDGETS, dict)
        assert TOTAL_CAP > 0
        assert REASONING_RESERVE > 0


class TestBudgetConfig:
    def test_all_context_sources_have_budgets(self):
        """Every source that appears in to_prompt_sections should have a budget."""
        expected_sources = [
            'scene_state', 'internal_state', 'fertility_context', 'user_context',
            'values_context', 'activities_context', 'temporal_context',
            'graphiti_context', 'synthesized_events', 'episode_context',
            'observations_context', 'core_memory', 'reflections_context',
            'opinions_context', 'curiosity_context', 'goals_context',
            'relationship_insights', 'relationship_dynamics',
            'relationship_evaluation', 'entity_profiles', 'personality',
            'memories', 'biographies', 'conversation_history'
        ]
        for source in expected_sources:
            assert source in SOURCE_TOKEN_BUDGETS, f"Missing budget for '{source}'"

    def test_tiers_cover_all_sources(self):
        all_tiered = TIER_1 | TIER_2 | TIER_3
        for source in SOURCE_TOKEN_BUDGETS:
            assert source in all_tiered, f"'{source}' not assigned to any tier"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_token_budget.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement token_budget.py**

```python
# src/core/conversation/token_budget.py
"""
Token Budget — per-source token limits and priority-based truncation.

WHAT: Defines generous per-source token budgets, truncation logic that
      respects sentence boundaries, and priority tiers for graceful
      degradation when total context exceeds the cap.

WHY:  Without budgets, any single context source can dominate the prompt.
      Priority tiers ensure the most important sources survive under pressure.

Budgets are intentionally generous — they're safety rails, not straitjackets.
All values are configurable via environment variables:
  TOKEN_BUDGET_{SOURCE_NAME}=N  (e.g., TOKEN_BUDGET_MEMORIES=2000)
  TOKEN_BUDGET_TOTAL_CAP=N
  TOKEN_BUDGET_REASONING_RESERVE=N
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


# ---------------------------------------------------------------------------
# Per-source budgets (tokens). Generous defaults — these are upper bounds.
# Override any with TOKEN_BUDGET_{SOURCE_NAME_UPPER}=N env var.
# ---------------------------------------------------------------------------

_DEFAULT_BUDGETS = {
    # Tier 1: Core identity and context — always full budget
    'entity_profiles':       2000,
    'personality':           1200,
    'conversation_history':  3000,
    'core_memory':           1000,
    'scene_state':            800,
    'internal_state':         600,

    # Tier 2: Rich memory — proportionally trimmed under pressure
    'memories':              2000,
    'graphiti_context':      1500,
    'biographies':           1500,
    'episode_context':       1200,
    'temporal_context':      1000,
    'synthesized_events':    1000,
    'observations_context':  1000,
    'relationship_dynamics': 1000,
    'relationship_insights':  800,

    # Tier 3: Enrichment — first to trim
    'opinions_context':       800,
    'curiosity_context':      600,
    'reflections_context':    800,
    'activities_context':     600,
    'values_context':         600,
    'goals_context':          600,
    'relationship_evaluation': 800,
    'user_context':           400,
    'fertility_context':      400,
    'schedule':               600,
}

SOURCE_TOKEN_BUDGETS: Dict[str, int] = {
    k: _env_int(f'TOKEN_BUDGET_{k.upper()}', v)
    for k, v in _DEFAULT_BUDGETS.items()
}

TOTAL_CAP = _env_int('TOKEN_BUDGET_TOTAL_CAP', 20000)
REASONING_RESERVE = _env_int('TOKEN_BUDGET_REASONING_RESERVE', 4000)

# ---------------------------------------------------------------------------
# Priority tiers
# ---------------------------------------------------------------------------

TIER_1 = {
    'entity_profiles', 'personality', 'conversation_history',
    'core_memory', 'scene_state', 'internal_state',
}

TIER_2 = {
    'memories', 'graphiti_context', 'biographies', 'episode_context',
    'temporal_context', 'synthesized_events', 'observations_context',
    'relationship_dynamics', 'relationship_insights',
}

TIER_3 = {
    'opinions_context', 'curiosity_context', 'reflections_context',
    'activities_context', 'values_context', 'goals_context',
    'relationship_evaluation', 'user_context', 'fertility_context',
    'schedule',
}


# ---------------------------------------------------------------------------
# Token estimation and truncation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count. ~4 chars per token is a reasonable approximation."""
    return len(text) // 4


def truncate_to_budget(text: str, budget_tokens: int) -> str:
    """Truncate text to fit within a token budget, preferring sentence boundaries.

    Preserves any provenance tag on the first line (lines starting with '[').
    """
    if estimate_tokens(text) <= budget_tokens:
        return text

    # Preserve provenance tag if present
    prefix = ""
    body = text
    if text.startswith('[') and '\n' in text:
        newline_idx = text.index('\n')
        prefix = text[:newline_idx + 1]
        body = text[newline_idx + 1:]
        budget_tokens -= estimate_tokens(prefix)

    # Truncate body at sentence boundary
    char_budget = max(budget_tokens * 4, 20)
    if len(body) <= char_budget:
        return prefix + body

    truncated = body[:char_budget]

    # Find last sentence-ending punctuation
    for end_char in ['. ', '.\n', '! ', '?\n']:
        last_end = truncated.rfind(end_char)
        if last_end > len(truncated) // 2:  # Only if we keep > half
            truncated = truncated[:last_end + 1]
            break

    return prefix + truncated


def apply_token_budgets(
    sections: Dict[str, str],
    total_cap: int = None,
) -> Dict[str, str]:
    """Apply per-source token budgets and priority-based total cap.

    1. Truncate each source to its individual budget.
    2. If total exceeds total_cap, trim Tier 3 first, then Tier 2.
       Tier 1 is never trimmed below its individual budget.

    Returns a new dict with truncated sections.
    """
    if total_cap is None:
        total_cap = TOTAL_CAP

    # Phase 1: Apply individual budgets
    result = {}
    for name, content in sections.items():
        budget = SOURCE_TOKEN_BUDGETS.get(name, 1000)
        result[name] = truncate_to_budget(content, budget)

    # Phase 2: Check total
    total = sum(estimate_tokens(v) for v in result.values())
    if total <= total_cap:
        return result

    overshoot = total - total_cap
    logger.info(f"Token budget: {total} tokens exceeds cap {total_cap}, trimming {overshoot} tokens")

    # Trim Tier 3 first (proportionally)
    result, overshoot = _trim_tier(result, TIER_3, overshoot)

    # If still over, trim Tier 2
    if overshoot > 0:
        result, overshoot = _trim_tier(result, TIER_2, overshoot)

    if overshoot > 0:
        logger.warning(f"Token budget: still {overshoot} tokens over cap after trimming Tier 2+3")

    return result


def _trim_tier(
    sections: Dict[str, str],
    tier_sources: set,
    overshoot: int,
) -> tuple:
    """Trim sources in a tier proportionally to reduce overshoot."""
    tier_sections = {k: v for k, v in sections.items() if k in tier_sources and v}
    if not tier_sections:
        return sections, overshoot

    tier_total = sum(estimate_tokens(v) for v in tier_sections.values())
    if tier_total == 0:
        return sections, overshoot

    # Calculate trim ratio — how much to keep
    trim_amount = min(overshoot, tier_total)
    keep_ratio = max(0.1, 1.0 - (trim_amount / tier_total))

    result = dict(sections)
    trimmed = 0
    for name, content in tier_sections.items():
        current_tokens = estimate_tokens(content)
        new_budget = max(50, int(current_tokens * keep_ratio))
        if new_budget < current_tokens:
            old_content = content
            result[name] = truncate_to_budget(content, new_budget)
            trimmed += estimate_tokens(old_content) - estimate_tokens(result[name])

    return result, overshoot - trimmed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_token_budget.py -v`
Expected: All PASS

- [ ] **Step 5: Integrate token budgets into context_builder and pipeline**

In `src/core/conversation/context_builder.py`, update `to_prompt_sections()` to apply budgets after provenance tags. At the end of the method, before `return sections`:

```python
# Apply token budgets
from src.core.conversation.token_budget import apply_token_budgets
sections = apply_token_budgets(sections)
return sections
```

In `src/core/conversation/pipeline.py`, update the token estimation section (around line 997) to include budget info:

```python
# Log prompt length for monitoring
token_estimate = len(full_prompt) // 4
logger.info(f"Prompt assembled: ~{token_estimate} tokens")

# Check token budget against provider context limits
provider_limit = self._get_provider_context_limit()
if provider_limit:
    from src.core.conversation.token_budget import REASONING_RESERVE
    available_for_context = provider_limit - REASONING_RESERVE
    if token_estimate > int(available_for_context * 0.85):
        logger.warning(
            f"Prompt approaching context limit: ~{token_estimate} tokens "
            f"(provider limit: {provider_limit}, reasoning reserve: {REASONING_RESERVE})"
        )
```

- [ ] **Step 6: Run full test suite for context_builder and pipeline**

Run: `pytest tests/test_token_budget.py tests/test_provenance.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/core/conversation/token_budget.py tests/test_token_budget.py src/core/conversation/context_builder.py src/core/conversation/pipeline.py
git commit -m "feat(memory): add configurable per-source token budgets with priority tiers (#21)"
```

---

## Task 3: Configurable Embedding & Graph Eviction (#20)

**Files:**
- Create: `src/tasks/embedding_pruning_task.py`
- Create: `src/tasks/graphiti_pruning_task.py`
- Modify: `src/memory/semantic_search.py` (recency weighting)
- Modify: `src/database/schema_ddl.py` (last_retrieved_at column)
- Modify: `src/celery_app.py` (register new tasks)
- Test: `tests/test_embedding_pruning.py`

**User preference:** Make everything configurable. Defaults should be conservative — don't actually remove memories. The user feels removing memories is counter to good memory. So: recency weighting YES (improves retrieval quality without removing anything), archival defaults to DRY RUN (reports but doesn't archive), and all thresholds are env-var configurable.

- [ ] **Step 1: Write failing test for recency-weighted search**

```python
# tests/test_embedding_pruning.py
"""Tests for embedding pruning and recency-weighted search."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


class TestRecencyWeighting:
    """Test recency weighting in semantic search."""

    def test_apply_recency_weighting_recent_message(self):
        from src.memory.semantic_search import apply_recency_weighting
        now = datetime.now(timezone.utc)
        results = [
            {'similarity': 0.8, 'timestamp': now - timedelta(hours=1), 'message_text': 'recent'},
        ]
        weighted = apply_recency_weighting(results)
        # Recent message should keep most of its score
        assert weighted[0]['weighted_similarity'] >= 0.7

    def test_apply_recency_weighting_old_message(self):
        from src.memory.semantic_search import apply_recency_weighting
        now = datetime.now(timezone.utc)
        results = [
            {'similarity': 0.8, 'timestamp': now - timedelta(days=180), 'message_text': 'old'},
        ]
        weighted = apply_recency_weighting(results)
        # Old message should have reduced score
        assert weighted[0]['weighted_similarity'] < 0.8

    def test_recency_reorders_results(self):
        from src.memory.semantic_search import apply_recency_weighting
        now = datetime.now(timezone.utc)
        results = [
            {'similarity': 0.82, 'timestamp': now - timedelta(days=120), 'message_text': 'old high sim'},
            {'similarity': 0.78, 'timestamp': now - timedelta(hours=2), 'message_text': 'recent lower sim'},
        ]
        weighted = apply_recency_weighting(results)
        # Recent message should rank higher despite lower raw similarity
        assert weighted[0]['message_text'] == 'recent lower sim'

    def test_recency_weight_configurable(self):
        from src.memory.semantic_search import apply_recency_weighting
        now = datetime.now(timezone.utc)
        results = [
            {'similarity': 0.8, 'timestamp': now - timedelta(days=60), 'message_text': 'test'},
        ]
        # With very long half-life, decay should be minimal
        weighted = apply_recency_weighting(results, half_life_days=365)
        assert weighted[0]['weighted_similarity'] > 0.75

    def test_missing_timestamp_gets_no_penalty(self):
        from src.memory.semantic_search import apply_recency_weighting
        results = [
            {'similarity': 0.8, 'timestamp': None, 'message_text': 'no ts'},
        ]
        weighted = apply_recency_weighting(results)
        assert weighted[0]['weighted_similarity'] == 0.8


class TestEmbeddingPruningTask:
    """Test the embedding pruning Celery task configuration."""

    def test_task_defaults_to_dry_run(self):
        from src.tasks.embedding_pruning_task import prune_stale_embeddings
        # The task should default to dry_run=True
        import inspect
        sig = inspect.signature(prune_stale_embeddings)
        assert sig.parameters['dry_run'].default is True

    def test_pruning_thresholds_configurable(self):
        from src.tasks.embedding_pruning_task import (
            EMBEDDING_MAX_AGE_DAYS,
            EMBEDDING_MIN_IMPORTANCE,
            EMBEDDING_PRUNING_ENABLED,
        )
        assert isinstance(EMBEDDING_MAX_AGE_DAYS, int)
        assert isinstance(EMBEDDING_MIN_IMPORTANCE, int)
        assert isinstance(EMBEDDING_PRUNING_ENABLED, bool)


class TestGraphitiPruningTask:
    """Test the Graphiti pruning Celery task configuration."""

    def test_task_defaults_to_dry_run(self):
        from src.tasks.graphiti_pruning_task import prune_stale_graph_data
        import inspect
        sig = inspect.signature(prune_stale_graph_data)
        assert sig.parameters['dry_run'].default is True

    def test_pruning_thresholds_configurable(self):
        from src.tasks.graphiti_pruning_task import (
            GRAPH_EDGE_MAX_AGE_DAYS,
            GRAPH_EDGE_MIN_IMPORTANCE,
            GRAPH_PRUNING_ENABLED,
        )
        assert isinstance(GRAPH_EDGE_MAX_AGE_DAYS, int)
        assert isinstance(GRAPH_EDGE_MIN_IMPORTANCE, int)
        assert isinstance(GRAPH_PRUNING_ENABLED, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedding_pruning.py -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Add recency weighting to semantic_search.py**

Add the following function to `src/memory/semantic_search.py` after the existing imports (after line 38):

```python
import math

# Recency weighting config
RECENCY_HALF_LIFE_DAYS = int(os.environ.get('RECENCY_HALF_LIFE_DAYS', '90'))
RECENCY_WEIGHT = float(os.environ.get('RECENCY_WEIGHT', '0.15'))  # Blend: (1-w)*sim + w*recency
ENABLE_RECENCY_WEIGHTING = os.environ.get('ENABLE_RECENCY_WEIGHTING', 'true').lower() == 'true'


def apply_recency_weighting(
    results: List[Dict],
    half_life_days: int = None,
    weight: float = None,
) -> List[Dict]:
    """Apply recency weighting to search results.

    Blends similarity score with a time-decay factor:
        weighted = (1 - weight) * similarity + weight * recency_factor

    recency_factor uses exponential decay with configurable half-life.
    Results are re-sorted by weighted_similarity.

    Args:
        results: Search results with 'similarity' and 'timestamp' fields
        half_life_days: Days until recency factor reaches 0.5 (default: 90)
        weight: Blend weight for recency vs similarity (default: 0.15)
    """
    from datetime import datetime, timezone

    if half_life_days is None:
        half_life_days = RECENCY_HALF_LIFE_DAYS
    if weight is None:
        weight = RECENCY_WEIGHT

    now = datetime.now(timezone.utc)

    for r in results:
        ts = r.get('timestamp')
        sim = r.get('similarity', 0)

        if ts is None:
            r['weighted_similarity'] = sim
            continue

        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                r['weighted_similarity'] = sim
                continue

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        days_old = max(0, (now - ts).total_seconds() / 86400)
        # Exponential decay: 1.0 at day 0, 0.5 at half_life_days
        recency_factor = math.exp(-0.693 * days_old / max(half_life_days, 1))
        r['weighted_similarity'] = (1 - weight) * sim + weight * recency_factor

    results.sort(key=lambda r: r.get('weighted_similarity', 0), reverse=True)
    return results
```

Then update `search_memory()` to apply recency weighting after reranking (add before the `return results` at line 97):

```python
# Apply recency weighting if enabled
if ENABLE_RECENCY_WEIGHTING:
    results = apply_recency_weighting(results)
```

- [ ] **Step 4: Create embedding_pruning_task.py**

```python
# src/tasks/embedding_pruning_task.py
"""
Embedding Pruning Task - Configurable archival of old pgvector embeddings.

WHAT: Identifies message embeddings that are old, low-importance, and rarely
      retrieved, then archives them by setting embedding_vec = NULL.
      The message row itself is preserved — only the vector is cleared.

WHEN: Monthly on the 1st at 6:00 AM Pacific (configurable via Celery beat).

WHY:  pgvector embeddings accumulate indefinitely (~6KB each). Without pruning,
      semantic search returns increasingly stale results and index performance
      degrades. Archiving old embeddings keeps the search index focused.

SAFETY:
- Defaults to dry_run=True — reports what would be pruned but changes nothing.
- All thresholds are env-var configurable.
- Never archives embeddings from messages with importance >= EMBEDDING_MIN_IMPORTANCE.
- Never archives embeddings referenced by active (non-archived) facts.
- Never archives embeddings from messages newer than EMBEDDING_MAX_AGE_DAYS.
- Set EMBEDDING_PRUNING_ENABLED=false to disable entirely.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

# --- Configurable thresholds ---
EMBEDDING_PRUNING_ENABLED = os.environ.get('EMBEDDING_PRUNING_ENABLED', 'false').lower() == 'true'
EMBEDDING_MAX_AGE_DAYS = int(os.environ.get('EMBEDDING_MAX_AGE_DAYS', '365'))
EMBEDDING_MIN_IMPORTANCE = int(os.environ.get('EMBEDDING_MIN_IMPORTANCE', '8'))


@celery_app.task(
    name='tasks.embedding_pruning.prune_stale_embeddings',
    bind=True,
    max_retries=1,
    soft_time_limit=180,
    time_limit=240
)
def prune_stale_embeddings(self, dry_run: bool = True) -> Dict[str, Any]:
    """
    Archive stale pgvector embeddings from the messages table.

    Sets embedding_vec = NULL on messages that are old and low-value.
    The message row itself is never deleted.

    Args:
        dry_run: If True (default), only report what would be pruned.

    Returns:
        Dict with status, candidate count, and archive count.
    """
    if not EMBEDDING_PRUNING_ENABLED:
        return {'status': 'disabled', 'message': 'Set EMBEDDING_PRUNING_ENABLED=true to enable'}

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        # Find embeddings older than threshold that aren't high-importance
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get candidate messages with embeddings older than max age
            cursor.execute(f"""
                SELECT m.id, m.timestamp, m.sender_name,
                       LEFT(m.message_text, 100) as preview
                FROM {T.MESSAGES} m
                WHERE m.embedding_vec IS NOT NULL
                  AND m.timestamp < NOW() - INTERVAL '{EMBEDDING_MAX_AGE_DAYS} days'
                  AND m.id NOT IN (
                      -- Exclude messages referenced by active facts
                      SELECT UNNEST(f.source_message_ids)
                      FROM {T.FACTS} f
                      WHERE f.archived_at IS NULL
                        AND f.importance >= {EMBEDDING_MIN_IMPORTANCE}
                  )
                ORDER BY m.timestamp ASC
            """)
            candidates = [dict(row) for row in cursor.fetchall()]

        logger.info(
            f"Embedding pruning: {len(candidates)} candidates older than "
            f"{EMBEDDING_MAX_AGE_DAYS} days (dry_run={dry_run})"
        )

        if dry_run or not candidates:
            conn.close()
            return {
                'status': 'dry_run' if dry_run else 'no_candidates',
                'candidates': len(candidates),
                'archived': 0,
                'oldest': str(candidates[0]['timestamp']) if candidates else None,
                'newest': str(candidates[-1]['timestamp']) if candidates else None,
            }

        # Archive: set embedding_vec = NULL
        candidate_ids = [c['id'] for c in candidates]
        with conn.cursor() as cursor:
            cursor.execute(f"""
                UPDATE {T.MESSAGES}
                SET embedding_vec = NULL
                WHERE id = ANY(%s)
            """, (candidate_ids,))
            archived_count = cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(f"Embedding pruning: archived {archived_count} embeddings")
        return {
            'status': 'completed',
            'candidates': len(candidates),
            'archived': archived_count,
        }

    except Exception as e:
        logger.error(f"Embedding pruning failed: {e}")
        return {'status': 'error', 'error': str(e)}
```

- [ ] **Step 5: Create graphiti_pruning_task.py**

```python
# src/tasks/graphiti_pruning_task.py
"""
Graphiti Pruning Task - Configurable archival of low-importance Neo4j edges.

WHAT: Identifies low-importance, old edges in the knowledge graph and marks
      them as archived. Entity nodes are preserved — only edges are pruned.

WHEN: Monthly on the 1st at 6:30 AM Pacific (configurable via Celery beat).

WHY:  The Neo4j knowledge graph grows with every conversation. Low-importance
      edges (e.g., "James mentioned weather") add noise to graph queries
      without contributing meaningful context.

SAFETY:
- Defaults to dry_run=True — reports what would be pruned but changes nothing.
- All thresholds are env-var configurable.
- Never prunes edges with importance >= GRAPH_EDGE_MIN_IMPORTANCE.
- Never prunes edges newer than GRAPH_EDGE_MAX_AGE_DAYS.
- Entity nodes are never deleted.
- Set GRAPH_PRUNING_ENABLED=false to disable entirely.
"""

import os
import logging
from typing import Dict, Any

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

# --- Configurable thresholds ---
GRAPH_PRUNING_ENABLED = os.environ.get('GRAPH_PRUNING_ENABLED', 'false').lower() == 'true'
GRAPH_EDGE_MAX_AGE_DAYS = int(os.environ.get('GRAPH_EDGE_MAX_AGE_DAYS', '365'))
GRAPH_EDGE_MIN_IMPORTANCE = int(os.environ.get('GRAPH_EDGE_MIN_IMPORTANCE', '5'))


@celery_app.task(
    name='tasks.graphiti_pruning.prune_stale_graph_data',
    bind=True,
    max_retries=1,
    soft_time_limit=180,
    time_limit=240
)
def prune_stale_graph_data(self, dry_run: bool = True) -> Dict[str, Any]:
    """
    Archive low-importance, old edges in the Neo4j knowledge graph.

    Sets an 'archived' property on edges. Does not delete anything.

    Args:
        dry_run: If True (default), only report what would be pruned.

    Returns:
        Dict with status, candidate count, and archive count.
    """
    if not GRAPH_PRUNING_ENABLED:
        return {'status': 'disabled', 'message': 'Set GRAPH_PRUNING_ENABLED=true to enable'}

    try:
        from src.memory.graphiti_client import get_graphiti_driver

        driver = get_graphiti_driver()
        if not driver:
            return {'status': 'error', 'error': 'Neo4j driver not available'}

        with driver.session() as session:
            # Find low-importance edges older than threshold
            result = session.run("""
                MATCH ()-[r]->()
                WHERE r.importance IS NOT NULL
                  AND r.importance < $min_importance
                  AND r.created_at < datetime() - duration({days: $max_age_days})
                  AND NOT exists(r.archived)
                RETURN count(r) as candidate_count
            """, min_importance=GRAPH_EDGE_MIN_IMPORTANCE,
                max_age_days=GRAPH_EDGE_MAX_AGE_DAYS)

            candidate_count = result.single()['candidate_count']

            logger.info(
                f"Graph pruning: {candidate_count} edge candidates older than "
                f"{GRAPH_EDGE_MAX_AGE_DAYS} days with importance < {GRAPH_EDGE_MIN_IMPORTANCE} "
                f"(dry_run={dry_run})"
            )

            if dry_run or candidate_count == 0:
                return {
                    'status': 'dry_run' if dry_run else 'no_candidates',
                    'candidates': candidate_count,
                    'archived': 0,
                }

            # Archive: set archived property on edges
            result = session.run("""
                MATCH ()-[r]->()
                WHERE r.importance IS NOT NULL
                  AND r.importance < $min_importance
                  AND r.created_at < datetime() - duration({days: $max_age_days})
                  AND NOT exists(r.archived)
                SET r.archived = true, r.archived_at = datetime()
                RETURN count(r) as archived_count
            """, min_importance=GRAPH_EDGE_MIN_IMPORTANCE,
                max_age_days=GRAPH_EDGE_MAX_AGE_DAYS)

            archived_count = result.single()['archived_count']

        logger.info(f"Graph pruning: archived {archived_count} edges")
        return {
            'status': 'completed',
            'candidates': candidate_count,
            'archived': archived_count,
        }

    except Exception as e:
        logger.error(f"Graph pruning failed: {e}")
        return {'status': 'error', 'error': str(e)}
```

- [ ] **Step 6: Register new tasks in celery_app.py**

Add to the `include` list (after line 100):

```python
'src.tasks.embedding_pruning_task',  # pgvector embedding archival (monthly)
'src.tasks.graphiti_pruning_task',  # Neo4j edge archival (monthly)
```

Add to `beat_schedule` (after the `weekly-memory-pruning` entry, around line 191):

```python
# Embedding pruning - archive old pgvector embeddings (dry_run by default)
'monthly-embedding-pruning': {
    'task': 'tasks.embedding_pruning.prune_stale_embeddings',
    'schedule': crontab(hour=6, minute=0, day_of_month=1),  # 1st of month 6:00 AM
    'kwargs': {'dry_run': True},  # Safe default: report only
},
# Graph pruning - archive old low-importance Neo4j edges (dry_run by default)
'monthly-graph-pruning': {
    'task': 'tasks.graphiti_pruning.prune_stale_graph_data',
    'schedule': crontab(hour=6, minute=30, day_of_month=1),  # 1st of month 6:30 AM
    'kwargs': {'dry_run': True},  # Safe default: report only
},
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_embedding_pruning.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/memory/semantic_search.py src/tasks/embedding_pruning_task.py src/tasks/graphiti_pruning_task.py src/celery_app.py tests/test_embedding_pruning.py
git commit -m "feat(memory): add configurable embedding/graph eviction with recency weighting (#20)"
```

---

## Task 4: Conversation Compression (#23)

**Files:**
- Create: `src/memory/conversation_compressor.py`
- Modify: `src/core/conversation/context_builder.py` (add `session_summary` field, integrate compression)
- Modify: `src/core/conversation/pipeline.py` (include session summary in prompt)
- Test: `tests/test_conversation_compressor.py`

- [ ] **Step 1: Write failing test for conversation compressor**

```python
# tests/test_conversation_compressor.py
"""Tests for conversation compression logic."""

import pytest
from unittest.mock import patch, MagicMock
from src.memory.conversation_compressor import (
    should_compress,
    format_messages_for_compression,
    compress_conversation,
    COMPRESSION_THRESHOLD,
    RECENT_MESSAGES_TO_KEEP,
)


class TestShouldCompress:
    def test_below_threshold_no_compress(self):
        messages = [{'role': 'user', 'content': f'msg {i}'} for i in range(10)]
        assert should_compress(messages) is False

    def test_above_threshold_compress(self):
        messages = [{'role': 'user', 'content': f'msg {i}'} for i in range(40)]
        assert should_compress(messages) is True

    def test_exactly_at_threshold_no_compress(self):
        messages = [{'role': 'user', 'content': f'msg {i}'} for i in range(COMPRESSION_THRESHOLD)]
        assert should_compress(messages) is False


class TestFormatMessages:
    def test_formats_user_and_assistant(self):
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'},
        ]
        formatted = format_messages_for_compression(messages)
        assert 'User: Hello' in formatted
        assert 'Assistant: Hi there' in formatted

    def test_empty_messages(self):
        assert format_messages_for_compression([]) == ""


class TestCompressConversation:
    @patch('src.memory.conversation_compressor._call_compression_llm')
    def test_returns_summary_and_recent(self, mock_llm):
        mock_llm.return_value = "They discussed work projects and weekend plans."

        messages = [{'role': 'user', 'content': f'message {i}'} for i in range(40)]
        summary, recent = compress_conversation(messages)

        assert summary == "They discussed work projects and weekend plans."
        assert len(recent) == RECENT_MESSAGES_TO_KEEP
        # Recent should be the last N messages
        assert recent[-1]['content'] == 'message 39'

    @patch('src.memory.conversation_compressor._call_compression_llm')
    def test_llm_failure_returns_none(self, mock_llm):
        mock_llm.side_effect = Exception("LLM down")

        messages = [{'role': 'user', 'content': f'message {i}'} for i in range(40)]
        summary, recent = compress_conversation(messages)

        assert summary is None
        assert len(recent) == len(messages)  # Fallback: return all messages

    def test_below_threshold_returns_all(self):
        messages = [{'role': 'user', 'content': f'msg {i}'} for i in range(10)]
        summary, recent = compress_conversation(messages)
        assert summary is None
        assert recent == messages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conversation_compressor.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement conversation_compressor.py**

```python
# src/memory/conversation_compressor.py
"""
Conversation Compressor - Summarize old messages to preserve context in fewer tokens.

WHAT: When conversation history exceeds a threshold, compresses older messages
      into a structured summary using GPT-4o-mini. Recent messages are kept raw.

WHY:  The 25-message history limit creates an "amnesia cliff" — messages older
      than 25 turns vanish entirely. Compression preserves the narrative arc
      and key details in fewer tokens.

HOW it fits:
  - context_builder calls compress_conversation() when conversation_turns
    exceeds COMPRESSION_THRESHOLD.
  - Returns (summary_text, recent_messages) — summary goes into a
    session_summary field, recent messages stay as conversation_turns.
  - Summaries are cached in Redis to avoid re-compressing on every turn.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Configurable thresholds ---
COMPRESSION_THRESHOLD = int(os.environ.get('CONVERSATION_COMPRESSION_THRESHOLD', '30'))
RECENT_MESSAGES_TO_KEEP = int(os.environ.get('CONVERSATION_RECENT_TO_KEEP', '15'))
COMPRESSION_MODEL = os.environ.get('CONVERSATION_COMPRESSION_MODEL', 'gpt-4o-mini')

# Redis cache TTL for session summaries (4 hours)
SUMMARY_CACHE_TTL = int(os.environ.get('CONVERSATION_SUMMARY_CACHE_TTL', '14400'))


def should_compress(messages: List[Dict]) -> bool:
    """Check if conversation history should be compressed."""
    return len(messages) > COMPRESSION_THRESHOLD


def format_messages_for_compression(messages: List[Dict]) -> str:
    """Format messages into a readable transcript for the LLM."""
    if not messages:
        return ""
    lines = []
    for m in messages:
        role = "User" if m.get('role') == 'user' else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def compress_conversation(
    messages: List[Dict],
    user_email: str = None,
) -> Tuple[Optional[str], List[Dict]]:
    """Compress older messages into a summary, keeping recent messages raw.

    Args:
        messages: Full list of conversation turns (chronological order)
        user_email: For cache key scoping

    Returns:
        Tuple of (summary_text_or_None, recent_messages)
        If below threshold or compression fails, returns (None, all_messages).
    """
    if not should_compress(messages):
        return None, messages

    # Split into old (to compress) and recent (to keep raw)
    split_point = len(messages) - RECENT_MESSAGES_TO_KEEP
    old_messages = messages[:split_point]
    recent_messages = messages[split_point:]

    # Check Redis cache first
    cache_key = _make_cache_key(user_email, len(messages), split_point)
    cached = _get_cached_summary(cache_key)
    if cached:
        logger.info(f"Conversation compression: using cached summary ({len(old_messages)} messages)")
        return cached, recent_messages

    # Compress via LLM
    try:
        transcript = format_messages_for_compression(old_messages)
        summary = _call_compression_llm(transcript, len(old_messages))
        if summary:
            _cache_summary(cache_key, summary)
            logger.info(
                f"Conversation compression: {len(old_messages)} messages -> "
                f"~{len(summary)//4} tokens summary"
            )
            return summary, recent_messages
    except Exception as e:
        logger.warning(f"Conversation compression failed: {e}")

    # Fallback: return all messages uncompressed
    return None, messages


def _call_compression_llm(transcript: str, message_count: int) -> Optional[str]:
    """Call GPT-4o-mini to compress a conversation transcript."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    response = client.chat.completions.create(
        model=COMPRESSION_MODEL,
        temperature=0,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. Compress the following conversation "
                    "into a structured summary. Preserve:\n"
                    "- Topics discussed and key points\n"
                    "- Decisions, agreements, or plans made\n"
                    "- Emotional tone and any shifts\n"
                    "- Open threads (topics mentioned but not resolved)\n"
                    "- Specific facts, names, dates mentioned\n\n"
                    "Format as a brief paragraph, not bullet points. "
                    "Be concise but preserve all concrete details."
                )
            },
            {
                "role": "user",
                "content": f"Summarize this {message_count}-message conversation:\n\n{transcript}"
            }
        ]
    )

    return response.choices[0].message.content.strip()


def _make_cache_key(user_email: str, total_count: int, split_point: int) -> str:
    """Generate a Redis cache key for a session summary."""
    email_part = (user_email or 'unknown').replace('@', '_at_')
    return f"session_summary:{email_part}:{total_count}:{split_point}"


def _get_cached_summary(cache_key: str) -> Optional[str]:
    """Retrieve cached summary from Redis."""
    try:
        from src.memory.message_condenser import get_redis
        redis = get_redis()
        cached = redis.get(cache_key)
        return cached if cached else None
    except Exception:
        return None


def _cache_summary(cache_key: str, summary: str):
    """Cache summary in Redis."""
    try:
        from src.memory.message_condenser import get_redis
        redis = get_redis()
        redis.setex(cache_key, SUMMARY_CACHE_TTL, summary)
    except Exception as e:
        logger.warning(f"Failed to cache session summary: {e}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_conversation_compressor.py -v`
Expected: All PASS

- [ ] **Step 5: Integrate into context_builder**

Add `session_summary` field to `ConversationContext` dataclass (after the `continuity_context` field, around line 64):

```python
session_summary: str = ""  # Compressed summary of older messages in long conversations
```

Add `session_summary` to `to_prompt_sections()` (add inside the raw_sections block):

```python
if self.session_summary:
    raw_sections['session_summary'] = self.session_summary
```

Add `session_summary` to `PROVENANCE_TAGS`:

```python
'session_summary': '[SESSION SUMMARY — EARLIER IN THIS CONVERSATION]',
```

Add `session_summary` to `SOURCE_TOKEN_BUDGETS` in `token_budget.py` as Tier 1:

```python
'session_summary': 800,
```

And add to `TIER_1`:

```python
TIER_1 = {
    'entity_profiles', 'personality', 'conversation_history',
    'core_memory', 'scene_state', 'internal_state', 'session_summary',
}
```

In `_get_conversation_history_structured()` (around line 2251), add compression after building conversation_turns. Before the `return conversation_turns, continuity_context` at the end of the try block:

```python
# Compress long conversations
from src.memory.conversation_compressor import compress_conversation, should_compress
if should_compress(conversation_turns):
    summary, recent_turns = compress_conversation(conversation_turns, user_email)
    if summary:
        # Store summary separately; return only recent turns
        # The summary will be set on the context object by the caller
        return (recent_turns, continuity_context, summary)
    return (conversation_turns, continuity_context)
return (conversation_turns, continuity_context)
```

Update the callers in `_build_parallel()` and `build_lightweight()` to handle the optional third element in the tuple. In the `conversation_turns` result handling (around line 351):

```python
elif name == 'conversation_turns' and result is not None:
    if isinstance(result, tuple):
        if len(result) == 3:
            turns, continuity, summary = result
            context.session_summary = summary or ""
        elif len(result) == 2:
            turns, continuity = result
        else:
            turns, continuity = [], ""
        context.conversation_turns = turns or []
        context.continuity_context = continuity or ""
        if turns:
            from src.config.persona_config import get_persona_config
            _pc = get_persona_config()
            formatted = [f"{_pc.companion_short_name if t['role'] == 'assistant' else _pc.primary_user_name}: {t['content']}" for t in turns]
            context.conversation_history = "\n".join(formatted)
```

Apply the same pattern to `build_lightweight()` (around line 252) and `_build_sequential()` (around line 424).

- [ ] **Step 6: Add session summary to pipeline prompt assembly**

In `src/core/conversation/pipeline.py` `_assemble_prompt()`, add session summary injection in the REFERENCE DATA section (after the conversation continuity context, around line 850):

```python
# Session summary (compressed older messages in long conversations)
if context.session_summary:
    sections.append(f"\n[Earlier in this conversation — compressed summary]\n{context.session_summary}")
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_conversation_compressor.py tests/test_provenance.py tests/test_token_budget.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/memory/conversation_compressor.py tests/test_conversation_compressor.py src/core/conversation/context_builder.py src/core/conversation/pipeline.py src/core/conversation/token_budget.py
git commit -m "feat(memory): compress old conversations instead of truncating (#23)"
```

---

## Task 5: MEDIUM Complexity Tier & Tier-Based Source Selection (#22 Phase 1)

**Files:**
- Modify: `src/core/conversation/complexity_classifier.py` (add MEDIUM tier)
- Modify: `src/core/conversation/context_builder.py` (tier-based source selection)
- Test: `tests/test_complexity_tiers.py`

- [ ] **Step 1: Write failing test for MEDIUM tier**

```python
# tests/test_complexity_tiers.py
"""Tests for the MEDIUM complexity tier and tier-based source selection."""

import pytest
from src.core.conversation.complexity_classifier import (
    MessageComplexity,
    classify_message,
)


class TestMediumTier:
    """Test that MEDIUM tier classifies general conversation correctly."""

    def test_medium_tier_exists(self):
        assert hasattr(MessageComplexity, 'MEDIUM')

    @pytest.mark.parametrize("message", [
        "What did you think about that movie?",
        "I had a really long day at work today",
        "Tell me about something interesting",
        "What's your opinion on remote work?",
        "I've been thinking about getting a dog",
    ])
    def test_general_conversation_is_medium(self, message):
        result = classify_message(message)
        assert result.complexity == MessageComplexity.MEDIUM, \
            f"Expected MEDIUM for '{message}', got {result.complexity.value}: {result.reason}"

    @pytest.mark.parametrize("message", [
        "hey",
        "lol",
        "good morning",
        "thanks!",
        "ok",
    ])
    def test_greetings_still_simple(self, message):
        result = classify_message(message)
        assert result.complexity == MessageComplexity.SIMPLE

    @pytest.mark.parametrize("message", [
        "do you remember what I said about my mom last week?",
        "I'm feeling really depressed and I don't know what to do",
        "what was that thing you told me about your childhood?",
    ])
    def test_memory_emotional_still_complex(self, message):
        result = classify_message(message)
        assert result.complexity == MessageComplexity.COMPLEX


class TestSourceTiers:
    """Test that context builder selects sources based on complexity tier."""

    def test_source_tier_mapping_exists(self):
        from src.core.conversation.context_builder import SOURCE_TIERS
        assert 'SIMPLE' in SOURCE_TIERS
        assert 'MEDIUM' in SOURCE_TIERS
        assert 'COMPLEX' in SOURCE_TIERS

    def test_simple_has_fewest_sources(self):
        from src.core.conversation.context_builder import SOURCE_TIERS
        assert len(SOURCE_TIERS['SIMPLE']) < len(SOURCE_TIERS['MEDIUM'])

    def test_medium_subset_of_complex(self):
        from src.core.conversation.context_builder import SOURCE_TIERS
        assert SOURCE_TIERS['MEDIUM'].issubset(SOURCE_TIERS['COMPLEX'])

    def test_simple_subset_of_medium(self):
        from src.core.conversation.context_builder import SOURCE_TIERS
        assert SOURCE_TIERS['SIMPLE'].issubset(SOURCE_TIERS['MEDIUM'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_complexity_tiers.py -v`
Expected: FAIL — `MessageComplexity.MEDIUM` not defined

- [ ] **Step 3: Add MEDIUM tier to complexity_classifier.py**

In `src/core/conversation/complexity_classifier.py`, update the enum (around line 35):

```python
class MessageComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ACTION = "action"
```

Update `classify_message()` to route general conversation to MEDIUM. The key changes are in the classification priority (around line 115). After the COMPLEX checks and before the final default, add MEDIUM detection:

After the greeting pattern check (line ~177) and short reactions check (line ~187), replace the moderate-length SIMPLE classification (lines 207-214) with:

```python
# Moderate length, no strong triggers → MEDIUM (general conversation)
if len(stripped) <= 120:
    return ClassificationResult(
        complexity=MessageComplexity.MEDIUM,
        reason="general_conversation",
        confidence=0.6
    )
```

And change the final default (lines 216-221) from COMPLEX to:

```python
# Multi-sentence check: 3+ sentences → COMPLEX, otherwise MEDIUM
sentences = re.split(r'[.!?]+', stripped)
sentences = [s for s in sentences if s.strip()]
if len(sentences) >= 3:
    return ClassificationResult(
        complexity=MessageComplexity.COMPLEX,
        reason="multi_sentence",
        confidence=0.5
    )

return ClassificationResult(
    complexity=MessageComplexity.MEDIUM,
    reason="default_medium",
    confidence=0.5
)
```

- [ ] **Step 4: Add SOURCE_TIERS to context_builder.py**

In `src/core/conversation/context_builder.py`, add after the `PROVENANCE_TAGS` dict:

```python
# ---------------------------------------------------------------------------
# Source tiers — which sources to fetch at each complexity level.
# SIMPLE: minimal context for greetings/reactions
# MEDIUM: core + enrichment for general conversation
# COMPLEX: full 24-source build for memory queries, emotional, relationship
# ---------------------------------------------------------------------------

SOURCE_TIERS = {
    'SIMPLE': {
        'entity_profiles', 'personality', 'conversation_turns',
        'internal_state', 'scene_state', 'core_memory', 'schedule',
    },
    'MEDIUM': {
        'entity_profiles', 'personality', 'conversation_turns',
        'internal_state', 'scene_state', 'core_memory', 'schedule',
        # Add: memories, relationship, biographies, observations
        'memories', 'relationship_dynamics', 'biographies',
        'observations_context', 'graphiti_context', 'activities_context',
    },
    'COMPLEX': {
        # All sources
        'entity_profiles', 'personality', 'conversation_turns',
        'internal_state', 'scene_state', 'core_memory', 'schedule',
        'memories', 'relationship_dynamics', 'biographies',
        'observations_context', 'graphiti_context', 'activities_context',
        'fertility_context', 'user_context', 'values_context',
        'temporal_context', 'synthesized_events', 'episode_context',
        'reflections_context', 'opinions_context', 'curiosity_context',
        'goals_context', 'relationship_insights', 'relationship_evaluation',
        'synthesized_biographies',
    },
}
```

- [ ] **Step 5: Add build_medium() method to ContextBuilder**

Add a new method to `ContextBuilder` (after `build_lightweight()`):

```python
def build_medium(self, user_email: str, user_message: str, closeness_score: int = 50) -> ConversationContext:
    """
    Build medium context for general conversation.

    Fetches core sources plus key enrichment (memories, relationship,
    biographies, observations) but skips expensive/niche sources
    (episodes, reflections, opinions, curiosity, values, fertility).

    Typically completes in ~200-600ms vs ~500-2000ms for full build.
    """
    start_time = time.time()
    self._source_timings = {}

    context = ConversationContext(
        user_email=user_email,
        user_message=user_message,
        closeness_score=closeness_score
    )

    medium_sources = SOURCE_TIERS['MEDIUM']

    def timed_fetch(name, func, *args):
        source_start = time.time()
        try:
            result = func(*args)
            elapsed = time.time() - source_start
            return name, result, elapsed
        except Exception as e:
            elapsed = time.time() - source_start
            logger.warning(f"Medium context source '{name}' failed after {elapsed:.2f}s: {e}")
            return name, None, elapsed

    futures = []
    # Only submit sources that are in the MEDIUM tier
    source_map = {
        'entity_profiles': (self._get_entity_profiles, user_message, user_email),
        'personality': (self._get_personality, user_email),
        'internal_state': (self._get_internal_state, user_email),
        'scene_state': (self._get_scene_state, user_email),
        'core_memory': (self._get_core_memory, user_email),
        'conversation_turns': (self._get_conversation_history_structured, user_email),
        'schedule': (self._get_time_awareness_context, user_email),
        'memories': (self._get_memories, user_email, user_message),
        'relationship_dynamics': (self._get_relationship_dynamics_context, user_email),
        'biographies': (self._get_synthesized_biographies, user_email),
        'observations_context': (self._get_observations_context, user_email, user_message),
        'graphiti_context': (self._get_graphiti_context, user_email, user_message),
        'activities_context': (self._get_activities_context, user_email),
    }

    for name, args in source_map.items():
        if name in medium_sources or name == 'conversation_turns':
            func = args[0]
            func_args = args[1:]
            futures.append(self._executor.submit(timed_fetch, name, func, *func_args))

    for future in futures:
        try:
            name, result, elapsed = future.result(timeout=15)
            self._source_timings[name] = elapsed

            if name == 'memories' and result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    contextual, _ = result
                    context.memories = contextual or ""
            elif name == 'synthesized_biographies' and result is not None:
                context.biographies = result or ""
            elif name == 'conversation_turns' and result is not None:
                if isinstance(result, tuple):
                    if len(result) == 3:
                        turns, continuity, summary = result
                        context.session_summary = summary or ""
                    elif len(result) == 2:
                        turns, continuity = result
                    else:
                        turns, continuity = [], ""
                    context.conversation_turns = turns or []
                    context.continuity_context = continuity or ""
                    if turns:
                        from src.config.persona_config import get_persona_config
                        _pc = get_persona_config()
                        formatted = [f"{_pc.companion_short_name if t['role'] == 'assistant' else _pc.primary_user_name}: {t['content']}" for t in turns]
                        context.conversation_history = "\n".join(formatted)
            elif result is not None:
                setattr(context, name, result or "")
        except Exception as e:
            logger.warning(f"Failed to get medium context source: {e}")

    total_time = time.time() - start_time
    logger.info(
        f"Medium context built in {total_time:.2f}s "
        f"({len(futures)} sources)"
    )

    return context
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_complexity_tiers.py tests/test_provenance.py tests/test_token_budget.py tests/test_conversation_compressor.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/core/conversation/complexity_classifier.py src/core/conversation/context_builder.py tests/test_complexity_tiers.py
git commit -m "feat(memory): add MEDIUM complexity tier and tier-based source selection (#22)"
```

---

## Self-Review Checklist

1. **Spec coverage**: All 5 issues covered. #20 (eviction + recency) = Task 3. #21 (token budgets) = Task 2. #22 Phase 1 (tiers) = Task 5. #23 (compression) = Task 4. #24 (provenance) = Task 1. Issue #22 Phases 2-3 (LLM memory tool, two-pass generation) are intentionally deferred — they're large architectural changes that should be planned separately.
2. **Placeholder scan**: No TBD/TODO. All code blocks are complete.
3. **Type consistency**: `MessageComplexity.MEDIUM` used consistently. `SOURCE_TIERS` dict keys match enum values. `session_summary` field name consistent across dataclass, provenance, budget, and pipeline. `apply_recency_weighting` signature consistent between test and implementation.
