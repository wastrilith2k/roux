"""Regression tests for issue #21: per-source token budgets.

The bug: Context sources had no per-source token limits. A single verbose
source (e.g. biographies at 12 paragraphs) could consume thousands of tokens
while relevant sources got squeezed. No total cap, no reasoning reserve,
no tier-based reallocation.

The fix: token_budget.py adds per-source budgets with sentence-aware
truncation, a total cap with tier-based reallocation, and a reasoning
reserve that reduces the effective provider limit.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePersonaConfig:
    companion_short_name: str = "Kai"
    companion_name: str = "Kai Tanaka"
    primary_user_name: str = "James"
    c_possessive: str = "her"
    c_pronoun_subject: str = "she"
    c_pronoun_object: str = "her"
    u_possessive: str = "his"
    u_pronoun_subject: str = "he"
    u_pronoun_object: str = "him"
    user_pronoun_subject: str = "he"
    user_pronoun_object: str = "him"
    user_pronoun_possessive: str = "his"
    primary_user_email: str = "test@test.com"


def _get_fake_persona_config():
    return FakePersonaConfig()


def _make_pipeline():
    """Create a ConversationPipeline with mocked dependencies."""
    with patch(
        'src.config.persona_config.get_persona_config',
        _get_fake_persona_config
    ):
        from src.core.conversation.pipeline import ConversationPipeline
        pipeline = ConversationPipeline()
    return pipeline


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self):
        from src.core.conversation.token_budget import estimate_tokens
        assert estimate_tokens("") == 0

    def test_known_length(self):
        from src.core.conversation.token_budget import estimate_tokens
        # 400 chars -> 100 tokens
        assert estimate_tokens("x" * 400) == 100

    def test_short_string(self):
        from src.core.conversation.token_budget import estimate_tokens
        # 3 chars -> 0 tokens (integer division)
        assert estimate_tokens("abc") == 0

    def test_four_chars_is_one_token(self):
        from src.core.conversation.token_budget import estimate_tokens
        assert estimate_tokens("abcd") == 1


# ---------------------------------------------------------------------------
# truncate_to_budget
# ---------------------------------------------------------------------------

class TestTruncateToBudget:
    def test_short_text_unchanged(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "Hello world."
        assert truncate_to_budget(text, 100) == text

    def test_truncates_at_sentence_boundary(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "First sentence. Second sentence. Third sentence."
        # Budget of 10 tokens = ~40 chars. "First sentence. Second sentence." is 32 chars.
        result = truncate_to_budget(text, 10)
        assert result.endswith(".")
        assert "Third" not in result

    def test_truncates_at_word_boundary_when_no_sentence_end(self):
        from src.core.conversation.token_budget import truncate_to_budget
        # No sentence-ending punctuation within budget
        text = "word " * 100  # 500 chars, no periods
        result = truncate_to_budget(text, 5)  # 20 chars
        assert not result.endswith(" ")
        assert len(result) <= 20

    def test_empty_text_returns_empty(self):
        from src.core.conversation.token_budget import truncate_to_budget
        assert truncate_to_budget("", 100) == ""

    def test_zero_budget_returns_empty(self):
        from src.core.conversation.token_budget import truncate_to_budget
        assert truncate_to_budget("Hello world.", 0) == ""

    def test_preserves_text_exactly_at_budget(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "x" * 400  # Exactly 100 tokens
        assert truncate_to_budget(text, 100) == text

    def test_handles_exclamation_and_question_marks(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "Really? Yes! And then some more text that goes on and on."
        result = truncate_to_budget(text, 5)  # ~20 chars
        assert result[-1] in '.!?'


# ---------------------------------------------------------------------------
# Issue #60: fragile sentence detection
# ---------------------------------------------------------------------------

class TestSentenceBoundaryHeuristic:
    """Regression tests for issue #60: the sentence-boundary detector must
    not break at abbreviations, decimals, URLs, or ellipsis."""

    def test_no_break_at_abbreviation(self):
        from src.core.conversation.token_budget import truncate_to_budget
        # "Dr." followed by a space and uppercase — the old code would break here.
        # Budget is tight enough to force truncation, so the function must skip "Dr."
        text = "Dr. Smith is a very qualified doctor and researcher in this field."
        # Budget ~10 tokens = 40 chars; "Dr." is at index 2, old code would stop there.
        result = truncate_to_budget(text, 10)
        assert result != "Dr."
        # Should fall back to a word boundary instead of breaking after "Dr."
        assert len(result) > 3

    def test_no_break_at_ellipsis(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "She paused... And then she continued talking for a while more."
        result = truncate_to_budget(text, 10)  # ~40 chars
        # Must not break after any dot in the ellipsis
        assert not result.endswith(".")  or "..." not in result

    def test_no_break_at_abbreviation_with_periods(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "The U.S. Government issued a statement about something important."
        result = truncate_to_budget(text, 10)  # ~40 chars
        # Must not break at "U." or "S."
        assert "U." not in result or "S." not in result or len(result) > 10

    def test_no_break_at_decimal(self):
        from src.core.conversation.token_budget import truncate_to_budget
        # Decimal followed by whitespace: "3. " — old code would break here
        text = "The value was approx 3. The experiment showed some results here."
        result = truncate_to_budget(text, 10)  # ~40 chars
        # "3" is a 1-char word, so the heuristic should reject this as a sentence end
        assert not result.endswith("3.")

    def test_still_breaks_at_real_sentence(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "First sentence here. Second sentence here. Third sentence here."
        result = truncate_to_budget(text, 8)  # ~32 chars
        assert result.endswith(".")
        assert "Third" not in result

    def test_exclamation_after_lowercase_rejected(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "wow! really cool stuff happening here and more and more text."
        result = truncate_to_budget(text, 8)  # ~32 chars
        # "wow!" followed by " really" — 'r' is lowercase, should not break
        assert not result.endswith("!")  or result != "wow!"

    def test_question_followed_by_uppercase_accepted(self):
        from src.core.conversation.token_budget import truncate_to_budget
        text = "Is it done? Yes it is and we should keep going with more text."
        result = truncate_to_budget(text, 5)  # ~20 chars
        assert result == "Is it done?"

    def test_is_sentence_boundary_direct(self):
        from src.core.conversation.token_budget import _is_sentence_boundary
        # Real sentence boundary: period + space + uppercase
        assert _is_sentence_boundary("Hello. World", 5) is True
        # Abbreviation: short word before period
        assert _is_sentence_boundary("Dr. Smith", 2) is False
        # Ellipsis: period preceded by period
        assert _is_sentence_boundary("Wait... Then", 6) is False
        # Embedded periods: "U.S."
        assert _is_sentence_boundary("U.S. Army", 3) is False
        # Period followed by lowercase
        assert _is_sentence_boundary("end. then", 3) is False
        # Period at end of string
        assert _is_sentence_boundary("The end.", 7) is True
        # Exclamation + space + uppercase
        assert _is_sentence_boundary("Go! Now", 2) is True
        # Question + space + lowercase
        assert _is_sentence_boundary("what? no", 4) is False


# ---------------------------------------------------------------------------
# apply_source_budgets
# ---------------------------------------------------------------------------

class TestApplySourceBudgets:
    def test_within_budget_unchanged(self):
        from src.core.conversation.token_budget import apply_source_budgets
        sections = {'personality': 'Short text.'}
        result = apply_source_budgets(sections)
        assert result['personality'] == 'Short text.'

    def test_over_budget_truncated(self):
        from src.core.conversation.token_budget import (
            apply_source_budgets, SOURCE_TOKEN_BUDGETS, estimate_tokens,
        )
        budget = SOURCE_TOKEN_BUDGETS['personality']  # 800 tokens
        # Create text way over budget
        over_text = "This is a sentence. " * 500  # ~2500 tokens
        sections = {'personality': over_text}
        result = apply_source_budgets(sections)
        assert estimate_tokens(result['personality']) <= budget + 1  # +1 for rounding

    def test_unknown_source_passed_through(self):
        from src.core.conversation.token_budget import apply_source_budgets
        sections = {'unknown_source': 'x' * 10000}
        result = apply_source_budgets(sections)
        assert result['unknown_source'] == 'x' * 10000

    def test_multiple_sources_independent(self):
        from src.core.conversation.token_budget import (
            apply_source_budgets, SOURCE_TOKEN_BUDGETS, estimate_tokens,
        )
        # One over, one under
        sections = {
            'personality': "Sentence. " * 500,  # Way over 800 token budget
            'entity_profiles': "Short.",         # Under 1500 token budget
        }
        result = apply_source_budgets(sections)
        assert estimate_tokens(result['personality']) <= SOURCE_TOKEN_BUDGETS['personality'] + 1
        assert result['entity_profiles'] == "Short."


# ---------------------------------------------------------------------------
# enforce_total_cap — tier-based reallocation
# ---------------------------------------------------------------------------

class TestEnforceTotalCap:
    def test_under_cap_unchanged(self):
        from src.core.conversation.token_budget import enforce_total_cap
        sections = {
            'entity_profiles': 'Short.',
            'memories': 'Also short.',
        }
        result = enforce_total_cap(sections, total_cap=50000)
        assert result == sections

    def test_tier3_trimmed_first(self):
        from src.core.conversation.token_budget import (
            enforce_total_cap, estimate_tokens,
        )
        # Tier 1 source + Tier 3 source, total over cap
        sections = {
            'entity_profiles': 'x' * 2000,     # Tier 1: 500 tokens
            'activities_context': 'y' * 4000,   # Tier 3: 1000 tokens
        }
        # Cap at 800 — need to cut 700 tokens from Tier 3
        result = enforce_total_cap(sections, total_cap=800)
        # Tier 1 untouched
        assert result['entity_profiles'] == 'x' * 2000
        # Tier 3 trimmed
        assert estimate_tokens(result['activities_context']) < 1000

    def test_tier1_never_trimmed(self):
        from src.core.conversation.token_budget import (
            enforce_total_cap, estimate_tokens,
        )
        # Only Tier 1 sources, over cap
        sections = {
            'entity_profiles': 'x' * 4000,   # 1000 tokens
            'personality': 'y' * 4000,        # 1000 tokens
            'core_memory': 'z' * 4000,        # 1000 tokens
        }
        # Cap of 500 — but Tier 1 is never trimmed
        result = enforce_total_cap(sections, total_cap=500)
        assert result['entity_profiles'] == 'x' * 4000
        assert result['personality'] == 'y' * 4000
        assert result['core_memory'] == 'z' * 4000

    def test_tier2_trimmed_after_tier3(self):
        from src.core.conversation.token_budget import (
            enforce_total_cap, estimate_tokens,
        )
        sections = {
            'entity_profiles': 'x' * 400,          # Tier 1: 100 tokens
            'memories': 'y' * 4000,                 # Tier 2: 1000 tokens
            'activities_context': 'z' * 4000,       # Tier 3: 1000 tokens
        }
        # Cap at 500 — need to cut 1600 tokens
        # Tier 3 has 1000, so after zeroing Tier 3, still need 600 from Tier 2
        result = enforce_total_cap(sections, total_cap=500)
        assert result['entity_profiles'] == 'x' * 400  # Tier 1 untouched
        assert estimate_tokens(result['activities_context']) < 1000  # Tier 3 trimmed
        assert estimate_tokens(result['memories']) < 1000  # Tier 2 also trimmed


# ---------------------------------------------------------------------------
# get_budget_diagnostics
# ---------------------------------------------------------------------------

class TestGetBudgetDiagnostics:
    def test_returns_per_source_info(self):
        from src.core.conversation.token_budget import get_budget_diagnostics
        sections = {'personality': 'x' * 400, 'memories': 'y' * 800}
        diag = get_budget_diagnostics(sections, provider_limit=10000)
        assert 'per_source' in diag
        assert 'personality' in diag['per_source']
        assert 'memories' in diag['per_source']
        assert diag['per_source']['personality']['tokens'] == 100
        assert diag['per_source']['personality']['tier'] == 1

    def test_reasoning_headroom_computed(self):
        from src.core.conversation.token_budget import get_budget_diagnostics
        sections = {'personality': 'x' * 400}  # 100 tokens
        diag = get_budget_diagnostics(sections, provider_limit=10000)
        assert diag['reasoning_headroom'] == 9900
        assert diag['total_tokens'] == 100

    def test_no_provider_limit(self):
        from src.core.conversation.token_budget import get_budget_diagnostics
        sections = {'personality': 'x' * 400}
        diag = get_budget_diagnostics(sections, provider_limit=None)
        assert diag['reasoning_headroom'] is None


# ---------------------------------------------------------------------------
# Integration: to_prompt_sections applies budgets
# ---------------------------------------------------------------------------

class TestToPromptSectionsAppliesBudgets:
    def test_oversized_source_truncated_by_to_prompt_sections(self):
        """to_prompt_sections() truncates sources that exceed their budget."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.token_budget import (
            SOURCE_TOKEN_BUDGETS, estimate_tokens,
        )

        ctx = ConversationContext()
        # Create a biographies section way over its 1000-token budget
        ctx.biographies = "This is a biography sentence. " * 500  # ~3750 tokens

        sections = ctx.to_prompt_sections()
        bio_tokens = estimate_tokens(sections['biographies'])
        assert bio_tokens <= SOURCE_TOKEN_BUDGETS['biographies'] + 1, \
            f"biographies should be truncated to ~{SOURCE_TOKEN_BUDGETS['biographies']} tokens, got {bio_tokens}"

    def test_small_source_unchanged_by_to_prompt_sections(self):
        """Sources within budget pass through unchanged."""
        from src.core.conversation.context_builder import ConversationContext

        ctx = ConversationContext()
        ctx.personality = "I am a personality."  # Well under 800 tokens

        sections = ctx.to_prompt_sections()
        assert "I am a personality." in sections['personality']
        assert sections['personality'].startswith("[YOUR PERSONALITY]")


# ---------------------------------------------------------------------------
# Integration: _assemble_prompt applies per-source budgets (Phase 1)
# ---------------------------------------------------------------------------

class TestAssemblePromptAppliesPerSourceBudgets:
    """Verify that per-source token budgets are enforced in the main prompt
    assembly path (_assemble_prompt), not just in to_prompt_sections().

    This is a regression test for the gap identified in the first attempt at
    issue #21: apply_source_budgets was only called from to_prompt_sections()
    which is never invoked by _assemble_prompt.
    """

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_oversized_source_truncated_in_assemble_prompt(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """A source that exceeds its per-source budget should be truncated
        in the prompt produced by _assemble_prompt."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.token_budget import (
            SOURCE_TOKEN_BUDGETS, estimate_tokens,
        )

        pipeline = _make_pipeline()

        bio_budget = SOURCE_TOKEN_BUDGETS['biographies']  # 1000 tokens
        # Create biographies WAY over budget (~3750 tokens).
        # Use a unique marker at the end so we can verify truncation removed it.
        oversized_bio = "This is a biography sentence. " * 500
        oversized_bio += "MARKER_END_OF_BIOGRAPHY."

        ctx = ConversationContext()
        ctx.biographies = oversized_bio

        with patch.object(pipeline, '_get_provider_context_limit', return_value=50000), \
             patch('src.config.persona_config.get_persona_config', _get_fake_persona_config):
            prompt = pipeline._assemble_prompt(ctx, "hello")

        # The biography content should appear in the prompt (start is kept)
        assert "biography sentence" in prompt

        # The end-of-text marker should be gone — truncation removed it
        assert "MARKER_END_OF_BIOGRAPHY" not in prompt, \
            "Oversized biography should be truncated, but the end marker survived"

        # The full original text should not be present
        assert oversized_bio not in prompt, \
            "The full oversized biography should not appear in the prompt"

        # Count occurrences to confirm significant truncation happened.
        # Original has 500 occurrences; budget of 1000 tokens (~4000 chars)
        # means roughly 133 occurrences should survive.
        occurrences = prompt.count("biography sentence")
        assert occurrences < 250, \
            f"Expected significant truncation of biography (got {occurrences} " \
            f"occurrences out of 500 original)"

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_within_budget_source_unchanged_in_assemble_prompt(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """A source within its budget should appear unchanged in the prompt."""
        from src.core.conversation.context_builder import ConversationContext

        pipeline = _make_pipeline()

        ctx = ConversationContext()
        ctx.personality = "I have a warm and caring personality."

        with patch.object(pipeline, '_get_provider_context_limit', return_value=50000), \
             patch('src.config.persona_config.get_persona_config', _get_fake_persona_config):
            prompt = pipeline._assemble_prompt(ctx, "hello")

        assert "I have a warm and caring personality." in prompt


# ---------------------------------------------------------------------------
# Integration: pipeline uses reasoning reserve
# ---------------------------------------------------------------------------

class TestReasoningReserve:
    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_reasoning_reserve_reduces_effective_limit(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """The reasoning reserve reduces the effective provider limit
        passed to _enforce_context_budget, ensuring headroom for generation."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.token_budget import REASONING_RESERVE

        pipeline = _make_pipeline()

        ctx = ConversationContext()
        ctx.entity_profiles = "ENTITY " + "x" * 200
        # Large low-priority section
        ctx.activities_context = "ACTIVITIES " + "x" * 20000

        # With provider_limit=10000, effective = 10000 - REASONING_RESERVE
        # The large activities section should be more aggressively trimmed
        with patch.object(pipeline, '_get_provider_context_limit', return_value=10000), \
             patch('src.config.persona_config.get_persona_config', _get_fake_persona_config):
            prompt = pipeline._assemble_prompt(ctx, "hello")

        # Entity profiles (Tier 1, high priority) should survive
        assert "ENTITY" in prompt

        # The total prompt should leave reasoning headroom
        from src.core.conversation.token_budget import estimate_tokens
        prompt_tokens = estimate_tokens(prompt)
        assert prompt_tokens < 10000 - REASONING_RESERVE + 500, \
            f"Prompt should leave reasoning headroom, got {prompt_tokens} tokens"


# ---------------------------------------------------------------------------
# Integration: diagnostics include token budget info
# ---------------------------------------------------------------------------

class TestDiagnosticsIncludeTokenBudgets:
    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_build_with_diagnostics_includes_token_budgets(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """build_with_diagnostics() should report per-source token budget info."""
        from src.core.conversation.context_builder import (
            ContextBuilder, ConversationContext,
        )

        builder = ContextBuilder.__new__(ContextBuilder)
        builder._source_timings = {}

        ctx = ConversationContext()
        ctx.personality = "Test personality."
        ctx.memories = "Test memories."

        with patch.object(builder, 'build', return_value=ctx):
            _, diagnostics = builder.build_with_diagnostics("test@test.com", "hello")

        assert 'token_budgets' in diagnostics
        assert 'per_source' in diagnostics['token_budgets']
        assert 'personality' in diagnostics['token_budgets']['per_source']
        assert 'total_tokens' in diagnostics['token_budgets']
        assert 'total_cap' in diagnostics['token_budgets']


# ---------------------------------------------------------------------------
# Regression: issue #64 — missing budget entries for context sources
# ---------------------------------------------------------------------------

class TestIssue64MissingBudgetEntries:
    """Regression tests for issue #64: derived_scene_context and schedule
    were missing from SOURCE_TOKEN_BUDGETS, bypassing per-source token caps.

    Without budget entries these sources pass through apply_source_budgets()
    unmodified and default to Tier 3 via get_tier(), meaning they get
    no per-source cap and incorrect tier assignment.
    """

    def test_derived_scene_context_has_budget_entry(self):
        """derived_scene_context must have a SOURCE_TOKEN_BUDGETS entry."""
        from src.core.conversation.token_budget import SOURCE_TOKEN_BUDGETS
        assert 'derived_scene_context' in SOURCE_TOKEN_BUDGETS, \
            "derived_scene_context missing from SOURCE_TOKEN_BUDGETS"
        assert SOURCE_TOKEN_BUDGETS['derived_scene_context'] > 0

    def test_schedule_has_budget_entry(self):
        """schedule must have a SOURCE_TOKEN_BUDGETS entry."""
        from src.core.conversation.token_budget import SOURCE_TOKEN_BUDGETS
        assert 'schedule' in SOURCE_TOKEN_BUDGETS, \
            "schedule missing from SOURCE_TOKEN_BUDGETS"
        assert SOURCE_TOKEN_BUDGETS['schedule'] > 0

    def test_derived_scene_context_not_default_tier(self):
        """derived_scene_context must be explicitly assigned to a tier,
        not fall through to the Tier 3 default."""
        from src.core.conversation.token_budget import (
            TIER_1, TIER_2, TIER_3, get_tier,
        )
        assert 'derived_scene_context' in (TIER_1 | TIER_2 | TIER_3), \
            "derived_scene_context not in any explicit tier set"
        # Should be Tier 2 (important derived context)
        assert get_tier('derived_scene_context') == 2

    def test_schedule_not_default_tier(self):
        """schedule must be explicitly assigned to a tier,
        not fall through to the Tier 3 default."""
        from src.core.conversation.token_budget import (
            TIER_1, TIER_2, TIER_3, get_tier,
        )
        assert 'schedule' in (TIER_1 | TIER_2 | TIER_3), \
            "schedule not in any explicit tier set"
        # Should be Tier 2 (can be large, important reference)
        assert get_tier('schedule') == 2

    def test_derived_scene_context_truncated_when_oversized(self):
        """Oversized derived_scene_context must be truncated by apply_source_budgets."""
        from src.core.conversation.token_budget import (
            apply_source_budgets, SOURCE_TOKEN_BUDGETS, estimate_tokens,
        )
        budget = SOURCE_TOKEN_BUDGETS['derived_scene_context']
        oversized = "This is a derived context sentence. " * 500  # Way over budget
        sections = {'derived_scene_context': oversized}
        result = apply_source_budgets(sections)
        result_tokens = estimate_tokens(result['derived_scene_context'])
        assert result_tokens <= budget + 1, \
            f"derived_scene_context should be capped at ~{budget} tokens, got {result_tokens}"

    def test_schedule_truncated_when_oversized(self):
        """Oversized schedule must be truncated by apply_source_budgets."""
        from src.core.conversation.token_budget import (
            apply_source_budgets, SOURCE_TOKEN_BUDGETS, estimate_tokens,
        )
        budget = SOURCE_TOKEN_BUDGETS['schedule']
        oversized = "Calendar event for today. " * 500  # Way over budget
        sections = {'schedule': oversized}
        result = apply_source_budgets(sections)
        result_tokens = estimate_tokens(result['schedule'])
        assert result_tokens <= budget + 1, \
            f"schedule should be capped at ~{budget} tokens, got {result_tokens}"

    def test_derived_scene_context_in_to_prompt_sections(self):
        """derived_scene_context must appear in to_prompt_sections() output."""
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext()
        ctx.derived_scene_context = "It is evening and the user is at home."
        sections = ctx.to_prompt_sections()
        assert 'derived_scene_context' in sections

    def test_schedule_in_to_prompt_sections(self):
        """schedule must appear in to_prompt_sections() output."""
        from src.core.conversation.context_builder import ConversationContext
        ctx = ConversationContext()
        ctx.schedule = "10:00 AM — Meeting with team. 2:00 PM — Dentist."
        sections = ctx.to_prompt_sections()
        assert 'schedule' in sections

    def test_all_to_prompt_sections_have_budget_entries(self):
        """Every source emitted by to_prompt_sections() must have a
        SOURCE_TOKEN_BUDGETS entry so it cannot bypass per-source caps."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.token_budget import SOURCE_TOKEN_BUDGETS

        # Populate every string field on ConversationContext with a non-empty value
        # so to_prompt_sections() emits all possible keys.
        ctx = ConversationContext()
        for field_name in vars(ctx):
            if field_name.startswith('_') or field_name == 'PROVENANCE_TAGS':
                continue
            current = getattr(ctx, field_name)
            if isinstance(current, str) and not current:
                setattr(ctx, field_name, f"test content for {field_name}")

        sections = ctx.to_prompt_sections()
        missing = [name for name in sections if name not in SOURCE_TOKEN_BUDGETS]
        assert not missing, \
            f"Sources emitted by to_prompt_sections() missing from SOURCE_TOKEN_BUDGETS: {missing}"
