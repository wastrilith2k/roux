"""Regression tests for issue #19: context budget enforcement.

The bug: _assemble_prompt() assembled all 24+ context sources unconditionally
and only logged a warning at 75% of the provider's context limit.  There was
no enforcement — the prompt could silently exceed the context window, causing
API errors or truncated input (context poisoning).

The fix: _enforce_context_budget() drops the lowest-priority reference-data
sections until the prompt fits within CONTEXT_BUDGET_FRACTION of the provider's
context window, reserving headroom for conversation turns and generation tokens.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


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


class TestEnforceContextBudget:
    """Test _enforce_context_budget drops low-priority sections correctly."""

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_no_drop_when_under_budget(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """When total tokens are under budget, all sections survive."""
        pipeline = _make_pipeline()

        fixed = ["identity block", "<reference_data>", "</reference_data>"]
        droppable = [
            ("memories", 2, "some memories"),
            ("opinions_context", 7, "some opinions"),
        ]

        # Provider limit large enough that nothing needs dropping
        result = pipeline._enforce_context_budget(
            fixed_sections=fixed,
            droppable_sections=droppable,
            provider_limit=100000,  # 100K tokens — plenty of room
        )

        # All sections should be present
        assert "some memories" in result
        assert "some opinions" in result

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_drops_lowest_priority_first(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Lowest-priority (highest number) sections are dropped first."""
        pipeline = _make_pipeline()
        # Override budget fraction for deterministic testing
        pipeline.CONTEXT_BUDGET_FRACTION = 0.6

        fixed = ["x" * 100]  # 25 tokens

        # Each section is ~250 tokens (1000 chars / 4)
        droppable = [
            ("entity_profiles", 1, "A" * 1000),    # priority 1 (highest)
            ("memories", 2, "B" * 1000),            # priority 2
            ("graphiti_context", 4, "C" * 1000),    # priority 4
            ("opinions_context", 7, "D" * 1000),    # priority 7
            ("activities_context", 9, "E" * 1000),  # priority 9 (lowest)
        ]

        # Total = 25 (fixed) + 5*250 (droppable) = 1275 tokens
        # Budget at provider_limit=1500 with 0.6 fraction = 900 tokens
        # Need to drop ~375 tokens worth = ~2 sections
        result = pipeline._enforce_context_budget(
            fixed_sections=fixed,
            droppable_sections=droppable,
            provider_limit=1500,
        )

        # Lowest priority sections (activities=9, opinions=7) should be dropped
        assert "A" * 1000 in result, "entity_profiles (priority 1) should survive"
        assert "B" * 1000 in result, "memories (priority 2) should survive"
        assert "C" * 1000 in result, "graphiti_context (priority 4) should survive"
        assert "E" * 1000 not in result, "activities_context (priority 9) should be dropped"
        assert "D" * 1000 not in result, "opinions_context (priority 7) should be dropped"

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_fixed_sections_never_dropped(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Fixed sections (identity, instructions, etc.) are never dropped."""
        pipeline = _make_pipeline()
        pipeline.CONTEXT_BUDGET_FRACTION = 0.6

        # Fixed sections alone are 500 tokens
        fixed = ["F" * 2000]  # 500 tokens

        droppable = [
            ("activities_context", 9, "G" * 400),  # 100 tokens
        ]

        # Budget = 600 * 0.6 = 360 tokens — fixed alone exceeds budget
        result = pipeline._enforce_context_budget(
            fixed_sections=fixed,
            droppable_sections=droppable,
            provider_limit=600,
        )

        # Fixed sections must survive even if over budget
        assert "F" * 2000 in result, "Fixed sections must never be dropped"
        # Droppable sections should be shed
        assert "G" * 400 not in result, "Droppable should be shed when over budget"

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_no_provider_limit_includes_everything(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """When provider limit is None, all sections are included."""
        pipeline = _make_pipeline()

        fixed = ["identity"]
        droppable = [
            ("memories", 2, "mem content"),
            ("opinions_context", 7, "opinions content"),
        ]

        result = pipeline._enforce_context_budget(
            fixed_sections=fixed,
            droppable_sections=droppable,
            provider_limit=None,
        )

        assert "mem content" in result
        assert "opinions content" in result


class TestAssemblePromptBudgetIntegration:
    """Integration tests: _assemble_prompt enforces budget end-to-end."""

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_oversized_context_drops_low_priority_sections(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """When context sources are large, low-priority sections are dropped."""
        from src.core.conversation.context_builder import ConversationContext

        pipeline = _make_pipeline()
        pipeline.CONTEXT_BUDGET_FRACTION = 0.6

        # Create context with many sources that together exceed the budget.
        # Issue #21 added per-source token budgets that truncate individual
        # sources before section-dropping runs.  To still trigger section
        # dropping, we need enough sources that the total (post-per-source
        # truncation) still exceeds the budget.
        # Note: issue #21 also added a reasoning reserve (default 4000 tokens)
        # that reduces the effective limit.
        context = ConversationContext()
        context.entity_profiles = "ENTITY_MARKER " + "x" * 200
        context.personality = "PERSONALITY_MARKER " + "x" * 200

        # Fill many low-priority sources to their per-source budgets so
        # the total droppable content is large.  Per-source budgets for
        # activities=400, curiosity=400, observations=600, reflections=500,
        # opinions=500, values=400, goals=400, temporal=400.
        # After per-source truncation these stay at their budgets (~3600
        # tokens total for low-priority).  Combined with fixed sections
        # (~1250 tokens) this should exceed a tight budget.
        context.activities_context = "ACTIVITIES_MARKER " + "x" * 10000   # budget 400
        context.curiosity_context = "CURIOSITY_MARKER " + "x" * 10000    # budget 400
        context.observations_context = "OBSERVATIONS_MARKER " + "x" * 10000  # budget 600
        context.reflections_context = "REFLECTIONS_MARKER " + "x" * 10000    # budget 500
        context.opinions_context = "OPINIONS_MARKER " + "x" * 10000         # budget 500
        context.values_context = "VALUES_MARKER " + "x" * 10000             # budget 400
        context.goals_context = "GOALS_MARKER " + "x" * 10000               # budget 400
        context.temporal_context = "TEMPORAL_MARKER " + "x" * 10000          # budget 400

        # Provider limit 8000 tokens, minus 4000 reasoning reserve = 4000 effective.
        # Budget at 0.6 fraction = 2400 tokens.
        # Fixed ~1250 + entity/personality ~100 = ~1350 (fits in 2400).
        # But low-priority sources add ~3600 tokens -> total ~4950 > 2400.
        with patch.object(pipeline, '_get_provider_context_limit', return_value=8000), \
             patch('src.config.persona_config.get_persona_config', _get_fake_persona_config):
            prompt = pipeline._assemble_prompt(context, "hello")

        # High-priority sections should survive
        assert "ENTITY_MARKER" in prompt, "entity_profiles (priority 1) must survive"
        assert "PERSONALITY_MARKER" in prompt, "personality (priority 2) must survive"

        # At least one low-priority section should be dropped to fit budget
        low_priority_present = sum(1 for marker in [
            "ACTIVITIES_MARKER", "CURIOSITY_MARKER", "OBSERVATIONS_MARKER",
            "REFLECTIONS_MARKER", "OPINIONS_MARKER", "VALUES_MARKER",
            "GOALS_MARKER", "TEMPORAL_MARKER",
        ] if marker in prompt)
        assert low_priority_present < 8, \
            "At least one low-priority section should be dropped to fit budget"

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_prompt_structure_preserved_after_budget_enforcement(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Prompt structure (identity -> reference_data -> instructions -> final_reminder)
        is maintained even when sections are dropped."""
        from src.core.conversation.context_builder import ConversationContext

        pipeline = _make_pipeline()

        context = ConversationContext()
        context.entity_profiles = "entity data here"
        context.memories = "memory data here"

        with patch.object(pipeline, '_get_provider_context_limit', return_value=100000), \
             patch('src.config.persona_config.get_persona_config', _get_fake_persona_config):
            prompt = pipeline._assemble_prompt(context, "hello")

        # Verify structural ordering
        identity_pos = prompt.find("<identity>")
        ref_data_pos = prompt.find("<reference_data>")
        ref_data_end = prompt.find("</reference_data>")
        instructions_pos = prompt.find("<instructions>")
        final_pos = prompt.find("<final_reminder>")

        assert identity_pos >= 0, "Identity section missing"
        assert ref_data_pos >= 0, "Reference data section missing"
        assert instructions_pos >= 0, "Instructions section missing"
        assert final_pos >= 0, "Final reminder section missing"

        assert identity_pos < ref_data_pos < ref_data_end < instructions_pos < final_pos, \
            "Prompt sections must appear in order: identity < reference_data < instructions < final_reminder"

        # Droppable content should be inside reference_data tags
        entity_pos = prompt.find("entity data here")
        memory_pos = prompt.find("memory data here")
        assert ref_data_pos < entity_pos < ref_data_end, \
            "Entity profiles should be inside <reference_data>"
        assert ref_data_pos < memory_pos < ref_data_end, \
            "Memories should be inside <reference_data>"

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_empty_context_still_produces_valid_prompt(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """An empty ConversationContext should still produce a valid prompt."""
        from src.core.conversation.context_builder import ConversationContext

        pipeline = _make_pipeline()

        context = ConversationContext()

        with patch.object(pipeline, '_get_provider_context_limit', return_value=100000), \
             patch('src.config.persona_config.get_persona_config', _get_fake_persona_config):
            prompt = pipeline._assemble_prompt(context, "hello")

        assert "<identity>" in prompt
        assert "<reference_data>" in prompt
        assert "</reference_data>" in prompt
        assert "<instructions>" in prompt
        assert "</instructions>" in prompt
        assert "<final_reminder>" in prompt
        assert "Kai" in prompt


class TestSectionPriority:
    """Verify section priority configuration is sensible (now tier-based via CoALA)."""

    def test_entity_profiles_highest_priority(self):
        """Entity profiles should be among the highest priority (never dropped early)."""
        from src.core.conversation.pipeline import ConversationPipeline
        prio = ConversationPipeline()._section_priority
        assert prio['entity_profiles'] <= 2  # PERMANENT tier

    def test_activities_lowest_priority(self):
        """Activities context should be among the lowest priority (dropped first)."""
        from src.core.conversation.pipeline import ConversationPipeline
        prio = ConversationPipeline()._section_priority
        assert prio['activities_context'] >= 6  # EPHEMERAL tier

    def test_memories_higher_than_opinions(self):
        """Core memories should have higher priority than opinions."""
        from src.core.conversation.pipeline import ConversationPipeline
        prio = ConversationPipeline()._section_priority
        assert prio['memories'] < prio['opinions_context']

    def test_all_context_sources_have_priority(self):
        """Every droppable context source that the pipeline uses should have a priority."""
        from src.core.conversation.pipeline import ConversationPipeline
        prio = ConversationPipeline()._section_priority

        # memory_validation uses the fallback default (ContextTier.EPHEMERAL) since
        # it is a pipeline-internal section not represented in SECTION_TIERS.
        expected_sources = [
            'entity_profiles', 'core_memory',
            'memories', 'personality', 'relationship_dynamics',
            'relationship_insights', 'relationship_evaluation',
            'scene_state', 'internal_state', 'fertility_context',
            'values_context', 'activities_context', 'temporal_context',
            'graphiti_context', 'synthesized_events', 'episode_context',
            'observations_context', 'reflections_context', 'opinions_context',
            'curiosity_context', 'goals_context', 'biographies',
            'user_context',
        ]

        for source in expected_sources:
            assert source in prio, f"Missing priority for '{source}'"
