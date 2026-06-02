"""Regression tests for issue #24: Source provenance markers in context prompt.

The bug: All context sources are injected as flat text with no provenance markers.
The LLM cannot distinguish between real memories, simulated activities, static
personality templates, or inferred values — leading to confabulation risk where
simulated activities are presented as genuine memories.

These tests verify that:
1. to_prompt_sections() prepends provenance tags to each section
2. _assemble_prompt() includes provenance-aware instructions in the final reminder
3. _format_records() in memory_validation_agent uses consistent provenance tags
4. Benchmark questions include provenance-aware test cases
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


@dataclass
class FakePersonaConfig:
    companion_short_name: str = "Kai"
    companion_name: str = "Kai Tanaka"
    primary_user_name: str = "James"
    primary_user_email: str = "james@test.com"
    c_possessive: str = "her"
    c_pronoun_subject: str = "she"
    c_pronoun_object: str = "her"
    u_possessive: str = "his"
    u_pronoun_subject: str = "he"
    u_pronoun_object: str = "him"
    user_pronoun_subject: str = "he"
    user_pronoun_object: str = "him"
    user_pronoun_possessive: str = "his"


def _get_fake_persona_config():
    return FakePersonaConfig()


# ============================================================
# Phase 1: to_prompt_sections() provenance tags
# ============================================================

class TestToPromptSectionsProvenance:
    """Verify to_prompt_sections() prepends provenance tags to each section."""

    def _make_context(self, **kwargs):
        from src.core.conversation.context_builder import ConversationContext
        return ConversationContext(**kwargs)

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_entity_profiles_tagged_as_verified_fact(self, mock_budgets):
        ctx = self._make_context(entity_profiles="Jesse is James's son.")
        sections = ctx.to_prompt_sections()
        assert '[VERIFIED FACT]' in sections['entity_profiles']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_memories_tagged_as_past_conversation(self, mock_budgets):
        ctx = self._make_context(memories="We talked about Portland yesterday.")
        sections = ctx.to_prompt_sections()
        assert '[FROM PAST CONVERSATION]' in sections['memories']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_activities_tagged_as_simulated(self, mock_budgets):
        ctx = self._make_context(activities_context="Spent the afternoon reading.")
        sections = ctx.to_prompt_sections()
        assert '[SIMULATED ACTIVITY' in sections['activities_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_values_tagged_as_inferred(self, mock_budgets):
        ctx = self._make_context(values_context="Values autonomy highly.")
        sections = ctx.to_prompt_sections()
        assert '[INFERRED VALUES' in sections['values_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_user_context_tagged_as_inferred(self, mock_budgets):
        ctx = self._make_context(user_context="User is probably sleeping.")
        sections = ctx.to_prompt_sections()
        assert '[INFERRED USER STATE' in sections['user_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_conversation_history_tagged_as_current_session(self, mock_budgets):
        ctx = self._make_context(conversation_history="User: hi\nAssistant: hey")
        sections = ctx.to_prompt_sections()
        assert '[CURRENT SESSION]' in sections['conversation_history']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_core_memory_tagged_as_curated(self, mock_budgets):
        ctx = self._make_context(core_memory="James loves his sons deeply.")
        sections = ctx.to_prompt_sections()
        assert '[YOUR CURATED MEMORY]' in sections['core_memory']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_opinions_tagged_as_inferred(self, mock_budgets):
        ctx = self._make_context(opinions_context="I think he's resilient.")
        sections = ctx.to_prompt_sections()
        assert '[YOUR OPINION' in sections['opinions_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_reflections_tagged_as_internal(self, mock_budgets):
        ctx = self._make_context(reflections_context="I've been thinking about boundaries.")
        sections = ctx.to_prompt_sections()
        assert '[YOUR REFLECTION' in sections['reflections_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_curiosity_tagged_as_internal(self, mock_budgets):
        ctx = self._make_context(curiosity_context="Want to ask about his new project.")
        sections = ctx.to_prompt_sections()
        assert '[YOUR CURIOSITY' in sections['curiosity_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_personality_tagged_as_personality(self, mock_budgets):
        ctx = self._make_context(personality="Witty, warm, playful.")
        sections = ctx.to_prompt_sections()
        assert '[YOUR PERSONALITY]' in sections['personality']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_graphiti_tagged_as_verified(self, mock_budgets):
        ctx = self._make_context(graphiti_context="James works in software.")
        sections = ctx.to_prompt_sections()
        assert '[KNOWLEDGE GRAPH' in sections['graphiti_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_episode_tagged_as_verified(self, mock_budgets):
        ctx = self._make_context(episode_context="A past episode about moving.")
        sections = ctx.to_prompt_sections()
        assert '[PAST EPISODE' in sections['episode_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_observations_tagged_as_from_conversations(self, mock_budgets):
        ctx = self._make_context(observations_context="He mentioned feeling stressed.")
        sections = ctx.to_prompt_sections()
        assert '[YOUR OBSERVATION' in sections['observations_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_biographies_tagged_as_derived(self, mock_budgets):
        ctx = self._make_context(biographies="James grew up in the Pacific Northwest.")
        sections = ctx.to_prompt_sections()
        assert '[BIOGRAPHICAL SUMMARY' in sections['biographies']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_relationship_dynamics_tagged(self, mock_budgets):
        ctx = self._make_context(relationship_dynamics="Closeness: 75")
        sections = ctx.to_prompt_sections()
        assert '[RELATIONSHIP STATE' in sections['relationship_dynamics']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_fertility_tagged_as_private(self, mock_budgets):
        ctx = self._make_context(fertility_context="Day 14 of cycle.")
        sections = ctx.to_prompt_sections()
        assert '[BIOLOGICAL TRACKING' in sections['fertility_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_original_content_preserved_after_tag(self, mock_budgets):
        """Provenance tag is prepended but original content is intact."""
        ctx = self._make_context(activities_context="Spent the afternoon reading.")
        sections = ctx.to_prompt_sections()
        assert 'Spent the afternoon reading.' in sections['activities_context']

    @patch('src.core.conversation.token_budget.apply_source_budgets', side_effect=lambda x: x)
    def test_empty_sections_still_excluded(self, mock_budgets):
        """Empty sections should not appear even with provenance system."""
        ctx = self._make_context(activities_context="", memories="")
        sections = ctx.to_prompt_sections()
        assert 'activities_context' not in sections
        assert 'memories' not in sections


# ============================================================
# Phase 2: _assemble_prompt() provenance-aware instructions
# ============================================================

class TestAssemblePromptProvenanceInstructions:
    """Verify _assemble_prompt() includes provenance-aware guidance."""

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_final_reminder_references_simulated_activity(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Final reminder should warn about SIMULATED ACTIVITY content."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()
            prompt = pipeline._assemble_prompt(context, "hello")

        assert 'SIMULATED' in prompt
        assert 'INFERRED' in prompt

    @patch('src.core.conversation.pipeline.get_context_builder')
    @patch('src.core.conversation.pipeline.get_memory_validation_agent')
    @patch('src.core.conversation.pipeline.get_message_validator_agent')
    def test_final_reminder_references_verified_content(
        self, mock_validator, mock_memory, mock_ctx_builder
    ):
        """Final reminder should guide LLM to trust VERIFIED and PAST CONVERSATION."""
        from src.core.conversation.context_builder import ConversationContext
        from src.core.conversation.pipeline import ConversationPipeline

        with patch(
            'src.config.persona_config.get_persona_config',
            _get_fake_persona_config
        ):
            pipeline = ConversationPipeline()
            context = ConversationContext()
            prompt = pipeline._assemble_prompt(context, "hello")

        assert 'VERIFIED' in prompt or 'PAST CONVERSATION' in prompt


# ============================================================
# Phase 3: memory_validation_agent provenance tags
# ============================================================

class TestMemoryValidationProvenance:
    """Verify _format_records uses provenance tags consistent with context_builder."""

    def test_postgres_source_uses_provenance_tag(self):
        from src.core.memory_validation_agent import MemoryValidationAgent
        from src.core.verified_memory import VerifiedMemory

        agent = MemoryValidationAgent.__new__(MemoryValidationAgent)
        records = [VerifiedMemory(content="We talked about work.", source='postgres', relevance=0.9)]
        formatted = agent._format_records(records)

        assert len(formatted) == 1
        assert '[FROM PAST CONVERSATION]' in formatted[0]

    def test_graphiti_source_uses_provenance_tag(self):
        from src.core.memory_validation_agent import MemoryValidationAgent
        from src.core.verified_memory import VerifiedMemory

        agent = MemoryValidationAgent.__new__(MemoryValidationAgent)
        records = [VerifiedMemory(content="James is a software engineer.", source='graphiti', relevance=0.8)]
        formatted = agent._format_records(records)

        assert len(formatted) == 1
        assert '[KNOWLEDGE GRAPH' in formatted[0]

    def test_entity_profile_source_uses_provenance_tag(self):
        from src.core.memory_validation_agent import MemoryValidationAgent
        from src.core.verified_memory import VerifiedMemory

        agent = MemoryValidationAgent.__new__(MemoryValidationAgent)
        records = [VerifiedMemory(content="Jesse is James's son.", source='entity_profile', relevance=1.0)]
        formatted = agent._format_records(records)

        assert len(formatted) == 1
        assert '[VERIFIED FACT]' in formatted[0]

    def test_search_tool_source_uses_provenance_tag(self):
        from src.core.memory_validation_agent import MemoryValidationAgent
        from src.core.verified_memory import VerifiedMemory

        agent = MemoryValidationAgent.__new__(MemoryValidationAgent)
        records = [VerifiedMemory(content="Some search result.", source='search_tool', relevance=0.6)]
        formatted = agent._format_records(records)

        assert len(formatted) == 1
        assert '[FROM PAST CONVERSATION]' in formatted[0] or '[SEARCH RESULT]' in formatted[0]


# ============================================================
# Phase 4: Benchmark questions with provenance awareness
# ============================================================

class TestBenchmarkProvenanceQuestions:
    """Verify benchmark questions include provenance-aware test cases."""

    def test_provenance_category_exists(self):
        from src.memory.benchmark_questions_data import get_categories
        categories = get_categories()
        assert 'provenance' in categories

    def test_provenance_questions_exist(self):
        from src.memory.benchmark_questions_data import get_questions_by_category
        questions = get_questions_by_category('provenance')
        assert len(questions) >= 3

    def test_provenance_questions_have_negative_keywords(self):
        """Provenance questions should test confabulation from simulated activities."""
        from src.memory.benchmark_questions_data import get_questions_by_category
        questions = get_questions_by_category('provenance')
        # At least one question should have negative keywords to catch confabulation
        has_negative = any(q.get('negative_keywords') for q in questions)
        assert has_negative
