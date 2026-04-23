"""
Strategy Tip Extractor Tests (STE01-STE04)

Tests LLM-based tip extraction from episodes.
Uses mocked LLM responses.
"""

import pytest
from unittest.mock import patch, MagicMock


MOCK_LLM_RESPONSE = """[
  {
    "tip_type": "strategy",
    "content": "When user mentions stress about parenting, matching his energy with empathy before advice creates space.",
    "trigger_condition": "user expresses parenting stress",
    "context_category": "family_children",
    "emotional_context": "stressed",
    "esme_authenticity": 0.9,
    "is_principled": false
  },
  {
    "tip_type": "recovery",
    "content": "When a playful comment lands wrong, acknowledging it directly works better than doubling down.",
    "trigger_condition": "Playful comment gets negative reaction",
    "context_category": "playful_fun",
    "emotional_context": "tense",
    "esme_authenticity": 0.8,
    "is_principled": false
  }
]"""


class TestStrategyTipExtraction:
    """Test LLM-based tip extraction from episodes."""

    def test_ste01_extract_tips_from_episode(self):
        """STE01: Extracts tips from episode with mocked LLM."""
        from src.memory.strategy_tip_extractor import StrategyTipExtractor
        from src.tasks.episode_learning_task import EpisodeLesson

        extractor = StrategyTipExtractor()

        lesson = EpisodeLesson(
            episode_id='ep-001',
            topic='parenting stress',
            emotional_context='stressed',
            satisfaction=0.85,
            successful_approach='Led with empathy',
            pitfalls_to_avoid='None identified',
            summary='user vented about parenting stress',
        )

        messages = [
            {'sender_name': 'user', 'message_text': 'The kids are driving me crazy today'},
            {'sender_name': 'companion', 'message_text': 'that sounds exhausting... what happened?'},
            {'sender_name': 'user', 'message_text': 'Jesse just refuses to listen to anything'},
            {'sender_name': 'companion', 'message_text': 'ugh, that push-pull phase is so draining'},
        ]

        with patch('src.memory.strategy_tip_extractor.generate_sync', return_value=MOCK_LLM_RESPONSE):
            with patch('src.memory.strategy_tip_extractor.get_resilient_provider_chain', return_value='mock'):
                with patch('src.memory.strategy_tip_extractor.track_llm_call'):
                    tips = extractor.extract_tips(
                        episode_id='ep-001',
                        messages=messages,
                        lesson=lesson,
                    )

        assert len(tips) == 2
        assert tips[0].tip_type == 'strategy'
        assert tips[1].tip_type == 'recovery'
        assert tips[0].source_ids == ['ep-001']
        assert tips[0].authenticity_avg == 0.9

    def test_ste02_skip_short_episodes(self):
        """STE02: Skips episodes with too few messages."""
        from src.memory.strategy_tip_extractor import StrategyTipExtractor
        from src.tasks.episode_learning_task import EpisodeLesson

        extractor = StrategyTipExtractor()

        lesson = EpisodeLesson(
            episode_id='ep-002', topic='greeting',
            emotional_context='neutral', satisfaction=0.5,
            successful_approach='', pitfalls_to_avoid='',
            summary='Quick hello',
        )

        tips = extractor.extract_tips(
            episode_id='ep-002',
            messages=[{'sender_name': 'user', 'message_text': 'hey'}],
            lesson=lesson,
        )

        assert tips == []

    def test_ste03_handles_llm_failure(self):
        """STE03: Returns empty list on LLM failure."""
        from src.memory.strategy_tip_extractor import StrategyTipExtractor
        from src.tasks.episode_learning_task import EpisodeLesson

        extractor = StrategyTipExtractor()

        lesson = EpisodeLesson(
            episode_id='ep-003', topic='test',
            emotional_context='neutral', satisfaction=0.5,
            successful_approach='', pitfalls_to_avoid='',
            summary='Test conversation',
        )

        messages = [
            {'sender_name': 'user', 'message_text': f'message {i}'}
            for i in range(5)
        ]

        with patch('src.memory.strategy_tip_extractor.generate_sync', return_value=None):
            with patch('src.memory.strategy_tip_extractor.get_resilient_provider_chain', return_value='mock'):
                with patch('src.memory.strategy_tip_extractor.track_llm_call'):
                    tips = extractor.extract_tips(
                        episode_id='ep-003',
                        messages=messages,
                        lesson=lesson,
                    )

        assert tips == []

    def test_ste04_caps_at_three_tips(self):
        """STE04: At most 3 tips per episode."""
        from src.memory.strategy_tip_extractor import StrategyTipExtractor, MAX_TIPS_PER_EPISODE
        assert MAX_TIPS_PER_EPISODE == 3
