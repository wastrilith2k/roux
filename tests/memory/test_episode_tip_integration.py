"""
Episode-Tip Integration Tests (ETI01-ETI02)

Tests that tip extraction runs as part of episode learning.
Uses mocked LLM and DB.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestEpisodeTipIntegration:
    """Test that episode learning triggers tip extraction."""

    def test_eti01_episode_learning_calls_tip_extractor(self):
        """ETI01: run_episode_learning calls tip extraction after lesson extraction."""
        from src.tasks.episode_learning_task import EpisodeLesson

        mock_lesson = EpisodeLesson(
            episode_id='ep-test',
            topic='test topic',
            emotional_context='neutral',
            satisfaction=0.8,
            successful_approach='was supportive',
            pitfalls_to_avoid='none',
            summary='test summary',
        )

        mock_extractor = MagicMock()
        mock_extractor.get_unanalyzed_episodes.return_value = [{
            'episode_id': 'ep-test',
            'topic': 'test',
            'emotional_state': 'neutral',
        }]
        mock_extractor.get_episode_messages.return_value = [
            {'sender_name': 'user', 'message_text': f'msg {i}'}
            for i in range(5)
        ]
        mock_extractor.analyze_episode.return_value = mock_lesson
        mock_extractor.save_episode_lesson.return_value = True

        mock_tip_extractor = MagicMock()
        mock_tip_extractor.extract_tips.return_value = []

        mock_tip_store = MagicMock()

        with patch('src.tasks.episode_learning_task.EpisodeLearningExtractor', return_value=mock_extractor):
            with patch('src.tasks.episode_learning_task.StrategyTipExtractor', return_value=mock_tip_extractor):
                with patch('src.tasks.episode_learning_task.StrategyTipStore', return_value=mock_tip_store):
                    from src.tasks.episode_learning_task import run_episode_learning
                    result = run_episode_learning()

        mock_tip_extractor.extract_tips.assert_called_once()

    def test_eti02_tip_extraction_failure_doesnt_break_learning(self):
        """ETI02: If tip extraction fails, episode learning still succeeds."""
        from src.tasks.episode_learning_task import EpisodeLesson

        mock_lesson = EpisodeLesson(
            episode_id='ep-test',
            topic='test', emotional_context='neutral',
            satisfaction=0.8, successful_approach='ok',
            pitfalls_to_avoid='', summary='test',
        )

        mock_extractor = MagicMock()
        mock_extractor.get_unanalyzed_episodes.return_value = [{
            'episode_id': 'ep-test', 'topic': 'test', 'emotional_state': 'neutral',
        }]
        mock_extractor.get_episode_messages.return_value = [
            {'sender_name': 'user', 'message_text': f'msg {i}'} for i in range(5)
        ]
        mock_extractor.analyze_episode.return_value = mock_lesson
        mock_extractor.save_episode_lesson.return_value = True

        mock_tip_extractor = MagicMock()
        mock_tip_extractor.extract_tips.side_effect = Exception("LLM exploded")

        with patch('src.tasks.episode_learning_task.EpisodeLearningExtractor', return_value=mock_extractor):
            with patch('src.tasks.episode_learning_task.StrategyTipExtractor', return_value=mock_tip_extractor):
                with patch('src.tasks.episode_learning_task.StrategyTipStore', return_value=MagicMock()):
                    from src.tasks.episode_learning_task import run_episode_learning
                    result = run_episode_learning()

        assert result['analyzed'] == 1
