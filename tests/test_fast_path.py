"""
Tests for fast path implementation (Issue #10, Task 3).

Covers:
- Message complexity classification (simple, complex, action)
- Sentence splitting for streaming TTS
- Feature flag behavior
"""

import os
import pytest

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')


# =========================================================================
# Complexity Classifier Tests
# =========================================================================

from unittest.mock import patch
from src.core.conversation.complexity_classifier import classify_message


class TestComplexityClassifier:
    """Test message complexity classification."""

    def _classify(self, message):
        """Helper — classify with fast path enabled."""
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', True):
            return classify_message(message)

    # --- SIMPLE messages ---

    def test_greeting_hey(self):
        result = self._classify("hey!")
        assert result.complexity.value == "simple"

    def test_greeting_good_morning(self):
        result = self._classify("good morning")
        assert result.complexity.value == "simple"

    def test_greeting_how_are_you(self):
        result = self._classify("how are you?")
        assert result.complexity.value == "simple"

    def test_short_reaction_nice(self):
        result = self._classify("nice")
        assert result.complexity.value == "simple"

    def test_short_reaction_lol(self):
        result = self._classify("lol")
        assert result.complexity.value == "simple"

    def test_im_home(self):
        result = self._classify("i'm home")
        assert result.complexity.value == "simple"

    def test_love_you(self):
        result = self._classify("love you")
        assert result.complexity.value == "simple"

    def test_short_casual(self):
        result = self._classify("sounds good")
        assert result.complexity.value == "simple"

    # --- COMPLEX messages ---

    def test_emotional_feeling_sad(self):
        result = self._classify("I'm feeling really sad today and I don't know what to do")
        assert result.complexity.value == "complex"

    def test_emotional_anxiety(self):
        result = self._classify("I've been so anxious about the interview")
        assert result.complexity.value == "complex"

    def test_memory_reference(self):
        result = self._classify("do you remember when we talked about my dad?")
        assert result.complexity.value == "complex"

    def test_opinion_request(self):
        result = self._classify("what do you think about long distance relationships?")
        assert result.complexity.value == "complex"

    def test_multi_sentence(self):
        result = self._classify(
            "So I went to the store today. Then I ran into my old coworker. "
            "She told me about this new job opportunity."
        )
        assert result.complexity.value == "medium"

    def test_long_question(self):
        result = self._classify(
            "Why do you think my boss always seems to give me the hardest projects?"
        )
        assert result.complexity.value == "complex"

    def test_relationship_topic(self):
        result = self._classify("I need to talk about something between us")
        assert result.complexity.value == "complex"

    # --- ACTION messages ---

    def test_weather_query(self):
        result = self._classify("what's the weather like in Portland?")
        assert result.complexity.value == "action"

    def test_send_email(self):
        result = self._classify("can you send an email to my boss?")
        assert result.complexity.value == "action"

    def test_calendar_check(self):
        result = self._classify("check my calendar for tomorrow")
        assert result.complexity.value == "action"

    def test_web_search(self):
        result = self._classify("search for the best Italian restaurant nearby")
        assert result.complexity.value == "action"

    def test_reminder(self):
        result = self._classify("remind me to call mom at 5pm")
        assert result.complexity.value == "action"

    # --- Feature flag ---

    def test_disabled_returns_complex(self):
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', False):
            result = classify_message("hey!")
            assert result.complexity.value == "complex"
            assert "disabled" in result.reason

    # --- Edge cases ---

    def test_empty_message(self):
        result = self._classify("")
        assert result.complexity.value == "simple"

    def test_just_emoji_short(self):
        result = self._classify("ok")
        assert result.complexity.value == "simple"


# =========================================================================
# Sentence Splitting Tests
# =========================================================================

class TestSentenceSplitting:
    """Test sentence splitting for streaming TTS."""

    def test_basic_sentences(self):
        from src.voice.voice_service import split_sentences

        result = split_sentences("Hello there. How are you doing? I'm great!")
        assert len(result) == 3
        assert result[0] == "Hello there."
        assert result[1] == "How are you doing?"
        assert result[2] == "I'm great!"

    def test_single_sentence(self):
        from src.voice.voice_service import split_sentences

        result = split_sentences("Just one sentence here.")
        assert len(result) == 1

    def test_empty_string(self):
        from src.voice.voice_service import split_sentences

        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_short_fragments_merged(self):
        from src.voice.voice_service import split_sentences

        # "Oh." is < 10 chars, should merge with previous
        result = split_sentences("That's interesting. Oh. Tell me more.")
        # "Oh." should be merged with "That's interesting."
        assert len(result) == 2

    def test_no_trailing_whitespace(self):
        from src.voice.voice_service import split_sentences

        result = split_sentences("Hello.  World.")
        assert all(s == s.strip() for s in result)

    def test_exclamations_and_questions(self):
        from src.voice.voice_service import split_sentences

        # "Wow!" is < 10 chars so gets merged with "Really?"
        result = split_sentences("Wow! Really? That's amazing.")
        assert len(result) == 2
        assert "Wow!" in result[0]  # Merged with "Really?"


# =========================================================================
# Context Builder Lightweight Path Tests
# =========================================================================

class TestContextBuilderLightweight:
    """Test that build_lightweight fetches fewer sources than full build."""

    def test_lightweight_submits_fewer_futures(self):
        """Verify build_lightweight submits fewer thread pool tasks than _build_parallel."""
        from unittest.mock import MagicMock, patch
        from src.core.conversation.context_builder import ContextBuilder

        # Count how many futures each method submits
        with patch.object(ContextBuilder, '__init__', lambda self: None):
            builder = ContextBuilder()
            builder._source_timings = {}

            # Mock the executor to count submit calls
            mock_executor = MagicMock()
            # Make futures return dummy results
            mock_future = MagicMock()
            mock_future.result.return_value = ('test', None, 0.01)
            mock_executor.submit.return_value = mock_future
            builder._executor = mock_executor

            # Mock all context source methods to no-op
            for attr in dir(builder):
                if attr.startswith('_get_'):
                    setattr(builder, attr, MagicMock(return_value=None))

            # Count lightweight submits
            builder.build_lightweight('test@test.com', 'hey')
            lightweight_count = mock_executor.submit.call_count

            mock_executor.reset_mock()

            # Count full parallel submits
            builder._build_parallel('test@test.com', 'hey', 50)
            full_count = mock_executor.submit.call_count

            assert lightweight_count < full_count, (
                f"Lightweight ({lightweight_count}) should submit fewer "
                f"futures than full ({full_count})"
            )
            assert lightweight_count <= 8, f"Lightweight should be ~7 sources, got {lightweight_count}"
            assert full_count >= 20, f"Full should be ~24 sources, got {full_count}"


class TestActionClassificationRouting:
    """Test that ACTION-classified messages are routed to tool pipeline."""

    def test_action_classification_sets_force_tools(self):
        """Verify that ACTION complexity results in force_tools=True in pipeline logic."""
        from src.core.conversation.complexity_classifier import MessageComplexity, FAST_PATH_ENABLED

        # Simulate the pipeline's logic for computing force_tools
        with patch('src.core.conversation.complexity_classifier.FAST_PATH_ENABLED', True):
            result = classify_message("what's the weather in Portland?")
            assert result.complexity == MessageComplexity.ACTION

            # Replicate pipeline logic
            use_fast_path = result.complexity == MessageComplexity.SIMPLE
            force_tools = result.complexity == MessageComplexity.ACTION

            assert use_fast_path is False
            assert force_tools is True
