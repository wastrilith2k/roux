"""Tests for MemoryQueryClassifier - regression tests for false positives."""

import pytest
from unittest.mock import patch
from src.core.memory_query_classifier import MemoryQueryClassifier, MemoryQueryResult


@pytest.fixture
def classifier():
    return MemoryQueryClassifier()


class TestFallbackFalsePositives:
    """Regression tests for issue #2: fallback_classify falsely triggers on non-memory queries."""

    @pytest.mark.parametrize("message", [
        "What's the weather like?",
        "What's up?",
        "What's for dinner?",
        "What's going on?",
        "What is the capital of France?",
        "What is two plus two?",
        "What is going on today?",
        "Who's up for pizza tonight?",
        "Who's going to the party?",
        "Who's ready?",
    ])
    def test_generic_queries_not_memory(self, classifier, message):
        result = classifier._fallback_classify(message)
        assert result.is_memory_query is False, f"'{message}' should not be a memory query"
        assert result.query_type == 'none'


class TestFallbackTruePositives:
    """Ensure legitimate factual memory queries still match in fallback."""

    @pytest.mark.parametrize("message", [
        "Who is Jesse?",
        "Where does my sister live?",
        "What's his last name?",
        "What's her birthday?",
        "What's my manager's name?",
        "What's their address?",
        "What is her job?",
        "What is my dog's name?",
        "What is their phone number?",
        "Who's her mother?",
        "How old is Kyler?",
        "Does she still work there?",
        "Does he have kids?",
        "Is she coming to visit?",
        "Is he still in Austin?",
        "Her name is what again?",
        "His name was something unusual",
    ])
    def test_factual_queries_are_memory(self, classifier, message):
        result = classifier._fallback_classify(message)
        assert result.is_memory_query is True, f"'{message}' should be a memory query"
        assert result.query_type == 'factual'

    @pytest.mark.parametrize("message", [
        "What is his favorite color?",
        "Who's his best friend?",
    ])
    def test_emotional_factual_queries_are_memory(self, classifier, message):
        """Queries with emotional keywords ('favorite', 'best') hit emotional_vague first — still memory queries."""
        result = classifier._fallback_classify(message)
        assert result.is_memory_query is True, f"'{message}' should be a memory query"
        assert result.query_type == 'emotional_vague'


class TestClassifyWithLLMFailure:
    """Test the real failure path: classify() -> LLM exception -> fallback."""

    def test_weather_not_memory_on_llm_failure(self, classifier):
        """Issue #2 regression: the actual production path."""
        with patch.object(classifier, '_classify_with_llm', side_effect=Exception("LLM unavailable")):
            result = classifier.classify("What's the weather like?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_whos_ready_not_memory_on_llm_failure(self, classifier):
        with patch.object(classifier, '_classify_with_llm', side_effect=Exception("LLM unavailable")):
            result = classifier.classify("Who's ready?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_what_is_generic_not_memory_on_llm_failure(self, classifier):
        with patch.object(classifier, '_classify_with_llm', side_effect=Exception("LLM unavailable")):
            result = classifier.classify("What is the time?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_who_is_factual_on_llm_failure(self, classifier):
        with patch.object(classifier, '_classify_with_llm', side_effect=Exception("LLM unavailable")):
            result = classifier.classify("Who is Jesse?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'


class TestEmptyLLMResponse:
    """Test fallback when LLM returns empty/None instead of raising."""

    def test_empty_response_falls_back_with_real_message(self, classifier):
        """Empty LLM response should fallback using the original message, not empty string."""
        with patch('src.llm.fireworks_models.call_fireworks', return_value=""), \
             patch.object(classifier, '_get_client', return_value=None), \
             patch.object(classifier, '_fallback_classify', wraps=classifier._fallback_classify) as mock_fb:
            result = classifier._classify_with_llm("Who is Jesse?")
        mock_fb.assert_called_once_with("Who is Jesse?")
        assert result.is_memory_query is True

    def test_none_response_falls_back_with_real_message(self, classifier):
        with patch('src.llm.fireworks_models.call_fireworks', return_value=None), \
             patch.object(classifier, '_get_client', return_value=None), \
             patch.object(classifier, '_fallback_classify', wraps=classifier._fallback_classify) as mock_fb:
            result = classifier._classify_with_llm("Who is Jesse?")
        mock_fb.assert_called_once_with("Who is Jesse?")
        assert result.is_memory_query is True


class TestPreFilterRouting:
    """Ensure pre-filter correctly routes to LLM or rejects."""

    def test_prefilter_rejects_whats_weather(self, classifier):
        with patch.object(classifier, '_has_memory_keywords', return_value=False), \
             patch.object(classifier, '_classify_with_llm') as mock_llm:
            result = classifier.classify("What's the weather like?")
        mock_llm.assert_not_called()
        assert result.is_memory_query is False

    def test_prefilter_accepts_whats_his_name(self, classifier):
        with patch.object(classifier, '_has_memory_keywords', return_value=False), \
             patch.object(classifier, '_classify_with_llm') as mock_llm:
            mock_llm.return_value = MemoryQueryResult(
                is_memory_query=True, query_type='factual',
                search_terms=['name'], confidence=0.9
            )
            result = classifier.classify("What's his name?")
        mock_llm.assert_called_once_with("What's his name?")

    def test_prefilter_accepts_does_she(self, classifier):
        with patch.object(classifier, '_has_memory_keywords', return_value=False), \
             patch.object(classifier, '_classify_with_llm') as mock_llm:
            mock_llm.return_value = MemoryQueryResult(
                is_memory_query=True, query_type='factual',
                search_terms=['work'], confidence=0.9
            )
            result = classifier.classify("Does she still work there?")
        mock_llm.assert_called_once_with("Does she still work there?")
