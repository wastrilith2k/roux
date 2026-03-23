"""Tests for MemoryQueryClassifier - regression tests for false positives."""

import pytest
from unittest.mock import patch
from src.core.memory_query_classifier import MemoryQueryClassifier


@pytest.fixture
def classifier():
    return MemoryQueryClassifier()


class TestFallbackFalsePositives:
    """Regression tests for issue #2: fallback_classify falsely triggers on non-memory queries."""

    def test_whats_the_weather_not_memory(self, classifier):
        """'What's the weather like?' should not be a memory query."""
        result = classifier._fallback_classify("What's the weather like?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_whats_up_not_memory(self, classifier):
        """'What's up?' should not be a memory query."""
        result = classifier._fallback_classify("What's up?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_whats_for_dinner_not_memory(self, classifier):
        """'What's for dinner?' should not be a memory query."""
        result = classifier._fallback_classify("What's for dinner?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_whats_going_on_not_memory(self, classifier):
        """'What's going on?' should not be a memory query."""
        result = classifier._fallback_classify("What's going on?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'


class TestFallbackTruePositives:
    """Ensure legitimate factual memory queries still match in fallback."""

    def test_who_is_factual(self, classifier):
        result = classifier._fallback_classify("Who is Jesse?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'

    def test_where_does_factual(self, classifier):
        result = classifier._fallback_classify("Where does my sister live?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'

    def test_whats_his_factual(self, classifier):
        result = classifier._fallback_classify("What's his last name?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'

    def test_whats_her_factual(self, classifier):
        result = classifier._fallback_classify("What's her birthday?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'

    def test_how_old_is_factual(self, classifier):
        result = classifier._fallback_classify("How old is Kyler?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'


class TestPreFilterFalsePositives:
    """Ensure the pre-filter also rejects generic 'what's' queries."""

    def test_prefilter_rejects_whats_weather(self, classifier):
        """'What's the weather like?' should be rejected by pre-filter too."""
        with patch.object(classifier, '_has_memory_keywords', return_value=False):
            result = classifier.classify("What's the weather like?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_prefilter_accepts_whats_his_name(self, classifier):
        """'What's his name?' should pass the pre-filter to LLM classification."""
        with patch.object(classifier, '_has_memory_keywords', return_value=False), \
             patch.object(classifier, '_classify_with_llm') as mock_llm:
            from src.core.memory_query_classifier import MemoryQueryResult
            mock_llm.return_value = MemoryQueryResult(
                is_memory_query=True, query_type='factual',
                search_terms=['name'], confidence=0.9
            )
            result = classifier.classify("What's his name?")
        mock_llm.assert_called_once()
