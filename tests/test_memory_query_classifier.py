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


class TestPreFilterRejection:
    """Verify pre-filter rejects generic queries before they reach the LLM."""

    @pytest.mark.parametrize("message", [
        "What's the weather like?",
        "Who's ready?",
    ])
    def test_generic_queries_rejected_by_prefilter(self, classifier, message):
        """These messages have no memory keywords and no factual patterns, so the
        pre-filter returns is_memory_query=False without ever calling the LLM."""
        with patch.object(classifier, '_classify_with_llm') as mock_llm:
            result = classifier.classify(message)
        mock_llm.assert_not_called()
        assert result.is_memory_query is False
        assert result.query_type == 'none'


class TestClassifyWithLLMFailure:
    """Test the real failure path: classify() -> LLM exception -> fallback."""

    def test_who_is_factual_on_llm_failure(self, classifier):
        """"Who is Jesse?" matches factual pattern, reaches LLM, falls back on error."""
        with patch.object(classifier, '_classify_with_llm', side_effect=Exception("LLM unavailable")):
            result = classifier.classify("Who is Jesse?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'

    def test_memory_keyword_non_memory_on_llm_failure(self, classifier):
        """"before" is a memory keyword so this passes pre-filter to LLM.
        On LLM failure, fallback finds no event/emotional/factual pattern → none."""
        with patch.object(classifier, '_classify_with_llm', side_effect=Exception("LLM unavailable")):
            result = classifier.classify("What should we do before dinner?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_what_is_the_time_non_memory_on_llm_failure(self, classifier):
        """"the time" matches MEMORY_KEYWORDS so this reaches the LLM.
        On LLM failure, fallback finds no matching pattern → none."""
        with patch.object(classifier, '_classify_with_llm', side_effect=Exception("LLM unavailable")):
            result = classifier.classify("What is the time?")
        assert result.is_memory_query is False
        assert result.query_type == 'none'


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


class TestLLMPostCorrection:
    """Regression tests for issue #1: LLM returns query_type != 'none' but is_memory_query=False."""

    @pytest.mark.parametrize("query_type", ['factual', 'specific_event', 'emotional_vague'])
    def test_llm_disagreement_corrected(self, classifier, query_type):
        """When LLM returns a real query_type but is_memory_query=False, the post-LLM
        override must force is_memory_query=True."""
        llm_response = f'{{"is_memory_query": false, "query_type": "{query_type}", "search_terms": ["test"], "confidence": 0.9, "reasoning": "test"}}'
        with patch('src.llm.fireworks_models.call_fireworks', return_value=llm_response), \
             patch.object(classifier, '_get_client', return_value=None):
            result = classifier._classify_with_llm("test query")
        assert result.is_memory_query is True, (
            f"query_type='{query_type}' must force is_memory_query=True"
        )
        assert result.query_type == query_type

    def test_llm_none_stays_false(self, classifier):
        """query_type='none' should keep is_memory_query=False."""
        llm_response = '{"is_memory_query": false, "query_type": "none", "search_terms": [], "confidence": 0.9, "reasoning": "not a memory query"}'
        with patch('src.llm.fireworks_models.call_fireworks', return_value=llm_response), \
             patch.object(classifier, '_get_client', return_value=None):
            result = classifier._classify_with_llm("Hello there")
        assert result.is_memory_query is False
        assert result.query_type == 'none'

    def test_who_is_factual_corrected(self, classifier):
        """Reproduce issue #1: 'Who is Jesse?' → LLM says factual but is_memory_query=False."""
        llm_response = '{"is_memory_query": false, "query_type": "factual", "search_terms": ["Jesse"], "confidence": 0.95, "reasoning": "Asking for factual information about a person"}'
        with patch('src.llm.fireworks_models.call_fireworks', return_value=llm_response), \
             patch.object(classifier, '_get_client', return_value=None):
            result = classifier._classify_with_llm("Who is Jesse?")
        assert result.is_memory_query is True
        assert result.query_type == 'factual'
        assert 'Jesse' in result.search_terms

    def test_what_did_i_tell_you_corrected(self, classifier):
        """'What did I tell you about my job?' with LLM disagreement."""
        llm_response = '{"is_memory_query": false, "query_type": "specific_event", "search_terms": ["job"], "confidence": 0.9, "reasoning": "Asking about a past conversation"}'
        with patch('src.llm.fireworks_models.call_fireworks', return_value=llm_response), \
             patch.object(classifier, '_get_client', return_value=None):
            result = classifier._classify_with_llm("What did I tell you about my job?")
        assert result.is_memory_query is True
        assert result.query_type == 'specific_event'

    def test_remember_that_thing_corrected(self, classifier):
        """'Remember that thing I mentioned?' with LLM disagreement."""
        llm_response = '{"is_memory_query": false, "query_type": "emotional_vague", "search_terms": ["mentioned"], "confidence": 0.8, "reasoning": "Vague reference to past"}'
        with patch('src.llm.fireworks_models.call_fireworks', return_value=llm_response), \
             patch.object(classifier, '_get_client', return_value=None):
            result = classifier._classify_with_llm("Remember that thing I mentioned?")
        assert result.is_memory_query is True
        assert result.query_type == 'emotional_vague'


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
