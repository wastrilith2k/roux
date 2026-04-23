"""Tests for Bayesian emotional memory weighting."""

import pytest
import math
from src.memory.emotional_memory import (
    EmotionalProfile,
    bayesian_update,
    classify_sentiment,
    get_emotional_profile_for_fact,
    apply_emotional_persistence,
)


class TestEmotionalProfile:
    """Test the EmotionalProfile dataclass."""

    def test_default_is_neutral(self):
        profile = EmotionalProfile()
        assert profile.valence == pytest.approx(0.0, abs=0.02)
        assert profile.intensity < 0.5

    def test_normalize(self):
        profile = EmotionalProfile(positive=0.6, negative=0.3, neutral=0.3)
        normalized = profile.normalize()
        total = normalized.positive + normalized.negative + normalized.neutral
        assert total == pytest.approx(1.0, abs=0.01)

    def test_valence_positive(self):
        profile = EmotionalProfile(positive=0.8, negative=0.1, neutral=0.1)
        assert profile.valence > 0.5

    def test_valence_negative(self):
        profile = EmotionalProfile(positive=0.1, negative=0.8, neutral=0.1)
        assert profile.valence < -0.5

    def test_intensity_strong(self):
        profile = EmotionalProfile(positive=0.8, negative=0.1, neutral=0.1)
        assert profile.intensity >= 0.8

    def test_intensity_neutral(self):
        profile = EmotionalProfile(positive=0.2, negative=0.2, neutral=0.6)
        assert profile.intensity < 0.3

    def test_entropy_clear_signal(self):
        """Clear emotional signal should have low entropy."""
        profile = EmotionalProfile(positive=0.9, negative=0.05, neutral=0.05)
        assert profile.entropy < 1.0

    def test_entropy_ambiguous_signal(self):
        """Ambiguous signal should have high entropy."""
        profile = EmotionalProfile(positive=0.34, negative=0.33, neutral=0.33)
        assert profile.entropy > 1.4

    def test_persistence_modifier_emotional(self):
        """Strong emotions should slow decay (modifier > 1.0)."""
        emotional = EmotionalProfile(positive=0.8, negative=0.1, neutral=0.1)
        assert emotional.persistence_modifier > 1.0

    def test_persistence_modifier_neutral(self):
        """Neutral facts should not get decay bonus."""
        neutral = EmotionalProfile(positive=0.2, negative=0.2, neutral=0.6)
        assert neutral.persistence_modifier < 1.2

    def test_to_dict_and_back(self):
        original = EmotionalProfile(positive=0.7, negative=0.2, neutral=0.1)
        d = original.to_dict()
        restored = EmotionalProfile.from_dict(d)
        assert restored.positive == pytest.approx(original.positive, abs=0.001)

    def test_to_json_and_back(self):
        original = EmotionalProfile(positive=0.6, negative=0.3, neutral=0.1)
        j = original.to_json()
        restored = EmotionalProfile.from_json(j)
        assert restored.positive == pytest.approx(original.positive, abs=0.001)


class TestBayesianUpdate:
    """Test the Bayesian emotional update formula."""

    def test_strong_evidence_shifts_profile(self):
        current = EmotionalProfile(positive=0.33, negative=0.33, neutral=0.34)
        new_sentiment = {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}

        updated = bayesian_update(current, new_sentiment, evidence_strength=0.8)
        assert updated.positive > current.positive

    def test_weak_evidence_barely_shifts(self):
        current = EmotionalProfile(positive=0.5, negative=0.3, neutral=0.2)
        new_sentiment = {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1}

        updated = bayesian_update(current, new_sentiment, evidence_strength=0.1)
        # Should shift slightly but not dramatically
        assert updated.positive > 0.3  # Still mostly positive

    def test_strong_prior_resists_change(self):
        """A strongly emotional profile should resist shifting."""
        strong = EmotionalProfile(positive=0.9, negative=0.05, neutral=0.05)
        new_sentiment = {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1}

        updated = bayesian_update(strong, new_sentiment, evidence_strength=0.3)
        # Should still be mostly positive (strong prior)
        assert updated.positive > updated.negative

    def test_multiple_updates_converge(self):
        """Repeated negative evidence should eventually shift positive profile."""
        profile = EmotionalProfile(positive=0.7, negative=0.2, neutral=0.1)
        negative_evidence = {'positive': 0.1, 'negative': 0.8, 'neutral': 0.1}

        for _ in range(10):
            profile = bayesian_update(profile, negative_evidence, evidence_strength=0.5)

        assert profile.negative > profile.positive

    def test_update_preserves_normalization(self):
        current = EmotionalProfile(positive=0.5, negative=0.3, neutral=0.2)
        new_sentiment = {'positive': 0.9, 'negative': 0.05, 'neutral': 0.05}

        updated = bayesian_update(current, new_sentiment, evidence_strength=0.7)
        total = updated.positive + updated.negative + updated.neutral
        assert total == pytest.approx(1.0, abs=0.01)


class TestClassifySentiment:
    """Test rule-based sentiment classification."""

    def test_positive_text(self):
        result = classify_sentiment("I'm so happy and excited about this amazing news!")
        assert result['positive'] > result['negative']

    def test_negative_text(self):
        result = classify_sentiment("I'm worried and stressed about this terrible situation")
        assert result['negative'] > result['positive']

    def test_neutral_text(self):
        result = classify_sentiment("The weather is cloudy today")
        assert result['neutral'] > 0.4

    def test_mixed_text(self):
        result = classify_sentiment("I'm happy but also worried about the future")
        # Both positive and negative should be elevated
        assert result['positive'] > 0.1
        assert result['negative'] > 0.1


class TestEmotionalPersistence:
    """Test how emotional weighting affects confidence."""

    def test_emotional_memory_resists_decay(self):
        """Highly emotional facts should have higher effective confidence."""
        emotional_profile = EmotionalProfile(positive=0.8, negative=0.1, neutral=0.1)
        neutral_profile = EmotionalProfile(positive=0.33, negative=0.33, neutral=0.34)

        base_confidence = 0.5

        emotional_result = apply_emotional_persistence(base_confidence, emotional_profile)
        neutral_result = apply_emotional_persistence(base_confidence, neutral_profile)

        assert emotional_result > neutral_result

    def test_negative_memory_also_persists(self):
        """Negative emotions should also slow decay (we remember bad things)."""
        negative_profile = EmotionalProfile(positive=0.05, negative=0.9, neutral=0.05)
        neutral_profile = EmotionalProfile(positive=0.33, negative=0.33, neutral=0.34)

        base_confidence = 0.4

        negative_result = apply_emotional_persistence(base_confidence, negative_profile)
        neutral_result = apply_emotional_persistence(base_confidence, neutral_profile)

        assert negative_result > neutral_result

    def test_persistence_clamped(self):
        """Result should be clamped to 0.1-0.99."""
        very_emotional = EmotionalProfile(positive=1.0, negative=0.0, neutral=0.0)
        result = apply_emotional_persistence(0.95, very_emotional)
        assert result <= 0.99

        result = apply_emotional_persistence(0.05, very_emotional)
        assert result >= 0.1


class TestFactEmotionalProfile:
    """Test getting emotional profiles from fact dicts."""

    def test_from_stored_profile(self):
        fact = {
            'emotional_profile': {'positive': 0.7, 'negative': 0.2, 'neutral': 0.1},
        }
        profile = get_emotional_profile_for_fact(fact)
        assert profile.positive == pytest.approx(0.7, abs=0.01)

    def test_from_json_string(self):
        fact = {
            'emotional_profile': '{"positive": 0.8, "negative": 0.1, "neutral": 0.1}',
        }
        profile = get_emotional_profile_for_fact(fact)
        assert profile.positive == pytest.approx(0.8, abs=0.01)

    def test_from_fact_text(self):
        """Should classify from fact text when no stored profile."""
        fact = {
            'subject': 'user',
            'predicate': 'experienced',
            'object': 'a wonderful happy celebration',
        }
        profile = get_emotional_profile_for_fact(fact)
        assert profile.positive > profile.negative

    def test_missing_everything_returns_neutral(self):
        fact = {}
        profile = get_emotional_profile_for_fact(fact)
        # Should be roughly neutral
        assert abs(profile.valence) < 0.3
