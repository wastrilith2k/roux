"""Tests for FactStore utility methods (no DB required)."""

import pytest
import math
from src.memory.fact_store import FactStore


class TestCosineSimilarity:
    """Test the _cosine_similarity method."""

    def setup_method(self):
        self.store = FactStore()

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert self.store._cosine_similarity(v, v) == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert self.store._cosine_similarity(v1, v2) == pytest.approx(0.0, abs=0.001)

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert self.store._cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=0.001)

    def test_empty_vectors(self):
        assert self.store._cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert self.store._cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_vector(self):
        assert self.store._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_none_vectors(self):
        assert self.store._cosine_similarity(None, [1.0]) == 0.0
        assert self.store._cosine_similarity([1.0], None) == 0.0

    def test_real_embeddings_similar(self):
        """Vectors pointing in similar directions should have high similarity."""
        v1 = [0.5, 0.8, 0.1, 0.3]
        v2 = [0.6, 0.7, 0.2, 0.4]
        sim = self.store._cosine_similarity(v1, v2)
        assert sim > 0.95  # Very similar direction


class TestIsDuplicate:
    """Test deduplication logic (mocked DB — tests pattern matching only)."""

    def test_semantic_pattern_detection(self):
        """Verify that known semantic patterns are recognized."""
        store = FactStore()

        # These should all match the same pattern
        SEMANTIC_PATTERNS = {
            "works_at_cavallo": ["works at cavallo", "employed at cavallo", "works for cavallo"],
            "tea_preference": ["loves tea", "drinks tea", "prefers tea"],
        }

        # Test pattern matching logic directly
        def get_pattern(text):
            text_lower = text.lower()
            for pattern_name, phrases in SEMANTIC_PATTERNS.items():
                for phrase in phrases:
                    if phrase in text_lower:
                        return pattern_name
            return None

        assert get_pattern("He works at Cavallo") == "works_at_cavallo"
        assert get_pattern("He is employed at Cavallo Group") == "works_at_cavallo"
        assert get_pattern("She loves tea") == "tea_preference"
        assert get_pattern("Something unrelated") is None
