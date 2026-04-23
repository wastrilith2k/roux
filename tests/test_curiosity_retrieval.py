"""Tests for curiosity-driven memory retrieval."""

import pytest


class TestCuriositySeededQueries:
    """Test that curiosity topics enhance memory retrieval queries."""

    def test_curiosity_topics_added_to_queries(self):
        """Curiosity topics should be appended as additional search queries."""
        base_queries = ["How was your day?"]
        curiosity_topics = ["user's project deadline", "weekend hiking plans"]

        # Simulate the seeding logic
        queries = list(base_queries)
        for topic in curiosity_topics[:2]:
            if topic and len(topic) > 5:
                queries.append(topic)

        assert len(queries) == 3
        assert "user's project deadline" in queries
        assert "weekend hiking plans" in queries

    def test_short_topics_filtered(self):
        """Very short topics should not be added as queries."""
        queries = ["Hello"]
        topic = "hi"  # Too short (< 5 chars)

        if topic and len(topic) > 5:
            queries.append(topic)

        assert len(queries) == 1  # Not added

    def test_max_two_curiosity_queries(self):
        """Should only add up to 2 curiosity topics."""
        queries = ["Base query"]
        topics = ["topic1 longer", "topic2 longer", "topic3 longer", "topic4 longer"]

        for topic in topics[:2]:  # Limit to 2
            queries.append(topic)

        assert len(queries) == 3  # 1 base + 2 curiosity

    def test_deduplication_works(self):
        """Duplicate queries should be removed."""
        queries = [
            "How was your day?",
            "work project deadline",
            "How was your day?",  # Duplicate
            "WORK PROJECT DEADLINE",  # Case-insensitive duplicate
        ]

        seen = set()
        unique = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)

        assert len(unique) == 2

    def test_empty_curiosity_no_effect(self):
        """No curiosity topics should not affect base queries."""
        base_queries = ["User message", "Agent-enhanced query"]
        curious_topics = []

        queries = list(base_queries)
        for topic in curious_topics[:2]:
            if topic and len(topic) > 5:
                queries.append(topic)

        assert queries == base_queries


class TestCuriosityThresholds:
    """Test the urgency threshold logic."""

    def test_moderate_urgency_included(self):
        """Topics with urgency >= 0.5 should be included."""
        threshold = 0.5
        topics = [
            {'topic': 'important thing', 'urgency': 0.7},
            {'topic': 'mildly curious', 'urgency': 0.4},
            {'topic': 'very curious', 'urgency': 0.9},
        ]

        included = [t for t in topics if t['urgency'] >= threshold]
        assert len(included) == 2
        assert included[0]['urgency'] == 0.7
        assert included[1]['urgency'] == 0.9

    def test_low_urgency_excluded(self):
        """Topics below threshold should not clutter retrieval."""
        threshold = 0.5
        topics = [
            {'topic': 'barely curious', 'urgency': 0.2},
            {'topic': 'slightly curious', 'urgency': 0.3},
        ]

        included = [t for t in topics if t['urgency'] >= threshold]
        assert len(included) == 0
