"""Tests for relationship store type system and validation."""

import pytest
from src.memory.relationship_store import (
    RelationshipType,
    INVERSE_RELATIONSHIPS,
)


class TestRelationshipType:
    """Test the relationship type enum."""

    def test_all_types_are_strings(self):
        for rt in RelationshipType:
            assert isinstance(rt.value, str)

    def test_family_types_exist(self):
        assert RelationshipType.PARENT_OF.value == "parent_of"
        assert RelationshipType.CHILD_OF.value == "child_of"
        assert RelationshipType.MARRIED_TO.value == "married_to"

    def test_professional_types_exist(self):
        assert RelationshipType.WORKS_AT.value == "works_at"
        assert RelationshipType.COWORKER_OF.value == "coworker_of"

    def test_from_string(self):
        assert RelationshipType("parent_of") == RelationshipType.PARENT_OF

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            RelationshipType("best_buddy_of")

    def test_no_generic_relates_to(self):
        """The store intentionally has no generic RELATES_TO — only typed relationships."""
        values = [rt.value for rt in RelationshipType]
        assert "relates_to" not in values


class TestInverseRelationships:
    """Test the inverse relationship mapping."""

    def test_parent_child_inverse(self):
        assert INVERSE_RELATIONSHIPS[RelationshipType.PARENT_OF] == RelationshipType.CHILD_OF
        assert INVERSE_RELATIONSHIPS[RelationshipType.CHILD_OF] == RelationshipType.PARENT_OF

    def test_symmetric_relationships(self):
        """Symmetric relationships are their own inverse."""
        assert INVERSE_RELATIONSHIPS[RelationshipType.MARRIED_TO] == RelationshipType.MARRIED_TO
        assert INVERSE_RELATIONSHIPS[RelationshipType.SIBLING_OF] == RelationshipType.SIBLING_OF

    def test_all_types_have_inverse(self):
        """Every relationship type should have an inverse defined."""
        for rt in RelationshipType:
            assert rt in INVERSE_RELATIONSHIPS, f"{rt} missing from INVERSE_RELATIONSHIPS"
