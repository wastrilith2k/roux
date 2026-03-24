"""Verify tables.py exports all expected constants and they're plain strings."""

import re
from src.database import tables as T


class TestTableConstants:
    """Validate table name constants are well-formed."""

    def _get_string_constants(self):
        """Return all ALL_CAPS string constants from tables module."""
        return {k: v for k, v in vars(T).items()
                if k.isupper() and not k.startswith('_') and isinstance(v, str)}

    def test_all_table_constants_are_strings(self):
        string_constants = self._get_string_constants()
        assert len(string_constants) >= 45, f"Expected 45+ table constants, got {len(string_constants)}"
        for name, value in string_constants.items():
            assert isinstance(value, str), f"{name} should be str, got {type(value)}"
            assert len(value) > 0, f"{name} should not be empty"

    def test_all_constants_are_valid_pg_identifiers(self):
        """Table names must be valid unquoted PostgreSQL identifiers."""
        for name, value in self._get_string_constants().items():
            assert re.match(r'^[a-z_][a-z0-9_]*$', value), (
                f"{name} = '{value}' is not a valid PostgreSQL identifier"
            )

    def test_no_duplicate_table_names(self):
        """No two constants should map to the same table name."""
        constants = self._get_string_constants()
        values = list(constants.values())
        dupes = [v for v in values if values.count(v) > 1]
        assert not dupes, f"Duplicate table names: {set(dupes)}"


class TestLookupSets:
    """Validate PUBLIC_TABLES and USER_SCHEMA_TABLES sets."""

    def test_public_schema_tables_listed(self):
        assert 'users' in T.PUBLIC_TABLES
        assert 'sessions' in T.PUBLIC_TABLES
        assert 'user_profiles' in T.PUBLIC_TABLES
        assert 'user_companions' in T.PUBLIC_TABLES
        assert 'service_links' in T.PUBLIC_TABLES

    def test_user_schema_tables_listed(self):
        assert 'messages' in T.USER_SCHEMA_TABLES
        assert 'facts' in T.USER_SCHEMA_TABLES
        assert 'companion_goals' in T.USER_SCHEMA_TABLES

    def test_no_overlap_between_sets(self):
        """A table should not be in both PUBLIC_TABLES and USER_SCHEMA_TABLES."""
        overlap = T.PUBLIC_TABLES & T.USER_SCHEMA_TABLES
        assert not overlap, f"Tables in both sets: {overlap}"

    def test_all_string_constants_in_one_set(self):
        """Every string constant should be in either PUBLIC_TABLES or USER_SCHEMA_TABLES."""
        all_constants = {v for k, v in vars(T).items()
                         if k.isupper() and not k.startswith('_') and isinstance(v, str)}
        all_sets = T.PUBLIC_TABLES | T.USER_SCHEMA_TABLES
        missing = all_constants - all_sets
        assert not missing, f"Table constants not in any set: {missing}"

    def test_set_counts(self):
        assert len(T.PUBLIC_TABLES) == 5
        assert len(T.USER_SCHEMA_TABLES) >= 42
