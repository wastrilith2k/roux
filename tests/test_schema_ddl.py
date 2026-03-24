"""Tests for schema_ddl.py — DDL generation for public and user schemas."""

import re
import pytest

from src.database import tables as T
from src.database.schema_ddl import get_public_schema_ddl, get_user_schema_ddl


class TestPublicSchemaDDL:
    """Tests for public schema DDL generation."""

    def test_returns_nonempty_string(self):
        ddl = get_public_schema_ddl()
        assert isinstance(ddl, str)
        assert len(ddl) > 100

    def test_contains_all_public_tables(self):
        ddl = get_public_schema_ddl()
        for table_name in T.PUBLIC_TABLES:
            assert f"public.{table_name}" in ddl, (
                f"Public table '{table_name}' not found in public DDL"
            )

    def test_does_not_contain_user_schema_tables(self):
        ddl = get_public_schema_ddl()
        # User-only tables should NOT appear in public DDL
        user_only = T.USER_SCHEMA_TABLES - T.PUBLIC_TABLES
        for table_name in user_only:
            # Should not have CREATE TABLE for user-only tables
            pattern = rf"CREATE TABLE IF NOT EXISTS public\.{re.escape(table_name)}\s*\("
            assert not re.search(pattern, ddl), (
                f"User-only table '{table_name}' should not be in public DDL"
            )

    def test_uses_create_table_if_not_exists(self):
        ddl = get_public_schema_ddl()
        # Every CREATE TABLE should be IF NOT EXISTS
        creates = re.findall(r"CREATE TABLE\b", ddl)
        creates_ine = re.findall(r"CREATE TABLE IF NOT EXISTS", ddl)
        assert len(creates) == len(creates_ine), (
            "All CREATE TABLE statements should use IF NOT EXISTS"
        )

    def test_user_companions_table_present(self):
        ddl = get_public_schema_ddl()
        assert "public.user_companions" in ddl
        assert "user_email TEXT NOT NULL" in ddl


class TestUserSchemaDDL:
    """Tests for user-scoped schema DDL generation."""

    SCHEMA = "user_testuser"

    def test_returns_nonempty_string(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        assert isinstance(ddl, str)
        assert len(ddl) > 500

    def test_contains_all_user_schema_tables(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        for table_name in T.USER_SCHEMA_TABLES:
            qualified = f"{self.SCHEMA}.{table_name}"
            assert qualified in ddl, (
                f"User schema table '{table_name}' not found as "
                f"'{qualified}' in user DDL"
            )

    def test_does_not_contain_public_only_tables(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        # Public-only tables should NOT appear in user DDL
        public_only = T.PUBLIC_TABLES - T.USER_SCHEMA_TABLES
        for table_name in public_only:
            pattern = rf"CREATE TABLE IF NOT EXISTS {re.escape(self.SCHEMA)}\.{re.escape(table_name)}\s*\("
            assert not re.search(pattern, ddl), (
                f"Public-only table '{table_name}' should not be in user DDL"
            )

    def test_schema_name_substituted(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        # Schema name should appear throughout
        assert ddl.count(self.SCHEMA) >= len(T.USER_SCHEMA_TABLES), (
            f"Schema name '{self.SCHEMA}' should appear at least once per table"
        )

    def test_different_schema_name(self):
        ddl1 = get_user_schema_ddl("user_alice")
        ddl2 = get_user_schema_ddl("user_bob")
        assert "user_alice" in ddl1
        assert "user_bob" in ddl2
        assert "user_alice" not in ddl2
        assert "user_bob" not in ddl1

    def test_uses_create_table_if_not_exists(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        creates = re.findall(r"CREATE TABLE\b", ddl)
        creates_ine = re.findall(r"CREATE TABLE IF NOT EXISTS", ddl)
        assert len(creates) == len(creates_ine)

    def test_uses_create_index_if_not_exists(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        indexes = re.findall(r"CREATE INDEX\b", ddl)
        indexes_ine = re.findall(r"CREATE INDEX IF NOT EXISTS", ddl)
        assert len(indexes) == len(indexes_ine)

    def test_no_cross_schema_foreign_keys(self):
        """User schema tables should not reference public.user_profiles."""
        ddl = get_user_schema_ddl(self.SCHEMA)
        assert "REFERENCES user_profiles" not in ddl, (
            "User schema DDL should not have cross-schema FK to user_profiles"
        )
        assert "REFERENCES public.user_profiles" not in ddl

    def test_facts_table_has_embedding_column(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        # The facts table should have a vector embedding column
        assert "embedding" in ddl.lower()

    def test_key_tables_have_indexes(self):
        ddl = get_user_schema_ddl(self.SCHEMA)
        # Messages table should have indexes
        assert "idx_messages_email" in ddl or "idx_messages_timestamp" in ddl


class TestDDLSyntax:
    """Basic syntax validation for generated DDL."""

    def test_balanced_parentheses_public(self):
        ddl = get_public_schema_ddl()
        assert ddl.count("(") == ddl.count(")"), "Unbalanced parentheses in public DDL"

    def test_balanced_parentheses_user(self):
        ddl = get_user_schema_ddl("user_test")
        assert ddl.count("(") == ddl.count(")"), "Unbalanced parentheses in user DDL"

    def test_no_sqlite_syntax_in_user_ddl(self):
        ddl = get_user_schema_ddl("user_test")
        assert "AUTOINCREMENT" not in ddl, "SQLite AUTOINCREMENT found in PostgreSQL DDL"

    def test_no_sqlite_syntax_in_public_ddl(self):
        ddl = get_public_schema_ddl()
        assert "AUTOINCREMENT" not in ddl

    def test_statements_end_with_semicolons(self):
        ddl = get_user_schema_ddl("user_test")
        # Each CREATE TABLE / CREATE INDEX should end with a semicolon
        # (possibly with whitespace between closing paren and semicolon)
        create_count = len(re.findall(r"CREATE (TABLE|INDEX)", ddl))
        semicolons = ddl.count(";")
        assert semicolons >= create_count, (
            f"Expected at least {create_count} semicolons, got {semicolons}"
        )
