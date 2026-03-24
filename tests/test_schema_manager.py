"""Test per-user schema management."""
import pytest
from unittest.mock import MagicMock, call

from src.database.schema_manager import schema_name_for_user, _quote_ident


class TestSchemaNameForUser:
    """Test email-to-schema-name derivation."""

    def test_basic_email(self):
        assert schema_name_for_user("james@example.com") == "user_james"

    def test_dotted_local_part(self):
        assert schema_name_for_user("alice.bob@example.com") == "user_alice_bob"

    def test_special_chars_sanitized(self):
        assert schema_name_for_user("test+user@example.com") == "user_test_user"
        assert schema_name_for_user("a'b@evil.com") == "user_a_b"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            schema_name_for_user("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            schema_name_for_user("   ")

    def test_no_at_sign_uses_whole_string(self):
        """An email without @ should use the whole string as local part."""
        assert schema_name_for_user("localonly") == "user_localonly"

    def test_uppercase_lowered(self):
        assert schema_name_for_user("Alice@Example.COM") == "user_alice"

    def test_consecutive_special_chars_collapsed(self):
        """Multiple special chars should collapse to a single underscore."""
        assert schema_name_for_user("a..b++c@x.com") == "user_a_b_c"

    def test_leading_trailing_specials_stripped(self):
        """Leading/trailing underscores after sanitization are stripped."""
        assert schema_name_for_user(".user.@x.com") == "user_user"

    def test_all_special_chars_raises(self):
        """An email local part that's ALL special chars should raise."""
        with pytest.raises(ValueError, match="empty schema name"):
            schema_name_for_user("+++@evil.com")

    def test_numeric_local_part(self):
        assert schema_name_for_user("12345@example.com") == "user_12345"

    def test_different_emails_different_schemas(self):
        """Two different emails must not collide."""
        s1 = schema_name_for_user("alice@example.com")
        s2 = schema_name_for_user("bob@example.com")
        assert s1 != s2


class TestQuoteIdent:
    """Test the SQL identifier quoting helper."""

    def test_simple_identifier(self):
        assert _quote_ident("user_james") == '"user_james"'

    def test_embedded_double_quotes_escaped(self):
        assert _quote_ident('a"b') == '"a""b"'

    def test_empty_string(self):
        assert _quote_ident("") == '""'


class TestEnsureUserSchema:
    """Test ensure_user_schema with a mocked connection."""

    def test_creates_schema_and_runs_ddl(self):
        from unittest.mock import patch
        from src.database.schema_manager import ensure_user_schema

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch('src.database.schema_ddl.get_user_schema_ddl', return_value='CREATE TABLE fake;'):
            schema = ensure_user_schema(mock_conn, "alice@example.com")

        assert schema == "user_alice"
        # Should have executed CREATE SCHEMA and the DDL
        assert mock_cursor.execute.call_count == 2
        create_schema_sql = mock_cursor.execute.call_args_list[0][0][0]
        assert 'CREATE SCHEMA IF NOT EXISTS' in create_schema_sql
        assert '"user_alice"' in create_schema_sql
        # DDL was passed through
        assert mock_cursor.execute.call_args_list[1][0][0] == 'CREATE TABLE fake;'
        mock_conn.commit.assert_called_once()

    def test_idempotent_uses_if_not_exists(self):
        """CREATE SCHEMA IF NOT EXISTS means calling twice is safe."""
        from unittest.mock import patch
        from src.database.schema_manager import ensure_user_schema

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch('src.database.schema_ddl.get_user_schema_ddl', return_value='-- DDL'):
            ensure_user_schema(mock_conn, "alice@example.com")
            ensure_user_schema(mock_conn, "alice@example.com")

        # Both calls should succeed without error
        assert mock_conn.commit.call_count == 2


class TestSetSearchPath:
    """Test set_search_path with a mocked connection."""

    def test_sets_correct_search_path(self):
        from src.database.schema_manager import set_search_path

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        schema = set_search_path(mock_conn, "bob@example.com")

        assert schema == "user_bob"
        sql = mock_cursor.execute.call_args[0][0]
        assert 'SET search_path TO' in sql
        assert '"user_bob"' in sql
        assert 'public' in sql
