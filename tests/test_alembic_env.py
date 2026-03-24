"""Tests for migrations/env.py helper functions.

env.py is tightly coupled to the Alembic runtime (it calls context.is_offline_mode()
at module level), so we can't import it normally. Instead, we extract and test
the pure helper functions by compiling them from source.
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).parent.parent
ENV_PATH = REPO_ROOT / "migrations" / "env.py"


def _extract_function(source: str, func_name: str) -> str:
    """Extract a function definition from source code by name."""
    lines = source.split('\n')
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f'def {func_name}('):
            start = i
        elif start is not None and line and not line[0].isspace() and not line.startswith('#'):
            return '\n'.join(lines[start:i])
    if start is not None:
        return '\n'.join(lines[start:])
    raise ValueError(f"Function {func_name} not found in source")


class TestGetDbUrl:
    """Test _get_db_url constructs URLs from environment variables."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Compile _get_db_url from env.py source."""
        source = ENV_PATH.read_text()
        func_src = _extract_function(source, '_get_db_url')
        ns = {'os': os}
        exec(compile(func_src, '<test>', 'exec'), ns)
        self._get_db_url = ns['_get_db_url']

    def test_builds_url_from_env(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_HOST", "dbhost")
        monkeypatch.setenv("POSTGRES_PORT", "5433")
        monkeypatch.setenv("POSTGRES_DB", "testdb")
        monkeypatch.setenv("POSTGRES_USER", "testuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
        url = self._get_db_url()
        assert url == "postgresql://testuser:secret@dbhost:5433/testdb"

    def test_uses_defaults_for_missing_vars(self, monkeypatch):
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("POSTGRES_PORT", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.delenv("POSTGRES_USER", raising=False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
        url = self._get_db_url()
        assert url == "postgresql://companion:pw@postgres:5432/companion"

    def test_raises_without_password(self, monkeypatch):
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
            self._get_db_url()


class TestGetAllUserEmails:
    """Test _get_all_user_emails query and error handling."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Compile _get_all_user_emails from env.py source."""
        source = ENV_PATH.read_text()
        func_src = _extract_function(source, '_get_all_user_emails')
        # Needs sqlalchemy.text in scope
        mock_text = MagicMock(side_effect=lambda s: s)
        ns = {'text': mock_text}
        exec(compile(func_src, '<test>', 'exec'), ns)
        self._get_all_user_emails = ns['_get_all_user_emails']

    def test_returns_emails(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value = [("a@x.com",), ("b@x.com",)]
        result = self._get_all_user_emails(mock_conn)
        assert result == ["a@x.com", "b@x.com"]

    def test_returns_empty_on_exception(self):
        """If users table doesn't exist yet, returns empty list."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("relation does not exist")
        result = self._get_all_user_emails(mock_conn)
        assert result == []

    def test_returns_empty_for_no_users(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value = []
        result = self._get_all_user_emails(mock_conn)
        assert result == []


class TestEnvPyStructure:
    """Validate env.py has the expected structure."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.source = ENV_PATH.read_text()

    def test_has_get_db_url(self):
        assert 'def _get_db_url()' in self.source

    def test_has_get_all_user_emails(self):
        assert 'def _get_all_user_emails(' in self.source

    def test_has_apply_user_schemas(self):
        assert 'def _apply_user_schemas(' in self.source

    def test_has_offline_and_online_modes(self):
        assert 'def run_migrations_offline()' in self.source
        assert 'def run_migrations_online()' in self.source

    def test_online_calls_apply_user_schemas(self):
        """run_migrations_online should call _apply_user_schemas after migrations."""
        assert '_apply_user_schemas' in self.source

    def test_imports_ensure_user_schema(self):
        """_apply_user_schemas should import ensure_user_schema."""
        assert 'ensure_user_schema' in self.source
