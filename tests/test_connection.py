"""Tests for centralized database connection parameters."""

import os
import pytest


class TestGetConnectionParams:
    def test_returns_params_from_env(self, monkeypatch):
        monkeypatch.setenv('POSTGRES_HOST', 'myhost')
        monkeypatch.setenv('POSTGRES_PORT', '5555')
        monkeypatch.setenv('POSTGRES_DB', 'mydb')
        monkeypatch.setenv('POSTGRES_USER', 'myuser')
        monkeypatch.setenv('POSTGRES_PASSWORD', 'secret')

        from src.database.connection import get_connection_params
        params = get_connection_params()

        assert params['host'] == 'myhost'
        assert params['port'] == '5555'
        assert params['dbname'] == 'mydb'
        assert params['user'] == 'myuser'
        assert params['password'] == 'secret'

    def test_raises_without_password(self, monkeypatch):
        monkeypatch.setenv('POSTGRES_PASSWORD', '')

        from src.database.connection import get_connection_params
        with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
            get_connection_params()

    def test_raises_with_missing_password(self, monkeypatch):
        monkeypatch.delenv('POSTGRES_PASSWORD', raising=False)

        from src.database.connection import get_connection_params
        with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
            get_connection_params()
