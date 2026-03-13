"""Tests for the admin CLI argument parsing and structure."""

import pytest
import subprocess
import sys


class TestAdminCLI:
    """Test that admin CLI parses arguments correctly."""

    def test_help_output(self):
        result = subprocess.run(
            [sys.executable, 'scripts/admin.py', '--help'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert 'status' in result.stdout
        assert 'errors' in result.stdout
        assert 'costs' in result.stdout
        assert 'companions' in result.stdout
        assert 'companion' in result.stdout
        assert 'memory' in result.stdout
        assert 'goals' in result.stdout
        assert 'db' in result.stdout

    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, 'scripts/admin.py'],
            capture_output=True, text=True, timeout=10,
        )
        # Should exit cleanly without error
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower() or 'Available commands' in result.stdout
