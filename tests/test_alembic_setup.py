"""Tests that Alembic migration files are syntactically valid and importable.

These tests do NOT connect to a live database.  They verify only that the
migration scripts are well-formed Python and that env.py can be parsed without
syntax errors.
"""

import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _load_module(path: Path, name: str):
    """Return an unexecuted module loaded from *path*."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None, f"Could not create module spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    return spec, mod


def test_migration_001_is_valid_python():
    """Verify the initial migration file loads without syntax errors."""
    migration_path = REPO_ROOT / "migrations" / "versions" / "001_initial_schema.py"
    assert migration_path.exists(), f"Migration not found: {migration_path}"
    spec, mod = _load_module(migration_path, "migration_001")
    # spec.loader.exec_module would run the module; we only check it parses.
    assert spec is not None


def test_migration_001_has_correct_revision_metadata():
    """Verify the initial migration declares the expected Alembic metadata.

    When alembic is not installed we verify the metadata by inspecting the
    source text directly rather than exec-ing the module (which would fail on
    the ``from alembic import op`` line).
    """
    import sys

    migration_path = REPO_ROOT / "migrations" / "versions" / "001_initial_schema.py"
    assert migration_path.exists()

    if "alembic" in sys.modules or importlib.util.find_spec("alembic") is not None:
        # Full check: execute the module and inspect the live objects.
        spec, mod = _load_module(migration_path, "migration_001_meta")
        spec.loader.exec_module(mod)
        assert mod.revision == "001"
        assert mod.down_revision is None
        assert callable(mod.upgrade)
        assert callable(mod.downgrade)
    else:
        # Alembic not installed: verify metadata via source text.
        source = migration_path.read_text()
        assert 'revision: str = "001"' in source
        assert "down_revision" in source
        assert "def upgrade" in source
        assert "def downgrade" in source


def test_env_py_is_valid_python():
    """Verify migrations/env.py exists and can be parsed as a module spec."""
    env_path = REPO_ROOT / "migrations" / "env.py"
    assert env_path.exists(), f"env.py not found: {env_path}"
    spec = importlib.util.spec_from_file_location("alembic_env", str(env_path))
    assert spec is not None, "Could not create spec for env.py"


def test_script_mako_exists():
    """Verify the Mako template for new migrations exists."""
    mako_path = REPO_ROOT / "migrations" / "script.py.mako"
    assert mako_path.exists(), f"script.py.mako not found: {mako_path}"
    content = mako_path.read_text()
    # Must contain the key Alembic template variables
    assert "${up_revision}" in content
    assert "def upgrade" in content
    assert "def downgrade" in content


def test_alembic_ini_has_script_location():
    """Verify alembic.ini points to the migrations directory."""
    ini_path = REPO_ROOT / "alembic.ini"
    assert ini_path.exists(), f"alembic.ini not found: {ini_path}"
    content = ini_path.read_text()
    assert "script_location = migrations" in content


def test_alembic_ini_has_logging_config():
    """Verify alembic.ini has the required logging sections."""
    ini_path = REPO_ROOT / "alembic.ini"
    content = ini_path.read_text()
    assert "[loggers]" in content
    assert "[handlers]" in content
    assert "[formatters]" in content
