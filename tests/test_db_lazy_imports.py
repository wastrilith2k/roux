"""Regression test for issue #14: arrow import should be lazy in db.py.

The module-level `import arrow` in db.py caused ModuleNotFoundError in test
environments where arrow is not installed, breaking any test that imported
from src.database.db — even tests that never touch arrow-dependent code paths.
"""
import importlib
import sys
import unittest.mock


def test_db_module_importable_without_arrow():
    """Importing src.database.db must not fail when arrow is not installed.

    This is a regression test for issue #14: three tests in
    test_companion_isolation.py failed with ModuleNotFoundError because
    db.py imported arrow at module level.
    """
    # Temporarily hide the arrow module to simulate it not being installed
    arrow_module = sys.modules.pop('arrow', None)
    # Also remove cached db module so it gets re-imported fresh
    db_module = sys.modules.pop('src.database.db', None)

    try:
        with unittest.mock.patch.dict(sys.modules, {'arrow': None}):
            # Re-import — this should NOT raise ModuleNotFoundError
            import src.database.db  # noqa: F811
            # Verify the module loaded successfully
            assert hasattr(src.database.db, 'CompanionDB')
    finally:
        # Restore original modules
        if arrow_module is not None:
            sys.modules['arrow'] = arrow_module
        if db_module is not None:
            sys.modules['src.database.db'] = db_module
