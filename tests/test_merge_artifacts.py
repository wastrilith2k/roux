"""Regression tests for issue #68: merge artifacts — misplaced import,
redundant import, non-portable strftime.

1. `import re` must be at module top, not after class definitions.
2. `from datetime import datetime, timezone` must not be duplicated inside
   search_memory_formatted().
3. `%-I` (GNU-only strftime) replaced with portable lstrip("0") approach.
"""

import ast
import inspect
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo


class TestImportReAtModuleLevel:
    """Issue #68 item 1: `import re` must live in the top-level import block."""

    def test_re_imported_at_module_level(self):
        """The `re` module should be available immediately after importing
        entity_profile_loader, not deferred to a function body or appended
        after class definitions."""
        import src.core.entity_profile_loader as mod

        # `re` should be in the module namespace
        assert hasattr(mod, 're'), (
            "entity_profile_loader does not have `re` at module level"
        )

    def test_import_re_not_after_class_definitions(self):
        """Verify `import re` appears before any class or function def in the
        source file (i.e. in the top-level import block, not appended at the
        bottom)."""
        import src.core.entity_profile_loader as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)

        first_class_line = None
        re_import_line = None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if first_class_line is None:
                    first_class_line = node.lineno
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 're':
                        re_import_line = node.lineno

        assert re_import_line is not None, "No top-level `import re` found"
        assert first_class_line is not None, "No class/function defs found"
        assert re_import_line < first_class_line, (
            f"`import re` at line {re_import_line} appears after first "
            f"class/function at line {first_class_line}"
        )


class TestNoRedundantDatetimeImport:
    """Issue #68 item 2: no duplicate datetime import inside function bodies
    in semantic_search.py."""

    def test_search_memory_formatted_no_local_datetime_import(self):
        """search_memory_formatted() must not re-import datetime — it's
        already available at module level."""
        import src.memory.semantic_search as mod
        source = inspect.getsource(mod.search_memory_formatted)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'datetime':
                pytest.fail(
                    "search_memory_formatted() contains a redundant "
                    "`from datetime import ...` — use the module-level import"
                )


class TestPortableStrftime:
    """Issue #68 item 3: get_current_time_context must not use %-I (GNU-only)."""

    def test_no_gnu_strftime_directive(self):
        """Source code must not contain %-I, %-d, %-m, or similar GNU-only
        format codes."""
        import src.core.entity_profile_loader as mod
        source = inspect.getsource(mod.get_current_time_context)
        assert '%-' not in source, (
            "get_current_time_context still uses GNU-only strftime directive "
            "(%-I or similar)"
        )

    def test_hour_formatting_no_leading_zero(self):
        """The fallback time string should show hours without a leading zero
        (e.g. '2:30 PM' not '02:30 PM')."""
        fixed_time = datetime(2025, 11, 25, 14, 30, tzinfo=ZoneInfo('America/Los_Angeles'))
        with patch('src.core.entity_profile_loader.datetime') as mock_dt:
            mock_dt.now.return_value = fixed_time
            mock_dt.fromisoformat = datetime.fromisoformat
            # strftime is called on the real datetime instance, not the mock
            from src.core.entity_profile_loader import get_current_time_context
            result = get_current_time_context()

        assert 'at 2:30 PM' in result, (
            f"Expected 'at 2:30 PM' (no leading zero) in: {result}"
        )

    def test_midnight_hour_shows_12(self):
        """At 12:00 AM the hour should be '12', not '' or '0'."""
        fixed_time = datetime(2025, 11, 25, 0, 5, tzinfo=ZoneInfo('America/Los_Angeles'))
        with patch('src.core.entity_profile_loader.datetime') as mock_dt:
            mock_dt.now.return_value = fixed_time
            mock_dt.fromisoformat = datetime.fromisoformat
            from src.core.entity_profile_loader import get_current_time_context
            result = get_current_time_context()

        assert 'at 12:05 AM' in result, (
            f"Expected 'at 12:05 AM' in: {result}"
        )
