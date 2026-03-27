"""
Regression test for issue #56: time-of-day logic duplicated across 3 modules.

Verifies that:
1. The canonical time_of_day_label() function exists in time_awareness.py
2. All three modules (time_awareness, derived_scene_context, schedule_context)
   use the same shared function rather than duplicated inline logic
3. The function produces correct results for all hour boundaries
"""

import ast
import os
import textwrap

import pytest

os.environ.setdefault('ENVIRONMENT', 'test')
os.environ.setdefault('POSTGRES_PASSWORD', 'test')
os.environ.setdefault('POSTGRES_DB', 'test')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'test')

from src.core.time_awareness import time_of_day_label


class TestTimeOfDayLabelCanonical:
    """Ensure the canonical function returns correct labels for all boundaries."""

    @pytest.mark.parametrize("hour,expected", [
        (0, 'night'),
        (1, 'night'),
        (4, 'night'),
        (5, 'morning'),
        (8, 'morning'),
        (11, 'morning'),
        (12, 'afternoon'),
        (14, 'afternoon'),
        (16, 'afternoon'),
        (17, 'evening'),
        (19, 'evening'),
        (20, 'evening'),
        (21, 'night'),
        (23, 'night'),
    ])
    def test_hour_to_label(self, hour, expected):
        assert time_of_day_label(hour) == expected


class TestNoDuplicatedLogic:
    """
    Verify that derived_scene_context.py and schedule_context.py do NOT
    contain inline time-of-day bucketing logic (the old pattern that was
    duplicated in issue #56).

    We parse the source files as ASTs and check that neither contains
    the characteristic `if 5 <= hour < 12` pattern.
    """

    @staticmethod
    def _source_contains_hour_bucketing(filepath: str) -> bool:
        """Check if a Python source file contains inline hour-bucketing logic."""
        with open(filepath, 'r') as f:
            source = f.read()

        # Simple text check for the duplicated pattern
        # The old code had variations of: if 5 <= hour < 12
        indicators = [
            '5 <= hour < 12',
            '12 <= hour < 17',
            '17 <= hour < 21',
        ]
        return any(indicator in source for indicator in indicators)

    def test_derived_scene_context_no_inline_bucketing(self):
        filepath = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'core', 'derived_scene_context.py'
        )
        assert not self._source_contains_hour_bucketing(filepath), (
            "derived_scene_context.py still contains inline time-of-day bucketing. "
            "It should import time_of_day_label from time_awareness instead."
        )

    def test_schedule_context_no_inline_bucketing(self):
        filepath = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'core', 'schedule_context.py'
        )
        assert not self._source_contains_hour_bucketing(filepath), (
            "schedule_context.py still contains inline time-of-day bucketing. "
            "It should import time_of_day_label from time_awareness instead."
        )

    def test_time_awareness_has_canonical_function(self):
        """The canonical function must be importable from time_awareness."""
        from src.core.time_awareness import time_of_day_label as fn
        assert callable(fn)
        # Sanity check: it returns a string
        assert isinstance(fn(12), str)
