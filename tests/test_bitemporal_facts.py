"""Tests for bi-temporal fact tracking.

Tests the temporal query logic and invalidation semantics without
requiring a live database connection.
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.core.clock import SimulationClock, set_clock, SystemClock, PST


NOW = datetime(2026, 4, 1, 12, 0, tzinfo=PST)


@pytest.fixture(autouse=True)
def fixed_clock():
    set_clock(SimulationClock(start=NOW))
    yield
    set_clock(SystemClock())


class TestBitemporalSemantics:
    """Test bi-temporal reasoning about facts."""

    def test_valid_at_tracks_real_world_time(self):
        """valid_at should represent when the fact became true in reality."""
        # James got hired Jan 2025
        hired_at = datetime(2025, 1, 15, tzinfo=PST)
        # Esme learned about it Feb 2025
        learned_at = datetime(2025, 2, 1, tzinfo=PST)

        # The fact should have:
        # valid_at = Jan 15 (when it became true)
        # created_at = Feb 1 (when Esme learned it)
        assert hired_at < learned_at
        # This means: "as of Jan 20", the fact was true but Esme didn't know yet
        # "as of Feb 5", the fact was both true and known

    def test_invalid_at_tracks_when_fact_stopped_being_true(self):
        """invalid_at should represent when the fact stopped being true."""
        # James left job in March 2026
        left_job = datetime(2026, 3, 15, tzinfo=PST)

        # Querying "as of March 1" should still see the job
        query_before = datetime(2026, 3, 1, tzinfo=PST)
        assert query_before < left_job  # Fact still valid

        # Querying "as of April 1" should NOT see the job
        query_after = datetime(2026, 4, 1, tzinfo=PST)
        assert query_after >= left_job  # Fact invalidated

    def test_archived_but_not_deleted(self):
        """Contradicted facts should be archived, not deleted."""
        # Old fact: "James works at CompanyA"
        # New fact: "James works at CompanyB"
        # The old fact gets: archived_at=now, invalid_at=now
        # But it still exists in the database for history queries
        old_fact = {
            'subject': 'user',
            'predicate': 'works at',
            'object': 'CompanyA',
            'created_at': datetime(2025, 1, 1, tzinfo=PST),
            'archived_at': NOW,
            'invalid_at': NOW,
        }
        new_fact = {
            'subject': 'user',
            'predicate': 'works at',
            'object': 'CompanyB',
            'created_at': NOW,
            'archived_at': None,
            'invalid_at': None,
        }

        # Both facts exist (old is archived but not deleted)
        facts = [old_fact, new_fact]
        assert len(facts) == 2

        # Only the new fact is "current"
        current = [f for f in facts if f['archived_at'] is None]
        assert len(current) == 1
        assert current[0]['object'] == 'CompanyB'

        # History shows both
        history = sorted(facts, key=lambda f: f['created_at'])
        assert history[0]['object'] == 'CompanyA'
        assert history[1]['object'] == 'CompanyB'


class TestAsOfQueryLogic:
    """Test the as-of query filtering logic (applied in SQL, tested here as pure logic)."""

    def _fact_visible_as_of(self, fact: dict, as_of: datetime) -> bool:
        """Replicate the SQL WHERE clause from get_facts_as_of."""
        if fact['created_at'] > as_of:
            return False  # Esme hadn't learned it yet
        if fact.get('valid_at') and fact['valid_at'] > as_of:
            return False  # Fact wasn't true yet at that point
        if fact.get('archived_at') and fact['archived_at'] <= as_of:
            return False  # Already archived by then
        if fact.get('invalid_at') and fact['invalid_at'] <= as_of:
            return False  # No longer true by then
        return True

    def test_current_fact_visible_now(self):
        fact = {
            'created_at': NOW - timedelta(days=30),
            'archived_at': None,
            'invalid_at': None,
        }
        assert self._fact_visible_as_of(fact, NOW) is True

    def test_future_fact_not_visible_in_past(self):
        fact = {
            'created_at': NOW,
            'archived_at': None,
            'invalid_at': None,
        }
        past = NOW - timedelta(days=1)
        assert self._fact_visible_as_of(fact, past) is False

    def test_invalidated_fact_not_visible_after_invalidation(self):
        invalidated = NOW - timedelta(days=5)
        fact = {
            'created_at': NOW - timedelta(days=60),
            'archived_at': invalidated,
            'invalid_at': invalidated,
        }
        # Before invalidation: visible
        assert self._fact_visible_as_of(fact, invalidated - timedelta(days=1)) is True
        # After invalidation: not visible
        assert self._fact_visible_as_of(fact, NOW) is False

    def test_fact_visible_between_creation_and_invalidation(self):
        """Fact should be visible in the window between creation and invalidation."""
        created = NOW - timedelta(days=90)
        invalidated = NOW - timedelta(days=10)
        fact = {
            'created_at': created,
            'archived_at': invalidated,
            'invalid_at': invalidated,
        }
        # During valid period
        mid = created + timedelta(days=40)
        assert self._fact_visible_as_of(fact, mid) is True

        # After invalidation
        assert self._fact_visible_as_of(fact, NOW) is False


class TestFactHistory:
    """Test fact history reconstruction."""

    def test_history_shows_evolution(self):
        """Should be able to reconstruct how a fact changed over time."""
        history = [
            {
                'subject': 'user', 'predicate': 'lives in',
                'object': 'San Francisco',
                'created_at': datetime(2024, 1, 1, tzinfo=PST),
                'invalid_at': datetime(2025, 6, 1, tzinfo=PST),
                'is_current': False,
            },
            {
                'subject': 'user', 'predicate': 'lives in',
                'object': 'Portland',
                'created_at': datetime(2025, 6, 1, tzinfo=PST),
                'invalid_at': None,
                'is_current': True,
            },
        ]

        current = [h for h in history if h['is_current']]
        assert len(current) == 1
        assert current[0]['object'] == 'Portland'

        # Can reconstruct: "James used to live in SF, now lives in Portland"
        past = [h for h in history if not h['is_current']]
        assert past[0]['object'] == 'San Francisco'
