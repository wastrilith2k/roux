"""Tests for multi-agent companion isolation.

These tests verify that companion_id correctly scopes data
so that Kai's queries never return Mira's data and vice versa.

Requires a running PostgreSQL instance to execute.
"""
import pytest
import os
from datetime import datetime
from zoneinfo import ZoneInfo

PST = ZoneInfo('America/Los_Angeles')

requires_db = pytest.mark.skipif(
    not os.getenv('POSTGRES_HOST'),
    reason="No database available (set POSTGRES_HOST to run)"
)


@requires_db
class TestCompanionIsolation:
    """Verify companion_id scopes data correctly."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up test data for two companions."""
        from src.database.db import get_db
        self.db = get_db()

        # Clean test data
        self.db.execute("DELETE FROM messages WHERE companion_id IN ('test_kai', 'test_mira')")

        # Insert test messages for each companion
        self.db.execute(
            "INSERT INTO messages (email, role, content, created_at, companion_id) VALUES (%s, %s, %s, %s, %s)",
            ('test@test.com', 'user', 'Hello from Kai perspective', datetime.now(PST), 'test_kai')
        )
        self.db.execute(
            "INSERT INTO messages (email, role, content, created_at, companion_id) VALUES (%s, %s, %s, %s, %s)",
            ('test@test.com', 'user', 'Hello from Mira perspective', datetime.now(PST), 'test_mira')
        )

        yield

        # Cleanup
        self.db.execute("DELETE FROM messages WHERE companion_id IN ('test_kai', 'test_mira')")

    def test_messages_scoped_by_companion_id(self):
        """Kai should only see Kai's messages."""
        result = self.db.execute(
            "SELECT content FROM messages WHERE companion_id = %s",
            ('test_kai',)
        )
        messages = result.fetchall()
        assert len(messages) == 1
        assert 'Kai' in messages[0]['content']

    def test_no_cross_contamination(self):
        """Mira should not see Kai's messages."""
        result = self.db.execute(
            "SELECT content FROM messages WHERE companion_id = %s",
            ('test_mira',)
        )
        messages = result.fetchall()
        for msg in messages:
            assert 'Kai perspective' not in msg['content']

    def test_both_companions_coexist(self):
        """Both companions can have messages for the same user email."""
        result = self.db.execute(
            "SELECT DISTINCT companion_id FROM messages WHERE email = 'test@test.com' AND companion_id LIKE 'test_%'"
        )
        companions = result.fetchall()
        companion_ids = [r['companion_id'] for r in companions]
        assert 'test_kai' in companion_ids
        assert 'test_mira' in companion_ids


class TestClockIsolation:
    """Verify simulation clock doesn't affect production path."""

    def test_system_clock_is_default(self):
        from src.core.clock import get_clock, SystemClock
        clock = get_clock()
        # After module import, should be system clock
        assert isinstance(clock, SystemClock)

    def test_simulation_clock_injectable(self):
        from src.core.clock import SimulationClock, set_clock, get_clock, SystemClock, now, PST

        original = get_clock()
        sim = SimulationClock(start=datetime(2026, 6, 1, tzinfo=PST))
        set_clock(sim)

        assert now().year == 2026
        assert now().month == 6

        # Restore
        set_clock(SystemClock())


class TestSeedHistoryFormat:
    """Validate seed history JSON files are well-formed."""

    def test_kai_seed_history_valid(self):
        import json
        from pathlib import Path
        seed_path = Path(__file__).parent.parent / 'instances' / 'kai' / 'seed_history.json'
        with open(seed_path) as f:
            data = json.load(f)

        assert 'companion_facts' in data
        assert 'user_facts' in data
        assert 'relationship_facts' in data
        assert 'seed_exchanges' in data
        assert len(data['companion_facts']) > 20
        assert len(data['user_facts']) > 15
        assert len(data['seed_exchanges']) > 5

    def test_mira_seed_history_valid(self):
        import json
        from pathlib import Path
        seed_path = Path(__file__).parent.parent / 'instances' / 'mira' / 'seed_history.json'
        with open(seed_path) as f:
            data = json.load(f)

        assert 'companion_facts' in data
        assert 'user_facts' in data
        assert len(data['companion_facts']) > 15
        assert len(data['user_facts']) > 20  # Mira's user_facts = Kai's companion_facts

    def test_seed_histories_are_mirrored(self):
        """Kai's companion_facts should be Mira's user_facts."""
        import json
        from pathlib import Path
        base = Path(__file__).parent.parent / 'instances'

        with open(base / 'kai' / 'seed_history.json') as f:
            kai = json.load(f)
        with open(base / 'mira' / 'seed_history.json') as f:
            mira = json.load(f)

        # Kai's self-knowledge = what Mira knows about Kai
        assert set(kai['companion_facts']) == set(mira['user_facts'])
        # Mira's self-knowledge = what Kai knows about Mira
        assert set(mira['companion_facts']) == set(kai['user_facts'])
        # Relationship facts are shared
        assert set(kai['relationship_facts']) == set(mira['relationship_facts'])
