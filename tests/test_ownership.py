"""Test companion ownership management with mocked database."""

from unittest.mock import MagicMock, call
from src.database.ownership import (
    user_owns_companion,
    assign_companion,
    get_user_companions,
    revoke_companion,
)


class TestUserOwnsCompanion:
    """Test ownership check."""

    def test_returns_true_when_row_exists(self):
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchone.return_value = (1,)
        assert user_owns_companion(mock_db, "alice@x.com", "kai") is True

    def test_returns_false_when_no_row(self):
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchone.return_value = None
        assert user_owns_companion(mock_db, "alice@x.com", "kai") is False

    def test_passes_correct_params(self):
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchone.return_value = None
        user_owns_companion(mock_db, "alice@x.com", "kai")
        args = mock_db.execute.call_args
        sql = args[0][0]
        params = args[0][1]
        assert "user_companions" in sql
        assert params == ("alice@x.com", "kai")


class TestAssignCompanion:
    """Test companion assignment."""

    def test_executes_insert_on_conflict(self):
        mock_db = MagicMock()
        assign_companion(mock_db, "alice@x.com", "kai", "Kai")
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "INSERT INTO" in sql
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql
        assert params == ("alice@x.com", "kai", "Kai")

    def test_display_name_optional(self):
        mock_db = MagicMock()
        assign_companion(mock_db, "alice@x.com", "kai")
        params = mock_db.execute.call_args[0][1]
        assert params == ("alice@x.com", "kai", None)


class TestGetUserCompanions:
    """Test listing user's companions."""

    def test_returns_fetchall_results(self):
        mock_db = MagicMock()
        expected = [("kai", "Kai", "2026-01-01"), ("mira", "Mira", "2026-01-02")]
        mock_db.execute.return_value.fetchall.return_value = expected
        result = get_user_companions(mock_db, "alice@x.com")
        assert result == expected

    def test_returns_empty_list_for_no_companions(self):
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        result = get_user_companions(mock_db, "alice@x.com")
        assert result == []

    def test_queries_by_email(self):
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        get_user_companions(mock_db, "alice@x.com")
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "WHERE user_email = %s" in sql
        assert params == ("alice@x.com",)

    def test_orders_by_created_at(self):
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        get_user_companions(mock_db, "alice@x.com")
        sql = mock_db.execute.call_args[0][0]
        assert "ORDER BY created_at" in sql


class TestRevokeCompanion:
    """Test companion revocation."""

    def test_executes_delete(self):
        mock_db = MagicMock()
        revoke_companion(mock_db, "alice@x.com", "kai")
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "DELETE FROM" in sql
        assert "user_companions" in sql
        assert params == ("alice@x.com", "kai")


class TestOwnershipWorkflow:
    """Test a full assign-check-revoke workflow."""

    def test_assign_then_check_then_revoke(self):
        """Simulate the lifecycle: assign, verify ownership, revoke, verify gone."""
        mock_db = MagicMock()

        # Assign
        assign_companion(mock_db, "alice@x.com", "kai", "Kai")
        assert mock_db.execute.called

        # Check ownership — simulate row exists
        mock_db.execute.return_value.fetchone.return_value = (1,)
        assert user_owns_companion(mock_db, "alice@x.com", "kai") is True

        # Revoke
        revoke_companion(mock_db, "alice@x.com", "kai")

        # Check ownership — simulate row gone
        mock_db.execute.return_value.fetchone.return_value = None
        assert user_owns_companion(mock_db, "alice@x.com", "kai") is False
