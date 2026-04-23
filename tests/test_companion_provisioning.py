import pytest
from unittest.mock import MagicMock, patch


def make_mock_db():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    return db


def test_provision_new_companion_creates_schema_and_ownership():
    """provision_new_companion calls ensure_user_schema and assign_companion."""
    db = make_mock_db()
    mock_conn = MagicMock()

    with patch("src.database.ownership.ensure_user_schema") as mock_schema, \
         patch("src.database.ownership.get_connection", return_value=mock_conn):

        from src.database.ownership import provision_new_companion
        provision_new_companion(db, user_email="alice@example.com", companion_id="alice")

        mock_schema.assert_called_once_with(mock_conn, "alice@companion.local")
        db.execute.assert_any_call(
            pytest.approx("INSERT INTO public.user_companions (user_email, companion_id, display_name)\n            VALUES (%s, %s, %s) ON CONFLICT (user_email, companion_id) DO NOTHING"),
            ("alice@example.com", "alice", "Alice"),
        )


def test_provision_new_companion_companion_email_format():
    """companion email is always {companion_id}@companion.local"""
    from src.database.ownership import companion_email_for
    assert companion_email_for("kai") == "kai@companion.local"
    assert companion_email_for("alice_x") == "alice_x@companion.local"


def test_provision_new_companion_returns_companion_id():
    """provision_new_companion returns the companion_id."""
    db = make_mock_db()
    mock_conn = MagicMock()

    with patch("src.database.ownership.ensure_user_schema"), \
         patch("src.database.ownership.get_connection", return_value=mock_conn):

        from src.database.ownership import provision_new_companion
        result = provision_new_companion(db, user_email="bob@example.com", companion_id="bob")
        assert result == "bob"


def test_provision_new_companion_closes_connection():
    """provision_new_companion closes the raw connection after use."""
    db = make_mock_db()
    mock_conn = MagicMock()

    with patch("src.database.ownership.ensure_user_schema"), \
         patch("src.database.ownership.get_connection", return_value=mock_conn):

        from src.database.ownership import provision_new_companion
        provision_new_companion(db, user_email="carol@example.com", companion_id="carol")

        mock_conn.close.assert_called_once()
