import pytest
import os
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    os.environ.setdefault("SECRET_KEY", "test-secret")
    os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
    from flask import Flask
    from src.routes.auth_routes import auth_bp
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    app.register_blueprint(auth_bp)
    return app.test_client()


def test_register_provisions_companion(client):
    """POST /api/auth/register creates user AND companion schema."""
    with patch("src.database.ownership.provision_new_companion") as mock_provision, \
         patch("src.database.db.get_db"), \
         patch("src.routes.auth_routes.create_user", return_value=True), \
         patch("src.routes.auth_routes.authenticate_user", return_value={
             "user_id": 1, "email": "alice@example.com", "session_token": "tok"
         }):
        resp = client.post("/api/auth/register", json={
            "email": "alice@example.com",
            "password": "password123",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["companion_id"] == "alice"
        mock_provision.assert_called_once()
        call_kwargs = mock_provision.call_args
        # companion_id derived from email prefix
        assert call_kwargs[1]["user_email"] == "alice@example.com"
        assert call_kwargs[1]["companion_id"] == "alice"


def make_mock_db():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None
    return db


def test_provision_new_companion_creates_schema_and_ownership():
    """provision_new_companion calls ensure_user_schema and assign_companion."""
    db = make_mock_db()
    mock_conn = MagicMock()

    with patch("src.database.schema_manager.ensure_user_schema") as mock_schema, \
         patch("src.database.connection.get_connection", return_value=mock_conn):

        from src.database.ownership import provision_new_companion
        provision_new_companion(db, user_email="alice@example.com", companion_id="alice")

        mock_schema.assert_called_once_with(mock_conn, "alice@companion.local")
        db.execute.assert_any_call(
            "INSERT INTO public.user_companions (user_email, companion_id, display_name)\n            VALUES (%s, %s, %s) ON CONFLICT (user_email, companion_id) DO NOTHING",
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

    with patch("src.database.schema_manager.ensure_user_schema"), \
         patch("src.database.connection.get_connection", return_value=mock_conn):

        from src.database.ownership import provision_new_companion
        result = provision_new_companion(db, user_email="bob@example.com", companion_id="bob")
        assert result == "bob"


def test_provision_new_companion_closes_connection():
    """provision_new_companion closes the raw connection after use."""
    db = make_mock_db()
    mock_conn = MagicMock()

    with patch("src.database.schema_manager.ensure_user_schema"), \
         patch("src.database.connection.get_connection", return_value=mock_conn):

        from src.database.ownership import provision_new_companion
        provision_new_companion(db, user_email="carol@example.com", companion_id="carol")

        mock_conn.close.assert_called_once()


def test_provision_new_companion_closes_connection_on_error():
    """Connection is closed even when ensure_user_schema raises."""
    db = make_mock_db()
    mock_conn = MagicMock()

    with patch("src.database.connection.get_connection", return_value=mock_conn), \
         patch("src.database.schema_manager.ensure_user_schema", side_effect=RuntimeError("DDL failed")):

        from src.database.ownership import provision_new_companion

        with pytest.raises(RuntimeError):
            provision_new_companion(db, user_email="dave@example.com", companion_id="dave")

        mock_conn.close.assert_called_once()


def test_login_returns_companion_id(client):
    """POST /api/auth/login returns companion_id for the user."""
    with patch("src.routes.auth_routes.authenticate_user", return_value={
        "user_id": 1, "email": "alice@example.com", "session_token": "tok"
    }), patch("src.database.ownership.get_user_companions", return_value=[
        {"companion_id": "alice", "display_name": "Alice", "created_at": None}
    ]) as mock_companions:
        resp = client.post("/api/auth/login", json={
            "email": "alice@example.com",
            "password": "password123",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["companion_id"] == "alice"
        mock_companions.assert_called_once()


def test_login_returns_null_companion_id_when_none(client):
    """POST /api/auth/login returns companion_id=None when user has no companions."""
    with patch("src.routes.auth_routes.authenticate_user", return_value={
        "user_id": 1, "email": "new@example.com", "session_token": "tok"
    }), patch("src.database.ownership.get_user_companions", return_value=[]):
        resp = client.post("/api/auth/login", json={
            "email": "new@example.com",
            "password": "password123",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("companion_id") is None
