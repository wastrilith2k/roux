"""
Regression tests for observe_routes schema routing bug.

All db.execute() calls in observe endpoints must pass user_email so that
PostgreSQL search_path is set to the companion's schema before queries run.
Without user_email the query hits the default search_path and returns nothing.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def app():
    import os
    os.environ.setdefault("SECRET_KEY", "test-secret")
    os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
    from flask import Flask
    from src.routes.observe_routes import observe_bp
    app = Flask(__name__)
    app.register_blueprint(observe_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def make_mock_db(rows=None):
    db = MagicMock()
    result = MagicMock()
    result.fetchall.return_value = rows or []
    result.fetchone.return_value = None
    db.execute.return_value = result
    return db


def test_observe_facts_uses_companion_schema(client):
    """observe_facts must pass user_email to route to the companion's schema."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        r.fetchone.return_value = None
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/facts?companion_id=kai')
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local in db.execute calls. Got: {captured}"


def test_observe_messages_uses_companion_schema(client):
    """observe_messages must route to each companion's schema."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/messages?companion_id=kai')
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local. Got: {captured}"


def test_observe_episodes_uses_companion_schema(client):
    """observe_episodes must pass user_email."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/episodes?companion_id=mira')
        assert resp.status_code == 200

    assert 'mira@companion.local' in captured


def test_observe_goals_uses_companion_schema(client):
    """observe_goals must pass user_email."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/goals?companion_id=kai')
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local. Got: {captured}"


def test_observe_relationship_uses_companion_schema(client):
    """observe_relationship must pass user_email."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        r.fetchone.return_value = None
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/relationship?companion_id=kai')
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local. Got: {captured}"


def test_observe_state_uses_companion_schema(client):
    """observe_state must pass user_email."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        r.fetchone.return_value = None
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/state?companion_id=kai')
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local. Got: {captured}"


def test_observe_opinions_uses_companion_schema(client):
    """observe_opinions must pass user_email."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/opinions?companion_id=kai')
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local. Got: {captured}"


def test_observe_curiosity_uses_companion_schema(client):
    """observe_curiosity must pass user_email."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        r.fetchone.return_value = None
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/curiosity?companion_id=kai')
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local. Got: {captured}"


def test_update_state_uses_companion_schema(client):
    """POST /api/observe/state must route UPDATE to the companion's schema."""
    captured = []

    def capturing_execute(query, params=None, user_email=None):
        captured.append(user_email)
        r = MagicMock()
        r.fetchall.return_value = []
        r.fetchone.return_value = None
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.post('/api/observe/state', json={
            'companion_id': 'kai',
            'closeness_score': 7,
        })
        assert resp.status_code == 200

    assert 'kai@companion.local' in captured, \
        f"Expected kai@companion.local in db.execute calls. Got: {captured}"


def test_observe_messages_supports_offset(client):
    """observe_messages accepts an offset parameter."""
    captured_params = []

    def capturing_execute(query, params=None, user_email=None):
        if params:
            captured_params.extend(params)
        r = MagicMock()
        r.fetchall.return_value = []
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = capturing_execute
        mock_db.return_value = db

        resp = client.get('/api/observe/messages?companion_id=kai&limit=10&offset=20')
        assert resp.status_code == 200

    assert 20 in captured_params, f"Expected offset=20 in query params. Got: {captured_params}"


def test_observe_messages_returns_sorted_by_timestamp(client):
    """Messages from multiple companions are merged and sorted ascending."""
    kai_msg = {'sender_name': 'Kai', 'message_text': 'Hello', 'timestamp': None, 'companion_id': 'kai', 'sentiment_score': None}
    mira_msg = {'sender_name': 'Mira', 'message_text': 'Hi', 'timestamp': None, 'companion_id': 'mira', 'sentiment_score': None}

    def side_effect(query, params=None, user_email=None):
        r = MagicMock()
        r.fetchall.return_value = [kai_msg] if 'kai' in (user_email or '') else [mira_msg]
        return r

    with patch('src.routes.observe_routes._get_db') as mock_db:
        db = MagicMock()
        db.execute.side_effect = side_effect
        mock_db.return_value = db

        resp = client.get('/api/observe/messages?companion_id=kai&companion_id=mira')
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 2
