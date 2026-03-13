"""
Observe Routes — Read-only dashboard API for watching simulations.

WHAT: Blueprint providing REST endpoints and a SocketIO namespace for the
      observation dashboard. Subscribes to Redis pub/sub 'simulation_events'
      and forwards events to connected /observe clients.
WHY:  The simulation runner emits events as Kai and Mira talk. This module
      bridges those events to the browser so you can watch in real-time.
HOW:  REST endpoints query PostgreSQL for aggregate data (facts, opinions,
      episodes, etc.). A background thread subscribes to Redis and pushes
      events through the /observe SocketIO namespace.
"""
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)
PST = ZoneInfo('America/Los_Angeles')

observe_bp = Blueprint('observe', __name__, url_prefix='/api/observe')


def _get_db():
    """Get database instance."""
    from src.database.db import get_db
    return get_db()


def _companion_email(companion_id: str) -> str:
    """Map companion_id to the email used in user_state/messages.

    In the simulation, Kai's state is stored with email='mira@companion.local'
    and companion_id='kai'. For querying messages/facts/opinions, we filter
    by companion_id directly.
    """
    other = {'kai': 'mira', 'mira': 'kai'}
    peer = other.get(companion_id, companion_id)
    return f"{peer}@companion.local"


# ---------------------------------------------------------------------------
# REST endpoints (all GET, no auth)
# ---------------------------------------------------------------------------

@observe_bp.route('/status')
def observe_status():
    """Simulation status — clock time, day, running state from Redis."""
    try:
        import redis
        r = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'redis'),
            port=int(os.environ.get('REDIS_PORT', 6379)),
            decode_responses=True
        )
        raw = r.get('simulation:status')
        if raw:
            return jsonify(json.loads(raw))
        return jsonify({'status': 'idle', 'message': 'No simulation running'})
    except Exception as e:
        return jsonify({'status': 'unknown', 'error': str(e)})


@observe_bp.route('/messages')
def observe_messages():
    """Recent messages for a companion."""
    companion_id = request.args.get('companion_id', 'kai')
    since = request.args.get('since')
    limit = min(int(request.args.get('limit', 50)), 200)

    try:
        db = _get_db()
        if since:
            result = db.execute(
                """SELECT sender_name, message_text, timestamp, companion_id, sentiment_score
                   FROM messages
                   WHERE companion_id = %s AND timestamp > %s
                   ORDER BY timestamp DESC LIMIT %s""",
                (companion_id, since, limit)
            )
        else:
            result = db.execute(
                """SELECT sender_name, message_text, timestamp, companion_id, sentiment_score
                   FROM messages
                   WHERE companion_id = %s
                   ORDER BY timestamp DESC LIMIT %s""",
                (companion_id, limit)
            )
        rows = result.fetchall() or []
        messages = []
        for r in rows:
            messages.append({
                'speaker': r.get('sender_name', ''),
                'content': r.get('message_text', ''),
                'timestamp': r['timestamp'].isoformat() if r.get('timestamp') else None,
                'companion_id': r.get('companion_id', ''),
                'sentiment': r.get('sentiment_score'),
            })
        return jsonify(list(reversed(messages)))
    except Exception as e:
        logger.error(f"observe_messages error: {e}")
        return jsonify([])


@observe_bp.route('/state')
def observe_state():
    """Internal state + user_state for a companion."""
    companion_id = request.args.get('companion_id', 'kai')

    try:
        db = _get_db()
        # user_state row
        result = db.execute(
            """SELECT closeness_score, romance_level, emotion_profile,
                      internal_state, cooldown_active
               FROM user_state
               WHERE companion_id = %s
               LIMIT 1""",
            (companion_id,)
        )
        row = result.fetchone()
        if not row:
            return jsonify({'companion_id': companion_id, 'state': None})

        internal = {}
        if row.get('internal_state'):
            internal = row['internal_state'] if isinstance(row['internal_state'], dict) else json.loads(row['internal_state'])

        # Extract mood/energy from internal state
        mood = internal.get('mood', {})
        energy = internal.get('energy', {})
        scene = internal.get('scene', {})
        mode = internal.get('conversation_mode', '')

        return jsonify({
            'companion_id': companion_id,
            'closeness': row.get('closeness_score', 0),
            'romance_level': row.get('romance_level', 0),
            'emotion_profile': row.get('emotion_profile', ''),
            'cooldown_active': row.get('cooldown_active', False),
            'mood': mood,
            'energy': energy,
            'scene': scene,
            'mode': mode,
            'internal_state': internal,
        })
    except Exception as e:
        logger.error(f"observe_state error: {e}")
        return jsonify({'companion_id': companion_id, 'error': str(e)})


@observe_bp.route('/facts')
def observe_facts():
    """Recent facts for a companion."""
    companion_id = request.args.get('companion_id', 'kai')
    limit = min(int(request.args.get('limit', 20)), 100)
    email = _companion_email(companion_id)

    try:
        db = _get_db()
        result = db.execute(
            """SELECT subject, predicate, object, confidence, importance, category,
                      created_at, updated_at
               FROM facts
               WHERE user_email = %s AND archived_at IS NULL
               ORDER BY updated_at DESC LIMIT %s""",
            (email, limit)
        )
        rows = result.fetchall() or []
        facts = []
        for r in rows:
            facts.append({
                'subject': r.get('subject', ''),
                'predicate': r.get('predicate', ''),
                'object': r.get('object', ''),
                'confidence': r.get('confidence'),
                'importance': r.get('importance'),
                'category': r.get('category', ''),
                'created_at': r['created_at'].isoformat() if r.get('created_at') else None,
                'updated_at': r['updated_at'].isoformat() if r.get('updated_at') else None,
            })
        return jsonify(facts)
    except Exception as e:
        logger.error(f"observe_facts error: {e}")
        return jsonify([])


@observe_bp.route('/opinions')
def observe_opinions():
    """Opinions for a companion."""
    companion_id = request.args.get('companion_id', 'kai')

    try:
        db = _get_db()
        result = db.execute(
            """SELECT topic, opinion, confidence, evidence_count, category,
                      evidence_summary, created_at, updated_at
               FROM companion_opinions
               WHERE companion_id = %s
               ORDER BY updated_at DESC""",
            (companion_id,)
        )
        rows = result.fetchall() or []
        opinions = []
        for r in rows:
            opinions.append({
                'topic': r.get('topic', ''),
                'opinion': r.get('opinion', ''),
                'confidence': r.get('confidence'),
                'evidence_count': r.get('evidence_count', 0),
                'category': r.get('category', ''),
                'evidence_summary': r.get('evidence_summary', ''),
                'created_at': r['created_at'].isoformat() if r.get('created_at') else None,
            })
        return jsonify(opinions)
    except Exception as e:
        logger.error(f"observe_opinions error: {e}")
        return jsonify([])


@observe_bp.route('/curiosity')
def observe_curiosity():
    """Active curiosity threads for a companion."""
    companion_id = request.args.get('companion_id', 'kai')
    email = _companion_email(companion_id)

    try:
        db = _get_db()
        # Curiosity is stored in the state table as JSON
        state_key = f'proactive_curiosity_{companion_id}'
        result = db.execute(
            "SELECT value FROM state WHERE key = %s",
            (state_key,)
        )
        row = result.fetchone()
        if row and row.get('value'):
            data = json.loads(row['value'])
            threads = data.get('curiosity_threads', [])
            # Filter to unresolved
            active = [t for t in threads if not t.get('resolved', False)]
            return jsonify(active)
        return jsonify([])
    except Exception as e:
        logger.error(f"observe_curiosity error: {e}")
        return jsonify([])


@observe_bp.route('/goals')
def observe_goals():
    """Active goals for a companion."""
    companion_id = request.args.get('companion_id', 'kai')
    email = _companion_email(companion_id)

    try:
        db = _get_db()
        result = db.execute(
            """SELECT id, goal, motivation, category, progress, status,
                      actions_taken, created_at
               FROM companion_goals
               WHERE user_email = %s AND status = 'active'
               ORDER BY created_at DESC""",
            (email,)
        )
        rows = result.fetchall() or []
        goals = []
        for r in rows:
            goals.append({
                'id': r.get('id', ''),
                'goal': r.get('goal', ''),
                'motivation': r.get('motivation', ''),
                'category': r.get('category', ''),
                'progress': r.get('progress', 0),
                'status': r.get('status', ''),
                'actions_taken': r.get('actions_taken', []),
                'created_at': r['created_at'].isoformat() if r.get('created_at') else None,
            })
        return jsonify(goals)
    except Exception as e:
        logger.error(f"observe_goals error: {e}")
        return jsonify([])


@observe_bp.route('/episodes')
def observe_episodes():
    """Recent episodes for a companion."""
    companion_id = request.args.get('companion_id', 'kai')
    limit = min(int(request.args.get('limit', 10)), 50)
    email = _companion_email(companion_id)

    try:
        db = _get_db()
        result = db.execute(
            """SELECT episode_id, started_at, ended_at, summary, significance,
                      emotional_arc, topics
               FROM episodes
               WHERE user_email = %s
               ORDER BY started_at DESC LIMIT %s""",
            (email, limit)
        )
        rows = result.fetchall() or []
        episodes = []
        for r in rows:
            episodes.append({
                'episode_id': str(r.get('episode_id', '')),
                'started_at': r['started_at'].isoformat() if r.get('started_at') else None,
                'ended_at': r['ended_at'].isoformat() if r.get('ended_at') else None,
                'summary': r.get('summary', ''),
                'significance': r.get('significance'),
                'emotional_arc': r.get('emotional_arc', ''),
                'topics': r.get('topics', []),
            })
        return jsonify(episodes)
    except Exception as e:
        logger.error(f"observe_episodes error: {e}")
        return jsonify([])


@observe_bp.route('/relationship')
def observe_relationship():
    """Relationship metrics for a companion."""
    companion_id = request.args.get('companion_id', 'kai')
    email = _companion_email(companion_id)

    try:
        db = _get_db()
        # Relationship evaluation
        result = db.execute(
            """SELECT evaluation, evaluated_at, evaluation_count
               FROM _companion_relationship_eval
               WHERE user_email = %s
               ORDER BY evaluated_at DESC LIMIT 1""",
            (email,)
        )
        row = result.fetchone()
        if row:
            return jsonify({
                'companion_id': companion_id,
                'evaluation': row.get('evaluation', {}),
                'evaluated_at': row['evaluated_at'].isoformat() if row.get('evaluated_at') else None,
                'count': row.get('evaluation_count', 0),
            })
        return jsonify({'companion_id': companion_id, 'evaluation': None})
    except Exception as e:
        logger.error(f"observe_relationship error: {e}")
        return jsonify({'companion_id': companion_id, 'error': str(e)})


# ---------------------------------------------------------------------------
# SocketIO namespace /observe — Redis subscriber
# ---------------------------------------------------------------------------

def start_observe_subscriber(socketio):
    """Start a background thread that subscribes to simulation_events
    and forwards them to all /observe SocketIO clients."""

    def _subscriber():
        import redis
        try:
            r = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                decode_responses=True
            )
            pubsub = r.pubsub()
            pubsub.subscribe('simulation_events')
            logger.info("Observe subscriber started for simulation_events")

            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        event_type = data.get('type', 'unknown')
                        event_data = data.get('data', {})
                        event_data['_ts'] = data.get('timestamp')

                        socketio.emit(
                            event_type,
                            event_data,
                            namespace='/observe'
                        )
                    except Exception as e:
                        logger.error(f"Observe subscriber error: {e}")
        except Exception as e:
            logger.error(f"Observe subscriber failed to start: {e}")

    thread = threading.Thread(target=_subscriber, daemon=True, name='observe-subscriber')
    thread.start()
    return thread


def register_observe_socketio(socketio):
    """Register /observe namespace handlers (read-only)."""

    @socketio.on('connect', namespace='/observe')
    def on_observe_connect():
        logger.info("Observer connected")

    @socketio.on('disconnect', namespace='/observe')
    def on_observe_disconnect():
        logger.info("Observer disconnected")
