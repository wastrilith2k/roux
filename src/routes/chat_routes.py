"""
Chat Routes -- WebSocket and HTTP endpoints for real-time conversation.

WHAT: Registers all Socket.IO event handlers (connect, disconnect, send_message,
      send_image, slash-commands, state/history/activity queries, fact approval,
      autonomy toggles) plus a small HTTP API (legacy chat, paginated history).

WHY:  The companion's primary interface is a WebSocket-based chat. This module is
      the front door for every message the user sends and every response the
      companion delivers. It also bridges the fact-approval pipeline and the
      autonomy system to the live WebSocket session.

HOW:  Flask-SocketIO events are registered inside `register_socketio_handlers()`,
      which is called once at app startup. Authentication supports Firebase ID
      tokens, legacy session tokens, and raw email (CLI). Redis tracks live
      connections so the autonomy layer knows when the user is online. Scene
      state is cleaned up on disconnect to reflect physical absence.

Architecture notes:
  - Message processing is delegated to MessageProcessor (handlers/message_handler)
  - Fact approval requests arrive via Redis pub/sub from Celery workers and are
    forwarded to the connected WebSocket client
  - The /say command stores a message silently and queues a Celery task for a
    delayed companion response (30s-3min)
"""
import logging
import os
import threading
import redis as _redis_mod
from flask import Blueprint, request
from flask_socketio import emit, join_room, leave_room
from src.database.db import get_db
from src.database import tables as T

logger = logging.getLogger(__name__)
from src.utils.timezone_utils import now_pacific_naive

# ---------------------------------------------------------------------------
# Module-level Redis singleton – avoids creating a new connection per call
# ---------------------------------------------------------------------------
_redis_client = None
_redis_lock = threading.Lock()


def _get_redis():
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
                _redis_client = _redis_mod.from_url(redis_url)
    return _redis_client


# ============================================================================
# HELPERS
# ============================================================================

def _get_default_state(email: str = None):
    """Return relationship state defaults, pulling closeness from the DB if possible."""
    closeness = 50
    if email:
        try:
            db = get_db()
            profile = db.get_profile(email)
            if profile:
                closeness = profile.get('closeness_score', 50)
        except Exception:
            pass
    return {
        'closeness_score': closeness,
        'emotion_profile': 'Caring, Trusting, and Personal',
        'cooldown_active': False,
        'badgering_count': 0,
        'romance_enabled': True,
        'attraction_latent_state': False
    }

from src.handlers.message_handler import get_message_processor
from src.utils.message_bundler import bundle_messages as _bundle_messages

# Blueprint for HTTP chat endpoints (WebSocket handlers registered separately)
chat_bp = Blueprint('chat', __name__)

# Maps Socket.IO session ID -> {'email': str, 'companion_id': str} for all live connections
# (Legacy callers may still store bare email strings; helpers handle both.)
connected_users = {}


def _get_user_email(sid):
    """Extract email from connected_users (handles both dict and legacy string)."""
    entry = connected_users.get(sid)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get('email')
    return entry  # legacy string


def _get_user_companion_id(sid):
    """Extract companion_id from connected_users (returns '' if not set)."""
    entry = connected_users.get(sid)
    if isinstance(entry, dict):
        return entry.get('companion_id', '')
    return ''

# ---------- Message bundling state ----------
# When user sends a message while the companion is still generating a response,
# the in-flight request is cancelled and all queued messages are bundled into
# a single coherent request so the companion responds to everything at once.
#
# _user_processing: email -> threading.Event (set = cancel requested)
# _user_message_queue: email -> list of queued message strings
_user_processing = {}
_user_message_queue = {}


def _process_with_bundling(socketio, data, email, sid):
    """
    Process a message with support for cancellation and re-bundling.

    If the user sends more messages while the companion is generating,
    the in-flight LLM call is cancelled, all pending messages are bundled,
    and the pipeline runs again with the full bundle.
    """
    from flask_socketio import emit

    cancel_event = threading.Event()
    _user_processing[email] = cancel_event

    current_message = data.get('message', '').strip()
    current_data = dict(data)

    try:
        while True:
            cancel_event.clear()

            processor = get_message_processor()
            result = processor.process_message(
                current_data, email, sid,
                cancel_check=lambda: cancel_event.is_set()
            )

            if result and result.get('cancelled'):
                # Pipeline was cancelled — bundle queued messages and retry
                import time as _time
                _time.sleep(0.3)  # Brief debounce for rapid-fire messages

                queued = _user_message_queue.pop(email, [])
                if not queued:
                    # Cancelled but no new messages? Shouldn't happen, but
                    # just re-process the original message.
                    logger.warning(f"Pipeline cancelled for {email} but queue empty — retrying")
                    continue

                all_messages = [current_message] + queued
                bundled = _bundle_messages(all_messages)
                current_message = bundled
                current_data = dict(data)
                current_data['message'] = bundled
                logger.info(
                    f"Bundled {len(all_messages)} messages for {email}, "
                    f"retrying pipeline ({len(bundled)} chars)"
                )
                continue

            # Pipeline completed (not cancelled) — emit result
            if result:
                if 'error' in result:
                    socketio.emit('error', {'message': result['error']}, room=sid)
                else:
                    socketio.emit('message', result, room=sid)
            else:
                socketio.emit('error', {'message': 'Failed to process message'}, room=sid)

            # Check for messages that arrived after the pipeline passed its
            # cancel checkpoints.  Without this, queued messages are silently
            # dropped in the finally block and the user never gets a response.
            queued = _user_message_queue.pop(email, [])
            if queued:
                bundled = _bundle_messages(queued)
                current_message = bundled
                current_data = dict(data)
                current_data['message'] = bundled
                logger.info(
                    f"Processing {len(queued)} late-queued message(s) for {email} "
                    f"({len(bundled)} chars)"
                )
                continue
            break

    finally:
        _user_processing.pop(email, None)
        _user_message_queue.pop(email, None)


# ============================================================================
# WEBSOCKET EVENT HANDLERS
# ============================================================================

def register_socketio_handlers(socketio):
    """
    Register all WebSocket event handlers with SocketIO.

    Call this from web_chat.py during app initialization:
        from routes.chat_routes import register_socketio_handlers
        register_socketio_handlers(socketio)

    Args:
        socketio: Flask-SocketIO instance
    """

    @socketio.on('connect')
    def handle_connect(auth):
        """
        Handle new WebSocket connection with token or email authentication.

        Supports:
        1. Firebase ID tokens (auth={'token': 'firebase_token'})
        2. Legacy session tokens (auth={'token': 'session_token'})
        3. Email-based auth for CLI/local testing (auth={'email': 'user@example.com'})
        """
        try:
            print(f"🔌 WebSocket connection attempt from {request.sid}")

            user_info = None
            auth_type = None

            # Check if email-based auth is provided (fallback for CLI)
            if auth and auth.get('email') and not auth.get('token'):
                email = auth.get('email')
                user_info = {'email': email}
                auth_type = 'Email'
                print(f"✅ Email-based auth accepted for: {email}")
            else:
                # Get token from auth dict
                token = auth.get('token') if auth else None
                if not token:
                    print(f"❌ No token or email provided")
                    return False

                # Try configured auth provider first (Firebase or Cognito)
                try:
                    from src.auth.auth_provider import verify_token as verify_auth_token

                    firebase_claims = verify_auth_token(token)
                    if firebase_claims:
                        email = firebase_claims.get('email')
                        if email:
                            user_info = {'email': email, 'uid': firebase_claims.get('uid')}
                            auth_type = 'Token'
                            print(f"✅ Auth token verified for: {email}")
                except Exception as e:
                    print(f"⚠️  Auth token verification failed: {e}")

                # Fall back to legacy session token
                if not user_info:
                    try:
                        from src.database.simple_auth import verify_session

                        try:
                            user_info = verify_session(token)
                            auth_type = 'Legacy'
                            if user_info:
                                print(f"✅ Legacy session token verified")
                        except Exception as db_error:
                            print(f"⚠️  Database error during verification: {db_error}")
                            user_info = None

                    except Exception as e:
                        print(f"⚠️  Legacy token verification failed: {e}")

                # Check if we got valid user info from either method
                if not user_info:
                    print(f"❌ Invalid or expired token (neither Firebase nor legacy)")
                    return False

            email = user_info['email']

            # Read companion_id from auth (optional, defaults to COMPANION_ID env var)
            selected_companion_id = (auth.get('companion_id') if auth else None) or os.environ.get('COMPANION_ID', '')

            # Store user connection with companion_id
            connected_users[request.sid] = {
                'email': email,
                'companion_id': selected_companion_id,
            }
            join_room(email)

            # Mark user as online in Redis (1h TTL) so autonomy tasks can
            # check presence before sending proactive messages
            try:
                _get_redis().set(f'ws_connected:{email}', request.sid, ex=3600)
            except Exception as _re:
                logger.debug(f"Could not set Redis connection key: {_re}")

            print(f'✅ {email} connected via WebSocket ({auth_type} auth, sid: {request.sid})')

            # Send initial status with companion identity
            state = _get_default_state(email)
            try:
                from src.config.persona_config import get_persona_config
                pc = get_persona_config(companion_id=selected_companion_id or None)
                companion_name = pc.companion_short_name
                companion_id_resolved = selected_companion_id or pc.companion_entity_profile
            except Exception:
                companion_name = 'Companion'
                companion_id_resolved = selected_companion_id or ''

            emit('status', {
                'closeness_score': state['closeness_score'],
                'profile': state['emotion_profile'],
                'connected': True,
                'companion_name': companion_name,
                'companion_id': companion_id_resolved,
            })

            # REMOVED: Presence greetings disabled - was causing spam
            # Re-enable via extensions system when ready

            return True

        except Exception as e:
            print(f'❌ Connection error: {e}')
            import traceback
            traceback.print_exc()
            return False

    @socketio.on('disconnect')
    def handle_disconnect(reason=None):
        """Handle WebSocket disconnection and cleanup."""
        user_data = connected_users.pop(request.sid, None)
        email = user_data['email'] if isinstance(user_data, dict) else user_data
        if email:
            leave_room(email)

            # Remove Redis presence key so autonomy knows user is offline
            try:
                _get_redis().delete(f'ws_connected:{email}')
            except Exception as _re:
                logger.debug(f"Could not delete Redis connection key: {_re}")

            # Clear physical presence flags -- user is no longer "in the room"
            try:
                from src.core.scene_tracker import get_scene_tracker
                tracker = get_scene_tracker()
                scene = tracker.get_scene_state(email)
                if scene.physical_presence:
                    scene.physical_presence = False
                    scene.physical_state = None
                    scene.posture = None
                    scene.position_detail = None
                    scene.james_position = None
                    scene.companion_position = None
                    tracker.save_scene_state(email, scene)
                    logger.info("📱 WebSocket disconnect — cleared physical_presence")
            except Exception as _se:
                logger.debug(f"Could not clear scene on disconnect: {_se}")

            print(f'❌ {email} disconnected (sid: {request.sid}, reason: {reason})')

    @socketio.on('send_message')
    def handle_send_message(data):
        """
        Handle incoming chat message via WebSocket.

        If the companion is already generating a response for this user,
        the in-flight request is cancelled and all messages are bundled
        into a single coherent request.

        Data should contain:
            - message: The user's message text
            - message_type (optional): 'chat' (default) or 'test'

        Special prefixes:
            - /test: Process message but don't save to database (dry run)
        """
        # Get authenticated user
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        # Validate message
        message = data.get('message', '').strip()
        if not message:
            emit('error', {'message': 'Empty message'})
            return

        # Check for slash commands (handled by core command system)
        if message.startswith('/'):
            from src.core.commands import get_command_registry
            registry = get_command_registry()

            if registry.is_command(message):
                # Execute command
                result = registry.execute(message, context={
                    'user_email': email,
                    'interface': 'websocket',
                    'sid': request.sid
                })

                # Emit command response
                emit('command_response', {
                    'command': message.split()[0],
                    'text': result.get('text', ''),
                    'data': result.get('data'),
                    'error': result.get('error')
                })
                return

        # Handle /test prefix - dry run mode (no database saves)
        if message.lower().startswith('/test '):
            message = message[6:].strip()  # Remove '/test ' prefix
            data['message'] = message
            data['message_type'] = 'test'
            print(f"🧪 TEST MODE: {email} - message will not be saved")

        # Inject companion_id into data so message handler can use it
        companion_id = _get_user_companion_id(request.sid)
        if companion_id:
            data['companion_id'] = companion_id

        # --- Message bundling: if companion is already processing for this user,
        #     queue this message and cancel the in-flight pipeline ---
        if email in _user_processing:
            _user_message_queue.setdefault(email, []).append(message)
            _user_processing[email].set()  # Signal cancellation
            logger.info(f"Queued message for {email} (companion busy, {len(_user_message_queue[email])} in queue)")
            emit('message_queued', {
                'queued_count': len(_user_message_queue[email]),
                'message': 'Got it — will respond to everything together.'
            })
            return

        # --- Process (possibly with bundled retry loop) ---
        _process_with_bundling(socketio, data, email, request.sid)

    @socketio.on('send_image')
    def handle_send_image(data):
        """
        Handle image sent via WebSocket.

        Data should contain:
            - image: base64-encoded image data (no data URI prefix)
            - mime_type (optional): e.g. 'image/jpeg' (default), 'image/png'
            - caption (optional): text message accompanying the image
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        image_b64 = data.get('image', '')
        if not image_b64:
            emit('error', {'message': 'No image data provided'})
            return

        # Reject oversized payloads (~10 MB base64 ~ 7.5 MB raw image)
        if len(image_b64) > 10 * 1024 * 1024:
            emit('error', {'message': 'Image too large (max ~10MB)'})
            return

        mime_type = data.get('mime_type', 'image/jpeg')
        caption = data.get('caption', '').strip()

        # Describe the image
        from src.llm.vision import describe_image_from_base64
        description = describe_image_from_base64(image_b64, mime_type=mime_type)

        if not description:
            emit('error', {'message': "Couldn't process that image. Try again?"})
            return

        # Build pipeline text (same format as Telegram)
        from src.config.persona_config import get_persona_config
        _user_name = get_persona_config().primary_user_name
        if caption:
            pipeline_text = f'[{_user_name} sent a photo: {description}]\n\n{caption}'
        else:
            pipeline_text = f'[{_user_name} sent a photo: {description}]'

        logger.info(f"Image described via WebSocket for {email}: {description[:80]}...")

        # Process through the normal message pipeline
        processor = get_message_processor()
        result = processor.process_message(
            {'message': pipeline_text},
            email,
            request.sid
        )

        if result:
            if 'error' in result:
                emit('error', {'message': result['error']})
            else:
                emit('message', result)
        else:
            emit('error', {'message': 'Failed to process image message'})

    @socketio.on('request_status')
    def handle_request_status():
        """
        Handle status request from client.

        Returns current closeness score, emotional profile, and state information.
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        state = _get_default_state(email)

        # Get current presence display
        try:
            from src.core.presence_manager import get_presence_manager
            from src.database.db import get_db
            db = get_db()
            presence_mgr = get_presence_manager(db=db)
            presence_display = presence_mgr.get_presence_display_text()
        except Exception as e:
            logger.debug(f"⚠️  Could not get presence display: {e}")
            presence_display = None

        status_data = {
            'closeness_score': state['closeness_score'],
            'profile': state['emotion_profile'],
            'cooldown_active': state.get('cooldown_active', False),
            'badgering_count': state.get('badgering_count', 0)
        }

        if presence_display:
            status_data['presence_display'] = presence_display

        emit('status', status_data)

    @socketio.on('request_history')
    def handle_request_history(data=None):
        """
        Handle history request from client.

        Data can contain:
            - limit: Number of messages to return (default: 5)
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        limit = 5
        if data and isinstance(data, dict):
            limit = min(data.get('limit', 5), 50)  # Cap at 50

        companion_id = _get_user_companion_id(request.sid)

        try:
            from src.database.db import get_db
            db = get_db()

            if companion_id:
                # Filter messages by companion_id
                from psycopg2.extras import RealDictCursor
                with db._get_connection() as conn:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute(f'''
                        SELECT sender_name, message_text, timestamp
                        FROM {T.MESSAGES}
                        WHERE email = %s AND companion_id = %s
                        ORDER BY timestamp DESC, id DESC
                        LIMIT %s
                    ''', (email, companion_id, limit))
                    rows = list(cursor.fetchall())
                rows.reverse()  # Chronological order
                messages = rows
            else:
                messages = db.get_recent_messages(email, limit=limit)

            # Format for client (oldest first for display)
            formatted = []
            for msg in messages:
                sender = msg.get('sender_name', '')
                role = 'companion' if sender.lower() in ('companion',) else 'user'
                ts = msg.get('timestamp', '')
                if hasattr(ts, 'isoformat'):
                    ts = ts.isoformat()
                formatted.append({
                    'role': role,
                    'content': msg.get('message_text', ''),
                    'timestamp': ts
                })

            emit('history', {'messages': formatted})
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            emit('error', {'message': 'Failed to fetch history'})

    @socketio.on('request_state')
    def handle_request_state(data=None):
        """
        Handle state request from client - returns scene state and internal state.
        This is a debug/info command, doesn't affect conversation.
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.database.db import get_db
            db = get_db()

            # Get user state which contains scene_state, internal_state, and relationship_state
            with db._get_connection() as conn:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    f'SELECT scene_state, internal_state, relationship_state FROM {T.USER_STATE} WHERE email = %s',
                    (email,)
                )
                row = cursor.fetchone()

            state_data = {
                'scene_state': row.get('scene_state') if row else None,
                'internal_state': row.get('internal_state') if row else None,
                'relationship_state': row.get('relationship_state') if row else None
            }

            emit('state', state_data)
        except Exception as e:
            logger.error(f"Error fetching state: {e}")
            emit('error', {'message': 'Failed to fetch state'})

    @socketio.on('request_images')
    def handle_request_images(data=None):
        """
        Handle image list request from client - returns recent generated images.
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            limit = 20
            if data and isinstance(data, dict):
                limit = min(data.get('limit', 20), 100)

            from src.database.db import get_db
            db = get_db()

            with db._get_connection() as conn:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(f'''
                    SELECT task_id, prompt, workflow_type, status,
                           cloudinary_url, runcomfy_url, requested_at
                    FROM {T.IMAGE_GENERATION_REQUESTS}
                    WHERE email = %s
                    ORDER BY requested_at DESC
                    LIMIT %s
                ''', (email, limit))
                rows = cursor.fetchall()

            images = []
            for row in rows:
                images.append({
                    'task_id': row.get('task_id'),
                    'prompt': row.get('prompt'),
                    'workflow_type': row.get('workflow_type'),
                    'status': row.get('status'),
                    'url': row.get('cloudinary_url') or row.get('runcomfy_url'),
                    'created_at': row.get('requested_at').isoformat() if row.get('requested_at') else None
                })

            emit('images', {'images': images})
        except Exception as e:
            logger.error(f"Error fetching images: {e}")
            emit('error', {'message': 'Failed to fetch images'})

    @socketio.on('request_activity')
    def handle_request_activity(data=None):
        """
        Handle activity request from client - returns current and upcoming activities.
        This is an info command, doesn't affect conversation.
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.scheduling.companion_schedule import get_companion_schedule
            from datetime import datetime

            schedule = get_companion_schedule()
            activity_status = schedule.get_current_activity_status()

            # Get upcoming activities from companion_autonomous_tasks table
            db = get_db()
            activities = []
            try:
                with db._get_connection() as conn:
                    from psycopg2.extras import RealDictCursor
                    cursor = conn.cursor(cursor_factory=RealDictCursor)

                    # Get activities for today
                    cursor.execute(f'''
                        SELECT task_name, category, scheduled_at, started_at, completed_at,
                               status, description, base_duration_minutes
                        FROM {T.COMPANION_AUTONOMOUS_TASKS}
                        WHERE (user_id = %s OR user_id IS NULL)
                          AND (scheduled_at >= CURRENT_DATE OR started_at >= CURRENT_DATE)
                          AND scheduled_at < CURRENT_DATE + INTERVAL '1 day'
                        ORDER BY COALESCE(scheduled_at, started_at) ASC
                        LIMIT 10
                    ''', (email,))
                    activities = cursor.fetchall()
            except Exception as db_err:
                logger.warning(f"Could not fetch activities from DB: {db_err}")

            activity_data = {
                'current_status': activity_status,
                'activities': [
                    {
                        'name': a.get('task_name'),
                        'type': a.get('category'),
                        'start': a.get('scheduled_at').isoformat() if a.get('scheduled_at') else (a.get('started_at').isoformat() if a.get('started_at') else None),
                        'end': a.get('completed_at').isoformat() if a.get('completed_at') else None,
                        'description': a.get('description'),
                        'duration_minutes': a.get('base_duration_minutes'),
                        'status': a.get('status')
                    }
                    for a in activities
                ] if activities else []
            }

            emit('activity', activity_data)
        except Exception as e:
            logger.error(f"Error fetching activity: {e}")
            emit('error', {'message': 'Failed to fetch activity'})

    @socketio.on('request_autopilot')
    def handle_request_autopilot(data=None):
        """
        Handle autopilot status request - returns James's current routine status.
        This is an info command, doesn't affect conversation.
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.core.user_context import get_user_probable_activity, get_energy_from_time_of_day, is_user_autopilot_enabled
            from datetime import datetime
            from zoneinfo import ZoneInfo

            tz = ZoneInfo('America/Los_Angeles')
            now = datetime.now(tz)
            activity = get_user_probable_activity(now)
            energy = get_energy_from_time_of_day(now)

            autopilot_data = {
                'time': now.strftime('%A %I:%M %p %Z'),
                'activity': activity['activity'],
                'location': activity['location'],
                'interruptibility': activity['interruptibility'],
                'natural_gap': activity['natural_gap'],
                'description': activity['description'],
                'energy': energy,
                'is_awake': activity.get('is_awake', False),
                'autopilot_enabled': is_user_autopilot_enabled()
            }

            emit('autopilot', autopilot_data)
        except Exception as e:
            logger.error(f"Error fetching autopilot status: {e}")
            emit('error', {'message': f'Failed to fetch autopilot status: {e}'})

    @socketio.on('toggle_user_autopilot')
    def handle_toggle_user_autopilot(data=None):
        """
        Toggle user autopilot (schedule assumptions) on/off.

        When OFF: companion won't assume "user is at work" etc.
        Useful during crises or unusual situations.

        data: {'enabled': true/false} or empty to toggle
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.core.user_context import is_user_autopilot_enabled, set_user_autopilot_enabled

            current = is_user_autopilot_enabled()

            if data and 'enabled' in data:
                new_state = bool(data['enabled'])
            else:
                new_state = not current

            set_user_autopilot_enabled(new_state)

            status_msg = "Schedule assumptions ON - the companion will use your normal routine" if new_state else "Schedule assumptions OFF - the companion won't assume what you're doing"

            emit('user_autopilot_status', {
                'enabled': new_state,
                'message': status_msg
            })

            logger.info(f"User autopilot toggled to {new_state} by {email}")

        except Exception as e:
            logger.error(f"Error toggling James autopilot: {e}")
            emit('error', {'message': f'Failed to toggle autopilot: {e}'})

    @socketio.on('toggle_autonomy')
    def handle_toggle_autonomy(data=None):
        """
        Toggle the companion's autonomy (proactive messaging) on/off.

        data: {'enabled': true/false} or empty to toggle
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.autonomy.reach_out_engine import is_autonomy_enabled, set_autonomy_enabled

            # Get current state
            current = is_autonomy_enabled()

            # Toggle or set explicitly
            if data and 'enabled' in data:
                new_state = bool(data['enabled'])
            else:
                new_state = not current

            set_autonomy_enabled(new_state)

            emit('autonomy_status', {
                'enabled': new_state,
                'message': f"Autonomy {'enabled' if new_state else 'disabled'}"
            })

            logger.info(f"Autonomy toggled to {new_state} by {email}")

        except Exception as e:
            logger.error(f"Error toggling autonomy: {e}")
            emit('error', {'message': f'Failed to toggle autonomy: {e}'})

    @socketio.on('request_autonomy_status')
    def handle_request_autonomy_status(data=None):
        """Get current autonomy status."""
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.autonomy.reach_out_engine import is_autonomy_enabled
            emit('autonomy_status', {'enabled': is_autonomy_enabled()})
        except Exception as e:
            logger.error(f"Error fetching autonomy status: {e}")
            emit('error', {'message': f'Failed to fetch autonomy status: {e}'})

    @socketio.on('set_presence_mode')
    def handle_set_presence_mode(data=None):
        """
        Set the presence mode (in_person vs texting).

        data: {'mode': 'in_person' | 'texting'}
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.core.presence_mode import (
                get_presence_mode_manager, PresenceMode
            )

            mode_str = (data or {}).get('mode', '')
            try:
                mode = PresenceMode(mode_str)
            except ValueError:
                emit('error', {'message': f'Invalid presence mode: {mode_str}. Use "in_person" or "texting".'})
                return

            manager = get_presence_mode_manager()
            success = manager.set_presence_mode(email, mode)

            if success:
                emit('presence_mode', {
                    'mode': mode.value,
                    'message': f"Presence mode set to {mode.value}"
                })
                logger.info(f"Presence mode set to {mode.value} by {email}")
            else:
                emit('error', {'message': 'Failed to save presence mode'})

        except Exception as e:
            logger.error(f"Error setting presence mode: {e}")
            emit('error', {'message': f'Failed to set presence mode: {e}'})

    @socketio.on('get_presence_mode')
    def handle_get_presence_mode(data=None):
        """Get the current presence mode."""
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.core.presence_mode import get_presence_mode_manager
            manager = get_presence_mode_manager()
            mode = manager.get_presence_mode(email)
            emit('presence_mode', {'mode': mode.value})
        except Exception as e:
            logger.error(f"Error fetching presence mode: {e}")
            emit('error', {'message': f'Failed to fetch presence mode: {e}'})

    @socketio.on('request_costs')
    def handle_request_costs(data=None):
        """
        Handle /costs command - returns API cost tracking information.
        Shows daily and weekly costs for tool calls and main responses.
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.core.cost_tracker import get_cost_tracker
            tracker = get_cost_tracker()

            # Get weekly summary
            weekly = tracker.get_weekly_summary()

            # Get recent calls
            recent = tracker.get_recent_calls(limit=5)

            cost_data = {
                'weekly_summary': {
                    'total_cost': f"${weekly['total_cost']:.4f}",
                    'total_calls': weekly['total_calls'],
                    'input_tokens': weekly['input_tokens'],
                    'output_tokens': weekly['output_tokens'],
                },
                'daily_breakdown': [
                    {
                        'date': day['date'],
                        'cost': f"${day.get('total_cost', 0):.4f}",
                        'calls': day.get('total_calls', 0)
                    }
                    for day in weekly.get('daily', [])
                ],
                'recent_calls': [
                    {
                        'time': call['timestamp'].split('T')[1][:8] if 'T' in call.get('timestamp', '') else '',
                        'model': call.get('model', 'unknown'),
                        'purpose': call.get('purpose', 'unknown'),
                        'cost': f"${call.get('cost_usd', 0):.6f}"
                    }
                    for call in recent
                ]
            }

            emit('costs', cost_data)
        except Exception as e:
            logger.error(f"Error fetching costs: {e}")
            emit('error', {'message': f'Failed to fetch costs: {e}'})

    @socketio.on('say_message')
    def handle_say_message(data):
        """
        Handle /say command - store message without immediate response.

        After random delay (30s-3min), the companion might naturally break the silence
        with a response based on the conversation context.

        Data should contain:
            - message: The user's silent message
        """
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        message = data.get('message', '').strip()
        if not message:
            emit('error', {'message': 'Empty message'})
            return

        try:
            # Get user's display name
            db = get_db()
            user_profile = db.get_profile(email)
            user_display_name = user_profile.get('display_name', email.split('@')[0])

            print(f"[{email}] ({user_display_name}) Silent message (/say): {message}")

            # Store message in conversation history (user's silent message only)
            db.store_message(
                email=email,
                sender_name=user_display_name,
                message_text=message,
                source='chat',
                message_type='normal'
            )

            # Get conversation context for delayed response decision
            recent_messages = db.get_recent_messages(email, limit=10)
            conversation_context = "\n".join([
                f"{msg['sender_name']}: {msg['message_text']}"
                for msg in recent_messages
            ])

            # Get current state
            state = _get_default_state(email)
            closeness_score = state.get('closeness_score', 50)
            emotion_profile = state.get('emotion_profile', 'Caring, Trusting, and Personal')

            # Acknowledge receipt
            emit('say_acknowledged', {
                'message': message,
                'timestamp': now_pacific_naive().isoformat()
            })

            # Queue background task for potential delayed response
            try:
                from celery_app import app as celery_app

                # Use Celery to handle delayed response decision
                from tasks.chat_tasks import generate_delayed_response

                # Schedule for later (randomized delay between 30s and 3min)
                import random
                delay_seconds = random.randint(30, 180)

                celery_app.send_task(
                    'tasks.chat_tasks.generate_delayed_response',
                    args=[email, request.sid, conversation_context, closeness_score],
                    countdown=delay_seconds
                )

                print(f"📅 Scheduled delayed response for {email} in {delay_seconds}s")
            except Exception as e:
                print(f"⚠️  Could not queue delayed response: {e}")

        except Exception as e:
            print(f"❌ Error handling /say message: {e}")
            import traceback
            traceback.print_exc()
            emit('error', {'message': 'Failed to process silent message'})

    # ========================================================================
    # FACT APPROVAL WEBSOCKET HANDLERS
    # ========================================================================

    @socketio.on('request_pending_facts')
    def handle_request_pending_facts(data=None):
        """Get pending facts awaiting approval."""
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        try:
            from src.memory.fact_approval import get_approval_service
            service = get_approval_service()
            pending = service.get_pending_facts(user_email=email, limit=50)

            # Convert datetime fields to strings for JSON serialization
            for fact in pending:
                for key, value in fact.items():
                    if hasattr(value, 'isoformat'):
                        fact[key] = value.isoformat()

            emit('pending_facts', {
                'count': len(pending),
                'facts': pending
            })
        except Exception as e:
            logger.error(f"Error fetching pending facts: {e}")
            emit('error', {'message': 'Failed to fetch pending facts'})

    @socketio.on('approve_fact')
    def handle_approve_fact(data):
        """Approve a pending fact."""
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        fact_id = data.get('fact_id')
        if not fact_id:
            emit('error', {'message': 'fact_id required'})
            return

        try:
            from src.memory.fact_approval import get_approval_service
            service = get_approval_service()
            success = service.approve_fact(fact_id, reviewed_by=email)

            if success:
                emit('fact_approved', {'fact_id': fact_id, 'status': 'approved'})
            else:
                emit('error', {'message': f'Failed to approve fact {fact_id}'})
        except Exception as e:
            logger.error(f"Error approving fact: {e}")
            emit('error', {'message': str(e)})

    @socketio.on('reject_fact')
    def handle_reject_fact(data):
        """Reject a pending fact."""
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        fact_id = data.get('fact_id')
        reason = data.get('reason', '')

        if not fact_id:
            emit('error', {'message': 'fact_id required'})
            return

        try:
            from src.memory.fact_approval import get_approval_service
            service = get_approval_service()
            success = service.reject_fact(fact_id, reviewed_by=email, reason=reason)

            if success:
                emit('fact_rejected', {'fact_id': fact_id, 'status': 'rejected'})
            else:
                emit('error', {'message': f'Failed to reject fact {fact_id}'})
        except Exception as e:
            logger.error(f"Error rejecting fact: {e}")
            emit('error', {'message': str(e)})

    @socketio.on('edit_fact')
    def handle_edit_fact(data):
        """Edit and approve a pending fact."""
        email = _get_user_email(request.sid)
        if not email:
            emit('error', {'message': 'Not authenticated'})
            return

        fact_id = data.get('fact_id')
        new_text = data.get('fact_text')

        if not fact_id or not new_text:
            emit('error', {'message': 'fact_id and fact_text required'})
            return

        try:
            from src.memory.fact_approval import get_approval_service
            service = get_approval_service()
            success = service.edit_and_approve_fact(fact_id, new_text, reviewed_by=email)

            if success:
                emit('fact_edited', {
                    'fact_id': fact_id,
                    'status': 'edited',
                    'new_text': new_text
                })
            else:
                emit('error', {'message': f'Failed to edit fact {fact_id}'})
        except Exception as e:
            logger.error(f"Error editing fact: {e}")
            emit('error', {'message': str(e)})

    # ========================================================================
    # REDIS PUB/SUB LISTENER FOR FACT APPROVAL REQUESTS
    # Celery workers publish to 'approval_requests:*' when new facts need
    # user review. This greenlet listens and forwards to the correct socket.
    # ========================================================================
    def start_approval_listener():
        """Listen for approval requests from Celery and forward to WebSocket."""
        import eventlet
        import redis
        import json

        def listener():
            try:
                redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
                r = redis.from_url(redis_url)
                pubsub = r.pubsub()
                pubsub.psubscribe('approval_requests:*')

                logger.info("Started approval request listener")

                for message in pubsub.listen():
                    if message['type'] == 'pmessage':
                        try:
                            data = json.loads(message['data'])
                            user_email = data.get('user_email')

                            # Find connected socket for this user
                            for sid, email in connected_users.items():
                                if email == user_email:
                                    socketio.emit('fact_approval_request', data, room=sid)
                                    logger.info(f"Forwarded approval request to {email}")
                                    break
                        except Exception as e:
                            logger.error(f"Error processing approval message: {e}")
            except Exception as e:
                logger.error(f"Approval listener error: {e}")

        eventlet.spawn(listener)

    # Start the listener when handlers are registered
    try:
        start_approval_listener()
    except Exception as e:
        logger.warning(f"Could not start approval listener: {e}")


# ============================================================================
# HTTP CHAT ENDPOINTS
# ============================================================================

@chat_bp.route('/api/chat', methods=['POST'])
def http_chat():
    """
    HTTP chat endpoint for simulation full-stack mode.

    Requires the SIMULATION_API_SECRET env var to be set and a matching
    X-Simulation-Secret header on every request. Returns 403 if the secret
    is missing or does not match, and 501 if the env var is not configured.

    Expects JSON body:
        - email: User email (required)
        - message: Message text (required)
        - companion_id: Companion ID for response generation (optional)
        - conversation_id: Conversation ID (optional)

    Returns JSON:
        - response: Companion's full response text
        - messages: Split messages (if multi-message)
        - timestamp: Response timestamp
        - processing_time: Time taken to generate response
    """
    from flask import jsonify

    # Authenticate via shared secret — prevents unauthenticated access
    expected_secret = os.environ.get('SIMULATION_API_SECRET')
    if not expected_secret:
        return jsonify({'error': 'Simulation API not configured'}), 501

    provided_secret = request.headers.get('X-Simulation-Secret', '')
    if provided_secret != expected_secret:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    email = data.get('email', '').strip()
    message = data.get('message', '').strip()
    if not email or not message:
        return jsonify({'error': 'email and message are required'}), 400

    processor = get_message_processor()
    result = processor.process_message(data, email, sid='http')

    if not result:
        return jsonify({'error': 'Failed to process message'}), 500

    if 'error' in result:
        return jsonify({'error': result['error']}), 500

    return jsonify(result), 200


@chat_bp.route('/api/history', methods=['GET'])
def get_history():
    """
    Get chat history for authenticated user.

    Query params:
        - limit: Number of messages to return (default: 50)
        - offset: Starting position (default: 0)

    Authentication: Bearer token in Authorization header or ?token= query param
    Supports both Firebase ID tokens and legacy session tokens
    """
    try:
        from src.database.simple_auth import verify_session
        from src.auth.auth_provider import verify_token as verify_auth_token

        # Try Authorization header first (standard REST pattern), then query param (fallback)
        token = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]  # Remove "Bearer " prefix
        else:
            token = request.args.get('token')

        if not token:
            return {'error': 'No session token provided'}, 401

        # Try configured auth provider first (Firebase or Cognito)
        user_info = verify_auth_token(token)
        if not user_info:
            # Fall back to legacy session token
            user_info = verify_session(token)

        if not user_info:
            return {'error': 'Invalid or expired session'}, 401

        email = user_info.get('email')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        db = get_db()
        raw_messages = db.get_messages_paginated(email, limit=limit, offset=offset)

        # Transform messages to match frontend Message type
        messages = []
        for msg in raw_messages:
            transformed = {
                'role': msg['sender_name'],  # companion name, user name, or other sender names
                'content': msg['message_text'],
                'timestamp': msg['timestamp'],
                'message_id': str(msg['id'])
            }
            messages.append(transformed)

        # Reverse to show oldest first (bottom) to newest last (top)
        messages.reverse()

        return {
            'messages': messages,
            'count': len(messages),
            'limit': limit,
            'offset': offset
        }

    except Exception as e:
        print(f"Error fetching history: {e}")
        return {'error': str(e)}, 500
