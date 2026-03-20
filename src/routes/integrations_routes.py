"""
Integration Routes -- external service webhooks and asset serving.

WHAT: Inbound webhooks (Twilio SMS, Google Chat, n8n workflow results),
      proactive-message endpoint, mood-based avatar selection, image serving
      (local + Cloudinary CDN redirect), and companion schedule API.

WHY:  The companion is reachable through multiple channels beyond the primary
      WebSocket chat. Each channel's webhook normalises the inbound message
      and funnels it through the same MessageProcessor pipeline. The avatar
      and image routes power the frontend's visual display. The n8n handler
      provides a centralised callback for Google Workspace automation flows.

HOW:  All routes live on a single Blueprint (`integrations_bp`). Auth varies:
      SMS uses Twilio's request signing, Google Chat trusts the webhook secret,
      the proactive endpoint uses Bearer tokens, and the n8n handler has its
      own webhook-key decorator (currently a stub after n8n module removal).

NOTE: The eight `_handle_*_response` helpers for n8n are structurally identical.
      They could be collapsed into a single generic handler with a workflow-name
      map, but are kept separate for per-workflow logging clarity.
"""
from flask import Blueprint, request, jsonify, send_from_directory, redirect
from functools import wraps
from src.database.db import get_db
# Closeness system removed - always use high-trust state (100)
from src.utils.timezone_utils import now_pacific_naive
import logging
import os


def n8n_webhook_auth_required(f):
    """Stub decorator - n8n integration module was removed during cleanup.

    This decorator now just passes through without authentication.
    The n8n webhook routes should be reviewed and possibly removed.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.warning("n8n_webhook_auth_required: Auth check skipped (module removed)")
        return f(*args, **kwargs)
    return decorated_function

logger = logging.getLogger(__name__)

integrations_bp = Blueprint('integrations', __name__)


# ============================================================================
# MOOD-BASED AVATAR SELECTION
# ============================================================================

def _get_mood_based_avatar():
    """
    Select an avatar image based on the companion's current unified emotional state.

    Uses the unified mood resolver which harmonizes:
    1. EmotionalState (per-message moods)
    2. MoodPersistence (long-lasting event-based moods)
    3. AvatarPersonality (emotion-to-avatar mapping)

    Then uses avatar personality system to:
    1. Map mood to specific emotion preferences
    2. Find avatars matching those emotions
    3. Rotate through preferred avatars (maintaining personality while adding variety)
    4. Exclude recently-shown avatars to prevent repetition

    Returns:
        Avatar URL path (e.g., '/img/avatar_filename.png')
    """
    try:
        from src.core.avatar_personality import get_avatar_for_mood
        from src.core.mood_resolver import get_resolved_mood

        # Get unified mood from resolver (harmonizes all three mood systems)
        primary_mood, mood_info = get_resolved_mood()

        logger.info(
            f"🎭 Avatar mood: {primary_mood} (intensity: {mood_info['intensity']:.0%}, "
            f"sources: {', '.join(mood_info['sources'])}, category: {mood_info['category']})"
        )

        # Get avatar using personality system (handles emotion matching, rotation, recent tracking)
        # The avatar_for_mood function normalizes the mood to its avatar personality equivalent
        avatar_url = get_avatar_for_mood(primary_mood, exclude_recent=True)

        if avatar_url:
            # avatar_url is either a Cloudinary URL or a local filename
            # If it's already a full URL (starts with http), use as-is
            # Otherwise, prepend /img/ for local files
            if not avatar_url.startswith('http'):
                avatar_url = f"/img/{avatar_url}"
            return avatar_url
        else:
            logger.warning(f"⚠️ No avatar found for mood '{primary_mood}'")
            return None

    except Exception as e:
        logger.error(f"❌ Error selecting mood-based avatar: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ============================================================================
# SMS WEBHOOK
# ============================================================================

@integrations_bp.route('/sms/incoming', methods=['POST'])
def handle_incoming_sms():
    """
    Handle inbound SMS from Twilio webhook.

    Twilio sends:
    - From: sender's phone number
    - To: recipient phone number
    - Body: message text
    - MessageSid: unique message ID
    """
    try:
        from services.sms_service import get_sms_service
        from handlers.message_handler import get_message_processor

        # Get SMS data from webhook
        result = get_sms_service().process_inbound_sms(request.form.to_dict())

        if not result['success']:
            logger.warning(f"Invalid SMS webhook data")
            return jsonify({'error': 'Invalid message'}), 400

        phone_number = result['phone_number']
        message_text = result['message']

        # Find or create user by phone number
        # This requires a phone_number field in user profile
        db = get_db()
        user = db.find_user_by_phone(phone_number)

        if not user:
            logger.info(f"SMS from unknown number {phone_number}, creating entry")
            # Could create a guest user or store SMS for manual review
            return jsonify({'status': 'stored_for_review'}), 202

        email = user['email']

        # Process message through normal pipeline
        processor = get_message_processor()
        response = processor.process_message(
            {'message': message_text, 'message_type': 'sms'},
            email,
            f"sms_{phone_number}"
        )

        if response and 'response' in response:
            # Send response back via SMS
            sms_response = get_sms_service().send_sms(
                phone_number,
                response['response'][:160]  # SMS character limit
            )

            logger.info(f"SMS response sent to {phone_number}")

        return jsonify({'status': 'processed'}), 200

    except Exception as e:
        logger.error(f"Error handling SMS webhook: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# GOOGLE CHAT WEBHOOK
# ============================================================================

@integrations_bp.route('/api/google-chat-webhook', methods=['POST'])
def handle_google_chat_webhook():
    """
    Handle inbound Google Chat message via webhook.

    Google Chat sends JSON with message content and sender info.
    """
    try:
        from services.google_workspace_service import get_google_workspace_service
        from handlers.message_handler import get_message_processor

        webhook_data = request.get_json()

        if not webhook_data:
            return jsonify({'error': 'No data provided'}), 400

        # Process webhook
        result = get_google_workspace_service().process_google_chat_webhook(webhook_data)

        if not result['success']:
            logger.warning(f"Invalid Google Chat webhook")
            return jsonify({'error': 'Invalid message'}), 400

        user_email = result['user_email']
        message_text = result['message']

        # Process through message pipeline
        processor = get_message_processor()
        response = processor.process_message(
            {'message': message_text, 'message_type': 'google_chat'},
            user_email,
            f"google_chat_{user_email}"
        )

        # Return response to Google Chat
        if response and 'response' in response:
            return jsonify({
                'text': response['response'],
                'thread_ts': webhook_data.get('message', {}).get('name')
            })

        return jsonify({'text': 'I processed your message.'}), 200

    except Exception as e:
        logger.error(f"Error handling Google Chat webhook: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PROACTIVE MESSAGES
# ============================================================================

@integrations_bp.route('/api/proactive', methods=['POST'])
def send_proactive_message():
    """
    Send a proactive message from the companion to user.

    Requires authentication. Used when the companion wants to initiate conversation
    (e.g., responding after silence, sharing something important).

    Body:
    {
        "message": "Message text",
        "urgency": "normal|high|low",
        "context": "optional context about why sending"
    }
    """
    try:
        from middleware.auth_middleware import token_required, get_auth_info

        auth_info = get_auth_info()
        if not auth_info:
            return jsonify({'error': 'Not authenticated'}), 401

        email = auth_info['email']
        data = request.get_json()

        if not data or not data.get('message'):
            return jsonify({'error': 'Message required'}), 400

        message = data['message']
        urgency = data.get('urgency', 'normal')

        db = get_db()

        # Store companion's proactive message
        from src.config.persona_config import get_persona_config
        message_id = db.store_message(
            email=email,
            sender_name=get_persona_config().companion_short_name,
            message_text=message,
            source='proactive',
            message_type='normal'
        )

        logger.info(f"Proactive message sent to {email} (urgency: {urgency})")

        # Queue notification to user via WebSocket or push
        try:
            from flask_socketio import socketio
            socketio.emit('proactive_message', {
                'message': message,
                'urgency': urgency,
                'timestamp': now_pacific_naive().isoformat()
            }, room=email)
        except Exception as e:
            logger.warning(f"Could not send WebSocket notification: {e}")

        return jsonify({
            'success': True,
            'message_id': message_id,
            'delivered': True
        }), 200

    except Exception as e:
        logger.error(f"Error sending proactive message: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# AVATAR AND IMAGE SERVING
# ============================================================================

@integrations_bp.route('/api/avatar', methods=['GET', 'POST'])
def manage_avatar():
    """
    Get or upload user's avatar image.

    GET: Returns avatar for logged-in user
    POST: Upload new avatar

    Supports both Firebase ID tokens and legacy session tokens
    """
    try:
        from src.database.simple_auth import verify_session
        from src.auth.auth_provider import verify_token as verify_auth_token

        # Get token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Not authenticated'}), 401

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Try configured auth provider first (Firebase or Cognito)
        user_info = verify_auth_token(token)
        if not user_info:
            # Fall back to legacy session token
            user_info = verify_session(token)

        if not user_info:
            return jsonify({'error': 'Invalid or expired session'}), 401

        email = user_info.get('email')
        db = get_db()

        if request.method == 'GET':
            # Get the companion's avatar based on current mood
            avatar_url = _get_mood_based_avatar()

            return jsonify({
                'avatar': avatar_url,
                'email': 'companion'
            }), 200

        elif request.method == 'POST':
            # Upload new avatar
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']

            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400

            # Validate file type
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
            if not any(file.filename.lower().endswith(f'.{ext}') for ext in allowed_extensions):
                return jsonify({'error': 'Invalid file type'}), 400

            # Save file
            import uuid
            filename = f"avatar_{email}_{uuid.uuid4().hex}.png"
            upload_folder = os.path.join(os.path.dirname(__file__), '../public/avatars')

            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            # Update user profile
            avatar_url = f"/avatars/{filename}"
            db.update_profile(email, {'avatar_url': avatar_url})

            logger.info(f"Avatar uploaded for {email}: {filename}")

            return jsonify({
                'success': True,
                'avatar_url': avatar_url
            }), 200

    except Exception as e:
        logger.error(f"Error managing avatar: {e}")
        return jsonify({'error': str(e)}), 500


@integrations_bp.route('/img/<path:filename>')
def serve_image(filename):
    """
    Serve uploaded images (generated, avatars, etc).

    Supports both local and Cloudinary URLs:
    - Checks if avatar has cloudinary_url in avatar_emotions.json
    - If yes, redirects to Cloudinary CDN (faster, more efficient)
    - If no, falls back to local file serving

    Args:
        filename: Image filename to serve
    """
    try:
        # First, check if this is an avatar with a Cloudinary URL
        try:
            import json
            avatar_metadata_path = os.path.join(os.path.dirname(__file__), '../../avatar_emotions.json')
            with open(avatar_metadata_path, 'r') as f:
                avatar_data = json.load(f)

            # Search for this filename in the avatars list
            for avatar in avatar_data.get('avatars', []):
                if avatar.get('filename') == filename:
                    # Found matching avatar
                    if avatar.get('cloudinary_url'):
                        # Redirect to Cloudinary URL (better for CDN delivery)
                        logger.debug(f"Redirecting to Cloudinary URL for: {filename}")
                        return redirect(avatar['cloudinary_url'], code=302)
                    break
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # If we can't load avatar metadata, just fall through to local serving
            logger.debug(f"Could not load avatar metadata, will serve locally: {filename}")

        # Fall back to serving locally from project img/ directory
        upload_folder = os.path.join(os.path.dirname(__file__), '../../img')
        upload_folder = os.path.abspath(upload_folder)  # Convert to absolute path
        logger.debug(f"Serving image from: {upload_folder}, filename: {filename}")

        # Verify file exists before trying to serve
        file_path = os.path.join(upload_folder, filename)
        if not os.path.exists(file_path):
            logger.error(f"Image file not found at: {file_path}")
            return jsonify({'error': 'Image not found'}), 404

        return send_from_directory(upload_folder, filename)

    except FileNotFoundError:
        logger.error(f"Image not found: {filename} in {upload_folder}")
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to serve image'}), 500


# ============================================================================
# COMPANION SCHEDULE
# ============================================================================

@integrations_bp.route('/api/companion-schedule', methods=['GET'])
def get_companion_schedule_endpoint():
    """
    Get the companion's schedule and availability status.

    Query Parameters:
    - days_offset: Number of days from today (0 = today, 1 = tomorrow, -1 = yesterday)

    Returns:
    - is_asleep: Whether the companion is currently sleeping (only relevant for today)
    - is_available: Whether the companion is available for chat
    - schedule: Schedule with work hours, tasks, meetings
    - current_status: Description of what the companion is currently doing
    - current_time: Current time in the companion's timezone
    """
    try:
        from src.scheduling.companion_schedule import get_companion_schedule
        from datetime import datetime, timedelta

        # Get days_offset parameter (default 0 = today)
        days_offset = int(request.args.get('days_offset', 0))

        schedule_mgr = get_companion_schedule()
        now = now_pacific_naive()
        target_date = now + timedelta(days=days_offset)

        # Closeness system removed - always use high-trust (100)
        closeness = 100

        # Get schedule for the target date
        daily_schedule = schedule_mgr.generate_daily_schedule(target_date)

        # Get day info
        day_name = target_date.strftime('%A')

        # Get work hours from preferences
        prefs = schedule_mgr.schedules.get('work_preferences', {})

        # For days other than today, don't include real-time availability
        if days_offset == 0:
            # Check current availability with closeness score
            availability_result = schedule_mgr.is_available_for_chat(closeness, target_date)
            is_available = availability_result[0] if isinstance(availability_result, tuple) else availability_result
            availability_reason = availability_result[1] if isinstance(availability_result, tuple) else "unknown"
            is_working_now = schedule_mgr.is_working_now(target_date)
            is_on_lunch = schedule_mgr.is_on_lunch(target_date)
            current_status = schedule_mgr.get_current_status()
            is_asleep = schedule_mgr.is_asleep(target_date)
        else:
            # For future/past days, show availability as if it's work hours
            is_available = True
            availability_reason = "not working" if not daily_schedule['is_workday'] else "working"
            is_working_now = False  # Not relevant for other days
            is_on_lunch = False
            current_status = schedule_mgr.get_schedule_description(target_date)
            is_asleep = False

        return jsonify({
            'is_asleep': is_asleep,
            'is_available': is_available,
            'availability_reason': availability_reason,
            'is_working_now': is_working_now,
            'is_on_lunch': is_on_lunch,
            'current_status': current_status,
            'current_time': now.isoformat(),
            'schedule': daily_schedule
        }), 200

    except Exception as e:
        logger.error(f"Error fetching companion schedule: {e}")
        import traceback
        traceback.print_exc()
        from datetime import datetime
        now = now_pacific_naive()
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][now.weekday()]

        return jsonify({
            'is_asleep': False,
            'is_available': True,
            'availability_reason': 'error',
            'is_working_now': False,
            'is_on_lunch': False,
            'current_status': 'Error loading schedule',
            'current_time': now.isoformat(),
            'schedule': {
                'day_name': day_name,
                'date': now.strftime('%Y-%m-%d'),
                'is_workday': True,
                'work_start': '09:00',
                'work_end': '17:00',
                'tasks': [],
                'meetings': [],
                'notes': 'Schedule error'
            }
        }), 200


# ============================================================================
# N8N WORKFLOW RESPONSE HANDLER
# ============================================================================

@integrations_bp.route('/webhook/n8n-response', methods=['POST'])
@n8n_webhook_auth_required
def handle_n8n_response():
    """
    Handle response from completed n8n workflows.

    All n8n workflows POST their results to this centralized endpoint:
    POST /webhook/n8n-response

    This provides a single point of authentication and response routing.

    Expected payload:
    {
        "workflow": "workflow-name",  # e.g., "gmail-search", "calendar-create"
        "results": {...}              # Workflow-specific result data
    }

    Authentication:
    - Requires Authorization: Bearer <N8N_INCOMING_WEBHOOK_AUTH_KEY> header
    - Validates against env var N8N_INCOMING_WEBHOOK_AUTH_KEY
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        workflow_name = data.get('workflow')
        results = data.get('results', data.get('data', {}))  # Handle both "results" and "data" keys

        if not workflow_name:
            logger.warning("⚠️  n8n response missing workflow name")
            return jsonify({'error': 'Missing workflow name'}), 400

        # Strip quotes if n8n sent them as JSON string
        if isinstance(workflow_name, str) and workflow_name.startswith('"') and workflow_name.endswith('"'):
            workflow_name = workflow_name[1:-1]

        logger.info(f"✅ Received n8n response from workflow: {workflow_name}")
        logger.debug(f"   Results: {results}")

        # Route results based on workflow type
        if workflow_name == 'gmail-search':
            return _handle_gmail_search_response(results)
        elif workflow_name == 'gmail-send':
            return _handle_gmail_send_response(results)
        elif workflow_name == 'calendar-list':
            return _handle_calendar_list_response(results)
        elif workflow_name == 'calendar-create':
            return _handle_calendar_create_response(results)
        elif workflow_name == 'drive-list':
            return _handle_drive_list_response(results)
        elif workflow_name == 'sheets-read':
            return _handle_sheets_read_response(results)
        elif workflow_name == 'docs-read':
            return _handle_docs_read_response(results)
        elif workflow_name == 'create-doc':
            return _handle_create_doc_response(results)
        else:
            # Generic handler for unknown workflows
            logger.warning(f"⚠️  Unknown workflow type: {workflow_name}")
            return jsonify({
                'success': True,
                'workflow': workflow_name,
                'message': 'Response received and logged'
            }), 200

    except Exception as e:
        logger.error(f"❌ Error handling n8n response: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


def _handle_gmail_search_response(results):
    """Handle Gmail search workflow results."""
    try:
        logger.info(f"📧 Gmail search raw results: {results}")
        logger.info(f"📧 Results type: {type(results)}")
        logger.info(f"📧 Results keys: {results.keys() if isinstance(results, dict) else 'N/A'}")

        messages = results.get('messages', []) if isinstance(results, dict) else results
        logger.info(f"📧 Gmail search returned {len(messages) if isinstance(messages, list) else 0} results")

        # Store in database for later retrieval or emit via WebSocket
        # For now, just log and acknowledge
        return jsonify({
            'success': True,
            'workflow': 'gmail-search',
            'message_count': len(messages) if isinstance(messages, list) else 0
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing Gmail search response: {e}")
        return jsonify({'error': str(e)}), 500


def _handle_gmail_send_response(results):
    """Handle Gmail send workflow results."""
    try:
        message_id = results.get('messageId') if isinstance(results, dict) else results
        logger.info(f"📧 Gmail send completed: {message_id}")
        return jsonify({
            'success': True,
            'workflow': 'gmail-send',
            'message_id': message_id
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing Gmail send response: {e}")
        return jsonify({'error': str(e)}), 500


def _handle_calendar_list_response(results):
    """Handle Calendar list events workflow results."""
    try:
        events = results.get('events', []) if isinstance(results, dict) else results
        logger.info(f"📅 Calendar list returned {len(events) if isinstance(events, list) else 0} events")
        return jsonify({
            'success': True,
            'workflow': 'calendar-list',
            'event_count': len(events) if isinstance(events, list) else 0
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing Calendar list response: {e}")
        return jsonify({'error': str(e)}), 500


def _handle_calendar_create_response(results):
    """Handle Calendar create event workflow results."""
    try:
        event_id = results.get('eventId') if isinstance(results, dict) else results
        logger.info(f"📅 Calendar event created: {event_id}")
        return jsonify({
            'success': True,
            'workflow': 'calendar-create',
            'event_id': event_id
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing Calendar create response: {e}")
        return jsonify({'error': str(e)}), 500


def _handle_drive_list_response(results):
    """Handle Drive list files workflow results."""
    try:
        files = results.get('files', []) if isinstance(results, dict) else results
        logger.info(f"📁 Drive list returned {len(files) if isinstance(files, list) else 0} files")
        return jsonify({
            'success': True,
            'workflow': 'drive-list',
            'file_count': len(files) if isinstance(files, list) else 0
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing Drive list response: {e}")
        return jsonify({'error': str(e)}), 500


def _handle_sheets_read_response(results):
    """Handle Sheets read data workflow results."""
    try:
        data = results.get('values', []) if isinstance(results, dict) else results
        logger.info(f"📊 Sheets read returned data")
        return jsonify({
            'success': True,
            'workflow': 'sheets-read',
            'row_count': len(data) if isinstance(data, list) else 0
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing Sheets read response: {e}")
        return jsonify({'error': str(e)}), 500


def _handle_docs_read_response(results):
    """Handle Docs read document workflow results."""
    try:
        content = results.get('body', '') if isinstance(results, dict) else results
        logger.info(f"📝 Docs read completed")
        return jsonify({
            'success': True,
            'workflow': 'docs-read',
            'content_length': len(str(content))
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing Docs read response: {e}")
        return jsonify({'error': str(e)}), 500


def _handle_create_doc_response(results):
    """Handle create-doc workflow results."""
    try:
        document_id = results.get('document_id') if isinstance(results, dict) else results
        logger.info(f"📝 Google Doc created: {document_id}")
        return jsonify({
            'success': True,
            'workflow': 'create-doc',
            'document_id': document_id
        }), 200
    except Exception as e:
        logger.error(f"❌ Error processing create-doc response: {e}")
        return jsonify({'error': str(e)}), 500
