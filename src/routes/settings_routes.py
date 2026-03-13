"""
Settings and Profile Routes -- user profile CRUD and admin personality management.

WHAT: Endpoints for reading/updating user display name, romance toggle, closeness
      stats, and a deprecated admin personality endpoint that now redirects to
      YAML entity profiles.

WHY:  The web frontend needs a way to persist user preferences (display name,
      romance toggle) and show relationship metrics. The admin personality
      endpoint is kept for backward compatibility but all personality config
      now lives in data/entity_profiles/*.yaml.

HOW:  Uses UserProfile (src.user.user_profile) for display-name storage and
      legacy session-token auth via `token_required` decorator.

NOTE: `_save_state` is a no-op -- closeness is now read-only from the DB.
      `_get_default_state` is duplicated in chat_routes.py; a shared helper
      would be cleaner but the coupling risk isn't worth it yet.
"""

import os
import sys

# Legacy path manipulation for bare imports; see cost_routes.py note.
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(src_dir, 'user'))
sys.path.insert(0, os.path.join(src_dir, 'core'))
sys.path.insert(0, src_dir)

from flask import Blueprint, request, jsonify
from src.database.simple_auth import token_required
from src.user.user_profile import UserProfile


# ============================================================================
# HELPERS
# ============================================================================

def _get_default_state(email: str = None):
    """Return relationship state defaults, pulling closeness from the DB if possible."""
    closeness = 50
    if email:
        try:
            from src.database.db import get_db
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

def _save_state(state, username):
    """No-op stub -- closeness state saving was removed with Neo4j."""
    pass

# Create blueprint
settings_bp = Blueprint('settings', __name__, url_prefix='/api')


@settings_bp.route('/profile', methods=['GET', 'POST'])
@token_required
def user_profile():
    """Get or update user profile (display name)"""
    email = request.user_email
    profile_manager = UserProfile()

    if request.method == 'POST':
        # Update display name
        data = request.get_json()
        display_name = data.get('display_name', '').strip()

        if not display_name:
            return jsonify({'error': 'Display name cannot be empty'}), 400

        profile_manager.set_display_name(email, display_name)
        profile = profile_manager.get_profile(email)
        return jsonify(profile)

    else:
        # Get profile
        profile = profile_manager.get_profile(email)
        profile_manager.update_last_seen(email)
        return jsonify(profile)


@settings_bp.route('/admin/personality', methods=['GET', 'POST', 'DELETE'])
@token_required
def manage_personality():
    """
    Admin-only: Manage the companion's personality facts.

    DEPRECATED: Neo4j has been removed. Personality facts are now managed
    via YAML entity profiles in data/entity_profiles/.

    This endpoint now returns a message directing admins to edit the YAML files.
    """
    email = request.user_email
    profile_manager = UserProfile()

    # Check admin status
    if not profile_manager.is_admin(email):
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Only admins can modify the companion\'s personality'
        }), 403

    return jsonify({
        'message': 'Personality management has moved to YAML profiles.',
        'instructions': 'Edit data/entity_profiles/ YAML files to modify the companion\'s personality.',
        'facts': [],
        'count': 0
    })


@settings_bp.route('/settings/<username>', methods=['GET'])
@token_required
def get_settings(username):
    """Get user settings (display name, romance toggle, closeness stats)"""
    email = request.user_email

    # Security check: users can only access their own settings
    if email != username:
        return jsonify({'error': 'Unauthorized'}), 403

    profile_manager = UserProfile()

    # Get display name from profile
    profile = profile_manager.get_profile(email)
    display_name = profile.get('display_name', '')

    # Get state
    state = _get_default_state(email)
    romance_enabled = state.get('romance_enabled', False)
    closeness_score = state.get('closeness_score', 15)
    emotion_profile = state.get('emotion_profile', 'Guarded')
    attraction_cue_count = state.get('attraction_cue_count', 0)
    attraction_latent_state = state.get('attraction_latent_state', False)

    return jsonify({
        'display_name': display_name,
        'romance_enabled': romance_enabled,
        'closeness_score': closeness_score,
        'emotion_profile': emotion_profile,
        'attraction_cue_count': attraction_cue_count,
        'attraction_latent_state': attraction_latent_state
    })


@settings_bp.route('/settings', methods=['POST'])
@token_required
def update_settings():
    """Update user settings (display name, romance toggle)"""
    email = request.user_email
    data = request.get_json()

    requested_username = data.get('username', '')

    # Security check: users can only update their own settings
    if email != requested_username:
        return jsonify({'error': 'Unauthorized'}), 403

    # Update display name
    display_name = data.get('display_name', '').strip()
    if display_name:
        profile_manager = UserProfile()
        profile_manager.set_display_name(email, display_name)

    # Update romance toggle
    romance_enabled = data.get('romance_enabled', False)
    state = _get_default_state(email)
    state['romance_enabled'] = romance_enabled

    # Romance level is preserved when disabled (just paused)
    # It will resume from the same level if re-enabled

    _save_state(state, email)

    return jsonify({
        'success': True,
        'display_name': display_name,
        'romance_enabled': romance_enabled
    })
