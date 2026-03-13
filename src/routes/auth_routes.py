"""
Authentication Routes

Handles user authentication, registration, and session management.
Supports both Firebase (primary) and legacy password-based auth (deprecated).
"""

import os
import sys

# Add src/ subdirectories to Python path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(src_dir, 'user'))
sys.path.insert(0, src_dir)

from flask import Blueprint, request, jsonify, session
from src.database.simple_auth import authenticate_user, create_user, logout_user, token_required, change_password
from src.auth.firebase_auth import (
    firebase_token_required,
    firebase_auth_optional,
    verify_firebase_token,
    get_firebase_user_info,
    initialize_firebase,
    create_custom_token
)

# Initialize Firebase on module load
initialize_firebase()

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


@auth_bp.route('/login', methods=['POST'])
def login():
    """Handle user login with SQLite auth."""
    data = request.json
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    # Use simple_auth for authentication
    result = authenticate_user(email, password)

    if result:
        print(f"✅ Login successful for user: {email}")
        return jsonify({
            'success': True,
            'user_id': result['user_id'],
            'email': result['email'],
            'session_token': result['session_token']
        })
    else:
        print(f"❌ Login failed for user: {email}")
        return jsonify({'error': 'Invalid credentials'}), 401


@auth_bp.route('/register', methods=['POST'])
def register():
    """Handle user registration with SQLite auth."""
    data = request.json
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    # Use simple_auth to create user
    if create_user(email, password):
        # Now authenticate to get session token
        result = authenticate_user(email, password)
        print(f"✅ Registration successful for user: {email}")
        return jsonify({
            'success': True,
            'user_id': result['user_id'],
            'email': result['email'],
            'session_token': result['session_token']
        })
    else:
        return jsonify({'error': 'Email already exists'}), 400


@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout_endpoint():
    """Handle user logout."""
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization', '')
    token = auth_header.replace('Bearer ', '').strip()

    if token:
        logout_user(token)

    return jsonify({'success': True})


@auth_bp.route('/status', methods=['GET'])
def auth_status():
    """Check if user is logged in (legacy endpoint for old session-based auth)."""
    from flask import session

    username = session.get('username')
    print(f"🔍 Auth status check - Session username: {username}, Session ID: {session.get('session_id')}")
    if 'username' in session:
        return jsonify({'logged_in': True, 'username': session['username']})
    else:
        return jsonify({'logged_in': False})


@auth_bp.route('/change-password', methods=['POST'])
@token_required
def change_password_endpoint():
    """Change user password"""
    email = request.user_email
    data = request.get_json()

    old_password = data.get('old_password', '')
    new_password = data.get('new_password', '')

    if not old_password or not new_password:
        return jsonify({'error': 'Both old and new passwords are required'}), 400

    if len(new_password) < 6:
        return jsonify({'error': 'New password must be at least 6 characters'}), 400

    if change_password(email, old_password, new_password):
        return jsonify({'success': True, 'message': 'Password changed successfully'})
    else:
        return jsonify({'error': 'Current password is incorrect'}), 401


# ========== Firebase Authentication Endpoints ==========

@auth_bp.route('/firebase/verify', methods=['POST'])
def firebase_verify():
    """Verify Firebase ID token and return user info.

    Request body: {"idToken": "<firebase_id_token>"}
    Response: User info if valid, error if not
    """
    data = request.json
    id_token = data.get('idToken')

    if not id_token:
        return jsonify({'error': 'idToken required'}), 400

    # Verify token
    claims = verify_firebase_token(id_token)
    if not claims:
        return jsonify({'error': 'Invalid or expired token'}), 401

    # Get full user info
    user_info = get_firebase_user_info(claims.get('uid'))
    if not user_info:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'success': True,
        'user': user_info,
        'email': user_info['email'],
        'uid': user_info['uid']
    })


@auth_bp.route('/firebase/user-info', methods=['GET'])
@firebase_token_required
def firebase_user_info():
    """Get authenticated user's info from Firebase token."""
    user_info = get_firebase_user_info(request.user_id)
    if not user_info:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'success': True,
        'user': user_info
    })


@auth_bp.route('/firebase/logout', methods=['POST'])
@firebase_token_required
def firebase_logout():
    """Logout Firebase user.

    Note: Firebase doesn't require server-side logout.
    Frontend should discard the ID token.
    """
    return jsonify({
        'success': True,
        'message': 'Firebase tokens should be discarded on client side'
    })
