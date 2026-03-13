"""
Authentication Middleware

Handles token validation, session management, and authentication decorators.
Extracted from web_chat.py for centralized auth logic.
"""
from functools import wraps
from flask import request, jsonify, session
from src.database.simple_auth import verify_session
import logging

logger = logging.getLogger(__name__)


def token_required(f):
    """
    Decorator for routes that require authentication via token.

    Validates the token in the Authorization header or query params.
    Adds 'username' to request context if valid.

    Usage:
        @app.route('/api/protected')
        @token_required
        def protected_route():
            username = request.username
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Try Authorization header first
        if 'Authorization' in request.headers:
            try:
                auth_header = request.headers['Authorization']
                token = auth_header.split(' ')[1]  # "Bearer <token>"
            except IndexError:
                return jsonify({'error': 'Invalid Authorization header'}), 401

        # Fall back to query parameter or session
        if not token:
            token = request.args.get('token')

        if not token:
            token = session.get('token')

        if not token:
            return jsonify({'error': 'No authentication token provided'}), 401

        try:
            user_info = verify_session(token)
            if not user_info:
                return jsonify({'error': 'Invalid or expired token'}), 401

            # Attach user info to request for use in route
            request.user_email = user_info.get('email')
            request.user_id = user_info.get('id')
            request.username = user_info.get('username')

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401

    return decorated


def session_required(f):
    """
    Decorator for routes that require session authentication.

    Checks Flask session for authentication.
    Adds username to request context if valid.

    Usage:
        @app.route('/api/dashboard')
        @session_required
        def dashboard():
            username = request.username
            ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'error': 'Not authenticated'}), 401

        request.username = session['username']
        request.user_email = session.get('email')

        return f(*args, **kwargs)

    return decorated


def optional_auth(f):
    """
    Decorator for routes that work with or without authentication.

    Validates token if provided, but doesn't fail if absent.
    Adds username to request context only if authenticated.

    Usage:
        @app.route('/api/public')
        @optional_auth
        def public_route():
            if hasattr(request, 'username'):
                # User is authenticated
                ...
            else:
                # User is not authenticated
                ...
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Try Authorization header
        if 'Authorization' in request.headers:
            try:
                auth_header = request.headers['Authorization']
                token = auth_header.split(' ')[1]
            except IndexError:
                pass

        # Try query parameter or session
        if not token:
            token = request.args.get('token')

        if not token:
            token = session.get('token')

        # If we have a token, try to validate it
        if token:
            try:
                user_info = verify_session(token)
                if user_info:
                    request.user_email = user_info.get('email')
                    request.user_id = user_info.get('id')
                    request.username = user_info.get('username')
                    logger.debug(f"Optional auth: User {request.username} authenticated")
            except Exception as e:
                logger.debug(f"Optional auth: Token validation failed: {e}")
                # Not authenticated, but that's OK

        return f(*args, **kwargs)

    return decorated


class AuthContext:
    """
    Context manager for handling authentication in a code block.

    Usage:
        with AuthContext(token) as auth:
            if auth.is_valid():
                username = auth.username
                ...
    """

    def __init__(self, token: str):
        """
        Initialize auth context with token.

        Args:
            token: Session token to validate
        """
        self.token = token
        self.user_info = None
        self.valid = False

    def __enter__(self):
        """Validate token on context entry"""
        try:
            if self.token:
                self.user_info = verify_session(self.token)
                self.valid = bool(self.user_info)
            else:
                self.valid = False
        except Exception as e:
            logger.debug(f"Auth context error: {e}")
            self.valid = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        pass

    def is_valid(self) -> bool:
        """Check if authentication is valid"""
        return self.valid

    @property
    def email(self) -> str:
        """Get authenticated user's email"""
        return self.user_info.get('email') if self.user_info else None

    @property
    def username(self) -> str:
        """Get authenticated user's username"""
        return self.user_info.get('username') if self.user_info else None

    @property
    def user_id(self) -> str:
        """Get authenticated user's ID"""
        return self.user_info.get('id') if self.user_info else None


def get_auth_info():
    """
    Get authentication info from current request context.

    Returns:
        Dict with email, username, user_id, or None if not authenticated
    """
    if hasattr(request, 'user_email'):
        return {
            'email': request.user_email,
            'username': request.username,
            'user_id': request.user_id
        }

    if 'email' in session:
        return {
            'email': session.get('email'),
            'username': session.get('username'),
            'user_id': session.get('user_id')
        }

    return None
