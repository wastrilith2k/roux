"""
Global Error Handlers Middleware

Centralized error handling for the application.
Provides consistent error responses across all routes and WebSocket events.
"""
from flask import jsonify
import logging
import sys

# Setup logger
logger = logging.getLogger(__name__)

# Suppress harmless Werkzeug/eventlet assertion errors during disconnect
# This is a known issue where eventlet doesn't properly handle WSGI response
# finalization during WebSocket disconnects. The error doesn't affect functionality.
# Default is now 'threading' mode which avoids this, but filter for safety.
werkzeug_logger = logging.getLogger('werkzeug')

class WorkzeugAssertionFilter(logging.Filter):
    """Filter to suppress harmless Werkzeug assertion errors during disconnect"""
    def filter(self, record):
        msg = record.getMessage()
        # Suppress only the specific "write() before start_response" assertion error
        # This occurs with eventlet/WebSocket disconnect and doesn't affect functionality
        if "write() before start_response" in msg and "assert status_set is not None" in msg:
            return False
        # Also suppress the error traceback line that precedes it
        if "Error on request:" in msg and "assert status_set" in msg:
            return False
        return True

werkzeug_logger.addFilter(WorkzeugAssertionFilter())
# Set werkzeug to WARNING to reduce noise, but allow real errors through
werkzeug_logger.setLevel(logging.WARNING)


def register_error_handlers(app):
    """
    Register global error handlers with Flask app.

    Call from web_chat.py during app initialization:
        from middleware.error_handlers import register_error_handlers
        register_error_handlers(app)

    Args:
        app: Flask application instance
    """

    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        logger.warning(f"Bad request: {error}")
        return jsonify({
            'error': 'Bad request',
            'message': str(error.description) if hasattr(error, 'description') else str(error)
        }), 400

    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors"""
        logger.warning(f"Unauthorized access attempt")
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Invalid or expired credentials'
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors"""
        logger.warning(f"Forbidden access attempt: {error}")
        return jsonify({
            'error': 'Forbidden',
            'message': 'You do not have permission to access this resource'
        }), 403

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        logger.debug(f"Resource not found: {error}")
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found'
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed (common from internet scanners)"""
        logger.debug(f"Method not allowed: {error}")
        return jsonify({
            'error': 'Method not allowed',
            'message': 'The method is not allowed for the requested URL'
        }), 405

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle 429 Too Many Requests errors"""
        logger.warning(f"Rate limit exceeded")
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.'
        }), 429

    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error"""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred. Please try again later.'
        }), 500

    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable errors"""
        logger.error(f"Service unavailable: {error}")
        return jsonify({
            'error': 'Service unavailable',
            'message': 'The service is temporarily unavailable. Please try again later.'
        }), 503

    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle unexpected exceptions"""
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        return jsonify({
            'error': 'Server error',
            'message': 'An unexpected error occurred'
        }), 500


def register_socketio_error_handlers(socketio):
    """
    Register WebSocket-specific error handlers.

    Call from web_chat.py during app initialization:
        from middleware.error_handlers import register_socketio_error_handlers
        register_socketio_error_handlers(socketio)

    Args:
        socketio: Flask-SocketIO instance
    """

    @socketio.on_error_default
    def default_error_handler(e):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {e}", exc_info=True)
        return {
            'error': 'WebSocket error',
            'message': 'An error occurred during communication'
        }

    @socketio.on_error()
    def socketio_error_handler(e):
        """Handle specific SocketIO errors"""
        logger.error(f"SocketIO error: {e}", exc_info=True)
        return {
            'error': 'Communication error',
            'message': 'Failed to process request'
        }


def format_error_response(error_code: str, message: str, details=None):
    """
    Format a standard error response.

    Args:
        error_code: Error code (e.g., 'AUTH_FAILED', 'INVALID_INPUT')
        message: Human-readable error message
        details: Additional details (optional)

    Returns:
        Dict formatted for JSON response
    """
    response = {
        'error': error_code,
        'message': message
    }
    if details:
        response['details'] = details
    return response
