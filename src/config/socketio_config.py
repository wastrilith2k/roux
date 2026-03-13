"""
Flask-SocketIO Configuration

Centralized SocketIO configuration including CORS, ping timeouts, and message queuing.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class SocketIOConfig:
    """
    Flask-SocketIO configuration.

    All SocketIO settings should be configured here rather than scattered
    throughout web_chat.py.
    """

    # CORS configuration for WebSocket
    CORS_ALLOWED_ORIGINS = os.getenv('SOCKETIO_CORS_ORIGINS', '*').split(',')
    CORS_CREDENTIALS = os.getenv('SOCKETIO_CORS_CREDENTIALS', True)

    # Connection management
    PING_TIMEOUT = int(os.getenv('SOCKETIO_PING_TIMEOUT', 60))
    PING_INTERVAL = int(os.getenv('SOCKETIO_PING_INTERVAL', 25))
    MAX_CONNECTIONS = int(os.getenv('SOCKETIO_MAX_CONNECTIONS', 1000))

    # Message queue (for production - uses Redis)
    MESSAGE_QUEUE = os.getenv('SOCKETIO_MESSAGE_QUEUE', 'redis://localhost:6379/0')
    QUEUE_PREFIX = 'socketio:'

    # Async mode
    # Using 'threading' as default since gevent isn't installed
    # eventlet is installed but causes Werkzeug assertion errors during disconnect
    # Threading is more compatible and sufficient for typical workloads
    ASYNC_MODE = os.getenv('SOCKETIO_ASYNC_MODE', 'threading')  # Can be: threading, eventlet, gevent

    # Logging
    LOGGER = os.getenv('SOCKETIO_LOGGER', True)
    ENGINEIO_LOGGER = os.getenv('SOCKETIO_ENGINEIO_LOGGER', False)

    # Namespaces to enable
    NAMESPACES = ['/']

    # Custom event settings
    EVENT_TIMEOUT = 30  # Timeout for event handlers (seconds)
    MAX_RECONNECTION_ATTEMPTS = 5
    RECONNECTION_DELAY = 1000  # Milliseconds
    RECONNECTION_DELAY_MAX = 5000  # Milliseconds

    @classmethod
    def get_kwargs(cls):
        """
        Get SocketIO initialization kwargs.

        Returns a dict suitable for passing to SocketIO(app, **kwargs)

        Returns:
            Dict with SocketIO configuration
        """
        return {
            'cors_allowed_origins': cls.CORS_ALLOWED_ORIGINS,
            'cors_credentials': cls.CORS_CREDENTIALS,
            'ping_timeout': cls.PING_TIMEOUT,
            'ping_interval': cls.PING_INTERVAL,
            'async_mode': cls.ASYNC_MODE,
            'logger': cls.LOGGER,
            'engineio_logger': cls.ENGINEIO_LOGGER,
        }

    @classmethod
    def get_client_config(cls):
        """
        Get SocketIO client-side configuration.

        This should be passed to the client for proper connection behavior.

        Returns:
            Dict with client configuration
        """
        return {
            'reconnection': True,
            'reconnectionDelay': cls.RECONNECTION_DELAY,
            'reconnectionDelayMax': cls.RECONNECTION_DELAY_MAX,
            'reconnectionAttempts': cls.MAX_RECONNECTION_ATTEMPTS,
            'transports': ['websocket', 'polling'],  # Fallback to polling if WebSocket unavailable
        }

    @classmethod
    def apply_to_socketio(cls, socketio):
        """
        Apply configuration to SocketIO instance.

        Usage:
            from flask_socketio import SocketIO
            from config.socketio_config import SocketIOConfig

            socketio = SocketIO(app, **SocketIOConfig.get_kwargs())
            SocketIOConfig.apply_to_socketio(socketio)

        Args:
            socketio: SocketIO instance
        """
        # Store config for access during runtime
        socketio.config = {
            'max_connections': cls.MAX_CONNECTIONS,
            'event_timeout': cls.EVENT_TIMEOUT,
            'namespaces': cls.NAMESPACES,
        }
        return socketio
