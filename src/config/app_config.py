"""
Flask Application Configuration — Companion Framework

WHAT: Centralized Flask app configuration for development, testing, and production.
WHY:  Keeps all Flask settings in one place rather than scattered across web_chat.py
      and various initialization functions. Environment-aware: FLASK_ENV selects the
      right config subclass automatically.
HOW:  AppConfig is the base class with shared defaults. DevelopmentConfig,
      TestingConfig, and ProductionConfig override security/debug settings.
      get_config() returns the right class based on FLASK_ENV.
      AppConfig.apply_to_app(app) copies all uppercase attributes into app.config.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Base configuration (shared across all environments)
# ---------------------------------------------------------------------------

class AppConfig:
    """Base Flask application configuration.

    All uppercase class attributes are automatically applied to the Flask
    app via apply_to_app(). Env vars provide runtime overrides.
    """

    # Flask core
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', False)
    TESTING = os.getenv('TESTING', False)

    # Session
    PERMANENT_SESSION_LIFETIME = 86400 * 7  # 7 days in seconds
    SESSION_TYPE = 'filesystem'
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', True)
    SESSION_COOKIE_HTTPONLY = os.getenv('SESSION_COOKIE_HTTPONLY', True)
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Uploads
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/companion-uploads')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp'}

    # CORS — comma-separated origins in env, defaults to allow-all
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    CORS_ALLOW_HEADERS = ['Content-Type', 'Authorization']
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', '/var/log/companion/app.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def apply_to_app(cls, app):
        """Copy all uppercase, non-callable attributes into app.config.

        Usage:
            app = Flask(__name__)
            AppConfig.apply_to_app(app)
        """
        for key in dir(cls):
            if key.isupper() and not key.startswith('_'):
                value = getattr(cls, key)
                if not callable(value):
                    app.config[key] = value
        return app


# ---------------------------------------------------------------------------
# Environment-specific overrides
# ---------------------------------------------------------------------------

class DevelopmentConfig(AppConfig):
    """Relaxed settings for local development."""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False  # No HTTPS on localhost


class TestingConfig(AppConfig):
    """Settings for automated test suites."""
    DEBUG = True
    TESTING = True
    SESSION_COOKIE_SECURE = False
    WTF_CSRF_ENABLED = False  # Disable CSRF for test convenience


class ProductionConfig(AppConfig):
    """Locked-down settings for production deployment."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True


# ---------------------------------------------------------------------------
# Config selector
# ---------------------------------------------------------------------------

def get_config():
    """Return the appropriate config class based on FLASK_ENV.

    Values: 'development', 'testing', 'production' (default).
    """
    env = os.getenv('FLASK_ENV', 'production')
    configs = {
        'development': DevelopmentConfig,
        'testing': TestingConfig,
    }
    return configs.get(env, ProductionConfig)
