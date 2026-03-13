"""
Logging Configuration — Companion Framework

WHAT: Sets up structured logging with both console and rotating-file output.
WHY:  The framework has 25+ subsystems running concurrently (Flask, Celery, SocketIO,
      background schedulers). Without centralized logging config, each module would
      configure its own handler, leading to duplicated or missing output.
HOW:  LoggingConfig.setup_logging() is called once during app initialization. It
      configures the root logger with a console handler and a rotating file handler,
      then sets custom log levels for noisy third-party libraries. Individual modules
      just use logging.getLogger(__name__).
"""

import logging
import logging.handlers
import os
from dotenv import load_dotenv

load_dotenv()


class LoggingConfig:
    """Centralized logging configuration.

    Controls log levels, formats, file rotation, and per-module overrides
    via environment variables.
    """

    # --- Log level and file path (overridable via env) ---
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', '/var/log/companion/app.log')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10485760))  # 10 MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 10))

    # --- Format strings (detailed for files, compact for console) ---
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    SIMPLE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    CONSOLE_FORMAT = '%(levelname)s - %(name)s - %(message)s'

    # --- Suppress noisy third-party loggers ---
    # Without these, Flask/werkzeug/engineio flood the logs on every request.
    CUSTOM_LEVELS = {
        'flask': 'WARNING',
        'werkzeug': 'WARNING',
        'socketio': 'INFO',
        'engineio': 'WARNING',
        'celery': 'INFO',
        'redis': 'WARNING',
    }

    @classmethod
    def setup_logging(cls, app=None):
        """Configure the root logger, file handler, and optional Flask app logger.

        Should be called once, early in app initialization:
            from src.config.logging_config import setup_logging
            setup_logging(app)

        Args:
            app: Optional Flask app instance. If provided, its logger is
                 reconfigured to use our handlers (replacing Flask's default).
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(cls.LOG_LEVEL)

        # Clear existing handlers to prevent duplicates on reload
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Formatters
        detailed_formatter = logging.Formatter(cls.DETAILED_FORMAT)
        console_formatter = logging.Formatter(cls.CONSOLE_FORMAT)

        # Console handler (always available)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(cls.LOG_LEVEL)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Rotating file handler (graceful failure if dir is not writable)
        file_handler = None
        try:
            log_dir = os.path.dirname(cls.LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                cls.LOG_FILE,
                maxBytes=cls.LOG_MAX_BYTES,
                backupCount=cls.LOG_BACKUP_COUNT
            )
            file_handler.setLevel(cls.LOG_LEVEL)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")

        # Quiet down third-party libraries
        for module_name, level in cls.CUSTOM_LEVELS.items():
            logging.getLogger(module_name).setLevel(level)

        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized (level: {cls.LOG_LEVEL}, file: {cls.LOG_FILE})")

        # Wire Flask's logger to our handlers (replaces Flask's default handler)
        if app:
            app.logger.setLevel(cls.LOG_LEVEL)
            for handler in app.logger.handlers[:]:
                app.logger.removeHandler(handler)
            app.logger.addHandler(console_handler)
            if file_handler:
                app.logger.addHandler(file_handler)

        return root_logger

    @classmethod
    def get_logger(cls, name):
        """Convenience: get a logger with the configured level.

        Usage:
            logger = LoggingConfig.get_logger(__name__)
        """
        logger = logging.getLogger(name)
        logger.setLevel(cls.LOG_LEVEL)
        return logger


# ---------------------------------------------------------------------------
# Module-level convenience (most callers just want this)
# ---------------------------------------------------------------------------

def setup_logging(app=None):
    """Shorthand for LoggingConfig.setup_logging(app)."""
    return LoggingConfig.setup_logging(app)
