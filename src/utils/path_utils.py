"""
Path Utilities -- environment-aware file path resolution.

WHAT: Helper functions for resolving data directory, config file, and user data
      paths that work in both Docker (/app/data) and local development
      ({project_root}/data) environments.

WHY:  The codebase runs in Docker (production) and directly on the host
      (development). Hardcoded paths would break one or the other. These
      helpers detect the environment and return the right path.

HOW:  `get_data_dir()` checks for /app/data (Docker) first, falls back to
      calculating project_root from this file's location. Module-level
      constants DATA_DIR and USERS_DIR are computed at import time for
      convenience.
"""
import os


def get_data_dir() -> str:
    """
    Get the data directory path (environment-aware)

    Returns:
        - /app/data if running in Docker
        - {project_root}/data if running on host
    """
    if os.path.exists('/app/data'):
        return '/app/data'
    else:
        # Calculate project root relative to this file
        # src/utils/path_utils.py -> go up 2 levels to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, 'data')


def get_data_file(filename: str) -> str:
    """
    Get full path to a file in the data directory

    Args:
        filename: Name of the file (e.g., 'companion.db', 'users.json')

    Returns:
        Full path to the file
    """
    return os.path.join(get_data_dir(), filename)


def get_user_data_dir(username: str = None) -> str:
    """
    Get the user data directory path

    Args:
        username: Optional username for user-specific directory

    Returns:
        Path to users directory or specific user directory
    """
    base_dir = os.path.join(get_data_dir(), 'users')

    if username:
        return os.path.join(base_dir, username)
    else:
        return base_dir


def get_config_file(filename: str) -> str:
    """
    Get full path to a config file

    Args:
        filename: Name of the config file (e.g., 'mcp_config.json')

    Returns:
        Full path to the config file
    """
    if os.path.exists('/app'):
        # Docker environment
        return f'/app/{filename}'
    else:
        # Host environment - config files are in project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, filename)


def ensure_dir_exists(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


# Commonly used paths
DATA_DIR = get_data_dir()
USERS_DIR = get_user_data_dir()
