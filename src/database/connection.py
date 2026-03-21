"""
Centralized PostgreSQL connection parameters.

Every module that needs a direct psycopg2 connection should call
get_connection_params() instead of reading env vars and hardcoding defaults.
This ensures consistent configuration and no leaked default passwords.

For most use cases, prefer src.database.db.get_db() instead — it provides
managed connections with cursor wrapping. Use this module only when you need
a raw psycopg2 connection (e.g., in modules that can't import the full DB layer).
"""

import os


def get_connection_params() -> dict:
    """Return psycopg2.connect() keyword arguments from environment.

    Raises ValueError if POSTGRES_PASSWORD is not set.
    """
    password = os.environ.get('POSTGRES_PASSWORD', '')
    if not password:
        raise ValueError(
            "POSTGRES_PASSWORD environment variable is required. "
            "Set it in .env or your environment."
        )
    return {
        'host': os.environ.get('POSTGRES_HOST', 'postgres'),
        'port': os.environ.get('POSTGRES_PORT', '5432'),
        'dbname': os.environ.get('POSTGRES_DB', 'companion'),
        'user': os.environ.get('POSTGRES_USER', 'companion'),
        'password': password,
    }


def get_connection():
    """Create and return a new psycopg2 connection using environment config."""
    import psycopg2
    return psycopg2.connect(**get_connection_params())


def get_user_connection(email: str):
    """Create connection with search_path set to user's schema."""
    conn = get_connection()
    from src.database.schema_manager import set_search_path
    set_search_path(conn, email)
    return conn
