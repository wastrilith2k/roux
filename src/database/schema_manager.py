"""
Per-user PostgreSQL schema management.

Each user gets their own schema (e.g., user_james) containing all companion
data tables. Shared tables (auth, user registry) remain in the public schema.

Usage:
    from src.database.schema_manager import schema_name_for_user, ensure_user_schema

    schema = schema_name_for_user(user_email)
    ensure_user_schema(conn, user_email)  # creates schema + tables if needed
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def schema_name_for_user(email: str) -> str:
    """Derive a PostgreSQL schema name from a user email.

    Takes the local part (before @), replaces non-alphanumeric chars with
    underscores, lowercases, and prefixes with 'user_'.

    Raises ValueError if email is empty.
    """
    if not email or not email.strip():
        raise ValueError("Email cannot be empty")

    local_part = email.split("@")[0].lower()
    sanitized = re.sub(r"[^a-z0-9]", "_", local_part)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")

    if not sanitized:
        raise ValueError(f"Email '{email}' produced empty schema name")

    return f"user_{sanitized}"


def ensure_user_schema(conn, email: str) -> str:
    """Create the user's schema and all tables if they don't exist.

    Returns the schema name. Safe to call multiple times (idempotent).
    """
    from src.database.schema_ddl import get_user_schema_ddl

    schema = schema_name_for_user(email)

    with conn.cursor() as cursor:
        # Use identifier quoting to prevent SQL injection
        cursor.execute(
            f"CREATE SCHEMA IF NOT EXISTS {_quote_ident(schema)}"
        )
        # Create all user-scoped tables within this schema
        ddl = get_user_schema_ddl(schema)
        cursor.execute(ddl)

    conn.commit()
    logger.info(f"Ensured schema '{schema}' exists with all tables")
    return schema


def set_search_path(conn, email: str) -> str:
    """Set the connection's search_path to the user's schema + public.

    Returns the schema name. Call this after authenticating the user,
    before executing any queries.
    """
    schema = schema_name_for_user(email)
    with conn.cursor() as cursor:
        cursor.execute(
            f"SET search_path TO {_quote_ident(schema)}, public"
        )
    return schema


def _quote_ident(identifier: str) -> str:
    """Quote a PostgreSQL identifier to prevent injection.

    Doubles any embedded double-quotes per SQL standard.
    """
    return '"' + identifier.replace('"', '""') + '"'
