"""
Companion ownership management.

Tracks which users own which companions via the public.user_companions table.
This table lives in the public schema since it's shared across all user schemas.
"""
import logging
from src.database import tables as T
from src.database.connection import get_connection
from src.database.schema_manager import ensure_user_schema

logger = logging.getLogger(__name__)


def user_owns_companion(db, email: str, companion_id: str) -> bool:
    """Check if a user owns a specific companion.

    Args:
        db: CompanionDB instance
        email: User's email
        companion_id: Companion identifier

    Returns:
        True if the user owns the companion
    """
    result = db.execute(
        f"SELECT 1 FROM public.{T.USER_COMPANIONS} WHERE user_email = %s AND companion_id = %s",
        (email, companion_id)
    )
    return result.fetchone() is not None


def assign_companion(db, email: str, companion_id: str, display_name: str = None):
    """Assign a companion to a user. Idempotent (no-op if already assigned).

    Args:
        db: CompanionDB instance
        email: User's email
        companion_id: Companion identifier
        display_name: Optional display name for the companion
    """
    db.execute(
        f"""INSERT INTO public.{T.USER_COMPANIONS} (user_email, companion_id, display_name)
            VALUES (%s, %s, %s) ON CONFLICT (user_email, companion_id) DO NOTHING""",
        (email, companion_id, display_name)
    )
    logger.info(f"Assigned companion '{companion_id}' to user '{email}'")


def get_user_companions(db, email: str) -> list:
    """Get all companions owned by a user.

    Returns:
        List of dicts with companion_id and display_name
    """
    result = db.execute(
        f"SELECT companion_id, display_name, created_at FROM public.{T.USER_COMPANIONS} WHERE user_email = %s ORDER BY created_at",
        (email,)
    )
    return result.fetchall()


def revoke_companion(db, email: str, companion_id: str):
    """Remove a companion from a user's ownership."""
    db.execute(
        f"DELETE FROM public.{T.USER_COMPANIONS} WHERE user_email = %s AND companion_id = %s",
        (email, companion_id)
    )
    logger.info(f"Revoked companion '{companion_id}' from user '{email}'")


def companion_email_for(companion_id: str) -> str:
    """Canonical email for a companion's data schema."""
    return f"{companion_id}@companion.local"


def provision_new_companion(db, user_email: str, companion_id: str) -> str:
    """Create schema + ownership record for a new user's companion.

    Called once on registration. Idempotent — safe to retry.

    Returns the companion_id.
    """
    c_email = companion_email_for(companion_id)

    # Create the companion's schema (all user-scoped tables)
    conn = get_connection()
    try:
        ensure_user_schema(conn, c_email)
    finally:
        conn.close()

    # Record ownership
    assign_companion(db, user_email, companion_id, display_name=companion_id.capitalize())

    logger.info(f"Provisioned companion '{companion_id}' for user '{user_email}'")
    return companion_id
