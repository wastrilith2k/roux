"""Alembic migration environment.

Reads database configuration from environment variables (POSTGRES_*) and runs
migrations against the public schema.  After public-schema migrations complete,
iterates all user schemas and ensures they are up to date via ensure_user_schema().

This environment uses raw SQL migrations (op.execute()) because the project uses
psycopg2 directly rather than the SQLAlchemy ORM — autogenerate is not used.
"""

import os
import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text
from alembic import context

# ---------------------------------------------------------------------------
# Alembic Config object — provides access to alembic.ini values
# ---------------------------------------------------------------------------
config = context.config

# Wire up Python logging from alembic.ini [loggers] section.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# ---------------------------------------------------------------------------
# Build the database URL from environment variables.
# We override whatever is in alembic.ini so that the actual credentials always
# come from the environment, never from a checked-in config file.
# ---------------------------------------------------------------------------


def _get_db_url() -> str:
    """Construct a SQLAlchemy connection URL from POSTGRES_* env vars."""
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "companion")
    user = os.environ.get("POSTGRES_USER", "companion")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    if not password:
        raise ValueError(
            "POSTGRES_PASSWORD environment variable is required for migrations."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_all_user_emails(connection) -> list[str]:
    """Return the email for every row in the public users table.

    Returns an empty list if the table does not yet exist (e.g., before the
    initial migration has run).
    """
    try:
        result = connection.execute(text("SELECT email FROM public.users"))
        return [row[0] for row in result]
    except Exception:
        return []


def _apply_user_schemas(connection) -> None:
    """Ensure every registered user's private schema exists and is current."""
    try:
        from src.database.schema_manager import ensure_user_schema
    except ImportError:
        logger.warning(
            "src.database.schema_manager not importable — skipping user schema sync."
        )
        return

    emails = _get_all_user_emails(connection)
    if not emails:
        logger.info("No user emails found — skipping user schema sync.")
        return

    # ensure_user_schema expects a psycopg2 connection; extract the raw
    # connection from the SQLAlchemy connection proxy.
    raw_conn = connection.connection
    for email in emails:
        try:
            schema = ensure_user_schema(raw_conn, email)
            logger.info(f"Ensured user schema: {schema}")
        except Exception as exc:
            logger.error(f"Failed to ensure schema for {email!r}: {exc}")


# ---------------------------------------------------------------------------
# Offline mode (alembic revision / show — no live DB needed)
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations without a live database connection.

    Emits SQL to stdout so it can be inspected or piped to psql.
    """
    url = _get_db_url()
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode (normal migration execution against a live DB)
# ---------------------------------------------------------------------------


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    url = _get_db_url()

    # Override the URL from alembic.ini with the env-var-derived one.
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = url

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None,
        )

        with context.begin_transaction():
            context.run_migrations()

        # After public schema migrations succeed, sync all user schemas.
        _apply_user_schemas(connection)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
