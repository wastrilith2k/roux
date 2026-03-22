"""Alembic migration environment.

Reads database configuration from environment variables (POSTGRES_*) and runs
migrations against the public schema.
"""

import os
import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text
from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")


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


def run_migrations_offline() -> None:
    """Run migrations without a live database connection."""
    url = _get_db_url()
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    url = _get_db_url()
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


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
