"""Initial public schema tables.

Creates all shared tables in the public schema:
  - users
  - sessions
  - user_profiles
  - user_companions
  - service_links

User-private schemas are created on demand by ensure_user_schema(), which is
called from migrations/env.py after this migration runs.

Revision ID: 001
Revises:
Create Date: 2026-03-21
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all public schema tables."""
    from src.database.schema_ddl import get_public_schema_ddl

    op.execute(get_public_schema_ddl())


def downgrade() -> None:
    """Drop all public schema tables in reverse dependency order."""
    # service_links has no dependencies
    op.execute("DROP TABLE IF EXISTS public.service_links")
    # user_companions depends on user_profiles
    op.execute("DROP TABLE IF EXISTS public.user_companions")
    # user_profiles has no FK dependencies
    op.execute("DROP TABLE IF EXISTS public.user_profiles")
    # sessions depends on users
    op.execute("DROP TABLE IF EXISTS public.sessions")
    # users — drop last
    op.execute("DROP TABLE IF EXISTS public.users")
