"""Add quality_score column to per-user messages tables.

quality_score (FLOAT, nullable) stores the ConversationPipeline's computed
quality score for companion responses so it can be analysed offline.

Revision ID: 003
Revises: 002
Create Date: 2026-06-01
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _user_schemas(connection) -> list[str]:
    """Return all schemas whose names begin with 'user_'."""
    result = connection.execute(
        text(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name LIKE 'user_%' ORDER BY schema_name"
        )
    )
    return [row[0] for row in result]


def upgrade() -> None:
    connection = op.get_bind()
    for schema in _user_schemas(connection):
        connection.execute(
            text(
                f'ALTER TABLE "{schema}".messages '
                f"ADD COLUMN IF NOT EXISTS quality_score FLOAT"
            )
        )


def downgrade() -> None:
    connection = op.get_bind()
    for schema in _user_schemas(connection):
        connection.execute(
            text(
                f'ALTER TABLE "{schema}".messages '
                f"DROP COLUMN IF EXISTS quality_score"
            )
        )
