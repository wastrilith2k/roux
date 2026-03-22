"""Add companion_work_projects table.

Persistent work projects that give the companion specific things to be
working on, excited about, or frustrated by. Projects evolve day-to-day
and are referenced in schedule generation and conversation.

Revision ID: 002
Revises: 001
Create Date: 2026-03-22
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS companion_work_projects (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            progress_pct INTEGER DEFAULT 0,
            feeling TEXT DEFAULT '',
            deadline TEXT DEFAULT '',
            last_worked DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_work_projects_status
            ON companion_work_projects (status);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS companion_work_projects")
