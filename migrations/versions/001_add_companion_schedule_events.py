"""Add companion_schedule_events table.

Internal-first schedule storage: events are generated and stored in PostgreSQL
as the primary source. Google Calendar sync is optional and secondary.

Revision ID: 001
Revises:
Create Date: 2026-03-22
"""
from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS companion_schedule_events (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            summary TEXT NOT NULL,
            start_time TIME NOT NULL,
            end_time TIME NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT 'personal',
            status TEXT DEFAULT 'planned',
            google_event_id TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_schedule_events_date
            ON companion_schedule_events (date);

        CREATE INDEX IF NOT EXISTS idx_schedule_events_date_status
            ON companion_schedule_events (date, status);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS companion_schedule_events")
