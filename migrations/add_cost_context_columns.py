"""Migration: add call context columns to LLM cost tracking tables.

Adds call_purpose, conversation_id, message_id, and companion_id to
openrouter_usage, fireworks_usage, and openai_usage.

Safe to run on existing databases — checks for column existence before
altering. All new columns are nullable or have defaults, so no data loss.

Usage:
    python -m migrations.add_cost_context_columns
    # or
    python migrations/add_cost_context_columns.py
"""
import os
import sqlite3
import sys


# Columns to add: (column_name, column_definition)
NEW_COLUMNS = [
    ("call_purpose", "TEXT DEFAULT 'conversation'"),
    ("conversation_id", "TEXT"),
    ("message_id", "TEXT"),
    ("companion_id", "TEXT"),
]

# Tables that receive the new columns
TABLES = ["openrouter_usage", "fireworks_usage", "openai_usage"]


def _get_existing_columns(cursor: sqlite3.Cursor, table: str) -> set:
    """Return the set of column names for a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def migrate(db_path: str) -> None:
    """Add call-context columns to cost tracking tables (idempotent)."""
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path} — skipping migration.")
        return

    conn = sqlite3.connect(db_path, timeout=10.0)
    cursor = conn.cursor()

    for table in TABLES:
        # Verify table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cursor.fetchone():
            print(f"  Table {table} does not exist — skipping.")
            continue

        existing = _get_existing_columns(cursor, table)

        for col_name, col_def in NEW_COLUMNS:
            if col_name in existing:
                continue
            stmt = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"
            cursor.execute(stmt)
            print(f"  Added {table}.{col_name}")

    conn.commit()
    conn.close()
    print("Migration complete.")


def _resolve_db_path() -> str:
    """Resolve the cost-tracking SQLite database path."""
    if os.path.exists("/app/data"):
        return "/app/data/companion.db"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data", "companion.db")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else _resolve_db_path()
    print(f"Migrating cost tracking database: {path}")
    migrate(path)
