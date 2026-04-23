#!/usr/bin/env bash
# Import exported demo data on the production server.
# Usage: ./scripts/import_demo_data.sh <dump_dir>
#
# Run after 'alembic upgrade head' and before starting the app.
# Note: If data already exists, psql will error on duplicate keys.
# This is acceptable — the most recent data will be used.

set -euo pipefail

DUMP_DIR="${1:?Usage: $0 <dump_dir>}"
DB_URL="${DATABASE_URL:?DATABASE_URL must be set}"

echo "Importing public schema..."
psql "$DB_URL" -f "$DUMP_DIR/public_schema.sql"

for SQL_FILE in "$DUMP_DIR"/user_*.sql; do
  [ -f "$SQL_FILE" ] || continue
  echo "Importing $SQL_FILE..."
  psql "$DB_URL" -f "$SQL_FILE"
done

echo "Import complete."
