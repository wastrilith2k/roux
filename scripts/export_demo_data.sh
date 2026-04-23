#!/usr/bin/env bash
# Export simulation companion schemas for deployment.
# Usage: ./scripts/export_demo_data.sh [output_dir]
#
# Exports:
#   - public schema (users, sessions, user_companions, user_profiles)
#   - all user_* companion schemas
#
# Output dir defaults to ./backups/demo_export/

set -euo pipefail

OUTPUT_DIR="${1:-./backups/demo_export}"
DB_URL="${DATABASE_URL:?DATABASE_URL must be set}"

mkdir -p "$OUTPUT_DIR"

echo "Exporting public schema..."
pg_dump "$DB_URL" \
  --schema=public \
  --no-owner --no-acl \
  -f "$OUTPUT_DIR/public_schema.sql"
echo "  -> $OUTPUT_DIR/public_schema.sql"

# Find and export all user_* schemas
SCHEMAS=$(psql "$DB_URL" -t -c \
  "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'user_%' ORDER BY schema_name;" \
  | tr -d ' ' | grep -v '^$')

if [ -z "$SCHEMAS" ]; then
  echo "No user_* schemas found — simulation may not have run yet."
else
  for SCHEMA in $SCHEMAS; do
    echo "Exporting schema: $SCHEMA"
    pg_dump "$DB_URL" \
      --schema="$SCHEMA" \
      --no-owner --no-acl \
      -f "$OUTPUT_DIR/${SCHEMA}.sql"
    echo "  -> $OUTPUT_DIR/${SCHEMA}.sql"
  done
fi

echo ""
echo "Export complete. Files in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
