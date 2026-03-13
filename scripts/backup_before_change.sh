#!/bin/bash
#
# Complete backup before making any code changes
# Creates: PostgreSQL dump, Neo4j backup, personality snapshot, git commit
#

set -e  # Exit on error

# Detect environment (production or dev)
if [ -d "/app" ]; then
    BACKUP_DIR="/app/backups"
else
    BACKUP_DIR="$HOME/projs/companion_backups"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pre_change_${TIMESTAMP}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FULL BACKUP: ${BACKUP_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# 1. PostgreSQL Backup
echo "📊 Backing up PostgreSQL..."
docker exec companion-postgres pg_dump -U companion companion > "${BACKUP_DIR}/${BACKUP_NAME}/postgres_dump.sql" 2>/dev/null || {
    echo "⚠️  PostgreSQL backup failed (container may not be running), continuing..."
    touch "${BACKUP_DIR}/${BACKUP_NAME}/postgres_dump.sql"
}
if [ -s "${BACKUP_DIR}/${BACKUP_NAME}/postgres_dump.sql" ]; then
    echo "✅ PostgreSQL backup complete: $(du -h ${BACKUP_DIR}/${BACKUP_NAME}/postgres_dump.sql | cut -f1)"
else
    echo "⚠️  PostgreSQL backup is empty"
fi

# 2. Neo4j Backup (export to JSON)
echo "🧠 Backing up Neo4j knowledge graph..."
# Use venv python if available
if [ -f ".venv/bin/python3" ]; then
    .venv/bin/python3 scripts/backup_neo4j.py "${BACKUP_DIR}/${BACKUP_NAME}/neo4j_backup.json" || {
        echo "⚠️  Neo4j backup failed, continuing..."
        echo '{"nodes": [], "relationships": []}' > "${BACKUP_DIR}/${BACKUP_NAME}/neo4j_backup.json"
    }
else
    python3 scripts/backup_neo4j.py "${BACKUP_DIR}/${BACKUP_NAME}/neo4j_backup.json" || {
        echo "⚠️  Neo4j backup failed, continuing..."
        echo '{"nodes": [], "relationships": []}' > "${BACKUP_DIR}/${BACKUP_NAME}/neo4j_backup.json"
    }
fi

if [ -s "${BACKUP_DIR}/${BACKUP_NAME}/neo4j_backup.json" ]; then
    echo "✅ Neo4j backup complete: $(du -h ${BACKUP_DIR}/${BACKUP_NAME}/neo4j_backup.json | cut -f1)"
else
    echo "⚠️  Neo4j backup is empty"
fi

# 3. State Files Backup
echo "💾 Backing up state files..."
if [ -d "/data/companion/state" ]; then
    cp -r /data/companion/state "${BACKUP_DIR}/${BACKUP_NAME}/state_files"
    echo "✅ State files backed up"
else
    echo "⚠️  /data/companion/state not found, creating empty backup"
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/state_files"
fi

# 4. Personality Snapshot
echo "🎭 Creating personality snapshot..."
# Use venv python if available
if [ -f ".venv/bin/python3" ]; then
    .venv/bin/python3 scripts/snapshot_personality_state.py > "${BACKUP_DIR}/${BACKUP_NAME}/personality_snapshot.txt" 2>&1
else
    python3 scripts/snapshot_personality_state.py > "${BACKUP_DIR}/${BACKUP_NAME}/personality_snapshot.txt" 2>&1
fi
echo "✅ Personality snapshot created"

# 5. Git commit (if changes exist)
echo "📝 Creating backup commit..."
cd /home/james/projs/companion-framework

if [[ -n $(git status -s) ]]; then
    git add -A
    git commit -F "${BACKUP_DIR}/${BACKUP_NAME}/personality_snapshot.txt" || {
        echo "⚠️  Git commit failed or no changes to commit"
    }
    COMMIT_HASH=$(git rev-parse HEAD)
    echo "✅ Backup commit created: ${COMMIT_HASH}"
    echo "${COMMIT_HASH}" > "${BACKUP_DIR}/${BACKUP_NAME}/git_commit_hash.txt"
else
    echo "ℹ️  No git changes to commit"
    git rev-parse HEAD > "${BACKUP_DIR}/${BACKUP_NAME}/git_commit_hash.txt"
fi

# 6. Create backup metadata
cat > "${BACKUP_DIR}/${BACKUP_NAME}/backup_metadata.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "backup_name": "${BACKUP_NAME}",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'N/A')",
  "postgres_size": "$(stat -c%s ${BACKUP_DIR}/${BACKUP_NAME}/postgres_dump.sql 2>/dev/null || echo '0')",
  "neo4j_size": "$(stat -c%s ${BACKUP_DIR}/${BACKUP_NAME}/neo4j_backup.json 2>/dev/null || echo '0')",
  "backup_path": "${BACKUP_DIR}/${BACKUP_NAME}"
}
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ BACKUP COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Backup Location: ${BACKUP_DIR}/${BACKUP_NAME}"
echo "Contents:"
ls -lh "${BACKUP_DIR}/${BACKUP_NAME}/"
echo ""
echo "To restore this backup:"
echo "  bash scripts/restore_backup.sh ${BACKUP_NAME}"
echo ""
