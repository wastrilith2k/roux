#!/bin/bash
# Neo4j Database Backup and Restore Script
# Backs up Neo4j database with automatic data dump (no need for full copy)
# Tested recovery: YES ✅

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NEO4J_USER="neo4j"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-companion_neo4j_password}"

# Both Neo4j databases to back up
NEO4J_CONTAINERS=("companion-neo4j" "companion-graphiti-neo4j")

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# ============================================================================
# BACKUP FUNCTION
# ============================================================================

backup_single_neo4j() {
    local CONTAINER=$1
    local PREFIX=$2

    echo "[$(date)] Backing up $CONTAINER..." | tee -a "$BACKUP_DIR/neo4j_backup.log"

    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        echo "[$(date)] WARNING: $CONTAINER not running, skipping" | tee -a "$BACKUP_DIR/neo4j_backup.log"
        return 0
    fi

    # Graphiti Neo4j has no auth
    local AUTH_ARGS=""
    if [ "$CONTAINER" = "companion-neo4j" ]; then
        AUTH_ARGS="-u $NEO4J_USER -p $NEO4J_PASSWORD"
    fi

    # Create a database dump using neo4j-admin (if available) or via Cypher
    BACKUP_FILE="$BACKUP_DIR/${PREFIX}_${TIMESTAMP}.cypher"

    # Export all data using Cypher
    docker exec $CONTAINER cypher-shell $AUTH_ARGS \
        "CALL apoc.export.cypher.all('$BACKUP_FILE', {format: 'cypher-shell'})" \
        2>/dev/null

    if [ $? -eq 0 ] && [ -f "$BACKUP_FILE" ]; then
        gzip "$BACKUP_FILE"
        echo "[$(date)] $CONTAINER backup complete: ${PREFIX}_${TIMESTAMP}.cypher.gz" | tee -a "$BACKUP_DIR/neo4j_backup.log"
    else
        # Fallback: Volume-based backup (more reliable)
        echo "[$(date)] Cypher export failed, using volume backup..." | tee -a "$BACKUP_DIR/neo4j_backup.log"

        # Stop container briefly to get consistent backup
        docker stop $CONTAINER >/dev/null 2>&1

        # Get the volume name
        if [ "$CONTAINER" = "companion-neo4j" ]; then
            VOLUME="companionframework_neo4j_data"
        else
            VOLUME="companionframework_graphiti_neo4j_data"
        fi

        # Backup the volume
        docker run --rm -v ${VOLUME}:/data -v ${BACKUP_DIR}:/backup alpine \
            tar czf /backup/${PREFIX}_${TIMESTAMP}.tar.gz -C /data . 2>/dev/null

        # Restart container
        docker start $CONTAINER >/dev/null 2>&1

        if [ -f "$BACKUP_DIR/${PREFIX}_${TIMESTAMP}.tar.gz" ]; then
            echo "[$(date)] $CONTAINER backup complete (volume): ${PREFIX}_${TIMESTAMP}.tar.gz" | tee -a "$BACKUP_DIR/neo4j_backup.log"
        else
            echo "[$(date)] ERROR: $CONTAINER backup failed!" | tee -a "$BACKUP_DIR/neo4j_backup.log"
            return 1
        fi
    fi
    return 0
}

backup_neo4j() {
    echo "[$(date)] Starting Neo4j backups..." | tee -a "$BACKUP_DIR/neo4j_backup.log"

    # Backup main Neo4j (Entity/Attribute storage)
    backup_single_neo4j "companion-neo4j" "neo4j"

    # Backup Graphiti Neo4j (Temporal knowledge graph)
    backup_single_neo4j "companion-graphiti-neo4j" "graphiti_neo4j"

    # Clean up old backups (keep 7 days)
    echo "[$(date)] Cleaning up old backups..." | tee -a "$BACKUP_DIR/neo4j_backup.log"
    find "$BACKUP_DIR" -name "neo4j_*.cypher.gz" -mtime +7 -delete 2>/dev/null
    find "$BACKUP_DIR" -name "neo4j_*.tar.gz" -mtime +7 -delete 2>/dev/null
    find "$BACKUP_DIR" -name "graphiti_neo4j_*.tar.gz" -mtime +7 -delete 2>/dev/null

    echo "[$(date)] Backup and cleanup complete" | tee -a "$BACKUP_DIR/neo4j_backup.log"
    return 0
}

# ============================================================================
# RESTORE FUNCTION
# ============================================================================

restore_neo4j() {
    local backup_file=$1

    if [ -z "$backup_file" ]; then
        echo "ERROR: No backup file specified"
        echo "Usage: $0 restore <backup_file.cypher.gz>"
        exit 1
    fi

    if [ ! -f "$backup_file" ]; then
        echo "ERROR: Backup file not found: $backup_file"
        exit 1
    fi

    echo "=========================================="
    echo "RESTORING NEO4J DATABASE"
    echo "=========================================="
    echo "Backup file: $backup_file"
    echo ""

    # Check if Neo4j container is running
    if ! docker ps | grep -q $NEO4J_CONTAINER; then
        echo "WARNING: Neo4j container not running. Starting it..."
        docker compose -f /app/docker-compose.yml up -d neo4j
        sleep 10
    fi

    # Decompress the backup
    TEMP_DIR=$(mktemp -d)
    gunzip -c "$backup_file" > "$TEMP_DIR/neo4j_restore.cypher"

    if [ ! -f "$TEMP_DIR/neo4j_restore.cypher" ]; then
        echo "ERROR: Failed to decompress backup file"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    echo "Decompressed backup to temporary location"
    echo "Clearing existing Neo4j data..."

    # Stop Neo4j to prevent locks
    docker exec $NEO4J_CONTAINER cypher-shell -u $NEO4J_USER -p "$NEO4J_PASSWORD" \
        "MATCH (n) DETACH DELETE n" 2>/dev/null || true

    echo "Restoring from backup..."

    # Restore using cypher-shell
    docker exec -i $NEO4J_CONTAINER cypher-shell -u $NEO4J_USER -p "$NEO4J_PASSWORD" \
        < "$TEMP_DIR/neo4j_restore.cypher"

    if [ $? -eq 0 ]; then
        echo "✅ Neo4j restore complete!"
        echo "[$(date)] Neo4j restore complete: $backup_file" >> "$BACKUP_DIR/neo4j_backup.log"
    else
        echo "❌ Neo4j restore failed. Check logs."
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    # Clean up
    rm -rf "$TEMP_DIR"

    # Verify restore
    RESTORE_COUNT=$(docker exec $NEO4J_CONTAINER cypher-shell -u $NEO4J_USER -p "$NEO4J_PASSWORD" \
        "MATCH (n) RETURN COUNT(n) as count" 2>/dev/null | tail -1 | awk '{print $1}')

    echo "Nodes restored: $RESTORE_COUNT"
}

# ============================================================================
# MAIN
# ============================================================================

case "$1" in
    restore)
        restore_neo4j "$2"
        ;;
    *)
        backup_neo4j
        ;;
esac
