#!/usr/bin/env python3
"""
Restore Neo4j knowledge graph from JSON backup
"""

import json
import os
import sys
from neo4j import GraphDatabase


def restore_neo4j(input_file):
    """Restore Neo4j from JSON backup file"""

    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')

    if not password:
        print("❌ NEO4J_PASSWORD not set", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"❌ Backup file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"📖 Reading backup from {input_file}...", file=sys.stderr)
    with open(input_file, 'r') as f:
        backup = json.load(f)

    print(f"🧠 Connecting to Neo4j at {uri}...", file=sys.stderr)

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            # Clear existing data
            print("🗑️  Clearing existing data...", file=sys.stderr)
            session.run("MATCH (n) DETACH DELETE n")

            # Create ID mapping (old ID -> new ID)
            id_map = {}

            # Restore nodes
            print(f"📦 Restoring {len(backup['nodes'])} nodes...", file=sys.stderr)
            for node in backup['nodes']:
                labels = ':'.join(node['labels'])
                props = node['properties']

                # Create node
                query = f"CREATE (n:{labels}) SET n = $props RETURN id(n) as new_id"
                result = session.run(query, props=props)
                new_id = result.single()['new_id']
                id_map[node['id']] = new_id

            # Restore relationships
            print(f"🔗 Restoring {len(backup['relationships'])} relationships...", file=sys.stderr)
            for rel in backup['relationships']:
                start_id = id_map.get(rel['start_id'])
                end_id = id_map.get(rel['end_id'])

                if start_id is None or end_id is None:
                    print(f"⚠️  Skipping relationship - node IDs not found", file=sys.stderr)
                    continue

                rel_type = rel['type']
                props = rel['properties']

                query = f"""
                    MATCH (a), (b)
                    WHERE id(a) = $start_id AND id(b) = $end_id
                    CREATE (a)-[r:{rel_type}]->(b)
                    SET r = $props
                """
                session.run(query, start_id=start_id, end_id=end_id, props=props)

        driver.close()

        print(f"✅ Neo4j restore complete", file=sys.stderr)
        return True

    except Exception as e:
        print(f"❌ Neo4j restore failed: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: restore_neo4j.py <input_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    success = restore_neo4j(input_file)
    sys.exit(0 if success else 1)
