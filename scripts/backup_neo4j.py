#!/usr/bin/env python3
"""
Backup Neo4j knowledge graph to JSON
"""

import json
import os
import sys
from neo4j import GraphDatabase


def backup_neo4j(output_file):
    """Backup Neo4j to JSON file"""

    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')

    if not password:
        print("❌ NEO4J_PASSWORD not set", file=sys.stderr)
        sys.exit(1)

    print(f"🧠 Connecting to Neo4j at {uri}...", file=sys.stderr)

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        backup = {
            "nodes": [],
            "relationships": []
        }

        with driver.session() as session:
            # Export all nodes
            print("📦 Exporting nodes...", file=sys.stderr)
            result = session.run("MATCH (n) RETURN n")
            for record in result:
                node = record['n']
                backup['nodes'].append({
                    "id": node.id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                })

            # Export all relationships
            print("🔗 Exporting relationships...", file=sys.stderr)
            result = session.run("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end")
            for record in result:
                rel = record['r']
                start = record['start']
                end = record['end']
                backup['relationships'].append({
                    "id": rel.id,
                    "type": rel.type,
                    "start_id": start.id,
                    "end_id": end.id,
                    "properties": dict(rel)
                })

        driver.close()

        # Write to file
        print(f"💾 Writing to {output_file}...", file=sys.stderr)
        with open(output_file, 'w') as f:
            json.dump(backup, f, indent=2)

        print(f"✅ Neo4j backup complete: {len(backup['nodes'])} nodes, {len(backup['relationships'])} relationships", file=sys.stderr)
        return True

    except Exception as e:
        print(f"❌ Neo4j backup failed: {e}", file=sys.stderr)
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: backup_neo4j.py <output_file>", file=sys.stderr)
        sys.exit(1)

    output_file = sys.argv[1]
    success = backup_neo4j(output_file)
    sys.exit(0 if success else 1)
