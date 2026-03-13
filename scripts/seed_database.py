#!/usr/bin/env python3
"""
Seed Database — Companion Framework

WHAT: Bootstraps a companion's database with initial knowledge and conversation
      history from a seed_history.json file.
WHY:  A brand-new companion has no memories, no facts, no relationship context.
      Seeding provides a backstory so the companion can reference shared history
      from day one of a simulation or demo. Without this, the first few conversations
      feel hollow and amnesic.
HOW:  For each companion:
      1. Reads instances/<companion_id>/seed_history.json
      2. Inserts companion_facts, user_facts, and relationship_facts into Graphiti
         (the knowledge graph) as initial episodes
      3. Inserts seed_exchanges as backdated messages in PostgreSQL
      The companions.yaml registry maps companion IDs to their config paths
      and Graphiti group IDs.

Usage:
    python scripts/seed_database.py --companion kai
    python scripts/seed_database.py --companion mira
    python scripts/seed_database.py --all
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PST = ZoneInfo('America/Los_Angeles')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find project root by looking for data/ directory."""
    candidates = [
        Path(__file__).parent.parent,
        Path('/app'),
        Path.cwd(),
    ]
    for p in candidates:
        if (p / 'data').exists():
            return p
    raise RuntimeError("Cannot find project root (no data/ directory found)")


def load_seed_history(instance_dir: Path) -> dict:
    """Load seed_history.json for a companion instance."""
    seed_file = instance_dir / 'seed_history.json'
    if not seed_file.exists():
        raise FileNotFoundError(f"No seed_history.json found at {seed_file}")
    with open(seed_file) as f:
        return json.load(f)


def load_companions_registry(project_root: Path) -> dict:
    """Load the companions.yaml registry."""
    import yaml
    registry_path = project_root / 'data' / 'companions.yaml'
    if not registry_path.exists():
        raise FileNotFoundError(f"No companions.yaml at {registry_path}")
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data.get('companions', {})


# ---------------------------------------------------------------------------
# Seeding functions
# ---------------------------------------------------------------------------

def seed_knowledge_graph(companion_id: str, seed_data: dict, group_id: str):
    """Insert facts into Graphiti as seed episodes (graceful skip if unavailable)."""
    try:
        from src.memory.graphiti_search import get_graphiti_client
        client = get_graphiti_client()
    except Exception as e:
        logger.warning(f"Graphiti not available, skipping knowledge graph seeding: {e}")
        return

    all_facts = (
        seed_data.get('companion_facts', []) +
        seed_data.get('user_facts', []) +
        seed_data.get('relationship_facts', [])
    )

    logger.info(f"[{companion_id}] Seeding {len(all_facts)} facts into knowledge graph (group_id={group_id})")

    for i, fact in enumerate(all_facts):
        try:
            client.add_episode(
                name=f"seed_fact_{i}",
                episode_body=fact,
                group_id=group_id,
                source_description="seed_history",
            )
        except Exception as e:
            logger.warning(f"  Failed to seed fact {i}: {e}")

    logger.info(f"[{companion_id}] Knowledge graph seeding complete")


def seed_messages(companion_id: str, seed_data: dict, companion_email: str, user_email: str):
    """Insert seed conversation exchanges with backdated timestamps.

    Each exchange has a 'days_ago' offset that gets subtracted from now()
    to create realistic temporal spacing in the message history.
    """
    exchanges = seed_data.get('seed_exchanges', [])
    if not exchanges:
        logger.info(f"[{companion_id}] No seed exchanges to insert")
        return

    logger.info(f"[{companion_id}] Seeding {len(exchanges)} message exchanges")

    try:
        from src.database.db import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as e:
        logger.error(f"Database not available: {e}")
        return

    now = datetime.now(PST)

    for exchange in exchanges:
        days_ago = exchange.get('days_ago', 0)
        timestamp = now - timedelta(days=days_ago)
        sender = exchange.get('from', 'companion')
        text = exchange.get('text', '')

        role = 'assistant' if sender == 'companion' else 'user'

        try:
            cursor.execute("""
                INSERT INTO messages (email, role, content, created_at, companion_id)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_email, role, text, timestamp, companion_id))
        except Exception as e:
            logger.warning(f"  Failed to insert message: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"[{companion_id}] Message seeding complete")


def seed_companion(companion_id: str, project_root: Path, registry: dict):
    """Seed a single companion from its instance directory."""
    companion_config = registry.get(companion_id)
    if not companion_config:
        raise ValueError(f"Companion '{companion_id}' not found in companions.yaml")

    instance_dir = project_root / f'instances/{companion_id}'
    seed_data = load_seed_history(instance_dir)

    group_id = companion_config.get('graphiti_group_id', companion_id)

    # Load persona config for email addresses
    import yaml
    persona_path = project_root / companion_config['persona_config']
    with open(persona_path) as f:
        persona = yaml.safe_load(f)

    companion_email = persona.get('companion', {}).get('email', f'{companion_id}@companion.local')
    user_email = persona.get('primary_user', {}).get('email', 'user@companion.local')

    logger.info(f"=== Seeding companion: {companion_id} ===")

    # 1. Knowledge graph
    seed_knowledge_graph(companion_id, seed_data, group_id)

    # 2. Messages
    seed_messages(companion_id, seed_data, companion_email, user_email)

    logger.info(f"=== Done seeding: {companion_id} ===\n")


def main():
    parser = argparse.ArgumentParser(description='Seed companion database from seed_history.json')
    parser.add_argument('--companion', type=str, help='Companion ID to seed (e.g., "kai")')
    parser.add_argument('--all', action='store_true', help='Seed all enabled companions')
    args = parser.parse_args()

    if not args.companion and not args.all:
        parser.error("Specify --companion <id> or --all")

    project_root = find_project_root()
    sys.path.insert(0, str(project_root))

    registry = load_companions_registry(project_root)

    if args.all:
        for cid, config in registry.items():
            if config.get('enabled', True):
                seed_companion(cid, project_root, registry)
    else:
        seed_companion(args.companion, project_root, registry)


if __name__ == '__main__':
    main()
