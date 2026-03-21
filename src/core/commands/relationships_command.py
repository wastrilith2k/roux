"""
Relationships Command - View and manage structured relationships

Commands:
  /relationships         - Show all relationships
  /relationships <name>  - Show relationships for specific entity
  /relationships stats   - Show relationship statistics

Relationships are explicit, typed connections between entities
(parent_of, married_to, partner_of, etc.) - not generic RELATES_TO.
"""

import logging

from src.database import tables as T

logger = logging.getLogger(__name__)


def handle_relationships_command(args: str = "", context: dict = None) -> dict:
    """Handle /relationships command."""
    user_email = context.get('user_email') if context else None

    args = args.strip().lower()

    if args == 'stats':
        return _show_stats(user_email)
    elif args:
        return _show_entity_relationships(args.title(), user_email)
    else:
        return _show_all_relationships(user_email)


def _show_all_relationships(user_email: str = None) -> dict:
    """Show all relationships."""
    try:
        from src.memory.relationship_store import get_relationship_store

        store = get_relationship_store()

        # Get all relationships via stats first
        stats = store.get_stats()

        if stats.get('total_relationships', 0) == 0:
            return {
                'text': "No relationships stored yet.\n\nRelationships are extracted from conversations automatically.",
                'data': {'count': 0}
            }

        # Get relationships grouped by source entity
        conn = store._get_connection()
        from psycopg2.extras import RealDictCursor

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"""
                SELECT source_entity, relationship_type, target_entity,
                       confidence, mention_count
                FROM {T.RELATIONSHIPS}
                WHERE valid_until IS NULL
                ORDER BY source_entity, confidence DESC
                LIMIT 50
            """)
            relationships = cursor.fetchall()

        lines = ["🔗 **Structured Relationships**"]
        lines.append("(Explicit types with verification tracking)\n")

        # Group by source
        by_source = {}
        for rel in relationships:
            src = rel['source_entity']
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(rel)

        for source, rels in by_source.items():
            lines.append(f"**{source}:**")
            for rel in rels:
                rel_type = rel['relationship_type'].replace('_', ' ').upper()
                target = rel['target_entity']
                conf = rel['confidence']
                count = rel['mention_count']

                # Confidence indicator
                if count > 3:
                    indicator = "✓✓"
                elif conf >= 0.8:
                    indicator = "✓"
                else:
                    indicator = "?"

                lines.append(f"  {indicator} {rel_type} → {target} (conf={conf:.0%}, mentions={count})")
            lines.append("")

        lines.append(f"Total: {stats.get('active_relationships', 0)} active relationships")
        lines.append("\nCommands:")
        lines.append("  `/relationships <name>` - Show for specific entity")
        lines.append("  `/relationships stats` - Show statistics")

        return {
            'text': '\n'.join(lines),
            'data': stats
        }

    except Exception as e:
        logger.error(f"Error showing relationships: {e}")
        return {
            'text': f"Error loading relationships: {e}",
            'error': str(e)
        }


def _show_entity_relationships(entity: str, user_email: str = None) -> dict:
    """Show relationships for a specific entity."""
    try:
        from src.memory.relationship_store import get_relationship_store

        store = get_relationship_store()
        relationships = store.get_relationships_for_entity(
            entity,
            include_as_target=True,
            only_current=True,
            min_confidence=0.3,
            user_email=user_email
        )

        if not relationships:
            return {
                'text': f"No relationships found for '{entity}'.",
                'data': {'entity': entity, 'count': 0}
            }

        lines = [f"🔗 **Relationships for {entity}**\n"]

        for rel in relationships:
            source = rel['source_entity']
            target = rel['target_entity']
            rel_type = rel['relationship_type'].replace('_', ' ')
            conf = rel['confidence']
            count = rel['mention_count']
            context = rel.get('context', '')

            # Determine direction
            if source.lower() == entity.lower():
                direction = f"{rel_type} → {target}"
            else:
                direction = f"← {rel_type} ← {source}"

            lines.append(f"**{direction}**")
            lines.append(f"  Confidence: {conf:.0%} | Mentions: {count}")
            if context:
                lines.append(f"  Evidence: \"{context[:80]}...\"" if len(context) > 80 else f"  Evidence: \"{context}\"")
            lines.append("")

        return {
            'text': '\n'.join(lines),
            'data': {
                'entity': entity,
                'count': len(relationships),
                'relationships': relationships
            }
        }

    except Exception as e:
        logger.error(f"Error showing relationships for {entity}: {e}")
        return {
            'text': f"Error: {e}",
            'error': str(e)
        }


def _show_stats(user_email: str = None) -> dict:
    """Show relationship statistics."""
    try:
        from src.memory.relationship_store import get_relationship_store

        store = get_relationship_store()
        stats = store.get_stats()

        lines = ["📊 **Relationship Statistics**\n"]

        lines.append(f"Total relationships: {stats.get('total_relationships', 0)}")
        lines.append(f"Active relationships: {stats.get('active_relationships', 0)}")
        lines.append(f"Unique entities (source): {stats.get('unique_sources', 0)}")
        lines.append(f"Unique entities (target): {stats.get('unique_targets', 0)}")
        lines.append(f"Average confidence: {stats.get('avg_confidence', 0):.0%}")
        lines.append(f"Average mentions: {stats.get('avg_mentions', 0):.1f}")

        if stats.get('by_type'):
            lines.append("\n**By Type:**")
            for rel_type, count in stats['by_type'].items():
                lines.append(f"  {rel_type.replace('_', ' ')}: {count}")

        return {
            'text': '\n'.join(lines),
            'data': stats
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            'text': f"Error: {e}",
            'error': str(e)
        }


def register_relationships_command(registry):
    """Register the relationships command."""
    registry.register(
        name='relationships',
        handler=handle_relationships_command,
        description='View structured entity relationships',
        aliases=['rels', 'rel']
    )
