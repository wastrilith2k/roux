"""
Biographies Command - View and manage synthesized biographies

Commands:
  /biographies         - Show all synthesized biography paragraphs
  /biographies refresh - Regenerate all biographies from facts
  /biographies <name>  - Show biographies for specific subject

Synthesized biographies group related facts into coherent paragraphs
with temporal decay applied to importance scores.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


def handle_biographies_command(args: str = "", context: dict = None) -> dict:
    """
    Handle /biographies command.

    Args:
        args: Command arguments (refresh, subject name, or empty)
        context: Optional context with user_email

    Returns:
        Dict with 'text' for display and optional 'data' for structured info
    """
    user_email = context.get('user_email') if context else None

    if args.strip().lower() == 'refresh':
        return _refresh_biographies(user_email)
    elif args.strip():
        return _show_subject_biographies(args.strip(), user_email)
    else:
        return _show_all_biographies(user_email)


def _show_all_biographies(user_email: str = None) -> dict:
    """Show all synthesized biography paragraphs."""
    try:
        from src.memory.synthesized_biographies import get_biography_store

        store = get_biography_store()
        paragraphs = store.get_all_paragraphs(
            min_importance=1.0,  # Show all for debugging
            user_email=user_email,
            limit=50
        )

        if not paragraphs:
            return {
                'text': "No synthesized biographies found.\n\nRun `/biographies refresh` to generate them from facts.",
                'data': {'count': 0}
            }

        lines = ["📚 **Synthesized Biographies**"]
        lines.append("(Importance-weighted with temporal decay)\n")

        # Group by subject
        by_subject = {}
        for p in paragraphs:
            subj = p.get('subject', 'Unknown')
            if subj not in by_subject:
                by_subject[subj] = []
            by_subject[subj].append(p)

        for subject, subj_paras in by_subject.items():
            lines.append(f"\n**{subject}:**")
            for p in subj_paras:
                theme = p.get('theme', '?')
                base_imp = p.get('base_importance', 0)
                eff_imp = p.get('effective_importance', 0)
                content = p.get('content', '')[:200]
                if len(p.get('content', '')) > 200:
                    content += "..."

                decay_pct = (eff_imp / base_imp * 100) if base_imp > 0 else 100
                lines.append(f"  [{theme}] imp: {eff_imp:.1f}/{base_imp:.1f} ({decay_pct:.0f}%)")
                lines.append(f"    {content}")

        lines.append(f"\nTotal: {len(paragraphs)} paragraphs")
        lines.append("\nCommands:")
        lines.append("  `/biographies refresh` - Regenerate all")
        lines.append("  `/biographies <name>` - Show for specific person")

        return {
            'text': '\n'.join(lines),
            'data': {
                'count': len(paragraphs),
                'by_subject': {k: len(v) for k, v in by_subject.items()}
            }
        }

    except Exception as e:
        logger.error(f"Error showing biographies: {e}")
        return {
            'text': f"Error loading biographies: {e}",
            'error': str(e)
        }


def _show_subject_biographies(subject: str, user_email: str = None) -> dict:
    """Show biographies for a specific subject."""
    try:
        from src.memory.synthesized_biographies import get_biography_store

        store = get_biography_store()
        paragraphs = store.get_paragraphs_for_subject(
            subject=subject,
            min_importance=0.0,
            user_email=user_email
        )

        if not paragraphs:
            return {
                'text': f"No biographies found for '{subject}'.\n\nRun `/biographies refresh` to generate them.",
                'data': {'count': 0, 'subject': subject}
            }

        lines = [f"📚 **Biographies for {subject}**\n"]

        for p in paragraphs:
            theme = p.get('theme', '?')
            base_imp = p.get('base_importance', 0)
            eff_imp = p.get('effective_importance', 0)
            content = p.get('content', '')
            fact_ids = p.get('fact_ids', [])
            updated = p.get('updated_at')

            decay_pct = (eff_imp / base_imp * 100) if base_imp > 0 else 100

            lines.append(f"**{theme.upper()}** (imp: {eff_imp:.1f}/{base_imp:.1f}, decay: {decay_pct:.0f}%)")
            lines.append(f"{content}")
            if updated:
                if isinstance(updated, datetime):
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=PST)
                    age = datetime.now(PST) - updated
                    age_str = f"{age.days}d" if age.days > 0 else f"{int(age.total_seconds()/3600)}h"
                    lines.append(f"  _Updated: {age_str} ago, from {len(fact_ids)} facts_")
            lines.append("")

        return {
            'text': '\n'.join(lines),
            'data': {
                'count': len(paragraphs),
                'subject': subject
            }
        }

    except Exception as e:
        logger.error(f"Error showing biographies for {subject}: {e}")
        return {
            'text': f"Error loading biographies: {e}",
            'error': str(e)
        }


def _refresh_biographies(user_email: str = None) -> dict:
    """Refresh all synthesized biographies."""
    try:
        from src.tasks.biography_refresh_task import refresh_biographies_sync

        result = refresh_biographies_sync(user_email)

        if result.get('success'):
            details = result.get('details', {})
            subjects = list(details.keys())

            lines = ["🔄 **Biography Refresh Complete**\n"]
            for subject, info in details.items():
                if info.get('success'):
                    para_count = info.get('paragraphs', 0)
                    lines.append(f"  ✓ {subject}: {para_count} paragraphs")
                else:
                    error = info.get('error', 'Unknown error')
                    lines.append(f"  ✗ {subject}: {error}")

            lines.append(f"\nTotal: {result.get('subjects_processed', 0)} subjects processed")

            return {
                'text': '\n'.join(lines),
                'data': result
            }
        else:
            return {
                'text': f"Biography refresh failed: {result.get('error', 'Unknown error')}",
                'error': result.get('error')
            }

    except Exception as e:
        logger.error(f"Error refreshing biographies: {e}")
        return {
            'text': f"Error refreshing biographies: {e}",
            'error': str(e)
        }


def register_biographies_command(registry):
    """Register the biographies command with the command registry."""
    registry.register(
        name='biographies',
        handler=handle_biographies_command,
        description='View and manage synthesized biographies (/biographies [refresh|<name>])',
        aliases=['bio', 'bios']
    )
