"""
/opinions command - View companion opinions and stances
"""

from src.database import tables as T


def handle_opinions_command(args: str, context: dict) -> dict:
    user_email = context.get('user_email')
    if not user_email:
        return {'text': 'No user session — cannot query opinions.', 'error': 'no_email'}

    search = args.strip() or None

    try:
        from src.database.db import get_db
        db = get_db()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            if search:
                cursor.execute(
                    f"""SELECT topic, opinion, confidence, category, evidence_count
                        FROM {T.COMPANION_OPINIONS}
                        WHERE user_email = %s
                          AND (topic ILIKE %s OR opinion ILIKE %s OR category ILIKE %s)
                        ORDER BY last_updated DESC LIMIT 20""",
                    (user_email, f'%{search}%', f'%{search}%', f'%{search}%')
                )
            else:
                cursor.execute(
                    f"""SELECT topic, opinion, confidence, category, evidence_count
                        FROM {T.COMPANION_OPINIONS}
                        WHERE user_email = %s
                        ORDER BY last_updated DESC LIMIT 20""",
                    (user_email,)
                )
            rows = cursor.fetchall() or []
            cursor.close()

        if not rows:
            hint = f' matching "{search}"' if search else ''
            return {'text': f'No opinions found{hint}.'}

        header = f'**Opinions** ({len(rows)}{", search: " + search if search else ""}):\n'
        lines = []

        # Group by category
        by_cat: dict = {}
        for r in rows:
            cat = r[3] or 'general'
            by_cat.setdefault(cat, []).append(r)

        for cat, items in by_cat.items():
            lines.append(f'\n  **{cat.title()}**')
            for r in items:
                topic  = r[0] or ''
                opinion = r[1] or ''
                conf   = r[2]
                evid   = r[4] or 0
                conf_str = f' ({conf:.0%} conf)' if conf is not None else ''
                evid_str = f' [{evid} evidence]' if evid else ''
                lines.append(f'    • **{topic}**: {opinion}{conf_str}{evid_str}')

        return {
            'text': header + '\n'.join(lines),
            'data': {'opinions': [{'topic': r[0], 'opinion': r[1], 'confidence': r[2],
                                    'category': r[3], 'evidence_count': r[4]} for r in rows]}
        }
    except Exception as e:
        return {'text': f'Error querying opinions: {e}', 'error': str(e)}


def register_opinions_command(registry):
    registry.register(
        name='opinions',
        handler=handle_opinions_command,
        description='View companion opinions (/opinions [topic search])',
        aliases=['opinion', 'op']
    )
