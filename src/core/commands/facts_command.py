"""
/facts command - Browse and search the fact store
"""

from src.database import tables as T


def handle_facts_command(args: str, context: dict) -> dict:
    user_email = context.get('user_email')
    if not user_email:
        return {'text': 'No user session — cannot query facts.', 'error': 'no_email'}

    args = args.strip()
    limit = 15
    search = None

    # Parse: /facts [n] [search term]
    parts = args.split(maxsplit=1)
    if parts:
        try:
            limit = min(int(parts[0]), 50)
            search = parts[1] if len(parts) > 1 else None
        except ValueError:
            search = args  # whole string is a search term

    try:
        from src.database.db import get_db
        db = get_db()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            if search:
                cursor.execute(
                    f"""SELECT subject, predicate, object, confidence, importance
                        FROM {T.FACTS}
                        WHERE user_email = %s AND archived_at IS NULL
                          AND (subject ILIKE %s OR predicate ILIKE %s OR object ILIKE %s)
                        ORDER BY updated_at DESC LIMIT %s""",
                    (user_email, f'%{search}%', f'%{search}%', f'%{search}%', limit)
                )
            else:
                cursor.execute(
                    f"""SELECT subject, predicate, object, confidence, importance
                        FROM {T.FACTS}
                        WHERE user_email = %s AND archived_at IS NULL
                        ORDER BY updated_at DESC LIMIT %s""",
                    (user_email, limit)
                )
            rows = cursor.fetchall() or []
            cursor.close()

        if not rows:
            msg = f'No facts found{" matching "" + search + """ if search else ""}.'
            return {'text': msg}

        header = f'**Facts** ({len(rows)} shown{", search: " + search if search else ""}):\n'
        lines = []
        for r in rows:
            subj = r[0] or ''
            pred = r[1] or ''
            obj  = r[2] or ''
            conf = r[3]
            imp  = r[4]
            conf_str = f' [{conf:.0%}]' if conf is not None else ''
            imp_str  = f' imp={imp}' if imp is not None else ''
            lines.append(f'  • {subj} {pred} {obj}{conf_str}{imp_str}')

        return {
            'text': header + '\n'.join(lines),
            'data': {'facts': [{'subject': r[0], 'predicate': r[1], 'object': r[2],
                                 'confidence': r[3], 'importance': r[4]} for r in rows]}
        }
    except Exception as e:
        return {'text': f'Error querying facts: {e}', 'error': str(e)}


def register_facts_command(registry):
    registry.register(
        name='facts',
        handler=handle_facts_command,
        description='Browse facts (/facts [n] [search term])',
        aliases=['f', 'fact']
    )
