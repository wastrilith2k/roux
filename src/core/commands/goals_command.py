"""
/goals command - View active companion goals and progress
"""

from src.database import tables as T


def handle_goals_command(args: str, context: dict) -> dict:
    user_email = context.get('user_email')
    if not user_email:
        return {'text': 'No user session — cannot query goals.', 'error': 'no_email'}

    include_completed = args.strip().lower() in ('all', 'done', 'completed')

    try:
        from src.database.db import get_db
        db = get_db()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            if include_completed:
                cursor.execute(
                    f"""SELECT goal, motivation, category, progress, status, created_at
                        FROM {T.COMPANION_GOALS}
                        WHERE user_email = %s
                        ORDER BY status, created_at DESC LIMIT 30""",
                    (user_email,)
                )
            else:
                cursor.execute(
                    f"""SELECT goal, motivation, category, progress, status, created_at
                        FROM {T.COMPANION_GOALS}
                        WHERE user_email = %s AND status = 'active'
                        ORDER BY created_at DESC LIMIT 20""",
                    (user_email,)
                )
            rows = cursor.fetchall() or []
            cursor.close()

        if not rows:
            hint = '' if include_completed else ' (try /goals all to include completed)'
            return {'text': f'No goals found{hint}.'}

        label = 'All Goals' if include_completed else 'Active Goals'
        lines = [f'**{label}** ({len(rows)}):\n']
        for r in rows:
            goal     = r[0] or ''
            motivatn = r[1] or ''
            category = r[2] or ''
            progress = r[3] or 0
            status   = r[4] or 'active'

            bar = _progress_bar(progress)
            cat_str = f'[{category}] ' if category else ''
            lines.append(f'  {bar} {cat_str}{goal}')
            if motivatn:
                lines.append(f'       ↳ {motivatn}')
            if status != 'active':
                lines.append(f'       status: {status}')

        return {
            'text': '\n'.join(lines),
            'data': {'goals': [{'goal': r[0], 'motivation': r[1], 'category': r[2],
                                 'progress': r[3], 'status': r[4]} for r in rows]}
        }
    except Exception as e:
        return {'text': f'Error querying goals: {e}', 'error': str(e)}


def _progress_bar(progress) -> str:
    try:
        pct = float(progress or 0)
        filled = round(pct / 10)
        return '[' + '█' * filled + '░' * (10 - filled) + f'] {pct:.0f}%'
    except Exception:
        return '[----------]   0%'


def register_goals_command(registry):
    registry.register(
        name='goals',
        handler=handle_goals_command,
        description='View active goals (/goals [all])',
        aliases=['goal', 'g']
    )
