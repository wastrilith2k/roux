"""
/episodes command - View recent conversation episodes
"""

from src.database import tables as T


def handle_episodes_command(args: str, context: dict) -> dict:
    user_email = context.get('user_email')
    if not user_email:
        return {'text': 'No user session — cannot query episodes.', 'error': 'no_email'}

    limit = 8
    if args.strip():
        try:
            limit = min(int(args.strip()), 30)
        except ValueError:
            pass

    try:
        from src.database.db import get_db
        db = get_db()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT topic, emotional_state, resolution, trigger,
                           started_at, ended_at
                    FROM {T.EPISODES}
                    WHERE user_email = %s
                    ORDER BY started_at DESC LIMIT %s""",
                (user_email, limit)
            )
            rows = cursor.fetchall() or []
            cursor.close()

        if not rows:
            return {'text': 'No episodes recorded yet.'}

        lines = [f'**Recent Episodes** (last {len(rows)}):\n']
        for r in rows:
            topic    = r[0] or '(untitled)'
            emotion  = r[1] or ''
            resolut  = r[2] or ''
            trigger  = r[3] or ''
            started  = r[4]
            ended    = r[5]

            date_str = ''
            if started:
                try:
                    date_str = started.strftime('%b %d')
                except Exception:
                    pass

            emotion_str  = f' | {emotion}' if emotion else ''
            trigger_str  = f' | triggered by: {trigger}' if trigger else ''
            resolut_str  = f'\n       ↳ {resolut}' if resolut else ''
            lines.append(f'  • **{date_str}** {topic}{emotion_str}{trigger_str}{resolut_str}')

        return {
            'text': '\n'.join(lines),
            'data': {'episodes': [{'topic': r[0], 'emotional_state': r[1],
                                    'resolution': r[2]} for r in rows]}
        }
    except Exception as e:
        return {'text': f'Error querying episodes: {e}', 'error': str(e)}


def register_episodes_command(registry):
    registry.register(
        name='episodes',
        handler=handle_episodes_command,
        description='View recent episodes (/episodes [n])',
        aliases=['ep', 'eps']
    )
