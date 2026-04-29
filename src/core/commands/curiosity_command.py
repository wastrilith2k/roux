"""
/curiosity command - View active curiosity threads
"""

from src.database import tables as T


def handle_curiosity_command(args: str, context: dict) -> dict:
    user_email = context.get('user_email')
    if not user_email:
        return {'text': 'No user session — cannot query curiosity threads.', 'error': 'no_email'}

    try:
        from src.database.db import get_db
        db = get_db()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT topic, category, questions, urgency, times_asked, last_discussed
                    FROM {T.CURIOSITY_THREADS}
                    WHERE user_email = %s
                    ORDER BY urgency DESC, last_discussed DESC LIMIT 20""",
                (user_email,)
            )
            rows = cursor.fetchall() or []
            cursor.close()

        if not rows:
            return {'text': 'No active curiosity threads.'}

        lines = [f'**Curiosity Threads** ({len(rows)}):\n']
        for r in rows:
            topic      = r[0] or ''
            category   = r[1] or ''
            questions  = r[2] or []
            urgency    = r[3] or 0
            times_asked = r[4] or 0

            urgency_bar = _urgency_str(urgency)
            cat_str = f' [{category}]' if category else ''
            asked_str = f' | asked {times_asked}×' if times_asked else ''
            lines.append(f'  {urgency_bar}{cat_str} **{topic}**{asked_str}')

            if isinstance(questions, list):
                for q in questions[:2]:
                    lines.append(f'       ? {q}')
            elif isinstance(questions, str) and questions:
                lines.append(f'       ? {questions}')

        return {
            'text': '\n'.join(lines),
            'data': {'threads': [{'topic': r[0], 'category': r[1], 'urgency': r[3]} for r in rows]}
        }
    except Exception as e:
        return {'text': f'Error querying curiosity threads: {e}', 'error': str(e)}


def _urgency_str(urgency) -> str:
    try:
        u = float(urgency or 0)
        if u >= 0.7:
            return '🔥'
        if u >= 0.4:
            return '💭'
        return '·'
    except Exception:
        return '·'


def register_curiosity_command(registry):
    registry.register(
        name='curiosity',
        handler=handle_curiosity_command,
        description='View active curiosity threads',
        aliases=['curious', 'threads']
    )
