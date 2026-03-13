"""
Episodic Embedding Task - Generate vector embeddings for messages.

WHAT: Embeds messages into vector space (stored in embedding_vec column) after
      they're saved to the database. Supports both batch embedding (recent
      unembedded messages) and single-message embedding (called immediately
      after a message is stored). Skips messages shorter than 20 characters
      (too short for meaningful embeddings).

WHEN: embed_single_message fires after each message is stored.
      embed_recent_messages can be called for batch backfilling.

WHY:  Vector embeddings enable episodic memory retrieval -- finding past
      conversations that are semantically similar to the current context.
      When the companion needs to recall "that time we talked about Jesse's
      school problems," vector similarity search finds the relevant messages
      even if the exact words differ.
"""

from celery import shared_task
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@shared_task(name='tasks.episodic_embedding.embed_messages')
def embed_recent_messages(user_email: str, message_ids: list[int] = None):
    """
    Celery task to embed recent messages for episodic memory.

    Args:
        user_email: User's email
        message_ids: Specific message IDs to embed (or None for recent unembedded)

    Returns:
        Dict with embedding results
    """
    conn = None
    try:
        from src.memory.episodic_memory import embed_and_store_message, get_postgres_connection

        conn = get_postgres_connection()
        cursor = conn.cursor()

        # Get messages to embed (skip already-embedded and too-short messages)
        if message_ids:
            # Targeted: embed specific message IDs (e.g., just-stored messages)
            cursor.execute("""
                SELECT id, message_text, sender_name
                FROM messages
                WHERE id = ANY(%s)
                AND embedding_vec IS NULL
                AND message_text IS NOT NULL
                AND LENGTH(message_text) > 20
            """, (message_ids,))
        else:
            # Batch: find recent unembedded messages for this user
            cursor.execute("""
                SELECT id, message_text, sender_name
                FROM messages
                WHERE email = %s
                AND embedding_vec IS NULL
                AND message_text IS NOT NULL
                AND LENGTH(message_text) > 20
                ORDER BY timestamp DESC
                LIMIT 20
            """, (user_email,))

        messages = cursor.fetchall()
        cursor.close()

        if not messages:
            return {'status': 'success', 'embedded': 0, 'message': 'No messages to embed'}

        embedded = 0
        for msg_id, text, sender in messages:
            if embed_and_store_message(msg_id, text, sender):
                embedded += 1

        logger.info(f"Embedded {embedded}/{len(messages)} messages for {user_email}")

        return {
            'status': 'success',
            'embedded': embedded,
            'total': len(messages)
        }

    except Exception as e:
        logger.error(f"Error in episodic embedding task: {e}")
        return {'status': 'error', 'error': str(e)}
    finally:
        if conn:
            conn.close()


@shared_task(name='tasks.episodic_embedding.embed_single')
def embed_single_message(message_id: int, message_text: str, sender: str):
    """
    Embed a single message immediately after it's stored.

    Called from message handler after saving user/companion messages.
    """
    try:
        from src.memory.episodic_memory import embed_and_store_message

        success = embed_and_store_message(message_id, message_text, sender)

        if success:
            logger.debug(f"Embedded message {message_id}")
        else:
            logger.warning(f"Failed to embed message {message_id}")

        return {'status': 'success' if success else 'failed', 'message_id': message_id}

    except Exception as e:
        logger.error(f"Error embedding message {message_id}: {e}")
        return {'status': 'error', 'error': str(e)}
