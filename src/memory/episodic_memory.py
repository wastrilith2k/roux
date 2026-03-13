"""
Episodic Memory - Embedding storage for conversation messages.

WHAT: Generates and stores embedding vectors for individual chat messages in
PostgreSQL's pgvector column on the messages table.

WHY: The companion needs to recall relevant past messages when responding.
Embedding each message at write-time lets semantic_search.py perform fast
cosine-similarity lookups later without re-embedding at query time.

HOW it fits: After each message is saved to the messages table (by the
message handler), embed_and_store_message() is called to compute an
OpenAI embedding and write it into the embedding_vec column. At query time,
semantic_search.py generates a query embedding and uses pgvector's <=>
operator to find the closest stored messages.

Usage:
    from src.memory.episodic_memory import embed_and_store_message

    success = embed_and_store_message(message_id, message_text, sender)
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def get_postgres_connection():
    """Get PostgreSQL connection."""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'companion'),
        user=os.getenv('POSTGRES_USER', 'companion'),
        password=os.getenv('POSTGRES_PASSWORD', '')
    )


def store_episode_embedding(message_id: int, embedding: List[float]) -> bool:
    """
    Store embedding for a message in the vector column.

    Args:
        message_id: ID of the message in messages table
        embedding: 1536-dim embedding vector

    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()

        # Store in vector format for pgvector search
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

        cursor.execute("""
            UPDATE messages
            SET embedding_vec = %s::vector
            WHERE id = %s
        """, (embedding_str, message_id))

        rows_updated = cursor.rowcount
        conn.commit()
        cursor.close()

        if rows_updated == 0:
            logger.warning(f"No message found with id {message_id} to store embedding")
            return False

        return True

    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        return False
    finally:
        if conn:
            conn.close()


def embed_and_store_message(
    message_id: int,
    message_text: str,
    sender: str
) -> bool:
    """
    Generate an embedding for a message and persist it to the messages table.

    The text sent to the embedding model is prefixed with the speaker's name
    (e.g. "James said: ...") so that semantic search can distinguish who said
    what. This is important because "I love tea" means different things
    depending on whether the user or the companion said it.

    Called by the message handler after each message is saved.

    Args:
        message_id: ID in messages table
        message_text: The message content
        sender: Who sent it (for context in embedding)

    Returns:
        True if successful
    """
    try:
        if not message_text or not message_text.strip():
            logger.warning(f"Empty message_text for message {message_id}, skipping embedding")
            return False

        if not sender:
            sender = 'Unknown'

        from src.memory.embeddings import generate_embedding

        # Prefix with speaker name so the embedding captures who said it
        sender_lower = sender.lower()
        if sender_lower in ('user',):
            embed_text = f"James said: {message_text}"
        else:
            from src.config.persona_config import get_persona_config as _gpc
            embed_text = f"{_gpc().companion_short_name} said: {message_text}"

        embedding = generate_embedding(embed_text)
        return store_episode_embedding(message_id, embedding)

    except Exception as e:
        logger.error(f"Error embedding message {message_id}: {e}")
        return False
