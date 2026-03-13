"""
Embedding Generation - Central embedding service for the memory subsystem.

WHAT: Thin wrapper around OpenAI's text-embedding-3-small model that provides
single-text and batch embedding generation, plus a cosine similarity utility.

WHY: Multiple modules need embeddings (episodic_memory for message storage,
fact_store for contradiction detection, semantic_search for query embedding,
synthesized_biographies for paragraph retrieval). Centralizing the client and
model config here avoids duplication and makes it easy to swap models later.

HOW it fits:
  - episodic_memory.py calls generate_embedding() for each message.
  - fact_store.py calls get_embedding() (alias) for contradiction detection.
  - semantic_search.py calls generate_embedding() for query vectors.
  - generate_embeddings_batch() is used by backfill scripts.

Model: text-embedding-3-small (1536 dimensions, ~$0.02 per 1M tokens)
"""
import os
from typing import List, Optional
from openai import OpenAI

# =============================================================================
# Client and model configuration
# =============================================================================

# Lazily initialized to avoid import-time API key validation
_client: Optional[OpenAI] = None

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def get_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string.

    Args:
        text: The text to embed

    Returns:
        List of floats (1536 dimensions)
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    client = get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.strip(),
        dimensions=EMBEDDING_DIMENSIONS
    )
    return response.data[0].embedding


# Alias: fact_store and other modules import this name
get_embedding = generate_embedding


def generate_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call (max 2048)

    Returns:
        List of embeddings in same order as input texts
    """
    if not texts:
        return []

    client = get_client()
    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Clean empty strings
        batch = [t.strip() if t else "" for t in batch]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIMENSIONS
        )

        # Extract embeddings in order
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score between -1 and 1
    """
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
