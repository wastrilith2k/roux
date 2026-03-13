"""
Fireworks Reranker - Rerank search results using Fireworks Qwen3 Reranker.

Strategy:
- Keep existing OpenAI text-embedding-3-small embeddings (no migration needed)
- Add Fireworks /v1/rerank endpoint to reorder vector search results
- Reranking uses fireworks/qwen3-reranker-8b (purpose-built for relevance)

Usage:
    from src.memory.fireworks_reranker import rerank_results

    # Get initial results from pgvector
    candidates = search_similar_messages(query, limit=30)

    # Rerank with Fireworks
    texts = [c["message_text"] for c in candidates]
    reranked = rerank_results(query, texts, top_n=10)

    # reranked is a list of {index, relevance_score} sorted by relevance
"""

import os
import logging
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

FIREWORKS_API_KEY = os.environ.get('FIREWORKS_API_KEY')
FIREWORKS_RERANK_URL = "https://api.fireworks.ai/inference/v1/rerank"
FIREWORKS_RERANKER_MODEL = "accounts/fireworks/models/qwen3-reranker-8b"


def rerank_results(
    query: str,
    documents: List[str],
    top_n: int = 10,
    return_documents: bool = False
) -> List[Dict]:
    """
    Rerank documents by relevance to query using Fireworks Qwen3 Reranker.

    Args:
        query: The search query to rank against
        documents: List of document texts to rerank
        top_n: Number of top results to return (default 10)
        return_documents: Whether to include document text in response

    Returns:
        List of dicts with:
        - index: Original index in documents list
        - relevance_score: Relevance score (0-1, higher is better)
        - document: (optional) The document text if return_documents=True

        Sorted by relevance_score descending.

    Example:
        >>> rerank_results("How is Jesse?", ["Jesse is fine", "Weather is nice"])
        [{"index": 0, "relevance_score": 0.95}, {"index": 1, "relevance_score": 0.12}]
    """
    if not FIREWORKS_API_KEY:
        logger.warning("FIREWORKS_API_KEY not set, skipping reranking")
        # Return original order with decreasing scores
        return [{"index": i, "relevance_score": 1.0 - (i * 0.05)} for i in range(min(top_n, len(documents)))]

    if not documents:
        return []

    if not query or len(query.strip()) < 3:
        logger.debug("Query too short for reranking")
        return [{"index": i, "relevance_score": 1.0 - (i * 0.05)} for i in range(min(top_n, len(documents)))]

    try:
        response = requests.post(
            FIREWORKS_RERANK_URL,
            headers={
                "Authorization": f"Bearer {FIREWORKS_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": FIREWORKS_RERANKER_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": return_documents
            },
            timeout=10
        )

        if response.status_code != 200:
            logger.warning(f"Fireworks rerank failed: {response.status_code} - {response.text[:200]}")
            return _fallback_ranking(documents, top_n)

        result = response.json()

        # Fireworks returns {"data": [{"index": 0, "relevance_score": 0.95}, ...]}
        # (Note: uses "data" key, not "results")
        results = result.get("data", result.get("results", []))

        if not results:
            logger.warning("Fireworks rerank returned empty results")
            return _fallback_ranking(documents, top_n)

        logger.debug(f"Reranked {len(documents)} documents, returning top {len(results)}")
        return results

    except requests.exceptions.Timeout:
        logger.warning("Fireworks rerank timed out")
        return _fallback_ranking(documents, top_n)
    except Exception as e:
        logger.warning(f"Fireworks rerank error: {e}")
        return _fallback_ranking(documents, top_n)


def _fallback_ranking(documents: List[str], top_n: int) -> List[Dict]:
    """Fallback to original order when reranking fails."""
    return [
        {"index": i, "relevance_score": 1.0 - (i * 0.05)}
        for i in range(min(top_n, len(documents)))
    ]


def rerank_and_reorder(
    query: str,
    items: List[Dict],
    text_key: str = "message_text",
    top_n: int = 10
) -> List[Dict]:
    """
    Rerank a list of dicts and return them in relevance order.

    This is a convenience function that combines reranking with reordering.

    Args:
        query: Search query
        items: List of dicts with a text field to rank by
        text_key: Key in each dict containing the text to rank
        top_n: Number of results to return

    Returns:
        List of items reordered by relevance, with 'rerank_score' added to each

    Example:
        >>> results = [{"id": 1, "message_text": "Jesse is doing well"}, ...]
        >>> reranked = rerank_and_reorder("How is Jesse?", results, "message_text", 5)
        >>> # Returns top 5 items ordered by relevance
    """
    if not items:
        return []

    # Extract texts for reranking
    texts = [item.get(text_key, "") for item in items]

    # Rerank
    ranked = rerank_results(query, texts, top_n=top_n)

    # Reorder items based on ranking
    reordered = []
    for r in ranked:
        idx = r["index"]
        if idx < len(items):
            item_copy = items[idx].copy()
            item_copy["rerank_score"] = r["relevance_score"]
            reordered.append(item_copy)

    return reordered


def batch_rerank(
    queries: List[str],
    documents: List[str],
    top_n: int = 10
) -> Dict[str, List[Dict]]:
    """
    Rerank documents against multiple queries.

    Useful when you want to find documents relevant to multiple aspects
    of a conversation.

    Args:
        queries: List of queries to rank against
        documents: List of documents to rank
        top_n: Number of top results per query

    Returns:
        Dict mapping query to its reranked results
    """
    results = {}

    for query in queries:
        results[query] = rerank_results(query, documents, top_n=top_n)

    return results


# Test function
def test_reranker():
    """Test the reranker with sample data."""
    query = "How is Jesse doing?"
    documents = [
        "The weather is nice today.",
        "Jesse went to the ER last night.",
        "James prefers tea over coffee.",
        "Jesse is doing better after therapy.",
        "The cat Tuck is sleeping.",
        "Jesse had a meltdown at school.",
    ]

    print(f"Query: {query}")
    print(f"Documents: {len(documents)}")
    print()

    results = rerank_results(query, documents, top_n=5)

    print("Reranked results:")
    for r in results:
        idx = r["index"]
        score = r["relevance_score"]
        print(f"  {score:.3f}: {documents[idx][:50]}...")


if __name__ == "__main__":
    test_reranker()
