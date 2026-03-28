"""
Memory Importance Scorer

Scores memories/facts for importance (1-10) to enable weighted retrieval.
Uses Fireworks API for cost-effective scoring.

Scoring guide:
- 9-10: Core identity moments (trauma, major life changes, relationship-defining, firsts)
- 7-8: Significant milestones (moving, job changes, meaningful conversations)
- 5-6: Notable preferences, habits, recurring patterns
- 3-4: Minor details, casual observations
- 1-2: Mundane daily activities (meals, weather, routine tasks)
"""

import os
import json
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Use Fireworks model for quality scoring
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL

IMPORTANCE_PROMPT = """Rate how important this fact is for understanding a person's identity and life story.

Scoring guide:
- 9-10: Core identity moments (trauma, major life changes, relationship-defining moments, firsts)
- 7-8: Significant milestones (moving in together, job changes, meaningful conversations)
- 5-6: Notable preferences, habits, recurring patterns
- 3-4: Minor details, casual observations
- 1-2: Mundane daily activities (meals, weather, routine tasks)

Fact: "{fact}"

/no_think
Reply with ONLY a single number 1-10, nothing else."""


def _parse_score(response: str) -> Optional[int]:
    """Extract score from LLM response."""
    response = response.strip()

    # Handle thinking tags
    if '<think>' in response:
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

    # Handle "10" specifically
    if response.startswith('10'):
        return 10

    # Find first digit
    for char in response:
        if char.isdigit():
            score = int(char)
            return min(max(score, 1), 10)  # Clamp to 1-10

    return None


def score_importance(fact: str) -> Optional[int]:
    """
    Score a fact's importance (1-10) using Fireworks API.

    Args:
        fact: The fact text to score

    Returns:
        Importance score 1-10, or None if scoring failed
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.getenv('FIREWORKS_API_KEY')
        )

        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            messages=[
                {"role": "user", "content": IMPORTANCE_PROMPT.format(fact=fact)}
            ],
            max_tokens=10,
            temperature=0.1  # Low temp for consistent scoring
        )

        result = response.choices[0].message.content

        try:
            from src.services.cost_tracker import get_cost_tracker
            usage = response.usage
            if usage:
                get_cost_tracker().track_fireworks_call(
                    user_id='system', prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=usage.completion_tokens or 0,
                    model=FIREWORKS_MODEL, call_purpose='importance_scoring')
        except Exception:
            pass

        return _parse_score(result)

    except Exception as e:
        logger.error(f"Importance scoring failed: {e}")
        return None


def score_importance_batch(facts: List[str]) -> List[Tuple[str, Optional[int]]]:
    """
    Score multiple facts for importance.

    Args:
        facts: List of facts to score

    Returns:
        List of (fact, score) tuples
    """
    results = []

    for fact in facts:
        score = score_importance(fact)
        results.append((fact, score))

    return results


def score_and_store(fact_id: int, fact_text: str) -> Optional[int]:
    """
    Score a fact and update its importance in the database.

    Args:
        fact_id: Database ID of the fact
        fact_text: The fact text to score

    Returns:
        The importance score, or None if failed
    """
    score = score_importance(fact_text)

    if score is not None:
        try:
            from src.memory.fact_store import get_fact_store
            store = get_fact_store()
            store.update_importance(fact_id, score)
            logger.info(f"Scored fact {fact_id} with importance {score}")
        except Exception as e:
            logger.error(f"Failed to store importance score: {e}")

    return score
