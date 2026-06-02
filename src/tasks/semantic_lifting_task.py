"""
Semantic Lifting Task — weekly generalization of high-importance facts.

WHAT: Clusters facts with importance >= 7 by subject and asks an LLM to
      generalize them into a single inferred_preference fact (stored with
      source='inferred_preference').

WHY:  Specific observations ("likes ramen", "enjoyed Thai food") should lift
      to durable semantic rules ("prefers Asian cuisine") — the CoALA
      hierarchical abstraction step missing from nightly consolidation.
      High-importance facts already survived pruning and reinforce cycles;
      generalising them adds higher-order knowledge without duplicating raw
      observations.

WHEN: Wednesday at 4:30 AM Pacific (after sleep-time consolidation).

HOW:
  1. Fetch facts with importance >= SEMANTIC_LIFT_IMPORTANCE_MIN (default 7).
  2. Group by subject (lowercase).
  3. For groups with >= MIN_CLUSTER_SIZE facts, prompt the LLM to produce one
     generalised "<subject> <predicate> <object>" rule.
  4. Store each generalisation via store_fact(..., source='inferred_preference').
  5. Return {'lifted': N, 'skipped': M}.
"""

import logging
import os
from celery import shared_task
from src.memory.fact_store import get_fact_store

logger = logging.getLogger(__name__)

IMPORTANCE_THRESHOLD = int(os.environ.get('SEMANTIC_LIFT_IMPORTANCE_MIN', '7'))
MIN_CLUSTER_SIZE = int(os.environ.get('SEMANTIC_LIFT_MIN_CLUSTER', '3'))
LLM_MODEL = os.environ.get('SEMANTIC_LIFT_MODEL', 'accounts/fireworks/models/deepseek-v3')


def _cluster_and_lift(facts: list[dict]) -> list[dict]:
    """Group facts by subject and lift each qualifying group to an inferred_preference.

    Returns a list of dicts with keys: subject, predicate, obj, source, confidence.
    """
    from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

    by_subject: dict[str, list[dict]] = {}
    for f in facts:
        subj = (f.get('subject') or 'unknown').lower()
        by_subject.setdefault(subj, []).append(f)

    lifted = []
    for subject, group in by_subject.items():
        if len(group) < MIN_CLUSTER_SIZE:
            logger.debug(
                "Skipping subject '%s': only %d facts (need %d)",
                subject, len(group), MIN_CLUSTER_SIZE,
            )
            continue

        fact_lines = '\n'.join(
            f"- {f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')}"
            for f in group
        )
        prompt = (
            f"These are {len(group)} related facts about {subject}:\n{fact_lines}\n\n"
            "Write ONE general preference rule capturing the pattern. "
            "Format: '<subject> <predicate> <object>' (e.g. 'James prefers Asian cuisine'). "
            "Only the fact, nothing else."
        )

        try:
            chain = get_resilient_provider_chain(model_override=LLM_MODEL)
            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=64,
                chain=chain,
            )
            generalisation = response.strip()
            parts = generalisation.split(None, 2)
            if len(parts) == 3:
                lifted.append({
                    'subject': parts[0],
                    'predicate': parts[1],
                    'obj': parts[2],
                    'source': 'inferred_preference',
                    'confidence': 0.7,
                })
            else:
                logger.warning(
                    "Semantic lift for '%s' produced unexpected format: %r",
                    subject, generalisation,
                )
        except Exception as e:
            logger.warning("Semantic lift failed for '%s': %s", subject, e)

    return lifted


def lift_facts(user_email: str) -> dict:
    """Lift high-importance facts to inferred_preference rules for one user.

    Args:
        user_email: The user whose facts to generalise.

    Returns:
        {'lifted': N, 'skipped': M}
    """
    store = get_fact_store()

    high_importance = store.get_important_facts(
        min_importance=IMPORTANCE_THRESHOLD,
        limit=200,
        user_email=user_email,
    )

    if not high_importance:
        logger.info("No high-importance facts found for %s — skipping lift", user_email)
        return {'lifted': 0, 'skipped': 0}

    logger.info(
        "Semantic lifting: %d high-importance facts for %s",
        len(high_importance), user_email,
    )

    candidates = _cluster_and_lift(high_importance)

    stored = 0
    for fact in candidates:
        try:
            store.store_fact(
                subject=fact['subject'],
                predicate=fact['predicate'],
                obj=fact['obj'],
                confidence=fact['confidence'],
                importance=7,
                source=fact['source'],
                user_email=user_email,
            )
            stored += 1
            logger.debug(
                "Stored inferred_preference: %s %s %s",
                fact['subject'], fact['predicate'], fact['obj'],
            )
        except Exception as e:
            logger.warning("Failed to store lifted fact %r: %s", fact, e)

    skipped = len(candidates) - stored
    logger.info(
        "Semantic lift complete for %s: %d stored, %d skipped",
        user_email, stored, skipped,
    )
    return {'lifted': stored, 'skipped': skipped}


@shared_task(name='src.tasks.semantic_lifting_task.run_semantic_lifting')
def run_semantic_lifting():
    """Celery task: lift high-importance facts for the primary user.

    Scheduled weekly (Wednesday 4:30 AM Pacific) in celery_app.py beat_schedule.
    """
    from src.config.persona_config import get_persona_config

    config = get_persona_config()
    user_email = config.primary_user_email

    if not user_email:
        logger.warning("run_semantic_lifting: no primary_user_email configured — aborting")
        return 0

    result = lift_facts(user_email=user_email)
    logger.info("run_semantic_lifting finished: %s", result)
    return result['lifted']
