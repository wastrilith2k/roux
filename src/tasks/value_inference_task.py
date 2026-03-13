"""
Value Inference Task - Infer the companion's emergent personality from her messages.

WHAT: Analyzes the companion's own messages to extract inferred values, preferences,
      boundaries, private thoughts, and relationship feelings. Three tiers:
      - Daily incremental: volatile categories only (private_thoughts, feelings)
      - Weekly: all categories from last 7 days (200 messages)
      - Monthly full history: samples across entire history (300 messages)

WHEN: Daily (incremental), weekly (comprehensive), monthly (full history).

WHY:  The companion's personality emerges from how she actually communicates,
      not just from her persona config. This task discovers patterns like
      "she tends to be protective when Jesse is mentioned" or "she avoids
      talking about her own needs." These inferred values shape her behavior
      more authentically than static rules.
"""

import logging
from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name='tasks.value_inference.run_weekly')
def run_weekly_value_inference():
    """
    Weekly value inference from recent messages.

    Analyzes last 7 days of the companion's messages to update her
    inferred values, preferences, boundaries, and private thoughts.
    """
    try:
        from src.autonomy.value_inference import get_value_inference

        inference = get_value_inference()

        # Analyze recent messages (last 7 days)
        logger.info("Running weekly value inference (recent messages)...")
        values = inference.analyze_recent_messages(hours=168, limit=200)

        if values:
            stored = inference.store_values(values)
            logger.info(f"Weekly value inference complete: {stored} values stored/updated")
        else:
            logger.info("No values extracted from recent messages")

        stats = inference.get_stats()
        logger.info(f"Value inference stats: {stats}")

        return {'status': 'success', 'stats': stats}

    except Exception as e:
        logger.error(f"Weekly value inference failed: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task(name='tasks.value_inference.run_incremental')
def run_incremental_value_inference():
    """
    Daily lightweight value update -- analyzes last 24h of messages.

    Only updates volatile categories (private_thoughts, relationship_feelings,
    unresolved) because core values and boundaries need more data to shift
    meaningfully. This prevents daily noise from overwriting stable traits.
    """
    try:
        from src.autonomy.value_inference import get_value_inference

        inference = get_value_inference()

        logger.info("Running daily incremental value inference (24h window)...")
        values = inference.analyze_recent_messages(hours=24, limit=50)

        stored_categories = []
        if values:
            # Only keep volatile categories for daily update
            volatile_categories = {'private_thoughts', 'relationship_feelings', 'unresolved'}
            filtered = {k: v for k, v in values.items() if k in volatile_categories}

            if filtered:
                stored = inference.store_values(filtered)
                stored_categories = list(filtered.keys())
                logger.info(f"Daily value inference: {stored} volatile values updated ({stored_categories})")
            else:
                logger.info("Daily value inference: no volatile category updates")
        else:
            logger.info("No values extracted from last 24h")

        return {'status': 'success', 'categories_updated': stored_categories}

    except Exception as e:
        logger.error(f"Daily incremental value inference failed: {e}")
        return {'status': 'error', 'error': str(e)}


@celery_app.task(name='tasks.value_inference.run_full_history')
def run_full_history_value_inference():
    """
    Monthly full history analysis.

    Samples across all of the companion's messages for comprehensive
    personality understanding. More expensive but catches
    long-term patterns.
    """
    try:
        from src.autonomy.value_inference import get_value_inference

        inference = get_value_inference()

        logger.info("Running full history value inference...")
        values = inference.analyze_full_history(sample_size=300)

        if values:
            stored = inference.store_values(values)
            logger.info(f"Full history value inference complete: {stored} values stored/updated")
        else:
            logger.info("No values extracted from full history")

        stats = inference.get_stats()
        logger.info(f"Value inference stats: {stats}")

        return {'status': 'success', 'stats': stats}

    except Exception as e:
        logger.error(f"Full history value inference failed: {e}")
        return {'status': 'error', 'error': str(e)}
