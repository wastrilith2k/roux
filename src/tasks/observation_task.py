"""
Background observation triggers for observational memory.

Called after each message (non-blocking) and daily for reflections.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def run_observer_if_needed(user_email: str):
    """
    Check if observations need to be generated and run if so.

    Called after each message response is sent. Runs in a background
    thread to avoid blocking the response pipeline.

    Args:
        user_email: User email to process observations for
    """
    from src.memory.observational_memory import get_observation_manager

    manager = get_observation_manager()
    if not manager.is_enabled():
        return

    try:
        observer = manager.observer
        if observer.should_observe(user_email):
            logger.info(f"Running observer for {user_email}...")
            observations = observer.observe(user_email)
            logger.info(f"Observer created {len(observations)} observations")
    except Exception as e:
        logger.error(f"Observer task failed: {e}")


def run_observer_background(user_email: str):
    """
    Fire-and-forget version that runs in a background thread.

    Use this from the pipeline to avoid blocking response delivery.
    """
    thread = threading.Thread(
        target=run_observer_if_needed,
        args=(user_email,),
        daemon=True,
        name=f"observer-{user_email[:8]}"
    )
    thread.start()


def run_daily_reflector(user_email: str) -> Optional[dict]:
    """
    Run the daily reflector to consolidate older observations.

    Called from the observation scheduler or always_on_service.

    Returns:
        Dict with reflection details if created, None otherwise
    """
    from src.memory.observational_memory import get_observation_manager

    manager = get_observation_manager()
    if not manager.is_enabled():
        logger.debug("Observational memory disabled, skipping reflector")
        return None

    try:
        reflector = manager.reflector
        if reflector.needs_reflection(user_email):
            logger.info(f"Running reflector for {user_email}...")
            reflection = reflector.reflect(user_email)
            if reflection:
                logger.info(
                    f"Reflector created reflection for "
                    f"{reflection.period_start} to {reflection.period_end}"
                )
                return {
                    "period_start": str(reflection.period_start),
                    "period_end": str(reflection.period_end),
                    "observation_count": reflection.observation_count,
                    "themes": reflection.themes,
                }
        else:
            logger.debug("No reflection needed yet")
    except Exception as e:
        logger.error(f"Reflector task failed: {e}")

    return None
