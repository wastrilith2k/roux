#!/usr/bin/env python3
"""
Standalone Observation Scheduler - Run observation tasks outside Celery.

WHAT: Runs two observation tasks in sequence:
      1. Observer: processes any unobserved messages (extracts observations)
      2. Reflector: consolidates old observations into reflections

WHEN: Designed to run as a cron job (e.g., daily at 3 AM) or be imported
      by the always-on service. Not a Celery task -- standalone process.

WHY:  This exists as a standalone entry point so observation tasks can run
      independently of the Celery infrastructure. Useful for cron-based
      deployments or manual testing without spinning up a full Celery worker.

Usage:
    cron:   0 3 * * * cd /root/companion && python -m src.tasks.observation_scheduler
    manual: python -m src.tasks.observation_scheduler
"""

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default user email
def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


def run_scheduled_tasks():
    """Run all scheduled observation tasks."""
    from src.tasks.observation_task import run_observer_if_needed, run_daily_reflector

    user_email = _get_default_user_email()

    logger.info(f"Running scheduled observation tasks for {user_email}")

    # 1. Run observer (process any unobserved messages)
    try:
        run_observer_if_needed(user_email)
    except Exception as e:
        logger.error(f"Observer failed: {e}")

    # 2. Run reflector (consolidate old observations)
    try:
        result = run_daily_reflector(user_email)
        if result:
            logger.info(f"Reflection created: {result}")
        else:
            logger.info("No reflection needed today")
    except Exception as e:
        logger.error(f"Reflector failed: {e}")

    logger.info("Scheduled observation tasks complete")


if __name__ == "__main__":
    run_scheduled_tasks()
