"""
Interaction Outcome Tracking - Learn what conversational approaches "land."

WHAT: After each exchange, classifies the companion's action type (advice,
      emotional support, curiosity followup, etc.), James's engagement level
      (enthusiastic/engaged/brief/deflected/ignored), whether the topic
      continued, and emotional resonance (-1.0 to 1.0). Stores results in
      the interaction_outcomes table.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  The companion needs to learn what works. If curiosity follow-ups get
      "enthusiastic" engagement but unsolicited advice gets "deflected," that
      pattern should influence future behavior. Outcome data feeds into daily
      reflections and strategy adjustment.

Feature flag: COMPANION_OUTCOME_TRACKING_ENABLED (default: true)
"""

import os
import logging
import json
from typing import Optional, Dict, Any

from src.celery_app import celery_app
from src.database import tables as T

logger = logging.getLogger(__name__)

COMPANION_OUTCOME_TRACKING_ENABLED = os.environ.get('COMPANION_OUTCOME_TRACKING_ENABLED', 'true').lower() == 'true'

def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email



@celery_app.task(
    name='tasks.interaction_outcome.analyze_outcome',
    bind=True,
    max_retries=1,
    soft_time_limit=30,
    time_limit=45
)
def analyze_interaction_outcome(
    self,
    user_email: str,
    companion_message: str,
    user_response: str,
    companion_msg_id: int = None,
    user_msg_id: int = None
):
    """
    Analyze how James responded to the companion's previous message.

    Args:
        user_email: User's email
        companion_message: What the companion said previously
        user_response: How James responded
        companion_msg_id: Optional companion message ID
        user_msg_id: Optional user message ID
    """
    if not COMPANION_OUTCOME_TRACKING_ENABLED:
        return {'status': 'disabled'}

    if not companion_message or not user_response:
        return {'status': 'skipped', 'reason': 'missing messages'}

    try:
        from src.llm.provider_factory import generate_sync, get_resilient_provider_chain

        prompt = f"""Analyze this conversation exchange:

The companion said: {companion_message[:500]}

James responded: {user_response[:500]}

Classify:
1. ACTION_TYPE: What was the companion doing? (curiosity_followup, advice, emotional_support, topic_introduction, question, playful, general)
2. TOPIC: Brief topic/subject (max 5 words)
3. ENGAGEMENT: How engaged was James? (enthusiastic, engaged, brief, deflected, ignored)
4. CONTINUED: Did the topic continue? (yes/no)
5. RESONANCE: Emotional resonance -1.0 to 1.0 (negative=disconnect, 0=neutral, positive=connection)
6. NOTES: Brief observation (1 sentence)

Format:
ACTION_TYPE: ...
TOPIC: ...
ENGAGEMENT: ...
CONTINUED: yes/no
RESONANCE: N.N
NOTES: ..."""

        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[
                {"role": "system", "content": "Classify conversation interaction outcomes. Be brief."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150,
            chain=chain
        )

        # --- Parse structured LLM response ---
        action_type = 'general'
        topic = ''
        engagement = 'engaged'
        continued = False
        resonance = 0.0
        notes = ''

        for line in response.strip().split('\n'):
            line = line.strip()
            if line.upper().startswith('ACTION_TYPE:'):
                action_type = line.split(':', 1)[1].strip().lower()
            elif line.upper().startswith('TOPIC:'):
                topic = line.split(':', 1)[1].strip()[:255]
            elif line.upper().startswith('ENGAGEMENT:'):
                engagement = line.split(':', 1)[1].strip().lower()
            elif line.upper().startswith('CONTINUED:'):
                continued = 'yes' in line.lower()
            elif line.upper().startswith('RESONANCE:'):
                try:
                    # Take first token after colon (handles "0.8 - good connection")
                    resonance = float(line.split(':', 1)[1].strip().split()[0])
                    resonance = max(-1.0, min(1.0, resonance))  # Clamp to valid range
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith('NOTES:'):
                notes = line.split(':', 1)[1].strip()

        # Validate engagement level
        valid_levels = {'enthusiastic', 'engaged', 'brief', 'deflected', 'ignored'}
        if engagement not in valid_levels:
            engagement = 'engaged'

        # Store result
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.INTERACTION_OUTCOMES} (
                        user_email, companion_message_id, user_response_id,
                        companion_message_text, user_response_text,
                        companion_action_type, companion_topic, engagement_level,
                        topic_continued, emotional_resonance, analysis_notes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_email, companion_msg_id, user_msg_id,
                    companion_message[:1000], user_response[:1000],
                    action_type, topic, engagement,
                    continued, resonance, notes
                ))
                conn.commit()
        finally:
            conn.close()

        logger.info(
            f"Outcome tracked: action={action_type}, engagement={engagement}, "
            f"resonance={resonance:.1f}, topic={topic[:40]}"
        )

        return {
            'status': 'success',
            'action_type': action_type,
            'engagement': engagement,
            'resonance': resonance
        }

    except Exception as e:
        logger.error(f"Outcome tracking failed: {e}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)
        return {'status': 'error', 'error': str(e)}
