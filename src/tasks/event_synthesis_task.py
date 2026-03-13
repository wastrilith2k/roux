"""
Event Synthesis Task - Detect significant life events and build narrative timelines.

WHAT: Scans messages for event triggers (crisis keywords like "ER," career
      keywords like "interview," milestones like "birthday"). When triggered,
      gathers related facts and messages from the past 14-30 days, groups
      them into a timeline, and uses LLM to synthesize a narrative summary.
      Stores as synthesized_events for context retrieval.

WHEN: detect_and_synthesize_events fires after every message exchange.
      batch_process_events can be run manually for backfilling.

WHY:  Individual facts ("Jesse went to the ER," "Jesse ran away," "police were
      called") are scattered. Event synthesis connects them into coherent
      narratives ("Jesse's Mental Health Crisis - January 2026") that give
      the companion a holistic understanding of ongoing situations rather
      than fragmented data points.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, '/app')

logger = logging.getLogger(__name__)

from src.celery_app import celery_app

PST = ZoneInfo('America/Los_Angeles')


def get_related_facts(
    user_email: str,
    event_type: str,
    subject: str,
    keywords: List[str],
    days_back: int = 30
) -> List[Dict]:
    """
    Get facts related to a detected event.

    Searches the fact store for facts matching the event keywords and subject.
    """
    try:
        from src.memory.fact_store import get_fact_store

        store = get_fact_store()

        # Get facts for the subject (fact_store doesn't filter by user_email)
        all_facts = store.get_facts_for_subject(subject)

        # Filter to relevant facts (within timeframe and matching keywords)
        cutoff = datetime.now(PST) - timedelta(days=days_back)
        relevant_facts = []

        for fact in all_facts:
            created_at = fact.get('created_at')
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=PST)

            # Skip old facts
            if created_at and created_at < cutoff:
                continue

            # Check if fact content matches any keywords
            fact_text = fact.get('object', '').lower()
            if any(kw in fact_text for kw in keywords):
                relevant_facts.append(fact)

        return relevant_facts

    except Exception as e:
        logger.error(f"Failed to get related facts: {e}")
        return []


def get_related_messages(
    user_email: str,
    keywords: List[str],
    days_back: int = 14
) -> List[Dict]:
    """
    Get messages related to an event.

    Used to build event timeline from conversation history.
    """
    try:
        from src.database.db import get_db

        db = get_db()
        cutoff = datetime.now(PST) - timedelta(days=days_back)

        with db._get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build keyword patterns for LIKE query
            patterns = [f'%{kw}%' for kw in keywords]

            cursor.execute("""
                SELECT id, sender_name, message_text, timestamp
                FROM messages
                WHERE email = %s
                  AND timestamp > %s
                  AND LOWER(message_text) LIKE ANY(%s)
                ORDER BY timestamp ASC
                LIMIT 50
            """, (user_email, cutoff, patterns))

            messages = [dict(row) for row in cursor.fetchall()]
            cursor.close()

        return messages

    except Exception as e:
        logger.error(f"Failed to get related messages: {e}")
        return []


def generate_event_title(event_type: str, subject: str, keywords: List[str]) -> str:
    """Generate a descriptive title for an event."""
    keyword = keywords[0] if keywords else event_type

    title_templates = {
        'crisis': f"{subject}'s {keyword.title()} Situation",
        'career': f"{subject}'s {keyword.title()} at Work",
        'milestone': f"{subject}'s {keyword.title()}",
        'relationship': f"{subject}'s Relationship {keyword.title()}"
    }

    return title_templates.get(event_type, f"{subject}'s {keyword.title()}")


@celery_app.task(
    name='tasks.event_synthesis.detect_and_synthesize',
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=120,
    time_limit=180
)
def detect_and_synthesize_events(
    self,
    user_email: str,
    user_message: str,
    companion_response: str,
    message_id: int = None
):
    """
    Detect events in a message and synthesize if found.

    This task is triggered after each message exchange to detect
    new events or updates to existing events.
    """
    try:
        from src.memory.synthesized_events import (
            detect_event_type,
            extract_event_subject,
            get_event_synthesizer,
            get_event_store,
            EVENT_TRIGGERS
        )

        logger.info(f"Checking for events in message from {user_email}")

        # Check user message for event triggers
        detection = detect_event_type(user_message)
        if not detection:
            # Also check the companion's response (might contain event context)
            detection = detect_event_type(companion_response)

        if not detection:
            logger.debug("No event triggers detected")
            return {'status': 'no_event', 'message': 'No event triggers detected'}

        event_type, keywords = detection
        # Format with speaker labels so LLM knows who said what
        formatted_text = f"James: {user_message}\nCompanion: {companion_response}"
        subject = extract_event_subject(formatted_text, event_type)

        # SKIP means the subject isn't applicable (e.g., "Tuck the cat ran away"
        # triggers crisis keywords but cats can't be subjects of human crisis events)
        if subject == "SKIP":
            logger.info(f"Event skipped - subject not applicable (e.g., pet behavior)")
            return {'status': 'skipped', 'message': 'Event subject not applicable'}

        logger.info(f"Detected {event_type} event for {subject} (keywords: {keywords})")

        # Get related facts and messages
        related_facts = get_related_facts(
            user_email=user_email,
            event_type=event_type,
            subject=subject,
            keywords=keywords,
            days_back=30
        )

        related_messages = get_related_messages(
            user_email=user_email,
            keywords=keywords,
            days_back=14
        )

        # Convert messages to fact-like format for synthesis
        facts_for_synthesis = []

        for fact in related_facts:
            facts_for_synthesis.append({
                'id': fact.get('id'),
                'object': fact.get('object', ''),
                'created_at': fact.get('created_at'),
                'importance': fact.get('importance', 5)
            })

        for msg in related_messages:
            facts_for_synthesis.append({
                'id': msg.get('id'),
                'object': f"{msg['sender_name']}: {msg['message_text'][:200]}",
                'created_at': msg.get('timestamp'),
                'importance': 5
            })

        if not facts_for_synthesis:
            # Create fact from current message
            facts_for_synthesis.append({
                'id': message_id,
                'object': user_message[:500],
                'created_at': datetime.now(PST),
                'importance': 6
            })

        # Generate title and synthesize
        title = generate_event_title(event_type, subject, keywords)

        synthesizer = get_event_synthesizer()
        event = synthesizer.synthesize_event(
            event_type=event_type,
            subject=subject,
            title=title,
            facts=facts_for_synthesis,
            user_email=user_email
        )

        if event:
            logger.info(f"Synthesized event: {event.title} (id={event.id})")
            return {
                'status': 'success',
                'event_id': event.id,
                'event_type': event_type,
                'subject': subject,
                'title': title,
                'facts_used': len(facts_for_synthesis)
            }

        return {'status': 'failed', 'message': 'Synthesis returned no event'}

    except Exception as e:
        logger.error(f"Event synthesis task failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


@celery_app.task(
    name='tasks.event_synthesis.batch_process',
    bind=True,
    soft_time_limit=600,
    time_limit=900
)
def batch_process_events(self, user_email: str, days_back: int = 7):
    """
    Batch process messages to detect and synthesize events.

    Used for backfilling or periodic event detection.
    """
    try:
        from src.database.db import get_db
        from src.memory.synthesized_events import (
            detect_event_type,
            extract_event_subject,
            get_event_synthesizer,
            EVENT_TRIGGERS
        )

        logger.info(f"Batch processing events for {user_email} (last {days_back} days)")

        db = get_db()
        cutoff = datetime.now(PST) - timedelta(days=days_back)

        # Get messages with event keywords
        all_keywords = []
        for config in EVENT_TRIGGERS.values():
            all_keywords.extend(config['keywords'])

        patterns = [f'%{kw}%' for kw in all_keywords]

        with db._get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT id, sender_name, message_text, timestamp
                FROM messages
                WHERE email = %s
                  AND timestamp > %s
                  AND sender_name = 'User'
                  AND LOWER(message_text) LIKE ANY(%s)
                ORDER BY timestamp ASC
            """, (user_email, cutoff, patterns))

            messages = [dict(row) for row in cursor.fetchall()]
            cursor.close()

        logger.info(f"Found {len(messages)} messages with event keywords")

        # Group messages by detected event type
        events_detected = {}  # {(event_type, subject): [messages]}

        for msg in messages:
            detection = detect_event_type(msg['message_text'])
            if detection:
                event_type, keywords = detection
                subject = extract_event_subject(msg['message_text'], event_type)
                key = (event_type, subject)

                if key not in events_detected:
                    events_detected[key] = []
                events_detected[key].append(msg)

        logger.info(f"Detected {len(events_detected)} unique event groups")

        # Synthesize each event group
        synthesizer = get_event_synthesizer()
        events_created = 0

        for (event_type, subject), msgs in events_detected.items():
            # Get keywords from first message
            detection = detect_event_type(msgs[0]['message_text'])
            keywords = detection[1] if detection else [event_type]

            # Convert messages to facts
            facts = []
            for msg in msgs:
                facts.append({
                    'id': msg['id'],
                    'object': msg['message_text'][:500],
                    'created_at': msg['timestamp'],
                    'importance': 5
                })

            title = generate_event_title(event_type, subject, keywords)

            event = synthesizer.synthesize_event(
                event_type=event_type,
                subject=subject,
                title=title,
                facts=facts,
                user_email=user_email
            )

            if event:
                events_created += 1
                logger.info(f"Created event: {title}")

        return {
            'status': 'success',
            'messages_processed': len(messages),
            'event_groups_found': len(events_detected),
            'events_created': events_created
        }

    except Exception as e:
        logger.error(f"Batch event processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


# Convenience function for testing
def test_event_detection():
    """Test event detection with sample messages."""
    test_messages = [
        "Jesse ran away last night and I called the police",
        "I have an interview at Act-On tomorrow at 2pm",
        "Kyler's birthday is next week",
        "Alia and I got into a huge fight yesterday"
    ]

    from src.memory.synthesized_events import detect_event_type, extract_event_subject

    for msg in test_messages:
        detection = detect_event_type(msg)
        if detection:
            event_type, keywords = detection
            subject = extract_event_subject(msg, event_type)
            print(f"Message: {msg[:50]}...")
            print(f"  Event: {event_type}, Subject: {subject}, Keywords: {keywords}")
        else:
            print(f"Message: {msg[:50]}... -> No event detected")


if __name__ == "__main__":
    test_event_detection()
