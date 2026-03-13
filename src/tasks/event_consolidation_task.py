"""
Event Consolidation Task - Merge related events into single coherent narratives.

WHAT: Groups synthesized_events by (subject, event_type), and when a group has
      2+ events, uses LLM to merge them into a single consolidated narrative
      with a unified timeline, outcome, and impact statement. Original events
      are marked as superseded (not deleted) for audit trail.

WHEN: Runs daily at 3:00 AM Pacific via Celery beat.
      Can also be triggered per-subject via consolidate_events_for_subject.

WHY:  Without consolidation, the context window fills with redundant events.
      Five separate "Jesse crisis" events are noise; one consolidated narrative
      with a complete timeline is actionable context. This keeps event-based
      memory clean and focused.

Example:
- Before: 5 separate crisis events (ER visit, ran away, police called, etc.)
- After: 1 consolidated narrative with full timeline and current status
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, '/app')

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# Use Fireworks 235B for quality synthesis
FIREWORKS_MODEL = "accounts/fireworks/models/kimi-k2-instruct-0905"


def get_event_groups(days_back: int = 14) -> List[Dict]:
    """
    Find groups of related events that should be consolidated.

    Returns groups where (subject, event_type) has 2+ events.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        cutoff = datetime.now(PST) - timedelta(days=days_back)

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Find groups with multiple events
            cursor.execute("""
                SELECT
                    subject,
                    event_type,
                    COUNT(*) as event_count,
                    ARRAY_AGG(id ORDER BY created_at ASC) as event_ids,
                    MIN(created_at) as first_event,
                    MAX(created_at) as last_event
                FROM synthesized_events
                WHERE created_at > %s
                  AND subject != 'SKIP'
                  AND (superseded_by IS NULL OR superseded_by = 0)
                  AND (is_consolidated IS NULL OR is_consolidated = FALSE)
                GROUP BY subject, event_type
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
            """, (cutoff,))

            groups = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return groups

    except Exception as e:
        logger.error(f"Failed to get event groups: {e}")
        return []


def get_events_by_ids(event_ids: List[int]) -> List[Dict]:
    """Get full event details for a list of IDs."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, event_type, subject, title, timeline, narrative,
                       outcome, impact, base_importance, created_at, user_email
                FROM synthesized_events
                WHERE id = ANY(%s)
                ORDER BY created_at ASC
            """, (event_ids,))

            events = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return events

    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        return []


def synthesize_consolidated_narrative(
    subject: str,
    event_type: str,
    events: List[Dict]
) -> Tuple[str, str, str, str]:
    """
    Use LLM to synthesize a consolidated narrative from multiple events.

    Returns:
        Tuple of (title, narrative, outcome, impact)
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get('FIREWORKS_API_KEY')
        )

        # Build timeline from all events
        all_timeline_entries = []
        all_titles = []

        for event in events:
            all_titles.append(event.get('title', ''))
            timeline = event.get('timeline', [])
            if isinstance(timeline, list):
                all_timeline_entries.extend(timeline)

            # Also include the narrative as context
            if event.get('narrative'):
                created = event.get('created_at')
                date_str = created.strftime('%Y-%m-%d') if created else 'unknown'
                all_timeline_entries.append({
                    'date': date_str,
                    'description': f"[From event: {event.get('title')}] {event.get('narrative', '')[:300]}"
                })

        # Sort timeline by date
        all_timeline_entries.sort(key=lambda x: x.get('date', ''))

        # Format timeline for prompt
        timeline_text = "\n".join([
            f"- [{t.get('date', 'unknown')}] {t.get('description', '')[:200]}"
            for t in all_timeline_entries[:30]  # Limit to 30 entries
        ])

        prompt = f"""You are consolidating {len(events)} related events about {subject} into one coherent narrative.

SUBJECT: {subject}
EVENT TYPE: {event_type}
ORIGINAL TITLES: {', '.join(all_titles)}

COMBINED TIMELINE:
{timeline_text}

TASK: Create a consolidated view with:

1. TITLE: A single comprehensive title that captures the overall situation
   (e.g., "Jesse's Mental Health Crisis - January 2026" instead of multiple separate events)

2. NARRATIVE: A 3-5 sentence summary that:
   - Explains what started the situation
   - Describes how it has progressed
   - Notes the current status
   - Captures the emotional impact

3. OUTCOME: One of: "ongoing", "resolved", "unknown"

4. IMPACT: One sentence about how this affects the people involved

FORMAT YOUR RESPONSE AS:
TITLE: [consolidated title]
NARRATIVE: [synthesized narrative]
OUTCOME: [status]
IMPACT: [impact statement]

/no_think
Write ONLY the formatted response:"""

        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()

        # Handle thinking tags
        if '<think>' in content:
            if '</think>' in content:
                content = content.split('</think>')[-1].strip()

        # Parse response
        title = ""
        narrative = ""
        outcome = "ongoing"
        impact = ""

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('TITLE:'):
                title = line[6:].strip()
            elif line.startswith('NARRATIVE:'):
                narrative = line[10:].strip()
            elif line.startswith('OUTCOME:'):
                outcome = line[8:].strip().lower()
                if outcome not in ['ongoing', 'resolved', 'unknown']:
                    outcome = 'ongoing'
            elif line.startswith('IMPACT:'):
                impact = line[7:].strip()

        # Fallback if parsing failed
        if not title:
            title = f"{subject}'s {event_type.title()} - Consolidated"
        if not narrative:
            narrative = f"Consolidated from {len(events)} events. " + events[0].get('narrative', '')

        return title, narrative, outcome, impact

    except Exception as e:
        logger.error(f"LLM consolidation failed: {e}")
        # Fallback
        return (
            f"{subject}'s {event_type.title()} - Consolidated",
            f"Consolidated from {len(events)} related events.",
            "ongoing",
            ""
        )


def create_consolidated_event(
    subject: str,
    event_type: str,
    events: List[Dict],
    title: str,
    narrative: str,
    outcome: str,
    impact: str
) -> Optional[int]:
    """
    Create the consolidated event and mark originals as superseded.

    Returns the ID of the consolidated event.
    """
    try:
        import psycopg2
        from psycopg2.extras import Json

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        # Merge timelines from all events
        merged_timeline = []
        all_source_ids = []
        total_importance = 0
        user_email = None

        for event in events:
            timeline = event.get('timeline', [])
            if isinstance(timeline, list):
                merged_timeline.extend(timeline)

            source_ids = event.get('source_message_ids', [])
            if isinstance(source_ids, list):
                all_source_ids.extend(source_ids)

            total_importance += event.get('base_importance', 5)

            if not user_email:
                user_email = event.get('user_email')

        # Sort timeline by date
        merged_timeline.sort(key=lambda x: x.get('date', ''))

        # Consolidated events get a +1 importance boost because they represent
        # a pattern of recurring events, which is inherently more significant
        avg_importance = total_importance / len(events)
        base_importance = min(10, avg_importance + 1)

        # Original event IDs for tracking
        original_ids = [e['id'] for e in events]

        with conn.cursor() as cursor:
            # Insert consolidated event
            cursor.execute("""
                INSERT INTO synthesized_events (
                    event_type, subject, title, timeline, narrative,
                    outcome, impact, source_message_ids, base_importance,
                    user_email, is_consolidated, consolidated_from,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                RETURNING id
            """, (
                event_type,
                subject,
                title,
                Json(merged_timeline),
                narrative,
                outcome,
                impact,
                list(set(all_source_ids)),  # Deduplicate
                base_importance,
                user_email,
                original_ids
            ))

            result = cursor.fetchone()
            consolidated_id = result[0] if result else None

            if consolidated_id:
                # Mark original events as superseded
                cursor.execute("""
                    UPDATE synthesized_events
                    SET superseded_by = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ANY(%s)
                """, (consolidated_id, original_ids))

                logger.info(f"Created consolidated event {consolidated_id}, superseded {len(original_ids)} events")

            conn.commit()

        conn.close()
        return consolidated_id

    except Exception as e:
        logger.error(f"Failed to create consolidated event: {e}")
        import traceback
        traceback.print_exc()
        return None


@celery_app.task(
    name='tasks.consolidate_events',
    bind=True,
    max_retries=2,
    soft_time_limit=300,
    time_limit=600
)
def consolidate_events(self, days_back: int = 14):
    """
    Daily job to merge related events into consolidated narratives.

    1. Group events by subject + event_type
    2. For groups with 2+ events, synthesize merged narrative
    3. Keep individual events but mark as superseded

    Args:
        days_back: Number of days to look back for consolidation

    Returns:
        Dict with consolidation results
    """
    try:
        logger.info(f"[CONSOLIDATION] Starting event consolidation (looking back {days_back} days)")

        # Find groups that need consolidation
        groups = get_event_groups(days_back=days_back)

        if not groups:
            logger.info("[CONSOLIDATION] No event groups need consolidation")
            return {
                'status': 'success',
                'groups_found': 0,
                'events_consolidated': 0
            }

        logger.info(f"[CONSOLIDATION] Found {len(groups)} groups to consolidate")

        total_consolidated = 0
        total_events_merged = 0

        for group in groups:
            subject = group['subject']
            event_type = group['event_type']
            event_ids = group['event_ids']

            logger.info(f"[CONSOLIDATION] Processing: {subject} {event_type} ({len(event_ids)} events)")

            # Get full event details
            events = get_events_by_ids(event_ids)

            if len(events) < 2:
                logger.warning(f"[CONSOLIDATION] Skipping - only {len(events)} events found")
                continue

            # Synthesize consolidated narrative
            title, narrative, outcome, impact = synthesize_consolidated_narrative(
                subject=subject,
                event_type=event_type,
                events=events
            )

            # Create consolidated event
            consolidated_id = create_consolidated_event(
                subject=subject,
                event_type=event_type,
                events=events,
                title=title,
                narrative=narrative,
                outcome=outcome,
                impact=impact
            )

            if consolidated_id:
                total_consolidated += 1
                total_events_merged += len(events)
                logger.info(f"[CONSOLIDATION] Created: {title} (id={consolidated_id})")

        logger.info(f"[CONSOLIDATION] Complete: {total_consolidated} consolidated events, {total_events_merged} events merged")

        return {
            'status': 'success',
            'groups_found': len(groups),
            'events_consolidated': total_consolidated,
            'events_merged': total_events_merged
        }

    except Exception as e:
        logger.error(f"[CONSOLIDATION] Error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


@celery_app.task(
    name='tasks.consolidate_events_for_subject',
    bind=True,
    max_retries=2,
    soft_time_limit=120,
    time_limit=180
)
def consolidate_events_for_subject(self, subject: str, event_type: str = None, days_back: int = 30):
    """
    Consolidate events for a specific subject (e.g., "Jesse").

    Useful for manual consolidation of high-priority subjects.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        logger.info(f"[CONSOLIDATION] Consolidating events for {subject}")

        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        cutoff = datetime.now(PST) - timedelta(days=days_back)

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT
                    subject,
                    event_type,
                    COUNT(*) as event_count,
                    ARRAY_AGG(id ORDER BY created_at ASC) as event_ids
                FROM synthesized_events
                WHERE subject = %s
                  AND created_at > %s
                  AND (superseded_by IS NULL OR superseded_by = 0)
                  AND (is_consolidated IS NULL OR is_consolidated = FALSE)
            """
            params = [subject, cutoff]

            if event_type:
                query += " AND event_type = %s"
                params.append(event_type)

            query += " GROUP BY subject, event_type HAVING COUNT(*) > 1"

            cursor.execute(query, params)
            groups = [dict(row) for row in cursor.fetchall()]

        conn.close()

        if not groups:
            return {'status': 'success', 'message': f'No event groups found for {subject}'}

        total_consolidated = 0

        for group in groups:
            events = get_events_by_ids(group['event_ids'])

            if len(events) < 2:
                continue

            title, narrative, outcome, impact = synthesize_consolidated_narrative(
                subject=subject,
                event_type=group['event_type'],
                events=events
            )

            consolidated_id = create_consolidated_event(
                subject=subject,
                event_type=group['event_type'],
                events=events,
                title=title,
                narrative=narrative,
                outcome=outcome,
                impact=impact
            )

            if consolidated_id:
                total_consolidated += 1

        return {
            'status': 'success',
            'subject': subject,
            'groups_consolidated': total_consolidated
        }

    except Exception as e:
        logger.error(f"[CONSOLIDATION] Error for {subject}: {e}")
        return {'status': 'error', 'error': str(e)}


# Convenience function for manual testing
def test_consolidation():
    """Test event consolidation manually."""
    print("Finding event groups that need consolidation...")
    groups = get_event_groups(days_back=30)

    for group in groups:
        print(f"\n{group['subject']} - {group['event_type']}: {group['event_count']} events")
        print(f"  IDs: {group['event_ids']}")
        print(f"  First: {group['first_event']}, Last: {group['last_event']}")


if __name__ == "__main__":
    test_consolidation()
