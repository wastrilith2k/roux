"""
Synthesized Events - Event-based narrative summaries with temporal context.

WHAT: Detects significant life events in conversation messages, groups
related facts into chronological timelines, and synthesizes each event into
a narrative paragraph via LLM. Events are stored in PostgreSQL with
embeddings for retrieval.

WHY: Scattered facts like "Jesse went to the ER" and "Jesse ran away on
Tuesday" are hard for the companion to reason about in isolation. Grouping
them into a single "Jesse Ran Away" event with a timeline and narrative
gives the companion a coherent story she can reference naturally.

HOW it fits:
  - The message handler calls detect_event_type() on each incoming user
    message. If an event is detected, extract_event_subject() identifies
    who it is about, and the event synthesis pipeline creates or updates
    a SynthesizedEvent.
  - context_builder.py calls get_recent_events() + format_events_for_prompt()
    to inject ongoing/recent events into the LLM prompt.
  - Events are periodically consolidated to merge duplicates.

Key difference from synthesized_biographies:
  - Biographies: static, theme-grouped ("James's work", "Jesse's health").
  - Events: dynamic, timeline-based ("Jesse Ran Away Jan 15-17", "Act-On
    Interview Process").

Event types: crisis, career, milestone, relationship, health, school, activities
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')

# Event type definitions with trigger keywords, importance boosts, and
# relevance decay windows. importance_boost is added to the average fact
# importance when computing the event's base_importance. decay_days controls
# how long the event stays in the "recent events" context window.
EVENT_TRIGGERS = {
    'crisis': {
        'keywords': [
            'police', 'hospital', 'emergency', 'ambulance', '911', 'er ', ' er',
            'emergency room', 'ran away', 'running away', 'took off', 'missing', 'found him', 'found her',
            'arrested', 'court', 'evicted', 'accident', 'crisis', 'meltdown',
            'died', 'death', 'funeral', 'suicide', 'suicidal', 'overdose', 'self-harm', 'self harm',
            'hospitalized', 'psych ward', 'psychiatric', 'baker act', 'sectioned',
            'behavioral', 'put holes in', 'destroyed', 'violent', 'aggressive'
        ],
        'importance_boost': 3,  # Crises get importance boost
        'decay_days': 14,  # Stay relevant for 2 weeks
    },
    'career': {
        'keywords': [
            'interview', 'job offer', 'hired', 'fired', 'quit', 'resigned',
            'promotion', 'raise', 'salary', 'laid off', 'layoff',
            'new job', 'starting at', 'first day', 'last day',
            'act-on', 'leantaas', 'snapsheet', 'cavallo'  # Known companies
        ],
        'importance_boost': 2,
        'decay_days': 30,
    },
    'milestone': {
        'keywords': [
            'birthday', 'anniversary', 'graduation', 'pregnant', 'baby',
            'engaged', 'married', 'wedding', 'moved', 'moving',
            'bought', 'purchased', 'new house', 'new car'
        ],
        'importance_boost': 2,
        'decay_days': 60,
    },
    'relationship': {
        'keywords': [
            'broke up', 'breakup', 'divorce', 'separated', 'back together',
            'dating', 'girlfriend', 'boyfriend', 'partner',
            'fight', 'argument', 'apologized', 'forgave'
        ],
        'importance_boost': 1,
        'decay_days': 21,
    },
    'health': {
        'keywords': [
            'diagnosis', 'diagnosed', 'surgery', 'operation', 'procedure',
            'doctor', 'specialist', 'oncologist', 'cardiologist', 'therapist',
            'medication', 'meds', 'prescription', 'treatment', 'therapy',
            'cancer', 'tumor', 'broken', 'fracture', 'sprain', 'injury',
            'sick', 'illness', 'disease', 'chronic', 'flare up', 'flare-up',
            'adhd', 'autism', 'depression', 'anxiety', 'bipolar', 'ptsd',
            'recovery', 'recovering', 'rehab', 'rehabilitation',
            'test results', 'biopsy', 'scan', 'mri', 'ct scan', 'x-ray'
        ],
        'importance_boost': 2,
        'decay_days': 30,
    },
    'school': {
        'keywords': [
            'school', 'class', 'teacher', 'principal', 'counselor',
            'homework', 'assignment', 'test', 'exam', 'finals', 'midterm',
            'grades', 'report card', 'gpa', 'failing', 'passed', 'failed',
            'expelled', 'suspended', 'detention', 'truant', 'skipping',
            'college', 'university', 'accepted', 'rejected', 'application',
            'scholarship', 'tuition', 'semester', 'quarter',
            'iep', 'special ed', '504 plan', 'learning disability',
            'back to school', 'first day of school', 'last day of school'
        ],
        'importance_boost': 1,
        'decay_days': 21,
    },
    'activities': {
        'keywords': [
            # Sports
            'game', 'match', 'tournament', 'championship', 'playoffs',
            'practice', 'tryouts', 'made the team', 'cut from',
            'soccer', 'football', 'basketball', 'baseball', 'softball',
            'hockey', 'lacrosse', 'volleyball', 'tennis', 'swimming',
            'track', 'cross country', 'wrestling', 'gymnastics',
            'won', 'lost', 'scored', 'goal', 'touchdown', 'home run',
            # Other activities
            'recital', 'performance', 'concert', 'play', 'musical',
            'competition', 'contest', 'award', 'trophy', 'medal',
            'dance', 'ballet', 'piano', 'violin', 'guitar', 'band', 'choir',
            'scouts', 'camp', 'camping', 'field trip'
        ],
        'importance_boost': 1,
        'decay_days': 14,
    }
}


@dataclass
class SynthesizedEvent:
    """A synthesized event with timeline and narrative."""
    id: Optional[int] = None
    event_type: str = ""  # crisis, career, milestone, relationship
    subject: str = ""  # Who/what the event is about
    title: str = ""  # Short title ("Jesse Ran Away", "Act-On Interview")
    timeline: List[Dict] = field(default_factory=list)  # [{date, description}, ...]
    narrative: str = ""  # LLM-synthesized summary paragraph
    outcome: str = "ongoing"  # resolved, ongoing, unknown
    impact: str = ""  # How it affected people
    source_message_ids: List[int] = field(default_factory=list)
    base_importance: float = 5.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    user_email: Optional[str] = None


def detect_event_type_keyword(text: str) -> Optional[Tuple[str, List[str]]]:
    """
    Fast keyword-based event detection (fallback/filter).

    Returns:
        Tuple of (event_type, matched_keywords) or None if no event detected
    """
    text_lower = text.lower()

    for event_type, config in EVENT_TRIGGERS.items():
        matched = [kw for kw in config['keywords'] if kw in text_lower]
        if matched:
            return (event_type, matched)

    return None


def detect_event_type(text: str, use_llm: bool = True) -> Optional[Tuple[str, List[str]]]:
    """
    Detect if text contains event triggers using LLM classification.

    Uses an LLM to classify whether the text describes a significant life event,
    with keyword examples as guidance. Falls back to keyword matching if LLM fails.

    Args:
        text: Message text to analyze
        use_llm: If True, use LLM; if False, use keyword fallback

    Returns:
        Tuple of (event_type, matched_keywords) or None if no event detected
    """
    # Skip very short messages
    if len(text.strip()) < 15:
        return None

    # For short messages or when LLM disabled, use keyword fallback
    if not use_llm or len(text) < 30:
        return detect_event_type_keyword(text)

    try:
        from src.llm.provider_factory import generate_sync

        prompt = f"""Classify if this message describes a significant life event. Return ONLY the event type or "none".

EVENT TYPES:
- crisis: Emergencies, police, hospital, someone running away, missing, accidents, death, arrest
  Examples: {', '.join(EVENT_TRIGGERS['crisis']['keywords'][:8])}

- career: Job changes, interviews, hiring, firing, promotions, layoffs, new jobs
  Examples: {', '.join(EVENT_TRIGGERS['career']['keywords'][:8])}

- milestone: Birthdays, graduations, engagements, weddings, pregnancies, moves, major purchases
  Examples: {', '.join(EVENT_TRIGGERS['milestone']['keywords'][:8])}

- relationship: Breakups, reconciliations, fights, new relationships, conflicts
  Examples: {', '.join(EVENT_TRIGGERS['relationship']['keywords'][:8])}

- health: Medical issues, diagnoses, treatments, therapy, medications, injuries, mental health
  Examples: {', '.join(EVENT_TRIGGERS['health']['keywords'][:8])}

- school: Education, grades, teachers, exams, college, suspensions, IEP, academic issues
  Examples: {', '.join(EVENT_TRIGGERS['school']['keywords'][:8])}

- activities: Sports, games, competitions, performances, recitals, awards, extracurriculars
  Examples: {', '.join(EVENT_TRIGGERS['activities']['keywords'][:8])}

MESSAGE: {text[:500]}

Respond with ONLY ONE WORD: crisis, career, milestone, relationship, health, school, activities, or none"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )

        if response:
            event_type = response.strip().lower()
            if event_type in EVENT_TRIGGERS:
                # Extract relevant keywords for context
                keywords = _extract_event_keywords(text, event_type)
                logger.debug(f"LLM detected {event_type} event: {keywords}")
                return (event_type, keywords)
            elif event_type == 'none':
                return None

        # Fallback to keyword detection
        return detect_event_type_keyword(text)

    except Exception as e:
        logger.warning(f"LLM event detection failed, using fallback: {e}")
        return detect_event_type_keyword(text)


def _extract_event_keywords(text: str, event_type: str) -> List[str]:
    """
    Extract relevant keywords from text for a detected event type.

    Used after LLM classification to provide context about what triggered detection.
    """
    text_lower = text.lower()
    keywords = EVENT_TRIGGERS.get(event_type, {}).get('keywords', [])

    # Find which keywords (if any) are present
    matched = [kw for kw in keywords if kw in text_lower]

    if matched:
        return matched[:5]  # Limit to top 5

    # If no keyword matches, return the event type as the keyword
    # Don't extract random words - they're often misleading
    return [event_type]


def _extract_subject_via_llm(text: str, event_type: str) -> Optional[str]:
    """
    Use LLM to extract event subject for ambiguous cases.

    Returns the subject name or None if extraction fails.
    """
    try:
        from openai import OpenAI
        from src.config.persona_config import get_persona_config
        _companion_name = get_persona_config().companion_short_name

        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get('FIREWORKS_API_KEY')
        )

        prompt = f"""KNOWN PEOPLE:
- James: The user (male, adult). His messages are labeled "James:". When James says "I", "me", "my" - it refers to James.
- {_companion_name}: James's companion. Her messages are labeled "{_companion_name}:".
- Jesse: James's 16-year-old son. Often has behavioral issues.
- Kyler: James's 12-year-old son.
- Alia: James's ex-wife.
- Carol: James's mother.
- Tuck: James's CAT (not a person - cannot have jobs, relationships, or human milestones)

EVENT TYPE: {event_type}
CONVERSATION:
{text[:500]}

Who is this {event_type} event PRIMARILY ABOUT?
The subject should be the person most AFFECTED by or CENTRAL to the event.

RULES:
- Lines starting with "James:" are James speaking. "I" in those lines = James.
- CRITICAL: For career events like interviews/jobs, ask "whose interview/job is it?"
  - "I talked to Jesse about MY interview" → James (it's James's interview)
  - "Jesse has a job interview" → Jesse (it's Jesse's interview)
- If James says "I had an interview" or "my interview" → subject is James
- For crisis events, ask "who is in crisis?" (running away, hospital, etc.)
- "my son" in James's messages = Jesse (older) unless Kyler specified
- "my ex" or "ex-wife" = Alia
- "my mom" = Carol
- Tuck the cat CANNOT be the subject of career/milestone/relationship events → return SKIP
- Only use "Family" if the event truly affects everyone equally

Reply with ONLY one word: James, {_companion_name}, Jesse, Kyler, Alia, Carol, Family, or SKIP

/no_think"""

        response = client.chat.completions.create(
            model="accounts/fireworks/models/kimi-k2-instruct-0905",
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content.strip()

        # Clean up response - get first word only
        result = result.split()[0] if result else None
        if result:
            result = result.strip('.,!?"\'')

        # Handle thinking tags from Qwen (shouldn't happen with /no_think but just in case)
        if result and '<' in result:
            return None

        # Validate result is a known subject
        from src.config.persona_config import get_persona_config
        _pc = get_persona_config()
        valid_subjects = {_pc.primary_user_name, _pc.companion_short_name, 'Jesse', 'Kyler', 'Alia', 'Carol', 'Family', 'SKIP'}
        if result in valid_subjects:
            return result

        return None

    except Exception as e:
        logger.warning(f"LLM subject extraction failed: {e}")
        return None


def extract_event_subject(text: str, event_type: str) -> str:
    """
    Determine who a detected event is primarily about.

    Uses a hybrid heuristic-then-LLM approach that improved subject
    attribution accuracy from ~52% to ~86%:

      1. Tuck filter: cat cannot be subject of human events -> SKIP or LLM.
      2. Clear first-person career/milestone indicators -> primary user.
      3. Single entity mentioned -> that entity.
      4. Relationship phrases ("my son") -> LLM for disambiguation.
      5. Multiple entities or ambiguous -> LLM.
      6. Fallback heuristics (first person -> primary user, etc.).

    Returns a person name or "SKIP" if the event should be ignored (e.g.,
    cat behavior misidentified as a human event).
    """
    text_lower = text.lower()

    # === Rule 1: Tuck filter ===
    # If Tuck is mentioned and no other humans, use LLM to determine
    # This handles cases like "Tuck claimed my pillow" - event is about cat behavior, not James
    tuck_keywords = ['tuck']
    # Build entity list dynamically from persona config
    _pc = get_persona_config()
    human_entities = [_pc.companion_short_name.lower(), _pc.primary_user_name.lower()]
    first_person = ['i ', "i'", 'my ', 'me ']

    has_tuck = any(kw in text_lower for kw in tuck_keywords)
    has_human = any(ent in text_lower for ent in human_entities)
    has_first_person = any(fp in text_lower for fp in first_person)

    if has_tuck and not has_human:
        # Tuck mentioned, no humans - use LLM to determine if this is pet behavior (SKIP)
        llm_result = _extract_subject_via_llm(text, event_type)
        if llm_result:
            if llm_result == "SKIP":
                logger.debug(f"Skipping event - LLM determined this is about pet behavior")
            return llm_result
        # Fallback: if LLM fails and no humans mentioned, skip
        logger.debug(f"Skipping event - only Tuck (cat) mentioned, LLM unavailable")
        return "SKIP"

    # === Rule 2: Clear first-person career/milestone = primary user ===
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    primary_user = _pc.primary_user_name
    companion = _pc.companion_short_name

    first_person_career = ['i had', 'i got', 'i have', "i'm ", 'my interview', 'my job', 'my work', 'i start']
    if event_type in ['career', 'milestone'] and any(fp in text_lower for fp in first_person_career):
        return primary_user

    # === Rule 3: Single entity mentioned → that entity ===
    entity_map = {
        primary_user.lower(): primary_user,
        companion.lower(): companion,
    }
    mentioned = [ent for ent in human_entities if ent in text_lower]

    if len(mentioned) == 1:
        return entity_map[mentioned[0]]

    # === Rule 4: Relationship indicators — delegate to LLM for entity-specific resolution ===
    relationship_phrases = ['my son', 'my kid', 'my child', 'my mom', 'my mother', 'my ex', 'my dad', 'my brother', 'my sister']
    if any(phrase in text_lower for phrase in relationship_phrases):
        llm_result = _extract_subject_via_llm(text, event_type)
        if llm_result and llm_result != "SKIP":
            return llm_result
        return primary_user  # Fallback to primary user

    # === Rule 5: Ambiguous - use LLM ===
    # Multiple entities mentioned or unclear subject
    if len(mentioned) > 1 or (not mentioned and not has_first_person):
        llm_result = _extract_subject_via_llm(text, event_type)
        if llm_result:
            if llm_result == "SKIP":
                logger.debug(f"LLM determined event should be skipped")
                return "SKIP"
            return llm_result

    # === Fallback heuristics ===
    # First person without clear career context
    if has_first_person:
        return primary_user

    # Default by event type
    if event_type in ['career', 'milestone', 'relationship']:
        return primary_user

    return 'Family'


class SynthesizedEventStore:
    """
    Stores and retrieves synthesized events.

    Uses PostgreSQL for storage with embedding-based retrieval.
    """

    def __init__(self):
        self._conn = None

    def _get_connection(self):
        """Get database connection."""
        import psycopg2

        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def store_event(self, event: SynthesizedEvent) -> Optional[int]:
        """Store or update a synthesized event."""
        conn = self._get_connection()

        try:
            from psycopg2.extras import Json

            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO synthesized_events (
                        event_type, subject, title, timeline, narrative,
                        outcome, impact, source_message_ids, base_importance,
                        embedding_vec, user_email, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (event_type, subject, title, user_email)
                    DO UPDATE SET
                        timeline = EXCLUDED.timeline,
                        narrative = EXCLUDED.narrative,
                        outcome = EXCLUDED.outcome,
                        impact = EXCLUDED.impact,
                        source_message_ids = EXCLUDED.source_message_ids,
                        base_importance = EXCLUDED.base_importance,
                        embedding_vec = EXCLUDED.embedding_vec,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    event.event_type,
                    event.subject,
                    event.title,
                    Json(event.timeline),
                    event.narrative,
                    event.outcome,
                    event.impact,
                    event.source_message_ids,
                    event.base_importance,
                    event.embedding,
                    event.user_email
                ))

                result = cursor.fetchone()
                conn.commit()

                if result:
                    logger.info(f"Stored event: {event.title} (type={event.event_type})")
                    return result[0]
                return None

        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            conn.rollback()
            return None

    def get_recent_events(
        self,
        user_email: str,
        days_back: int = 14,
        event_types: List[str] = None,
        max_events: int = 10
    ) -> List[SynthesizedEvent]:
        """Get recent synthesized events."""
        conn = self._get_connection()

        try:
            from psycopg2.extras import RealDictCursor

            cutoff = datetime.now(PST) - timedelta(days=days_back)

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Exclude superseded events (they've been consolidated)
                if event_types:
                    cursor.execute("""
                        SELECT * FROM synthesized_events
                        WHERE (user_email = %s OR user_email IS NULL)
                          AND created_at > %s
                          AND event_type = ANY(%s)
                          AND (superseded_by IS NULL OR superseded_by = 0)
                        ORDER BY base_importance DESC, created_at DESC
                        LIMIT %s
                    """, (user_email, cutoff, event_types, max_events))
                else:
                    cursor.execute("""
                        SELECT * FROM synthesized_events
                        WHERE (user_email = %s OR user_email IS NULL)
                          AND created_at > %s
                          AND (superseded_by IS NULL OR superseded_by = 0)
                        ORDER BY base_importance DESC, created_at DESC
                        LIMIT %s
                    """, (user_email, cutoff, max_events))

                rows = cursor.fetchall()

                events = []
                for row in rows:
                    events.append(SynthesizedEvent(
                        id=row['id'],
                        event_type=row['event_type'],
                        subject=row['subject'],
                        title=row['title'],
                        timeline=row['timeline'] or [],
                        narrative=row['narrative'],
                        outcome=row['outcome'],
                        impact=row['impact'],
                        source_message_ids=row['source_message_ids'] or [],
                        base_importance=row['base_importance'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        user_email=row['user_email']
                    ))

                return events

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

    def get_event_by_title(
        self,
        title: str,
        user_email: str
    ) -> Optional[SynthesizedEvent]:
        """Get a specific event by title."""
        conn = self._get_connection()

        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM synthesized_events
                    WHERE title = %s
                      AND (user_email = %s OR user_email IS NULL)
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (title, user_email))

                row = cursor.fetchone()

                if row:
                    return SynthesizedEvent(
                        id=row['id'],
                        event_type=row['event_type'],
                        subject=row['subject'],
                        title=row['title'],
                        timeline=row['timeline'] or [],
                        narrative=row['narrative'],
                        outcome=row['outcome'],
                        impact=row['impact'],
                        source_message_ids=row['source_message_ids'] or [],
                        base_importance=row['base_importance'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        user_email=row['user_email']
                    )
                return None

        except Exception as e:
            logger.error(f"Failed to get event: {e}")
            return None

    def update_event_outcome(
        self,
        event_id: int,
        outcome: str,
        impact: str = None
    ) -> bool:
        """Update an event's outcome status."""
        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                if impact:
                    cursor.execute("""
                        UPDATE synthesized_events
                        SET outcome = %s, impact = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (outcome, impact, event_id))
                else:
                    cursor.execute("""
                        UPDATE synthesized_events
                        SET outcome = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (outcome, event_id))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to update event outcome: {e}")
            conn.rollback()
            return False


class EventSynthesizer:
    """
    Synthesizes events from facts into narrative summaries.

    Uses LLM to create coherent narratives from event timelines.
    """

    def __init__(self):
        self.store = SynthesizedEventStore()

    def synthesize_event(
        self,
        event_type: str,
        subject: str,
        title: str,
        facts: List[Dict],
        user_email: str = None
    ) -> Optional[SynthesizedEvent]:
        """
        Synthesize an event from a list of facts.

        Args:
            event_type: Type of event (crisis, career, milestone, relationship)
            subject: Who the event is about
            title: Short title for the event
            facts: List of fact dicts with 'object', 'created_at', 'importance'
            user_email: User email for storage

        Returns:
            SynthesizedEvent if successful, None otherwise
        """
        if not facts:
            return None

        # Sort facts chronologically
        sorted_facts = sorted(
            facts,
            key=lambda f: f.get('created_at', datetime.min)
        )

        # Build timeline
        timeline = []
        source_ids = []
        total_importance = 0

        for fact in sorted_facts:
            created_at = fact.get('created_at')
            if created_at:
                date_str = created_at.strftime('%Y-%m-%d') if hasattr(created_at, 'strftime') else str(created_at)[:10]
            else:
                date_str = 'unknown'

            timeline.append({
                'date': date_str,
                'description': fact.get('object', str(fact))
            })

            if fact.get('id'):
                source_ids.append(fact['id'])

            total_importance += fact.get('importance', 5)

        # Calculate average importance with event type boost
        avg_importance = total_importance / len(facts)
        boost = EVENT_TRIGGERS.get(event_type, {}).get('importance_boost', 0)
        base_importance = min(10, avg_importance + boost)

        # Generate narrative via LLM
        narrative = self._generate_narrative(event_type, subject, title, timeline)

        # Create event object
        event = SynthesizedEvent(
            event_type=event_type,
            subject=subject,
            title=title,
            timeline=timeline,
            narrative=narrative,
            outcome='ongoing',  # Default, can be updated later
            source_message_ids=source_ids,
            base_importance=base_importance,
            user_email=user_email
        )

        # Store and return
        event_id = self.store.store_event(event)
        if event_id:
            event.id = event_id
            return event

        return None

    def _generate_narrative(
        self,
        event_type: str,
        subject: str,
        title: str,
        timeline: List[Dict]
    ) -> str:
        """Generate narrative paragraph using LLM."""
        try:
            from src.llm.provider_factory import generate_sync

            # Format timeline for prompt
            timeline_text = "\n".join([
                f"- [{t['date']}] {t['description']}"
                for t in timeline
            ])

            prompt = f"""Synthesize these facts about {subject}'s {event_type} event "{title}" into a coherent 2-3 sentence narrative.

TIMELINE:
{timeline_text}

Write a brief narrative that captures:
1. What triggered the event
2. How it progressed
3. Current status (if known)

Write ONLY the narrative paragraph, nothing else:"""

            response = generate_sync(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            if response:
                return response.strip()

            # Fallback: simple concatenation
            return f"{subject}'s {event_type}: " + "; ".join([t['description'] for t in timeline[:3]])

        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}")
            # Fallback: simple summary
            return f"{subject}'s {event_type}: " + "; ".join([t['description'] for t in timeline[:3]])


# Singleton instances
_event_store: Optional[SynthesizedEventStore] = None
_event_synthesizer: Optional[EventSynthesizer] = None


def get_event_store() -> SynthesizedEventStore:
    """Get singleton event store."""
    global _event_store
    if _event_store is None:
        _event_store = SynthesizedEventStore()
    return _event_store


def get_event_synthesizer() -> EventSynthesizer:
    """Get singleton event synthesizer."""
    global _event_synthesizer
    if _event_synthesizer is None:
        _event_synthesizer = EventSynthesizer()
    return _event_synthesizer


def get_recent_events(
    user_email: str,
    days_back: int = 14,
    max_events: int = 5
) -> List[SynthesizedEvent]:
    """
    Convenience function to get recent events.

    This is the main entry point used by context_builder.
    """
    store = get_event_store()
    return store.get_recent_events(
        user_email=user_email,
        days_back=days_back,
        max_events=max_events
    )


def format_events_for_prompt(events: List[SynthesizedEvent]) -> str:
    """
    Format events for inclusion in prompt.

    Returns formatted string with event narratives and status.
    """
    if not events:
        return ""

    lines = ["[RECENT EVENTS - Ongoing situations to remember]"]

    for event in events:
        status_marker = ""
        if event.outcome == 'resolved':
            status_marker = " [RESOLVED]"
        elif event.outcome == 'ongoing':
            status_marker = " [ONGOING]"

        lines.append(f"\n**{event.title}**{status_marker}")
        lines.append(event.narrative)

        if event.impact:
            lines.append(f"Impact: {event.impact}")

    return "\n".join(lines)
