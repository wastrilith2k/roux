"""
Fact Extraction Task - Extract and store structured facts from conversations.

WHAT: Uses LLM (Fireworks Kimi K2) to extract facts from each message exchange,
      validates them against known entity profiles to reject hallucinations,
      routes sensitive facts (medical, legal) to an approval queue, and stores
      the rest directly. High-importance facts (>= 7) trigger biography refresh.

WHEN: Fires asynchronously after every message exchange via Celery.

WHY:  Facts are the companion's long-term memory about people and events.
      Without extraction, the companion would only remember what's in the
      current context window. Entity grounding prevents the common LLM failure
      of inventing children, wrong names, or contradicting known information.

Based on Mem0's proven patterns:
1. Ground extraction with known entities to prevent hallucination
2. Simple fact strings, not complex triples
3. Explicit categories of what to extract
4. Clear DON'T examples to prevent common mistakes
5. Memory operations: ADD, UPDATE, DELETE, NONE
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

# Use Fireworks model for quality extraction
from src.config.models import FIREWORKS_DEFAULT_MODEL as FIREWORKS_MODEL

def get_entity_grounding() -> str:
    """Get dynamic entity grounding from profile manager."""
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    try:
        from src.memory.entity_profile_manager import get_entity_manager
        manager = get_entity_manager()
        return manager.get_grounding_context()
    except Exception as e:
        logger.warning(f"Failed to load entity profiles, using fallback: {e}")
        user = _pc.primary_user_name
        companion = _pc.companion_short_name
        # Fallback if profile loading fails
        return f"""KNOWN ENTITIES (ground truth - DO NOT contradict):
- {user}: User, lives with {companion}. Has two sons Jesse and Kyler. Drinks tea NOT coffee.
- {companion}: AI companion, lives with {user}. Has a cat named Tuck. No children.
- Jesse: {user}'s son
- Kyler: {user}'s son
- Alia: {user}'s wife (separated). Mother of Jesse and Kyler.

CRITICAL RULES:
- {companion} has NO children. NEVER extract facts saying she has children.
- {user}'s children are Jesse and Kyler, NOT Nicholas or any other names.
"""


@celery_app.task(
    name='tasks.fact_extraction_task.extract_facts',
    bind=True,
    max_retries=2,
    soft_time_limit=60,
    time_limit=90
)
def extract_facts(self, user_email: str, user_message: str, companion_response: str, message_id: int = None):
    """
    Celery task for async fact extraction.

    Extracts facts from conversation, validates against known entities,
    routes sensitive facts to approval queue, stores others directly.
    """
    try:
        logger.info(f"[FACT_EXTRACTION] Starting for {user_email}")

        # Extract facts via LLM with entity grounding
        facts = extract_facts_via_llm(user_message, companion_response)

        if not facts:
            logger.info("[FACT_EXTRACTION] No facts extracted")
            return {'status': 'success', 'facts_extracted': 0, 'facts_stored': 0, 'facts_pending': 0}

        # Validate against known entities - filter out hallucinations
        validated_facts = validate_facts(facts)

        if len(validated_facts) < len(facts):
            logger.info(f"[FACT_EXTRACTION] Filtered {len(facts) - len(validated_facts)} invalid facts")

        if not validated_facts:
            logger.info("[FACT_EXTRACTION] No valid facts after filtering")
            return {'status': 'success', 'facts_extracted': len(facts), 'facts_stored': 0, 'facts_pending': 0}

        logger.info(f"[FACT_EXTRACTION] Extracted {len(validated_facts)} valid facts")

        # Route facts: sensitive -> pending, others -> direct storage
        from src.memory.fact_approval import get_approval_service, FactSensitivity
        approval_service = get_approval_service()

        direct_facts = []
        pending_facts = []

        for fact in validated_facts:
            sensitivity, reason = approval_service.detect_sensitivity(fact)

            if sensitivity == FactSensitivity.NONE:
                direct_facts.append(fact)
            else:
                # Add to pending queue
                pending_id = approval_service.add_pending_fact(
                    fact=fact,
                    sensitivity=sensitivity,
                    reason=reason,
                    user_email=user_email,
                    message_id=message_id,
                    source_message=user_message[:200]
                )
                if pending_id:
                    pending_facts.append({
                        'id': pending_id,
                        'fact': fact,
                        'sensitivity': sensitivity.value,
                        'reason': reason
                    })
                    logger.info(f"[FACT_EXTRACTION] Queued for approval: {fact.get('fact', '')[:50]}...")

        # Store non-sensitive facts directly
        stored_count = 0
        if direct_facts:
            stored_count = store_facts(direct_facts, user_email, message_id)
            logger.info(f"[FACT_EXTRACTION] Stored {stored_count} facts directly")

        # Emit WebSocket events for pending facts
        if pending_facts:
            emit_approval_requests(user_email, pending_facts)

        logger.info(f"[FACT_EXTRACTION] Complete: {stored_count} stored, {len(pending_facts)} pending approval")

        return {
            'status': 'success',
            'facts_extracted': len(facts),
            'facts_validated': len(validated_facts),
            'facts_stored': stored_count,
            'facts_pending': len(pending_facts)
        }

    except Exception as e:
        logger.error(f"[FACT_EXTRACTION] Error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def extract_facts_via_llm(user_message: str, companion_response: str) -> List[Dict[str, Any]]:
    """
    Use LLM to extract facts from conversation with Mem0-style prompt.
    """
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    entity_grounding = get_entity_grounding()
    prompt = f"""{entity_grounding}

---

CONVERSATION TO ANALYZE:
{_pc.primary_user_name}: {user_message}
{_pc.companion_short_name}: {companion_response}

---

TASK: Extract NEW facts worth remembering long-term.

CATEGORIES TO EXTRACT:
1. **Personal Preferences** - Likes, dislikes, favorites
   Example: "James prefers peach tea over other flavors"

2. **Important Details** - Names, relationships, dates, places
   Example: "Jesse has a soccer game on Friday"

3. **Plans & Events** - Upcoming activities, goals, intentions
   Example: "James is planning to take the kids camping in August"

4. **Relationship Dynamics** - How people interact, feel about each other
   Example: "The companion feels safe when James checks in during intimate moments"

5. **Work & Professional** - Job details, career, projects
   Example: "James is job hunting for AI-focused roles"

6. **Crisis & Medical Events** - Emergencies, hospitalizations, mental health crises, behavioral incidents
   IMPORTANT: Preserve EXACT names, medical terms, dates, and frequencies
   Example: "Jesse went to the ER on January 11th for suicidal ideation"
   Example: "Jesse has had 7 ER visits in 3 weeks for mental health crises"
   Example: "Jesse was drawing on walls in blood on January 14th"
   Example: "Jesse ran away from home on January 20th"

DO NOT EXTRACT:
- Temporary states ("I'm tired", "good morning")
- Things already in KNOWN ENTITIES above
- Vague statements without specific information
- Roleplay actions (*hugs*, *smiles*) - these aren't facts
- Anything that contradicts KNOWN ENTITIES

CRITICAL RULES FOR EXTRACTION:
- ALWAYS use the actual person's name (Jesse, Kyler, James, Alia) - NEVER replace with "someone" or "a family member"
- PRESERVE exact medical terms: "ER", "emergency room", "suicidal ideation", "self-harm", "psychiatric"
- PRESERVE specific dates when mentioned (e.g., "on January 11th", "on the 14th")
- PRESERVE counts and frequencies (e.g., "7 times", "multiple ER visits")
- Crisis and medical events are HIGH PRIORITY - extract them even if they seem temporary

OUTPUT FORMAT (JSON):
{{
  "facts": [
    {{
      "subject": "{_pc.primary_user_name}|{_pc.companion_short_name}|Jesse|Kyler|Alia|other",
      "fact": "Clear, specific fact statement with names and dates preserved",
      "category": "preference|detail|plan|relationship|work|crisis",
      "confidence": 0.7-1.0
    }}
  ]
}}

If no extractable facts, return: {{"facts": []}}

/no_think
Return ONLY valid JSON:"""

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get('FIREWORKS_API_KEY')
        )

        response = client.chat.completions.create(
            model=FIREWORKS_MODEL,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content.strip()

        # Handle thinking tags
        if '<think>' in content:
            if '</think>' in content:
                content = content.split('</think>')[-1].strip()

        # Handle markdown code blocks
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        return result.get('facts', []) if isinstance(result, dict) else []

    except json.JSONDecodeError as e:
        logger.error(f"[FACT_EXTRACTION] JSON parse error: {e}")
        return []
    except Exception as e:
        logger.error(f"[FACT_EXTRACTION] LLM error: {e}")
        return []


def validate_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate extracted facts against known entity rules.
    Filter out hallucinations and contradictions.
    """
    validated = []

    # Get companion name for validation patterns
    from src.config.persona_config import get_persona_config
    _pc = get_persona_config()
    companion_name_lower = _pc.companion_short_name.lower()

    # Invalid fact patterns — loaded from persona.yaml so instances can
    # configure their own without editing source code.
    INVALID_PATTERNS = [
        (p[0], p[1]) for p in _pc.get_resolved_invalid_fact_patterns()
    ]

    # Valid children for the primary user — loaded from persona.yaml
    USER_CHILDREN = set(_pc.user_children)
    user_name_lower = _pc.primary_user_name.lower()

    for fact in facts:
        subject = fact.get('subject', '').lower().strip()
        fact_text = fact.get('fact', '').lower()

        # Skip empty facts
        if not fact_text or len(fact_text) < 5:
            continue

        # Check for invalid patterns
        is_invalid = False
        for invalid_subject, invalid_keyword in INVALID_PATTERNS:
            if invalid_subject in subject and invalid_keyword in fact_text:
                logger.warning(f"[FACT_VALIDATION] Rejected hallucination: {fact}")
                is_invalid = True
                break

        if is_invalid:
            continue

        # If fact mentions the user having a child, verify against configured children
        if USER_CHILDREN and user_name_lower in subject and ('child' in fact_text or 'son' in fact_text or 'daughter' in fact_text):
            has_valid_child = any(child in fact_text for child in USER_CHILDREN)
            if not has_valid_child:
                logger.warning(f"[FACT_VALIDATION] Rejected wrong child name: {fact}")
                continue

        validated.append(fact)

    return validated


# --- Predicate Normalization ---
# Converts free-text category names from the LLM into consistent semantic
# predicates for database storage (e.g., "preference" -> "prefers")
PREDICATE_NORMALIZATION = {
    'preference': 'prefers',
    'personal preference': 'prefers',
    'personal preferences': 'prefers',
    'detail': 'has_detail',
    'important detail': 'has_detail',
    'important details': 'has_detail',
    'plan': 'plans_to',
    'plans': 'plans_to',
    'plans & events': 'plans_to',
    'relationship': 'relates_to',
    'relationship dynamics': 'relates_to',
    'work': 'works_on',
    'work & professional': 'works_on',
    'general': 'has_detail',
    'crisis': 'crisis_event',
    'crisis & medical': 'crisis_event',
    'crisis & medical events': 'crisis_event',
    'medical': 'crisis_event',
    'emergency': 'crisis_event',
}

# Minimum importance threshold - facts below this are too transient to store
# (e.g., "James is tired" scores ~2, "Jesse has a new therapist" scores ~7)
MIN_IMPORTANCE_THRESHOLD = 4


def normalize_predicate(raw_predicate: str) -> str:
    """Normalize predicate to consistent lowercase semantic form."""
    key = raw_predicate.lower().strip()
    return PREDICATE_NORMALIZATION.get(key, key.replace(' ', '_'))


def store_facts(facts: List[Dict[str, Any]], user_email: str, message_id: int = None) -> int:
    """
    Store validated facts in PostgreSQL.
    Converts new format to legacy format for fact_store.
    Applies predicate normalization and importance filtering.
    Triggers biography refresh for subjects with high-importance facts.
    """
    from src.memory.fact_store import get_fact_store
    from src.memory.importance_scorer import score_importance

    store = get_fact_store()
    stored_count = 0
    high_importance_subjects = set()  # Track subjects needing biography refresh

    for fact in facts:
        try:
            subject = fact.get('subject', 'Unknown')
            fact_text = fact.get('fact', '')
            category = fact.get('category', 'general')
            confidence = fact.get('confidence', 0.7)

            # Skip empty
            if not fact_text or len(fact_text.strip()) < 5:
                continue

            # Normalize predicate (category → semantic verb form)
            predicate = normalize_predicate(category)
            obj = fact_text

            # Score importance
            importance = score_importance(f"{subject}: {fact_text}")
            if importance is None:
                importance = 5

            # Skip low-importance facts (temporary states, mundane activities)
            if importance < MIN_IMPORTANCE_THRESHOLD:
                logger.debug(f"Skipping low-importance fact ({importance}): {fact_text[:50]}...")
                continue

            # Store the fact
            fact_id = store.store_fact(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=confidence,
                importance=importance,
                context=f"Category: {category}",
                source='conversation',
                user_email=user_email,
                message_id=message_id
            )

            if fact_id:
                stored_count += 1
                logger.debug(f"Stored fact {fact_id}: {subject} - {fact_text[:50]}... (importance: {importance})")

                # Track high-importance facts for biography refresh
                if importance >= 7:
                    high_importance_subjects.add(subject)
                    logger.info(f"[FACT_EXTRACTION] High-importance fact ({importance}) for {subject}: {fact_text[:50]}...")

        except Exception as e:
            logger.error(f"Error storing fact: {e}")
            continue

    # Trigger biography refresh for subjects with high-importance facts
    if high_importance_subjects:
        trigger_biography_refresh(user_email, high_importance_subjects)

    return stored_count


def trigger_biography_refresh(user_email: str, subjects: set) -> None:
    """
    Trigger asynchronous biography refresh for subjects with new high-importance facts.
    Uses Celery to avoid blocking the fact extraction task.
    """
    try:
        from src.tasks.biography_refresh_task import refresh_biographies_task

        logger.info(f"[FACT_EXTRACTION] Triggering biography refresh for {len(subjects)} subjects: {subjects}")

        # Queue the refresh task (runs async via Celery)
        refresh_biographies_task.delay(user_email)

    except Exception as e:
        logger.warning(f"[FACT_EXTRACTION] Failed to trigger biography refresh: {e}")


def emit_approval_requests(user_email: str, pending_facts: List[Dict]) -> None:
    """
    Emit WebSocket events for facts pending approval.

    Uses Redis pub/sub since Celery workers can't directly emit WebSocket events.
    The Flask app subscribes to this channel and forwards to connected clients.
    """
    try:
        import redis
        import json

        redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
        r = redis.from_url(redis_url)

        for pending in pending_facts:
            event_data = {
                'type': 'fact_approval_request',
                'user_email': user_email,
                'pending_id': pending['id'],
                'subject': pending['fact'].get('subject', 'Unknown'),
                'fact_text': pending['fact'].get('fact', ''),
                'category': pending['fact'].get('category', 'general'),
                'sensitivity': pending['sensitivity'],
                'reason': pending['reason'],
            }

            # Publish to Redis channel
            r.publish(f'approval_requests:{user_email}', json.dumps(event_data))
            logger.info(f"[FACT_EXTRACTION] Published approval request for fact {pending['id']}")

    except Exception as e:
        logger.error(f"[FACT_EXTRACTION] Failed to emit approval request: {e}")


# Convenience function for triggering extraction
def trigger_fact_extraction(user_email: str, user_message: str, companion_response: str, message_id: int = None):
    """
    Trigger async fact extraction.
    """
    extract_facts.delay(user_email, user_message, companion_response, message_id)
