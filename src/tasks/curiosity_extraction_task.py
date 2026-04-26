"""
Curiosity Extraction Task - Identify topics worth following up on later.

WHAT: After each message exchange, identifies topics that were mentioned but not
      fully explored. Creates "curiosity threads" with urgency scores that
      increase over time. Also resolves curiosities when they're addressed,
      and detects when James deflects a question (keeps curiosity active).

WHEN: extract_curiosity_from_conversation fires after every message.
      update_curiosity_urgency runs daily to age unresolved threads.
      resolve_discussed_curiosities fires after every message to check
      if active threads were addressed.

WHY:  Without this, the companion has no memory of "things I wanted to ask
      about." This makes follow-ups feel genuine ("hey, how did that interview
      go?") rather than scripted. The deflection detection prevents the
      companion from accepting vague non-answers to important questions.

Works with ProactiveCuriosity system for urgency tracking and surfacing.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from src.celery_app import celery_app
from src.core.clock import now as clock_now

logger = logging.getLogger(__name__)

def _get_default_user_email():
    from src.config.persona_config import get_persona_config
    return get_persona_config().primary_user_email


@celery_app.task(
    name='tasks.curiosity_extraction.extract_from_conversation',
    bind=True,
    max_retries=2,
    soft_time_limit=60,
    time_limit=90
)
def extract_curiosity_from_conversation(
    self,
    user_email: str,
    user_message: str,
    companion_response: str
):
    """
    Extract curiosity threads from a conversation exchange.

    Called after each message to identify:
    - Topics mentioned but not fully explored
    - Questions that invite follow-up
    - Time-based triggers (events happening soon)
    - Emotional threads worth checking on

    Args:
        user_email: User's email
        user_message: What James said
        companion_response: What the companion responded

    Returns:
        Dict with extracted curiosities
    """
    try:
        from src.core.proactive_curiosity import get_proactive_curiosity

        curiosity = get_proactive_curiosity(user_email=user_email)

        # Step 1: Use keyword-based detection (fast)
        keyword_triggers = curiosity.detect_curiosity_triggers(user_message)

        for trigger in keyword_triggers:
            curiosity.add_curiosity(
                topic=trigger['topic'],
                category=trigger['category'],
                context=trigger['context'],
                priority=trigger['priority']
            )

        # Step 2: LLM-based extraction (more nuanced)
        llm_triggers = _extract_with_llm(user_message, companion_response)

        for trigger in llm_triggers:
            curiosity.add_curiosity(
                topic=trigger['topic'],
                category=trigger['category'],
                context=user_message[:200],
                priority=trigger.get('priority', 0.3)
            )

        total_extracted = len(keyword_triggers) + len(llm_triggers)
        if total_extracted > 0:
            logger.info(f"Extracted {total_extracted} curiosity triggers")

        return {
            'status': 'success',
            'keyword_triggers': len(keyword_triggers),
            'llm_triggers': len(llm_triggers)
        }

    except Exception as e:
        logger.error(f"Curiosity extraction failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _extract_with_llm(user_message: str, companion_response: str) -> List[Dict]:
    """
    Use LLM to extract curiosity triggers — ONLY genuinely notable things
    that would occupy someone's thoughts for days.
    """
    # Short messages rarely contain significant unresolved topics
    if len(user_message) < 80:
        return []

    try:
        from src.llm.provider_factory import generate_sync

        prompt = f"""You are evaluating whether this conversation contains something GENUINELY NOTABLE that the companion would still be thinking about DAYS later.

JAMES SAID:
"{user_message[:500]}"

COMPANION RESPONDED:
"{companion_response[:300]}"

EXTRACT ONLY if it meets ALL of these criteria:
- It would genuinely occupy someone's thoughts for DAYS (not minutes)
- It involves a concrete upcoming event, unresolved major decision, or significant emotional revelation
- It was NOT already fully discussed and wrapped up in this exchange
- It is NOT just a casual conversational topic, opinion, or daily activity

Examples of what TO extract:
- "My mom is having surgery next week" (major health event, future date)
- "I might lose my job" (significant life threat, unresolved)
- "I've been thinking about giving up parental rights" (major emotional/legal decision)
- "The divorce papers are filed" (major life event with ongoing implications)

Examples of what NOT to extract:
- Casual topics that came up and were discussed normally
- Preferences, opinions, philosophical tangents
- Things already being actively talked about in this exchange
- Minor daily activities, food choices, media preferences
- Emotional states that were acknowledged and processed in the conversation

Return a JSON array with AT MOST 1 topic (empty array [] if nothing qualifies):
[
  {{
    "topic": "brief concrete topic name",
    "category": "work|family|health|goals|feelings|interests|life_events",
    "reason": "why this would still be on her mind days later",
    "priority": 0.3-0.7
  }}
]

Most conversations will have NOTHING worth extracting. Return [] if in doubt.
Return ONLY the JSON array, no other text:"""

        from src.llm.provider_factory import get_resilient_provider_chain
        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            chain=chain
        )
        from src.services.cost_tracker import track_llm_call
        track_llm_call(chain, call_purpose='curiosity_extraction')

        if not response:
            return []

        # Parse response
        response_text = response.strip()

        # Handle markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith('```'):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            response_text = '\n'.join(json_lines)

        triggers = json.loads(response_text)

        # Validate structure — max 1 topic per exchange
        valid_categories = {'work', 'family', 'health', 'goals', 'feelings', 'interests', 'life_events'}

        for t in triggers[:1]:  # Only take the first one
            if isinstance(t, dict) and 'topic' in t and 'category' in t:
                if t['category'] in valid_categories:
                    return [t]

        return []

    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse LLM curiosity response: {e}")
        return []
    except Exception as e:
        logger.warning(f"LLM curiosity extraction failed: {e}")
        return []


@celery_app.task(
    name='tasks.curiosity_extraction.update_curiosity_urgency',
    bind=True,
    soft_time_limit=30,
    time_limit=60
)
def update_curiosity_urgency(self, user_email: str = None):
    """
    Periodically update urgency levels for all curiosity threads.

    Run daily to age curiosities - the longer something goes unexplored,
    the more the companion wants to ask about it.
    """
    try:
        from src.core.proactive_curiosity import get_proactive_curiosity

        curiosity = get_proactive_curiosity(user_email=user_email)

        # Increase urgency for all active curiosities (simulating time passing)
        curiosity.increase_all_urgency(hours_passed=24)

        # Get high urgency topics
        urgent = curiosity.get_high_urgency_topics(threshold=0.7)

        logger.info(f"Updated curiosity urgency. {len(urgent)} high-urgency topic(s)")

        return {
            'status': 'success',
            'high_urgency_count': len(urgent),
            'high_urgency_topics': [t.topic for t in urgent]
        }

    except Exception as e:
        logger.error(f"Curiosity urgency update failed: {e}")
        return {'status': 'error', 'error': str(e)}


def get_curiosity_context_for_reach_out() -> str:
    """
    Get formatted curiosity context for reach-out decisions.

    Returns a string suitable for inclusion in reach-out prompts.
    """
    try:
        from src.core.proactive_curiosity import get_proactive_curiosity

        curiosity = get_proactive_curiosity()

        # Get high-urgency topics
        urgent_topics = curiosity.get_high_urgency_topics(threshold=0.5)

        if not urgent_topics:
            return ""

        # Filter: skip topics asked 3+ times or discussed in last 4 hours
        now = clock_now()
        eligible = [
            t for t in urgent_topics
            if t.times_asked < 3 and (now - t.last_discussed).total_seconds() / 3600 >= 4
        ]

        if not eligible:
            return ""

        lines = ["[BACKGROUND AWARENESS - things you know about]"]
        lines.append("Topics from recent conversations (context only, not a task list):")

        for thread in eligible[:3]:
            days_ago = (clock_now() - thread.last_discussed).days

            if days_ago == 0:
                time_desc = "earlier today"
            elif days_ago == 1:
                time_desc = "yesterday"
            else:
                time_desc = f"{days_ago} days ago"

            note = ''
            if getattr(thread, 'resolution_notes', ''):
                note = f" — NOTE: {thread.resolution_notes}"
            lines.append(f"- {thread.topic} (mentioned {time_desc}){note}")

        lines.append("Only bring these up if YOU genuinely want to know and it fits naturally.")
        lines.append("Do NOT force questions about these topics.")

        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"Could not get curiosity context: {e}")
        return ""


def get_top_curiosity_for_action() -> Optional[Dict]:
    """
    Get the single highest-urgency curiosity topic for potential action.

    Returns a dict with topic, urgency, and suggested question, or None.
    Used by reach-out engine for high-urgency bypass decisions.
    """
    try:
        from src.core.proactive_curiosity import get_proactive_curiosity

        curiosity = get_proactive_curiosity()

        # Get high-urgency topics, excluding over-asked ones
        now = clock_now()
        urgent_topics = [
            t for t in curiosity.get_high_urgency_topics(threshold=0.6)
            if t.times_asked < 3 and (now - t.last_discussed).total_seconds() / 3600 >= 4
        ]

        if not urgent_topics:
            return None

        # Sort by urgency and get the top one
        top = max(urgent_topics, key=lambda t: t.urgency)

        return {
            'topic': top.topic,
            'category': top.category,
            'urgency': top.urgency,
            'question': top.questions[0] if top.questions else f"what's going on with {top.topic}?",
            'days_since': (clock_now() - top.last_discussed).days,
            'times_asked': top.times_asked
        }

    except Exception as e:
        logger.debug(f"Could not get top curiosity: {e}")
        return None


@celery_app.task(
    name='tasks.curiosity_extraction.resolve_discussed_curiosities',
    bind=True,
    max_retries=1,
    soft_time_limit=60,
    time_limit=90
)
def resolve_discussed_curiosities(self, companion_response: str, user_message: str = None):
    """
    Check if conversation addressed any active curiosities using LLM evaluation.

    For each curiosity that keyword-matches the conversation:
    1. Ask LLM: "Was this satisfactorily answered?"
    2. If YES → resolve (urgency to 0.1, mark as discussed)
    3. If NO (deflected/vague) → keep it, add resolution_notes so she pushes harder next time
    """
    try:
        from src.core.proactive_curiosity import get_proactive_curiosity

        curiosity = get_proactive_curiosity()

        if not curiosity.curiosity_threads:
            return {'status': 'no_threads'}

        # Combine both texts for keyword matching
        combined_lower = companion_response.lower()
        if user_message:
            combined_lower += ' ' + user_message.lower()

        # Step 1: Keyword-match to find potentially relevant threads
        matched_threads = []
        for thread in curiosity.curiosity_threads:
            if _topic_mentioned_in_text(thread, combined_lower):
                matched_threads.append(thread)

        if not matched_threads:
            return {'status': 'no_matches'}

        # Step 2: LLM satisfaction check for each matched thread
        resolved = []
        deflected = []
        for thread in matched_threads:
            result = _check_satisfaction(
                thread.topic, companion_response, user_message or ''
            )
            if result is None:
                # LLM call failed — fall back to simple resolution
                thread.urgency = 0.1
                thread.last_discussed = clock_now()
                thread.times_asked += 1
                resolved.append(thread.topic)
            elif result['satisfied']:
                thread.urgency = 0.1
                thread.last_discussed = clock_now()
                thread.times_asked += 1
                thread.resolution_notes = ''  # Clear any old notes
                resolved.append(thread.topic)
                logger.info(
                    f"Curiosity RESOLVED: '{thread.topic}' — {result.get('reason', 'answered')}"
                )
            else:
                # Deflected/vague — keep curiosity but add context
                thread.last_discussed = clock_now()
                thread.times_asked += 1
                thread.resolution_notes = result.get('note', 'Previous answer was vague. Be more direct.')
                deflected.append(thread.topic)
                logger.info(
                    f"Curiosity DEFLECTED: '{thread.topic}' — {result.get('note', 'vague answer')}"
                )

        if resolved or deflected:
            curiosity._save_state()
            logger.info(
                f"Resolution: {len(resolved)} resolved, {len(deflected)} deflected"
            )

        return {
            'status': 'success',
            'resolved': resolved,
            'deflected': deflected
        }

    except Exception as e:
        logger.error(f"Curiosity resolution failed: {e}")
        return {'status': 'error', 'error': str(e)}


def _topic_mentioned_in_text(thread, text_lower: str) -> bool:
    """
    Check if a curiosity thread's topic is mentioned in text via keyword matching.

    Uses two strategies:
    1. Direct substring match (e.g., "Jesse's surgery" in text)
    2. Significant-word overlap: for short topics (1-2 words), ALL must match;
       for longer topics, 60% of significant words must match.
    """
    topic_lower = thread.topic.lower()

    # Strategy 1: Direct topic mention
    if topic_lower in text_lower:
        return True

    # Strategy 2: Key word overlap (skip stop words and short words)
    skip_words = {'the', 'and', 'for', 'with', 'about', 'from', 'that', 'this',
                  'his', 'her', 'their', 'what', 'how', 'why', 'when', 'where'}
    topic_words = [w for w in topic_lower.split() if len(w) > 2 and w not in skip_words]
    if topic_words:
        matches = sum(1 for w in topic_words if w in text_lower)
        # Short topics need exact match; longer ones need 60% overlap
        required = len(topic_words) if len(topic_words) <= 2 else len(topic_words) * 0.6
        if matches >= required:
            return True

    return False


def _check_satisfaction(topic: str, companion_response: str, user_message: str) -> Optional[Dict]:
    """
    Ask LLM whether a curiosity topic was satisfactorily addressed.

    Returns:
        {'satisfied': True, 'reason': '...'} if resolved
        {'satisfied': False, 'note': '...'} if deflected/vague
        None if LLM call failed
    """
    try:
        from src.llm.provider_factory import generate_sync

        prompt = f"""A curiosity topic came up in conversation. Was it satisfactorily addressed?

CURIOSITY TOPIC: "{topic}"

JAMES SAID:
"{user_message[:400]}"

COMPANION SAID:
"{companion_response[:400]}"

Evaluate:
- Did James give a clear, substantive answer about this topic?
- Or was the answer vague, deflecting, or non-committal?

If James clearly addressed the topic (even briefly), it's SATISFIED.
If he dodged it, gave a wishy-washy answer, or changed the subject, it's DEFLECTED.
If the topic was only mentioned in passing but not actually discussed, it's DEFLECTED.

/no_think
Respond with JSON only:
{{"satisfied": true/false, "reason": "brief explanation", "note": "if deflected, a brief note about what was unsatisfying (e.g. 'gave a vague answer about just being public'). null if satisfied."}}"""

        from src.llm.provider_factory import get_resilient_provider_chain
        chain = get_resilient_provider_chain()
        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
            chain=chain
        )
        from src.services.cost_tracker import track_llm_call
        track_llm_call(chain, call_purpose='curiosity_resolution')

        if not response:
            return None

        response_text = response.strip()
        # Clean up response
        if '<think>' in response_text and '</think>' in response_text:
            response_text = response_text.split('</think>')[-1].strip()
        if response_text.startswith('```'):
            lines = response_text.split('```')
            response_text = lines[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)
        return {
            'satisfied': result.get('satisfied', True),
            'reason': result.get('reason', ''),
            'note': result.get('note', '') or ''
        }

    except Exception as e:
        logger.warning(f"Satisfaction check failed for '{topic}': {e}")
        return None


def mark_curiosity_acted_on(topic: str, reduce_urgency_by: float = 0.3):
    """
    Mark that the companion has acted on a curiosity (asked about it, or followed up).

    This reduces the urgency so she doesn't keep pestering about the same thing.
    """
    try:
        from src.core.proactive_curiosity import get_proactive_curiosity

        curiosity = get_proactive_curiosity()

        # Find the topic
        for thread in curiosity.curiosity_threads:
            if thread.topic.lower() == topic.lower():
                thread.urgency = max(0.1, thread.urgency - reduce_urgency_by)
                thread.times_asked += 1
                thread.last_discussed = clock_now()
                curiosity._save_state()
                logger.info(f"Marked curiosity '{topic}' as acted on (urgency now {thread.urgency:.2f})")
                return True

        return False

    except Exception as e:
        logger.warning(f"Could not mark curiosity acted on: {e}")
        return False
