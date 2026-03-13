"""
Gmail Inbox Awareness - Surface interesting emails to the companion's awareness.

WHAT: Searches for recent unread emails (last 30 minutes) via Google API.
      Uses LLM to filter out spam/newsletters and identify emails worth
      thinking about or mentioning to James. Interesting items are stored
      as findings (autonomous_findings.json) and added to queued_thoughts.

WHEN: Every 4 hours during waking hours (8:30, 12:30, 16:30, 20:30 PST).

WHY:  Gives the companion awareness of the shared email context. If James
      gets an important email about an interview or a family member, the
      companion can naturally reference it in conversation rather than being
      blindsided.

Feature flag: COMPANION_GMAIL_CHECK_ENABLED (default: false -- enable after testing)
"""

import os
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from src.celery_app import celery_app

logger = logging.getLogger(__name__)

PST = ZoneInfo('America/Los_Angeles')


def _is_enabled() -> bool:
    return os.environ.get('COMPANION_GMAIL_CHECK_ENABLED', 'false').lower() in ('true', '1')


@celery_app.task(
    name='tasks.gmail_check.check_inbox',
    soft_time_limit=120,
    time_limit=180,
)
def check_inbox(user_email: str = None):
    """Check Gmail for interesting unread emails and surface them."""
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    if not _is_enabled():
        return {'status': 'disabled'}

    try:
        from src.integrations.google_service import get_google_service
        from src.llm.provider_factory import generate_sync

        google = get_google_service()

        # Search for recent unread emails
        emails = google.search_emails(query='is:unread newer_than:30m', max_results=5)

        if not emails:
            logger.debug("No recent unread emails")
            return {'status': 'success', 'processed': 0}

        # Build summary for LLM
        email_summaries = []
        for email in emails:
            email_summaries.append(
                f"- From: {email.get('from', 'unknown')}\n"
                f"  Subject: {email.get('subject', '(no subject)')}\n"
                f"  Preview: {email.get('snippet', '')[:200]}"
            )

        email_text = "\n".join(email_summaries)

        # LLM decides what's interesting
        prompt = f"""The companion just checked her email. Here are recent unread messages:

{email_text}

Which of these (if any) are interesting enough for the companion to:
1. Think about or remember
2. Potentially mention to James in conversation

Ignore spam, newsletters, and routine notifications.
Focus on personal emails, important updates, or things relevant to her life with James.

Return JSON array (empty if nothing interesting):
[
  {{
    "subject": "email subject",
    "summary": "brief 1-sentence summary of what it's about",
    "worth_mentioning": true/false,
    "thought": "what the companion might think or say about this"
  }}
]

Return ONLY valid JSON array:"""

        response = generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        if not response:
            return {'status': 'success', 'processed': 0}

        # Parse response
        response_text = response.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            json_lines = [l for l in lines if not l.startswith('```')]
            response_text = '\n'.join(json_lines)

        try:
            interesting = json.loads(response_text)
        except json.JSONDecodeError:
            logger.debug(f"Gmail check: LLM returned non-JSON response, skipping")
            return {'status': 'success', 'processed': 0}

        count = 0

        for item in interesting:
            if not isinstance(item, dict):
                continue

            subject = item.get('subject', '')
            summary = item.get('summary', '')
            thought = item.get('thought', '')

            if item.get('worth_mentioning') and summary:
                # Store as finding
                try:
                    from src.tasks.autonomous_action_task import _store_finding
                    _store_finding(
                        topic=f"Email: {subject}",
                        finding=summary,
                        source='gmail_check'
                    )
                except Exception as e:
                    logger.debug(f"Could not store gmail finding: {e}")

            if thought:
                # Add to queued thoughts
                try:
                    from src.core.internal_state import get_internal_state_manager
                    state_manager = get_internal_state_manager()
                    state_manager.add_queued_thought(user_email, thought)
                except Exception as e:
                    logger.debug(f"Could not add gmail thought: {e}")

            count += 1

        logger.info(f"Gmail check: {len(emails)} emails checked, {count} interesting items surfaced")
        return {'status': 'success', 'processed': count}

    except Exception as e:
        logger.error(f"Gmail check failed: {e}")
        return {'status': 'error', 'error': str(e)}
