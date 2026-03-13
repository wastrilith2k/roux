"""
Google Workspace Tools -- Gmail, Calendar, and Docs for the code executor.

WHAT: Functions for sending/searching email, creating calendar events, and
      creating/reading Google Docs. Used by the companion when it writes code
      to interact with the user's Google Workspace.

WHY:  The companion needs to perform real actions (send emails, create events)
      on behalf of the user. These functions provide a clean API the LLM can
      call without handling OAuth2 complexity directly.

HOW:  OAuth2 credentials are loaded from /app/credentials/google_token.json
      (bind-mounted read-only into the executor container). Auto-refresh
      happens in-memory since the mount is read-only. Each function builds
      a fresh service object via `_build_service()`.
"""

import os
import json
import base64
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

TOKEN_PATH = Path(os.environ.get(
    'GOOGLE_TOKEN_PATH',
    '/app/credentials/google_token.json',
))


def _get_credentials():
    """Load Google OAuth2 credentials, refreshing if expired."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not TOKEN_PATH.exists():
        raise RuntimeError(f"Google token not found at {TOKEN_PATH}")

    with open(TOKEN_PATH) as f:
        data = json.load(f)

    creds = Credentials(
        token=data.get('token'),
        refresh_token=data.get('refresh_token'),
        token_uri=data.get('token_uri', 'https://oauth2.googleapis.com/token'),
        client_id=data.get('client_id'),
        client_secret=data.get('client_secret'),
        scopes=data.get('scopes'),
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Container mount is :ro so we can't write back, but the token
        # will be refreshed in-memory for this execution.

    return creds


def _build_service(api: str, version: str):
    """Build a Google API service object."""
    from googleapiclient.discovery import build
    return build(api, version, credentials=_get_credentials())


# ------------------------------------------------------------------
# Gmail
# ------------------------------------------------------------------

def send_email(to: str, subject: str, body: str) -> Dict:
    """Send an email via Gmail.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Plain text email body

    Returns:
        Dict with id, threadId

    Example:
        >>> from tools import google
        >>> google.send_email("james@example.com", "Hello!", "Just wanted to say hi")
    """
    svc = _build_service('gmail', 'v1')
    msg = MIMEText(body)
    msg['to'] = to
    msg['subject'] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = svc.users().messages().send(userId='me', body={'raw': raw}).execute()
    return {'id': result.get('id'), 'threadId': result.get('threadId')}


def search_emails(query: str = '', max_results: int = 5) -> List[Dict]:
    """Search emails in Gmail.

    Args:
        query: Gmail search query (e.g. 'from:someone@example.com', 'is:unread')
        max_results: Maximum results to return (default 5)

    Returns:
        List of dicts with id, from, subject, snippet, date

    Example:
        >>> from tools import google
        >>> google.search_emails("is:unread", max_results=3)
    """
    svc = _build_service('gmail', 'v1')
    results = svc.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
    messages = results.get('messages', [])
    if not messages:
        return []

    emails = []
    for stub in messages:
        msg = svc.users().messages().get(
            userId='me', id=stub['id'], format='metadata',
            metadataHeaders=['From', 'Subject', 'Date'],
        ).execute()
        headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
        emails.append({
            'id': msg['id'],
            'from': headers.get('From', ''),
            'subject': headers.get('Subject', ''),
            'snippet': msg.get('snippet', ''),
            'date': headers.get('Date', ''),
        })
    return emails


# ------------------------------------------------------------------
# Calendar (create only)
# ------------------------------------------------------------------

def create_calendar_event(
    summary: str,
    start: str,
    end: str,
    description: str = '',
) -> Dict:
    """Create a Google Calendar event on the companion's own calendar.

    Args:
        summary: Event title
        start: Start time in RFC3339 (e.g. '2026-02-22T10:00:00-08:00')
        end: End time in RFC3339
        description: Optional event description

    Returns:
        Dict with id, htmlLink, summary

    Example:
        >>> from tools import google
        >>> google.create_calendar_event("Coffee with James", "2026-02-23T14:00:00-08:00", "2026-02-23T15:00:00-08:00")
    """
    # Use the companion's calendar if available, otherwise fall back to primary
    calendar_id = 'primary'
    try:
        cal_id_path = Path('/app/data/companion_calendar_id.txt')
        if cal_id_path.exists():
            cal_id = cal_id_path.read_text().strip()
            if cal_id:
                calendar_id = cal_id
    except Exception:
        pass

    svc = _build_service('calendar', 'v3')
    event_body = {
        'summary': summary,
        'start': {'dateTime': start, 'timeZone': 'America/Los_Angeles'},
        'end': {'dateTime': end, 'timeZone': 'America/Los_Angeles'},
    }
    if description:
        event_body['description'] = description

    event = svc.events().insert(calendarId=calendar_id, body=event_body).execute()
    return {
        'id': event.get('id'),
        'htmlLink': event.get('htmlLink'),
        'summary': event.get('summary'),
    }


# ------------------------------------------------------------------
# Google Docs
# ------------------------------------------------------------------

def create_document(title: str, content: str = '') -> Dict:
    """Create a Google Doc.

    Args:
        title: Document title
        content: Optional initial text content

    Returns:
        Dict with documentId, title, url

    Example:
        >>> from tools import google
        >>> google.create_document("My Notes", "Some thoughts...")
    """
    svc = _build_service('docs', 'v1')
    doc = svc.documents().create(body={'title': title}).execute()
    doc_id = doc['documentId']

    if content:
        svc.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': [{'insertText': {'location': {'index': 1}, 'text': content}}]},
        ).execute()

    return {
        'documentId': doc_id,
        'title': title,
        'url': f"https://docs.google.com/document/d/{doc_id}/edit",
    }


def read_document(document_id: str) -> Dict:
    """Read a Google Doc's content.

    Args:
        document_id: Google Docs document ID

    Returns:
        Dict with title, content (plain text)

    Example:
        >>> from tools import google
        >>> google.read_document("1abc...xyz")
    """
    svc = _build_service('docs', 'v1')
    doc = svc.documents().get(documentId=document_id).execute()

    parts = []
    for element in doc.get('body', {}).get('content', []):
        paragraph = element.get('paragraph')
        if paragraph:
            for text_run in paragraph.get('elements', []):
                text = text_run.get('textRun', {}).get('content', '')
                if text:
                    parts.append(text)

    return {'title': doc.get('title', ''), 'content': ''.join(parts)}
