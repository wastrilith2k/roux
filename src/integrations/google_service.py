"""
Google Service -- direct Google Workspace API client.

WHAT: OAuth2-authenticated client for Gmail (send, search, read), Google
      Calendar (create events, manage secondary calendars), and Google Docs
      (create, read). Handles credential loading, auto-refresh, and API
      service construction.

WHY:  The companion needs to interact with the user's Google Workspace --
      sending emails, creating calendar events, reading documents -- as part
      of its tool-use capabilities. This module provides the direct API calls
      that the code-executor tools and Celery tasks rely on.

HOW:  OAuth2 credentials are loaded from credentials/google_token.json (path
      configurable via GOOGLE_TOKEN_PATH env var). Tokens are auto-refreshed
      on expiry. Each API (gmail, calendar, docs) gets its own `build()`
      service object. Calendar operations support both the primary calendar
      and a dedicated companion schedule calendar (ID stored in a text file).

Singleton: `get_google_service()` at module bottom.
"""

import os
import json
import logging
import base64
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# Path to stored OAuth token
TOKEN_PATH = Path(os.environ.get(
    'GOOGLE_TOKEN_PATH',
    os.path.join(os.path.dirname(__file__), '..', '..', 'credentials', 'google_token.json')
))


class GoogleServiceError(Exception):
    """Raised when a Google API call fails."""
    pass


class GoogleService:
    """Direct Google API client using OAuth2 credentials."""

    SCOPES = [
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/documents',
    ]

    def __init__(self, token_path: str = None):
        self._token_path = Path(token_path) if token_path else TOKEN_PATH
        self._credentials: Optional[Credentials] = None
        self._gmail = None
        self._calendar = None
        self._docs = None

    # ------------------------------------------------------------------
    # Credential management
    # ------------------------------------------------------------------

    def _get_credentials(self) -> Credentials:
        """Load credentials from token file, refreshing if expired."""
        if self._credentials and self._credentials.valid:
            return self._credentials

        token_path = self._token_path.resolve()
        if not token_path.exists():
            raise GoogleServiceError(f"Token file not found: {token_path}")

        try:
            with open(token_path) as f:
                token_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise GoogleServiceError(f"Failed to read token file: {e}")

        self._credentials = Credentials(
            token=token_data.get('token'),
            refresh_token=token_data.get('refresh_token'),
            token_uri=token_data.get('token_uri', 'https://oauth2.googleapis.com/token'),
            client_id=token_data.get('client_id'),
            client_secret=token_data.get('client_secret'),
            scopes=token_data.get('scopes', self.SCOPES),
        )

        if self._credentials.expired and self._credentials.refresh_token:
            logger.info("Google token expired, refreshing...")
            try:
                self._credentials.refresh(Request())
                # Write refreshed token back to file
                token_data['token'] = self._credentials.token
                with open(token_path, 'w') as f:
                    json.dump(token_data, f, indent=2)
                logger.info("Google token refreshed and saved")
                # Invalidate cached service objects
                self._gmail = None
                self._calendar = None
                self._docs = None
            except Exception as e:
                raise GoogleServiceError(f"Token refresh failed: {e}")

        return self._credentials

    def _get_gmail(self):
        """Get cached Gmail service object."""
        if self._gmail is None:
            creds = self._get_credentials()
            self._gmail = build('gmail', 'v1', credentials=creds)
        return self._gmail

    def _get_calendar(self):
        """Get cached Calendar service object."""
        if self._calendar is None:
            creds = self._get_credentials()
            self._calendar = build('calendar', 'v3', credentials=creds)
        return self._calendar

    def _get_docs(self):
        """Get cached Docs service object."""
        if self._docs is None:
            creds = self._get_credentials()
            self._docs = build('docs', 'v1', credentials=creds)
        return self._docs

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    def create_calendar_event(
        self,
        summary: str,
        start: str,
        end: str,
        description: str = '',
        location: str = '',
        calendar_id: str = 'primary',
    ) -> Dict[str, Any]:
        """Create a Google Calendar event.

        Args:
            summary: Event title
            start: Start time in RFC3339 format (e.g. 2026-02-22T10:00:00-08:00)
            end: End time in RFC3339 format
            description: Optional event description
            location: Optional event location
            calendar_id: Calendar ID to create event on (default: 'primary')

        Returns:
            Dict with id, htmlLink, summary, start, end
        """
        try:
            event_body: Dict[str, Any] = {
                'summary': summary,
                'start': {'dateTime': start, 'timeZone': 'America/Los_Angeles'},
                'end': {'dateTime': end, 'timeZone': 'America/Los_Angeles'},
            }
            if description:
                event_body['description'] = description
            if location:
                event_body['location'] = location

            event = self._get_calendar().events().insert(
                calendarId=calendar_id,
                body=event_body,
            ).execute()

            logger.info(f"Calendar event created: {event.get('id')} on calendar {calendar_id}")
            return {
                'id': event.get('id'),
                'htmlLink': event.get('htmlLink'),
                'summary': event.get('summary'),
                'start': event.get('start'),
                'end': event.get('end'),
            }
        except Exception as e:
            logger.error(f"Calendar event creation failed: {e}")
            raise GoogleServiceError(f"Calendar event creation failed: {e}")

    def create_secondary_calendar(
        self,
        summary: str,
        timezone: str = 'America/Los_Angeles',
    ) -> str:
        """Create a secondary Google Calendar.

        Args:
            summary: Calendar name (e.g. "Companion's Schedule")
            timezone: Calendar timezone

        Returns:
            Calendar ID string
        """
        try:
            calendar_body = {
                'summary': summary,
                'timeZone': timezone,
            }
            created = self._get_calendar().calendars().insert(
                body=calendar_body,
            ).execute()
            cal_id = created['id']
            logger.info(f"Secondary calendar created: {summary} ({cal_id})")
            return cal_id
        except Exception as e:
            logger.error(f"Secondary calendar creation failed: {e}")
            raise GoogleServiceError(f"Secondary calendar creation failed: {e}")

    def list_calendars(self) -> List[Dict[str, Any]]:
        """List all calendars the user has access to.

        Returns:
            List of dicts with id, summary, primary, accessRole
        """
        try:
            result = self._get_calendar().calendarList().list().execute()
            calendars = []
            for item in result.get('items', []):
                calendars.append({
                    'id': item.get('id'),
                    'summary': item.get('summary', ''),
                    'primary': item.get('primary', False),
                    'accessRole': item.get('accessRole', ''),
                })
            return calendars
        except Exception as e:
            logger.error(f"Calendar list failed: {e}")
            raise GoogleServiceError(f"Calendar list failed: {e}")

    def list_calendar_events(
        self,
        calendar_id: str,
        time_min: str,
        time_max: str,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """List events from a Google Calendar within a time range.

        Args:
            calendar_id: Calendar ID to query
            time_min: Start of range in RFC3339 format
            time_max: End of range in RFC3339 format
            max_results: Maximum number of events to return

        Returns:
            List of event dicts with id, summary, start, end, description, location
        """
        try:
            result = self._get_calendar().events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime',
            ).execute()

            events = []
            for item in result.get('items', []):
                events.append({
                    'id': item.get('id'),
                    'summary': item.get('summary', ''),
                    'start': item.get('start', {}),
                    'end': item.get('end', {}),
                    'description': item.get('description', ''),
                    'location': item.get('location', ''),
                })
            return events
        except Exception as e:
            logger.error(f"Calendar event listing failed: {e}")
            raise GoogleServiceError(f"Calendar event listing failed: {e}")

    def list_all_calendar_events(
        self,
        time_min: str,
        time_max: str,
        max_results_per_calendar: int = 25,
    ) -> List[Dict[str, Any]]:
        """List events from ALL user calendars, aggregated and sorted.

        Args:
            time_min: Start of range in RFC3339 format
            time_max: End of range in RFC3339 format
            max_results_per_calendar: Max events per calendar

        Returns:
            Sorted list of event dicts from all calendars
        """
        calendars = self.list_calendars()
        all_events = []

        for cal in calendars:
            # Skip calendars we only have freeBusy access to
            if cal.get('accessRole') not in ('owner', 'writer', 'reader'):
                continue
            try:
                events = self.list_calendar_events(
                    calendar_id=cal['id'],
                    time_min=time_min,
                    time_max=time_max,
                    max_results=max_results_per_calendar,
                )
                for event in events:
                    event['calendar_name'] = cal.get('summary', '')
                all_events.extend(events)
            except GoogleServiceError:
                logger.warning(f"Failed to fetch events from calendar: {cal.get('summary', cal['id'])}")
                continue

        # Sort by start time
        def _sort_key(e):
            start = e.get('start', {})
            return start.get('dateTime', start.get('date', ''))

        all_events.sort(key=_sort_key)
        return all_events

    def delete_calendar_events(
        self,
        calendar_id: str,
        time_min: str,
        time_max: str,
    ) -> int:
        """Delete all events in a calendar within a time range.

        Args:
            calendar_id: Calendar ID
            time_min: Start of range in RFC3339 format
            time_max: End of range in RFC3339 format

        Returns:
            Number of events deleted
        """
        try:
            events = self.list_calendar_events(calendar_id, time_min, time_max)
            count = 0
            for event in events:
                self._get_calendar().events().delete(
                    calendarId=calendar_id,
                    eventId=event['id'],
                ).execute()
                count += 1
            logger.info(f"Deleted {count} events from calendar {calendar_id}")
            return count
        except GoogleServiceError:
            raise
        except Exception as e:
            logger.error(f"Calendar event deletion failed: {e}")
            raise GoogleServiceError(f"Calendar event deletion failed: {e}")

    # ------------------------------------------------------------------
    # Gmail
    # ------------------------------------------------------------------

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Send an email via Gmail.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (plain text)

        Returns:
            Dict with id, threadId
        """
        try:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

            result = self._get_gmail().users().messages().send(
                userId='me',
                body={'raw': raw},
            ).execute()

            logger.info(f"Email sent to {to}: {result.get('id')}")
            return {
                'id': result.get('id'),
                'threadId': result.get('threadId'),
            }
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            raise GoogleServiceError(f"Email send failed: {e}")

    def search_emails(self, query: str = '', max_results: int = 10) -> List[Dict[str, Any]]:
        """Search emails in Gmail.

        Args:
            query: Gmail search query (e.g. 'from:someone@example.com', 'subject:hello')
            max_results: Maximum number of results (default 10)

        Returns:
            List of dicts with id, from, subject, snippet, date
        """
        try:
            results = self._get_gmail().users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results,
            ).execute()

            messages = results.get('messages', [])
            if not messages:
                return []

            emails = []
            for msg_stub in messages:
                msg = self._get_gmail().users().messages().get(
                    userId='me',
                    id=msg_stub['id'],
                    format='metadata',
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
        except Exception as e:
            logger.error(f"Email search failed: {e}")
            raise GoogleServiceError(f"Email search failed: {e}")

    def get_email(self, message_id: str) -> Dict[str, Any]:
        """Get full email content by message ID.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with id, from, to, subject, date, body
        """
        try:
            msg = self._get_gmail().users().messages().get(
                userId='me',
                id=message_id,
                format='full',
            ).execute()

            headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}

            # Extract body text
            body = ''
            payload = msg.get('payload', {})
            if payload.get('body', {}).get('data'):
                body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
            elif payload.get('parts'):
                for part in payload['parts']:
                    if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                        break

            return {
                'id': msg['id'],
                'from': headers.get('From', ''),
                'to': headers.get('To', ''),
                'subject': headers.get('Subject', ''),
                'date': headers.get('Date', ''),
                'body': body,
            }
        except Exception as e:
            logger.error(f"Email get failed: {e}")
            raise GoogleServiceError(f"Email get failed: {e}")

    # ------------------------------------------------------------------
    # Google Docs
    # ------------------------------------------------------------------

    def create_document(self, title: str, content: str = '') -> Dict[str, Any]:
        """Create a Google Doc.

        Args:
            title: Document title
            content: Optional initial text content

        Returns:
            Dict with documentId, title, url
        """
        try:
            doc = self._get_docs().documents().create(body={'title': title}).execute()
            doc_id = doc['documentId']

            if content:
                self._get_docs().documents().batchUpdate(
                    documentId=doc_id,
                    body={
                        'requests': [{
                            'insertText': {
                                'location': {'index': 1},
                                'text': content,
                            }
                        }]
                    },
                ).execute()

            url = f"https://docs.google.com/document/d/{doc_id}/edit"
            logger.info(f"Document created: {doc_id}")
            return {
                'documentId': doc_id,
                'title': title,
                'url': url,
            }
        except Exception as e:
            logger.error(f"Document creation failed: {e}")
            raise GoogleServiceError(f"Document creation failed: {e}")

    def read_document(self, document_id: str) -> Dict[str, Any]:
        """Read a Google Doc's content.

        Args:
            document_id: Google Docs document ID

        Returns:
            Dict with title, content (plain text)
        """
        try:
            doc = self._get_docs().documents().get(documentId=document_id).execute()

            # Extract plain text from document body
            content_parts = []
            for element in doc.get('body', {}).get('content', []):
                paragraph = element.get('paragraph')
                if paragraph:
                    for text_run in paragraph.get('elements', []):
                        text = text_run.get('textRun', {}).get('content', '')
                        if text:
                            content_parts.append(text)

            return {
                'title': doc.get('title', ''),
                'content': ''.join(content_parts),
            }
        except Exception as e:
            logger.error(f"Document read failed: {e}")
            raise GoogleServiceError(f"Document read failed: {e}")


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_service: Optional[GoogleService] = None


def get_google_service() -> GoogleService:
    """Get or create the GoogleService singleton."""
    global _service
    if _service is None:
        _service = GoogleService()
    return _service
