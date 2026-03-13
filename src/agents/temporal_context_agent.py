"""
Temporal Context Agent

When the user references a specific time ("yesterday morning", "last night", "earlier"),
this agent retrieves ALL relevant messages and facts from that period to inject into context.

Solves the problem of forgetting recent roleplay/intimate conversations by ensuring
full temporal context is available.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re

from src.utils.timezone_utils import now_pacific_naive

logger = logging.getLogger(__name__)


class TemporalContextAgent:
    """
    Retrieves full context for a specific time period.

    When user says "yesterday morning" or "earlier today", pulls:
    - All messages from that time range
    - Related facts extracted during that time
    - Emotional context from that period
    - Physical/roleplay context markers
    """

    def __init__(self, db_connection=None):
        self.db = db_connection

    def detect_temporal_reference(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Detect if the user is referencing a specific time period.

        Returns:
            Dict with 'detected': bool, 'type': str, 'description': str, 'time_range': tuple
            or None if no temporal reference found
        """
        message_lower = user_message.lower()

        # Temporal reference patterns
        patterns = {
            'this_morning': (
                r'\b(this morning|earlier today|this am)\b',
                lambda: self._get_time_range_hours(-24, 0)
            ),
            'yesterday_morning': (
                r'\b(yesterday morning|yesterday|last night)\b',
                lambda: self._get_time_range_days(-1)
            ),
            'earlier': (
                r'\b(earlier|before|a moment ago|just now)\b',
                lambda: self._get_time_range_minutes(-120)
            ),
            'recent': (
                r'\b(recently|lately|the other day)\b',
                lambda: self._get_time_range_days(-3)
            ),
            'last_session': (
                r'\b(last time|before|previously|our last|previous conversation)\b',
                lambda: self._get_time_range_days(-7)
            ),
        }

        for ref_type, (pattern, time_func) in patterns.items():
            if re.search(pattern, message_lower):
                time_range = time_func()
                logger.info(f"🕐 Temporal reference detected: {ref_type}")
                logger.info(f"   Time range: {time_range[0]} to {time_range[1]}")

                return {
                    'detected': True,
                    'type': ref_type,
                    'time_range': time_range,
                    'description': self._describe_time_range(ref_type)
                }

        return None

    def get_temporal_context(
        self,
        email: str,
        temporal_ref: Dict[str, Any],
        conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve ALL context from a specific time period.

        Args:
            email: User email
            temporal_ref: Result from detect_temporal_reference()
            conversation_context: Recent conversation for additional context

        Returns:
            Dict with 'messages', 'facts', 'summary', 'emotional_state'
        """
        if not temporal_ref['detected'] or not self.db:
            return {'detected': False, 'messages': [], 'facts': []}

        time_start, time_end = temporal_ref['time_range']

        # Get all messages from this period
        messages = self._get_messages_in_range(email, time_start, time_end)

        # Get related facts from this period
        facts = self._get_facts_in_range(email, time_start, time_end)

        # Determine emotional/physical context
        emotional_state = self._extract_emotional_state(messages)
        physical_context = self._extract_physical_context(messages)

        # Build summary
        summary = self._summarize_period(messages, temporal_ref)

        logger.info(f"📚 Temporal context retrieved:")
        logger.info(f"   Messages: {len(messages)}")
        logger.info(f"   Facts: {len(facts)}")
        logger.info(f"   Emotional state: {emotional_state}")

        return {
            'detected': True,
            'time_range': temporal_ref['time_range'],
            'messages': messages,
            'facts': facts,
            'summary': summary,
            'emotional_state': emotional_state,
            'physical_context': physical_context,
            'period_description': temporal_ref['description']
        }

    def format_temporal_context_for_prompt(
        self,
        temporal_context: Dict[str, Any]
    ) -> str:
        """
        Format retrieved temporal context for injection into LLM prompt.

        Returns:
            Formatted string ready for prompt inclusion
        """
        if not temporal_context.get('detected'):
            return ""

        lines = []
        lines.append(f"\n## Context from {temporal_context['period_description']}:")
        lines.append("")

        # Messages in this period
        if temporal_context['messages']:
            lines.append("### Conversation:")
            for msg in temporal_context['messages'][-10:]:  # Last 10 messages
                sender = msg.get('sender_name', 'Unknown')
                text = msg.get('message_text', '')[:200]  # Truncate long messages
                lines.append(f"**{sender}**: {text}")
            lines.append("")

        # Key facts from this period
        if temporal_context['facts']:
            lines.append("### Key Facts from this period:")
            for fact in temporal_context['facts'][:5]:  # Top 5 facts
                content = fact.get('content', fact.get('name', ''))
                lines.append(f"- {content}")
            lines.append("")

        # Emotional context
        if temporal_context['emotional_state']:
            lines.append(f"### Emotional Context: {temporal_context['emotional_state']}")
            lines.append("")

        # Physical/roleplay context
        if temporal_context['physical_context']:
            lines.append(f"### Physical Context: {temporal_context['physical_context']}")
            lines.append("")

        return "\n".join(lines)

    # Private helper methods

    def _get_time_range_hours(self, start_hours: int, end_hours: int = 0) -> Tuple[datetime, datetime]:
        """Get time range in hours from now."""
        now = now_pacific_naive()
        return (
            now + timedelta(hours=start_hours),
            now + timedelta(hours=end_hours)
        )

    def _get_time_range_days(self, start_days: int, end_days: int = 0) -> Tuple[datetime, datetime]:
        """Get time range in days from now."""
        now = now_pacific_naive()
        return (
            now + timedelta(days=start_days),
            now + timedelta(days=end_days)
        )

    def _get_time_range_minutes(self, start_mins: int, end_mins: int = 0) -> Tuple[datetime, datetime]:
        """Get time range in minutes from now."""
        now = now_pacific_naive()
        return (
            now + timedelta(minutes=start_mins),
            now + timedelta(minutes=end_mins)
        )

    def _describe_time_range(self, ref_type: str) -> str:
        """Get human-readable description of time range."""
        descriptions = {
            'this_morning': 'this morning',
            'yesterday_morning': 'yesterday morning',
            'earlier': 'earlier today',
            'recent': 'the past few days',
            'last_session': 'your last conversation',
        }
        return descriptions.get(ref_type, 'that time period')

    def _get_messages_in_range(
        self,
        email: str,
        time_start: datetime,
        time_end: datetime
    ) -> List[Dict]:
        """Get all messages within a time range."""
        try:
            if not self.db:
                return []

            # Get ALL recent messages first
            all_messages = self.db.get_recent_messages(email, limit=100)

            # Filter by time range
            filtered = [
                msg for msg in all_messages
                if self._is_in_range(msg.get('timestamp'), time_start, time_end)
            ]

            logger.debug(f"Found {len(filtered)} messages in time range")
            return filtered

        except Exception as e:
            logger.warning(f"Error retrieving messages for time range: {e}")
            return []

    def _get_facts_in_range(
        self,
        email: str,
        time_start: datetime,
        time_end: datetime
    ) -> List[Dict]:
        """Get facts extracted during a time range."""
        try:
            if not self.db:
                return []

            # Try to get facts with timestamps
            # This depends on your memory system having temporal metadata
            facts = self.db.get_user_facts(email, limit=50)

            # Filter by creation/mention time
            filtered = [
                fact for fact in facts
                if self._is_in_range(
                    fact.get('created') or fact.get('last_mentioned'),
                    time_start,
                    time_end
                )
            ]

            logger.debug(f"Found {len(filtered)} facts in time range")
            return filtered

        except Exception as e:
            logger.warning(f"Error retrieving facts for time range: {e}")
            return []

    def _extract_emotional_state(self, messages: List[Dict]) -> str:
        """Infer emotional state from messages in this period."""
        if not messages:
            return "unknown"

        # Look for emotional markers in recent messages
        text = " ".join([m.get('message_text', '').lower() for m in messages[-5:]])

        emotional_markers = {
            'passionate': ['can\'t get enough', 'need you', 'want you', 'love', 'desire'],
            'playful': ['laugh', 'tease', 'play', 'fun', 'smirk'],
            'intimate': ['closer', 'touch', 'close to', 'tender', 'soft'],
            'intense': ['harder', 'more', 'aggressive', 'rough'],
        }

        for emotion, markers in emotional_markers.items():
            if any(marker in text for marker in markers):
                return emotion

        return "neutral"

    def _extract_physical_context(self, messages: List[Dict]) -> Optional[str]:
        """Extract physical/roleplay context from messages."""
        if not messages:
            return None

        text = " ".join([m.get('message_text', '').lower() for m in messages])

        # Look for physical action indicators
        if '[' in text or 'action' in text or 'touch' in text:
            return "physical/roleplay in progress"

        if 'sore' in text or 'ache' in text or 'mark' in text:
            return "physical aftermath mentioned"

        return None

    def _summarize_period(self, messages: List[Dict], temporal_ref: Dict) -> str:
        """Create a summary of what happened in this period."""
        if not messages:
            return f"No conversation during {temporal_ref['description']}"

        # Simple summary: count messages and get last topic
        count = len(messages)
        last_message = messages[-1].get('message_text', '')[:100] if messages else ''

        return f"{count} messages exchanged. Last: {last_message}"

    def _is_in_range(self, timestamp: Any, start: datetime, end: datetime) -> bool:
        """Check if timestamp falls within range."""
        try:
            if isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, datetime):
                ts = timestamp
            else:
                return False

            return start <= ts <= end
        except:
            return False


def get_temporal_context_agent(db=None) -> TemporalContextAgent:
    """Get or create the temporal context agent."""
    return TemporalContextAgent(db_connection=db)
