"""
Fact Approval System - Human-in-the-loop gating for sensitive facts.

WHAT: Intercepts extracted facts that touch sensitive categories (children,
trauma, relationship changes, corrections to ground truth) and routes them
through a pending approval queue before they reach permanent storage.

WHY: Not every fact the LLM extracts should be stored automatically. A
misheard "my son likes peanuts" could be dangerous if the kid is allergic.
Relationship changes, information about children, and trauma-related facts
all warrant a human check. This module implements that safety layer.

HOW it fits:
  - The fact extraction pipeline calls detect_sensitivity() on each
    candidate fact. If sensitivity != NONE, the fact is routed to
    add_pending_fact() instead of being stored directly.
  - Optionally, get_llm_review() provides a second-opinion recommendation
    (approve / reject / needs_review) that is displayed alongside the fact
    in the approval UI.
  - The frontend sends approve/reject/edit actions via WebSocket, which
    call approve_fact(), reject_fact(), or edit_and_approve_fact().
  - Approved facts are promoted to the permanent facts table (via
    fact_store) and optionally to entity profiles.

Flow: extraction -> sensitivity check -> pending_facts table
      -> [optional LLM review] -> WebSocket to user -> user decision
      -> facts table (if approved)
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

import psycopg2
from psycopg2.extras import RealDictCursor

from src.core.clock import now as clock_now
from src.database import tables as T

logger = logging.getLogger(__name__)


class FactSensitivity(Enum):
    """Categories of sensitive facts requiring approval."""
    NONE = "none"                    # Auto-approve
    RELATIONSHIP = "relationship"    # New people, relationship changes
    CHILDREN = "children"            # Facts about kids
    TRAUMA = "trauma"                # Trauma, boundaries, vulnerabilities
    CORRECTION = "correction"        # Contradicts existing hard fact
    ENTITY_UPDATE = "entity_update"  # Would update entity profile


class PendingFactStatus(Enum):
    """Status of a pending fact."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


# Keywords that trigger sensitivity detection
SENSITIVITY_KEYWORDS = {
    FactSensitivity.RELATIONSHIP: [
        "dating", "married", "divorced", "separated", "partner", "boyfriend",
        "girlfriend", "engaged", "broke up", "relationship", "seeing someone"
    ],
    FactSensitivity.CHILDREN: [
        "jesse", "kyler", "son", "daughter", "child", "kid", "custody",
        "parenting", "school", "medication"
    ],
    FactSensitivity.TRAUMA: [
        "trauma", "abuse", "anxiety", "depression", "therapy", "boundary",
        "trigger", "ptsd", "fear", "panic", "vulnerable", "hurt"
    ],
}


class FactApprovalService:
    """Manages the fact approval workflow."""

    def __init__(self):
        self._conn = None

    def _get_connection(self):
        """Get database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                dbname=os.environ.get('POSTGRES_DB', 'companion'),
                user=os.environ.get('POSTGRES_USER', 'companion'),
                password=os.environ.get('POSTGRES_PASSWORD', '')
            )
        return self._conn

    def detect_sensitivity(self, fact: Dict[str, Any]) -> tuple[FactSensitivity, str]:
        """
        Determine if a fact requires approval and why.

        Returns:
            (sensitivity_level, reason)
        """
        subject = fact.get('subject', '').lower()
        fact_text = fact.get('fact', '').lower()
        category = fact.get('category', '').lower()

        # Check each sensitivity category
        for sensitivity, keywords in SENSITIVITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in fact_text or keyword in subject:
                    reason = f"Contains '{keyword}' - may be {sensitivity.value}"
                    return sensitivity, reason

        # Category-based sensitivity
        if category == 'relationship':
            return FactSensitivity.RELATIONSHIP, "Relationship category fact"

        # Default: no approval needed
        return FactSensitivity.NONE, ""

    def add_pending_fact(
        self,
        fact: Dict[str, Any],
        sensitivity: FactSensitivity,
        reason: str,
        user_email: str,
        message_id: int = None,
        source_message: str = None
    ) -> Optional[int]:
        """Add a fact to the pending approval queue."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    INSERT INTO {T.PENDING_FACTS} (
                        subject, predicate, fact_text, category, confidence,
                        importance, sensitivity, sensitivity_reason,
                        status, user_email, message_id, source_message
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    fact.get('subject', 'Unknown'),
                    fact.get('category', 'general'),
                    fact.get('fact', ''),
                    fact.get('category', 'general'),
                    fact.get('confidence', 0.7),
                    fact.get('importance', 5),
                    sensitivity.value,
                    reason,
                    PendingFactStatus.PENDING.value,
                    user_email,
                    message_id,
                    source_message
                ))
                fact_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Added pending fact {fact_id}: {fact.get('fact', '')[:50]}...")
            return fact_id
        except Exception as e:
            logger.error(f"Error adding pending fact: {e}")
            conn.rollback()
            return None

    def get_pending_facts(self, user_email: str = None, limit: int = 10) -> List[Dict]:
        """Get pending facts awaiting approval."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if user_email:
                    cursor.execute(f"""
                        SELECT * FROM {T.PENDING_FACTS}
                        WHERE status = 'pending' AND user_email = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_email, limit))
                else:
                    cursor.execute(f"""
                        SELECT * FROM {T.PENDING_FACTS}
                        WHERE status = 'pending'
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting pending facts: {e}")
            return []

    def approve_fact(self, fact_id: int, reviewed_by: str = None) -> bool:
        """Approve a pending fact and move to permanent storage."""
        conn = self._get_connection()
        try:
            # Get the pending fact
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    f"SELECT * FROM {T.PENDING_FACTS} WHERE id = %s",
                    (fact_id,)
                )
                pending = cursor.fetchone()

            if not pending:
                logger.warning(f"Pending fact {fact_id} not found")
                return False

            # Store in permanent facts table
            from src.memory.fact_store import get_fact_store
            store = get_fact_store()

            store.store_fact(
                subject=pending['subject'],
                predicate=pending['predicate'],
                obj=pending['fact_text'],
                confidence=pending['confidence'],
                importance=pending['importance'],
                source='approved',
                user_email=pending['user_email'],
                message_id=pending['message_id']
            )

            # Update pending status
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.PENDING_FACTS}
                    SET status = %s, reviewed_at = %s, reviewed_by = %s
                    WHERE id = %s
                """, (
                    PendingFactStatus.APPROVED.value,
                    clock_now(),
                    reviewed_by,
                    fact_id
                ))
            conn.commit()

            logger.info(f"Approved fact {fact_id}")
            return True

        except Exception as e:
            logger.error(f"Error approving fact: {e}")
            conn.rollback()
            return False

    def reject_fact(self, fact_id: int, reviewed_by: str = None, reason: str = None) -> bool:
        """Reject a pending fact."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.PENDING_FACTS}
                    SET status = %s, reviewed_at = %s, reviewed_by = %s, edit_note = %s
                    WHERE id = %s
                """, (
                    PendingFactStatus.REJECTED.value,
                    clock_now(),
                    reviewed_by,
                    reason,
                    fact_id
                ))
            conn.commit()
            logger.info(f"Rejected fact {fact_id}")
            return True
        except Exception as e:
            logger.error(f"Error rejecting fact: {e}")
            conn.rollback()
            return False

    def edit_and_approve_fact(
        self,
        fact_id: int,
        new_fact_text: str,
        reviewed_by: str = None
    ) -> bool:
        """Edit a fact's text and approve it."""
        conn = self._get_connection()
        try:
            # Get the pending fact
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    f"SELECT * FROM {T.PENDING_FACTS} WHERE id = %s",
                    (fact_id,)
                )
                pending = cursor.fetchone()

            if not pending:
                return False

            # Store edited version in permanent facts table
            from src.memory.fact_store import get_fact_store
            store = get_fact_store()

            store.store_fact(
                subject=pending['subject'],
                predicate=pending['predicate'],
                obj=new_fact_text,  # Use edited text
                confidence=pending['confidence'],
                importance=pending['importance'],
                source='approved_edited',
                user_email=pending['user_email'],
                message_id=pending['message_id']
            )

            # Update pending status
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.PENDING_FACTS}
                    SET status = %s, reviewed_at = %s, reviewed_by = %s,
                        edit_note = %s
                    WHERE id = %s
                """, (
                    PendingFactStatus.EDITED.value,
                    clock_now(),
                    reviewed_by,
                    f"Original: {pending['fact_text']}\nEdited to: {new_fact_text}",
                    fact_id
                ))
            conn.commit()

            logger.info(f"Edited and approved fact {fact_id}")
            return True

        except Exception as e:
            logger.error(f"Error editing fact: {e}")
            conn.rollback()
            return False

    def get_llm_review(self, fact: Dict[str, Any], context: str = None) -> Dict[str, Any]:
        """
        Get LLM second opinion on a sensitive fact.

        Returns:
            {
                'recommendation': 'approve' | 'reject' | 'needs_review',
                'reasoning': str,
                'confidence': float
            }
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ.get('FIREWORKS_API_KEY')
            )

            prompt = f"""You are reviewing a fact extracted from a conversation for accuracy and appropriateness.

FACT TO REVIEW:
Subject: {fact.get('subject', 'Unknown')}
Fact: {fact.get('fact', '')}
Category: {fact.get('category', 'unknown')}
Confidence: {fact.get('confidence', 0.7)}

{f'CONTEXT: {context}' if context else ''}

REVIEW CRITERIA:
1. Is this fact specific and meaningful (not vague or temporary)?
2. Does it contradict known information about the subject?
3. Is it appropriate to store long-term?
4. Is the subject correctly identified?

Respond with JSON:
{{
    "recommendation": "approve" | "reject" | "needs_review",
    "reasoning": "Brief explanation",
    "confidence": 0.0-1.0,
    "suggested_edit": "optional corrected fact text if needed"
}}

/no_think
Return ONLY valid JSON:"""

            from src.llm.fireworks_models import call_fireworks

            content = call_fireworks(
                client,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )

            if not content:
                return {"recommendation": "approve", "reasoning": "LLM review unavailable", "confidence": 0.0}

            # Parse JSON
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)

        except Exception as e:
            logger.warning(f"LLM review failed: {e}")
            return {
                'recommendation': 'needs_review',
                'reasoning': f'LLM review failed: {e}',
                'confidence': 0.0
            }

    def store_llm_review(self, fact_id: int, review: Dict[str, Any]) -> bool:
        """Store LLM review results on a pending fact."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    UPDATE {T.PENDING_FACTS}
                    SET llm_review = %s, llm_recommendation = %s
                    WHERE id = %s
                """, (
                    json.dumps(review),
                    review.get('recommendation', 'needs_review'),
                    fact_id
                ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing LLM review: {e}")
            conn.rollback()
            return False


# Singleton
_service: Optional[FactApprovalService] = None

def get_approval_service() -> FactApprovalService:
    """Get the singleton FactApprovalService instance."""
    global _service
    if _service is None:
        _service = FactApprovalService()
    return _service
