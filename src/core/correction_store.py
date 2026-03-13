"""
Correction Store -- PostgreSQL persistence for user-reported corrections.

WHAT: Stores and retrieves corrections (wrong_claim, correct_info, subject,
      type, importance) in a PostgreSQL table. Provides LIKE-based search
      across subject, wrong_claim, and correct_info fields.

WHY:  The claim verifier needs a fast way to check "has the user already
      corrected the companion on this topic?" before each response. Storing
      corrections in a queryable table (rather than just in conversation
      history) makes that lookup efficient and reliable.

HOW:  `store_correction()` inserts a row. `search_corrections()` does a
      case-insensitive LIKE search across three columns. Direct psycopg2
      connections (not through the db abstraction layer) for simplicity.

Singleton: `get_correction_store()` at module bottom.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict
from dataclasses import dataclass

from .correction_detector import Correction

logger = logging.getLogger(__name__)


@dataclass
class StoredCorrection:
    """Result of storing a correction."""
    success: bool
    correction_text: str
    timestamp: str
    error: Optional[str] = None


class CorrectionStore:
    """
    Stores corrections in PostgreSQL for future retrieval.

    Corrections are stored with importance scores that allow
    critical corrections (preferences, relationships) to surface
    more prominently than trivial ones.
    """

    def __init__(self):
        self._db = None

    def _get_db(self):
        """Get database connection."""
        if self._db is None:
            from src.database.db import get_db
            self._db = get_db()
        return self._db

    def store_correction(
        self,
        correction: Correction,
        user_email: str = None
    ) -> StoredCorrection:
        """
        Store correction in PostgreSQL.

        Args:
            correction: The detected correction
            user_email: User's email for source tracking

        Returns:
            StoredCorrection with result
        """
        if user_email is None:
            from src.config.persona_config import get_persona_config
            user_email = get_persona_config().primary_user_email
        try:
            db = self._get_db()
            timestamp = datetime.now(timezone.utc)

            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO corrections
                    (email, subject, wrong_claim, correct_info, correction_type, importance, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    user_email,
                    correction.subject,
                    correction.wrong_claim,
                    correction.correct_info,
                    correction.correction_type,
                    correction.importance,
                    correction.confidence
                ))
                correction_id = cursor.fetchone()[0]
                conn.commit()

            logger.info(f"Stored correction #{correction_id} in PostgreSQL: {correction.subject}")
            logger.info(f"  Wrong: {correction.wrong_claim[:50]}...")
            logger.info(f"  Correct: {correction.correct_info[:50]}...")

            return StoredCorrection(
                success=True,
                correction_text=f"{correction.wrong_claim} -> {correction.correct_info}",
                timestamp=timestamp.isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to store correction: {e}")
            return StoredCorrection(
                success=False,
                correction_text="",
                timestamp="",
                error=str(e)
            )

    def get_corrections_for_subject(
        self,
        subject: str,
        email: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get stored corrections related to a subject.

        Args:
            subject: The subject to search for (e.g., "User", "coffee")
            email: Optional user email filter
            limit: Max corrections to return

        Returns:
            List of correction dicts
        """
        try:
            db = self._get_db()

            with db._get_connection() as conn:
                cursor = conn.cursor()

                if email:
                    cursor.execute("""
                        SELECT subject, wrong_claim, correct_info, correction_type,
                               importance, confidence, created_at
                        FROM corrections
                        WHERE email = %s
                          AND (LOWER(subject) LIKE LOWER(%s)
                               OR LOWER(wrong_claim) LIKE LOWER(%s)
                               OR LOWER(correct_info) LIKE LOWER(%s))
                        ORDER BY importance DESC, created_at DESC
                        LIMIT %s
                    """, (email, f"%{subject}%", f"%{subject}%", f"%{subject}%", limit))
                else:
                    cursor.execute("""
                        SELECT subject, wrong_claim, correct_info, correction_type,
                               importance, confidence, created_at
                        FROM corrections
                        WHERE LOWER(subject) LIKE LOWER(%s)
                           OR LOWER(wrong_claim) LIKE LOWER(%s)
                           OR LOWER(correct_info) LIKE LOWER(%s)
                        ORDER BY importance DESC, created_at DESC
                        LIMIT %s
                    """, (f"%{subject}%", f"%{subject}%", f"%{subject}%", limit))

                rows = cursor.fetchall()

                corrections = []
                for row in rows:
                    corrections.append({
                        'subject': row[0],
                        'wrong_claim': row[1],
                        'correct_info': row[2],
                        'correction_type': row[3],
                        'importance': row[4],
                        'confidence': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    })

                return corrections

        except Exception as e:
            logger.warning(f"Failed to get corrections: {e}")
            return []


# Singleton instance
_store: Optional[CorrectionStore] = None


def get_correction_store() -> CorrectionStore:
    """Get singleton CorrectionStore instance."""
    global _store
    if _store is None:
        _store = CorrectionStore()
    return _store


def store_correction(
    correction: Correction,
    user_email: str = None
) -> StoredCorrection:
    """
    Convenience function to store a correction.

    Args:
        correction: The detected correction
        user_email: User's email

    Returns:
        StoredCorrection with result
    """
    if user_email is None:
        from src.config.persona_config import get_persona_config
        user_email = get_persona_config().primary_user_email
    store = get_correction_store()
    return store.store_correction(correction, user_email)
