"""
Companion Reminder System -- internal to-do list and follow-up tracker.

WHAT: CRUD functions for the companion's personal reminders: add, complete,
      delete, list (with/without completed), get due/overdue, get by date.
      This is separate from Google Calendar -- it's the companion's private
      "things to remember" list.

WHY:  The companion needs to track follow-ups ("ask James about his interview
      tomorrow") independently from the user's calendar. These reminders drive
      the proactive-curiosity system and daily check-in topics.

HOW:  PostgreSQL table `companion_reminders` with columns: id, title, due_date,
      notes, completed, created_at, updated_at. All functions return dicts
      with serialized datetime fields for JSON compatibility.
"""
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


def _get_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'postgres'),
        port=os.environ.get('POSTGRES_PORT', 5432),
        database=os.environ.get('POSTGRES_DB', 'companion'),
        user=os.environ.get('POSTGRES_USER', 'companion'),
        password=os.environ.get('POSTGRES_PASSWORD', '')
    )


def get_reminders(include_completed: bool = False) -> List[Dict]:
    """Get all of the companion's reminders.

    Args:
        include_completed: Include completed reminders (default False)

    Returns:
        List of reminders with keys:
        - id: Reminder ID
        - title: What to remember
        - due_date: When it's due (or None)
        - notes: Additional notes
        - completed: Whether it's done
        - created_at: When it was created

    Example:
        >>> from tools import reminders
        >>> my_reminders = reminders.get_reminders()
        >>> for r in my_reminders:
        ...     print(f"{r['title']} - due: {r['due_date']}")
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if include_completed:
                cur.execute("""
                    SELECT id, title, due_date, notes, completed, created_at
                    FROM companion_reminders
                    ORDER BY
                        CASE WHEN due_date IS NULL THEN 1 ELSE 0 END,
                        due_date ASC,
                        created_at DESC
                """)
            else:
                cur.execute("""
                    SELECT id, title, due_date, notes, completed, created_at
                    FROM companion_reminders
                    WHERE completed = FALSE
                    ORDER BY
                        CASE WHEN due_date IS NULL THEN 1 ELSE 0 END,
                        due_date ASC,
                        created_at DESC
                """)

            results = cur.fetchall()
            conn.close()

            # Convert datetime objects to strings for JSON serialization
            reminders = []
            for r in results:
                reminder = dict(r)
                if reminder.get('due_date'):
                    reminder['due_date'] = reminder['due_date'].isoformat()
                if reminder.get('created_at'):
                    reminder['created_at'] = reminder['created_at'].isoformat()
                reminders.append(reminder)

            return reminders

    except Exception as e:
        return [{"error": f"Failed to get reminders: {str(e)}"}]


def add_reminder(
    title: str,
    due_date: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict:
    """Add a new reminder for the companion.

    Args:
        title: What to remember (e.g., "Ask James about his presentation")
        due_date: Optional due date in ISO format ("2024-01-15" or "2024-01-15 14:00")
        notes: Optional additional notes

    Returns:
        {"status": "success", "reminder_id": 123}
        or {"status": "error", "message": "..."}

    Example:
        >>> from tools import reminders
        >>> reminders.add_reminder(
        ...     "Follow up on James's interview",
        ...     due_date="2024-01-20",
        ...     notes="He mentioned it was at 2pm"
        ... )
        {'status': 'success', 'reminder_id': 42}
    """
    try:
        # Parse due_date if provided
        parsed_due = None
        if due_date:
            try:
                # Try datetime format first
                parsed_due = datetime.fromisoformat(due_date)
            except ValueError:
                try:
                    # Try date-only format
                    parsed_due = datetime.strptime(due_date, "%Y-%m-%d")
                except ValueError:
                    return {"status": "error", "message": f"Invalid date format: {due_date}"}

        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO companion_reminders (title, due_date, notes, completed, created_at)
                VALUES (%s, %s, %s, FALSE, NOW())
                RETURNING id
            """, (title, parsed_due, notes))

            reminder_id = cur.fetchone()[0]
            conn.commit()
            conn.close()

            return {"status": "success", "reminder_id": reminder_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def complete_reminder(reminder_id: int) -> Dict:
    """Mark a reminder as completed.

    Args:
        reminder_id: The ID of the reminder to complete

    Returns:
        {"status": "success"} or {"status": "error", "message": "..."}

    Example:
        >>> from tools import reminders
        >>> reminders.complete_reminder(42)
        {'status': 'success'}
    """
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE companion_reminders
                SET completed = TRUE, updated_at = NOW()
                WHERE id = %s
            """, (reminder_id,))

            if cur.rowcount == 0:
                conn.close()
                return {"status": "error", "message": f"Reminder {reminder_id} not found"}

            conn.commit()
            conn.close()

            return {"status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_reminder(reminder_id: int) -> Dict:
    """Delete a reminder.

    Args:
        reminder_id: The ID of the reminder to delete

    Returns:
        {"status": "success"} or {"status": "error", "message": "..."}

    Example:
        >>> from tools import reminders
        >>> reminders.delete_reminder(42)
        {'status': 'success'}
    """
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM companion_reminders
                WHERE id = %s
            """, (reminder_id,))

            if cur.rowcount == 0:
                conn.close()
                return {"status": "error", "message": f"Reminder {reminder_id} not found"}

            conn.commit()
            conn.close()

            return {"status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_due_reminders() -> List[Dict]:
    """Get reminders that are due today or overdue.

    Returns:
        List of reminders that need attention

    Example:
        >>> from tools import reminders
        >>> due = reminders.get_due_reminders()
        >>> if due:
        ...     print(f"You have {len(due)} reminders due!")
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, due_date, notes, created_at
                FROM companion_reminders
                WHERE completed = FALSE
                  AND due_date IS NOT NULL
                  AND due_date <= NOW() + INTERVAL '1 day'
                ORDER BY due_date ASC
            """)

            results = cur.fetchall()
            conn.close()

            reminders = []
            for r in results:
                reminder = dict(r)
                if reminder.get('due_date'):
                    reminder['due_date'] = reminder['due_date'].isoformat()
                if reminder.get('created_at'):
                    reminder['created_at'] = reminder['created_at'].isoformat()
                reminders.append(reminder)

            return reminders

    except Exception as e:
        return [{"error": f"Failed to get due reminders: {str(e)}"}]


def get_reminders_for_date(date: str) -> List[Dict]:
    """Get reminders due on a specific date.

    Args:
        date: Date in ISO format (e.g., "2024-01-15")

    Returns:
        List of reminders due on that date

    Example:
        >>> from tools import reminders
        >>> tomorrow = reminders.get_reminders_for_date("2024-01-16")
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()

        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, due_date, notes, created_at
                FROM companion_reminders
                WHERE completed = FALSE
                  AND DATE(due_date) = %s
                ORDER BY due_date ASC
            """, (target_date,))

            results = cur.fetchall()
            conn.close()

            reminders = []
            for r in results:
                reminder = dict(r)
                if reminder.get('due_date'):
                    reminder['due_date'] = reminder['due_date'].isoformat()
                if reminder.get('created_at'):
                    reminder['created_at'] = reminder['created_at'].isoformat()
                reminders.append(reminder)

            return reminders

    except Exception as e:
        return [{"error": f"Failed to get reminders: {str(e)}"}]
