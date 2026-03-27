"""
Error Tracker — Structured error recording for the companion framework.

WHAT: Captures exceptions from any subsystem with context (module, function,
      companion_id, stack trace) and stores them in PostgreSQL for admin review.
WHY:  The framework uses graceful degradation everywhere — subsystems catch
      exceptions to avoid crashing. This means errors are silently swallowed.
      The error tracker records them so admins can see what's actually failing
      without tailing logs.
HOW:  Call error_tracker.record(e, context) anywhere an exception is caught.
      The tracker batches writes and handles its own failures gracefully
      (logs to stderr if DB is down — never raises).

Usage:
    from src.utils.error_tracker import record_error

    try:
        do_something()
    except Exception as e:
        record_error(e, module='fact_store', companion_id='companion')
        # continue gracefully
"""

import logging
import os
import traceback
from datetime import datetime
from typing import Optional

from src.database import tables as T

logger = logging.getLogger(__name__)

# In-memory buffer for when DB is unavailable
_error_buffer = []
_MAX_BUFFER = 100


def _get_connection():
    """Get a DB connection for error storage. Returns None if unavailable."""
    try:
        from src.database.connection import get_connection
        return get_connection()
    except Exception:
        return None



def record_error(
    exception: Exception,
    module: str = '',
    function_name: str = '',
    companion_id: str = '',
    context: str = '',
) -> None:
    """Record an error for later admin review.

    This function NEVER raises — it's designed to be called from except blocks.
    If the DB is unavailable, errors are buffered in memory (up to 100).
    """
    try:
        error_type = type(exception).__name__
        error_message = str(exception)[:2000]
        stack = traceback.format_exc()[:4000]
        companion_id = companion_id or os.environ.get('COMPANION_ID', '')

        conn = _get_connection()
        if conn is None:
            _buffer_error(module, function_name, companion_id, error_type, error_message, stack, context)
            return

        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {T.ERROR_LOG}
                        (module, function_name, companion_id, error_type, error_message, stack_trace, context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (module, function_name, companion_id, error_type, error_message, stack, context))
                conn.commit()

            # Flush buffer if we have pending errors
            _flush_buffer(conn)

        except Exception as e:
            logger.debug(f"Error tracker DB write failed: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            _buffer_error(module, function_name, companion_id, error_type, error_message, stack, context)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    except Exception:
        # Absolute last resort — never let the tracker itself crash
        pass


def _buffer_error(module, function_name, companion_id, error_type, error_message, stack, context):
    """Buffer an error in memory when DB is unavailable."""
    global _error_buffer
    if len(_error_buffer) < _MAX_BUFFER:
        _error_buffer.append({
            'timestamp': datetime.now(tz=None).isoformat(),
            'module': module,
            'function_name': function_name,
            'companion_id': companion_id,
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack,
            'context': context,
        })


def _flush_buffer(conn):
    """Write buffered errors to DB."""
    global _error_buffer
    if not _error_buffer:
        return
    try:
        with conn.cursor() as cur:
            for err in _error_buffer:
                cur.execute(f"""
                    INSERT INTO {T.ERROR_LOG}
                        (timestamp, module, function_name, companion_id,
                         error_type, error_message, stack_trace, context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    err['timestamp'], err['module'], err['function_name'],
                    err['companion_id'], err['error_type'], err['error_message'],
                    err['stack_trace'], err['context'],
                ))
            conn.commit()
        _error_buffer = []
    except Exception:
        pass


def get_recent_errors(
    companion_id: str = None,
    limit: int = 50,
    module: str = None,
    unresolved_only: bool = False,
) -> list:
    """Retrieve recent errors for admin review.

    Returns list of dicts, newest first. Returns empty list on any failure.
    """
    try:
        conn = _get_connection()
        if conn is None:
            return []

        try:
            conditions = []
            params = []

            if companion_id:
                conditions.append("companion_id = %s")
                params.append(companion_id)
            if module:
                conditions.append("module = %s")
                params.append(module)
            if unresolved_only:
                conditions.append("resolved = FALSE")

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, timestamp, module, function_name, companion_id,
                           error_type, error_message, stack_trace, context, resolved
                    FROM {T.ERROR_LOG}
                    {where}
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, params)
                return cur.fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return []


def get_error_summary(companion_id: str = None, days: int = 7) -> dict:
    """Get error counts grouped by module for the last N days."""
    try:
        conn = _get_connection()
        if conn is None:
            return {'total': 0, 'by_module': {}, 'by_type': {}}

        try:
            _ensure_table(conn)
            from psycopg2.extras import RealDictCursor
            params = [days]
            companion_filter = ""
            if companion_id:
                companion_filter = "AND companion_id = %s"
                params.append(companion_id)

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT module, error_type, COUNT(*) as cnt
                    FROM error_log
                    WHERE timestamp > NOW() - INTERVAL '%s days'
                    {companion_filter}
                    GROUP BY module, error_type
                    ORDER BY cnt DESC
                """, params)
                rows = cur.fetchall()

            by_module = {}
            by_type = {}
            total = 0
            for row in rows:
                m = row['module'] or 'unknown'
                t = row['error_type'] or 'unknown'
                c = row['cnt']
                by_module[m] = by_module.get(m, 0) + c
                by_type[t] = by_type.get(t, 0) + c
                total += c

            return {'total': total, 'by_module': by_module, 'by_type': by_type}
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return {'total': 0, 'by_module': {}, 'by_type': {}}
