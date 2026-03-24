"""
Fact Approval Routes -- REST API for the human-in-the-loop fact review pipeline.

WHAT: CRUD-style endpoints that let the user approve, reject, or edit facts
      extracted from conversation before they are committed to long-term memory.

WHY:  Automatically extracted facts can be wrong or sensitive. This workflow
      gives the user veto power over what the companion "remembers" permanently.
      An optional LLM second-opinion endpoint helps triage ambiguous facts.

HOW:  Each endpoint delegates to `fact_approval.get_approval_service()` which
      manages the `pending_facts` table in PostgreSQL. The `/review` endpoint
      calls the LLM for a second opinion and caches the result on the row.

Endpoints:
  GET  /api/approvals/pending       -- List pending facts
  POST /api/approvals/<id>/approve  -- Approve a fact
  POST /api/approvals/<id>/reject   -- Reject a fact
  POST /api/approvals/<id>/edit     -- Edit and approve a fact
  GET  /api/approvals/<id>/review   -- Get LLM review for a fact
"""

from flask import Blueprint, request, jsonify, g
from functools import wraps
import logging

from src.database import tables as T

logger = logging.getLogger(__name__)

approval_bp = Blueprint('approvals', __name__, url_prefix='/api/approvals')


def require_auth(f):
    """Simple auth check - requires email in session or header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check for email in various places
        email = None

        # From session
        if hasattr(g, 'user_email'):
            email = g.user_email

        # From header
        if not email:
            email = request.headers.get('X-User-Email')

        # From query param (for testing)
        if not email:
            email = request.args.get('email')

        if not email:
            return jsonify({'error': 'Authentication required'}), 401

        g.user_email = email
        return f(*args, **kwargs)

    return decorated


@approval_bp.route('/pending', methods=['GET'])
@require_auth
def get_pending_facts():
    """Get all pending facts for the current user."""
    try:
        from src.memory.fact_approval import get_approval_service

        service = get_approval_service()
        pending = service.get_pending_facts(user_email=g.user_email, limit=50)

        return jsonify({
            'status': 'success',
            'count': len(pending),
            'pending_facts': pending
        })

    except Exception as e:
        logger.error(f"Error getting pending facts: {e}")
        return jsonify({'error': str(e)}), 500


@approval_bp.route('/<int:fact_id>/approve', methods=['POST'])
@require_auth
def approve_fact(fact_id: int):
    """Approve a pending fact."""
    try:
        from src.memory.fact_approval import get_approval_service

        service = get_approval_service()
        success = service.approve_fact(fact_id, reviewed_by=g.user_email)

        if success:
            return jsonify({
                'status': 'success',
                'message': f'Fact {fact_id} approved and stored'
            })
        else:
            return jsonify({'error': 'Failed to approve fact'}), 400

    except Exception as e:
        logger.error(f"Error approving fact: {e}")
        return jsonify({'error': str(e)}), 500


@approval_bp.route('/<int:fact_id>/reject', methods=['POST'])
@require_auth
def reject_fact(fact_id: int):
    """Reject a pending fact."""
    try:
        from src.memory.fact_approval import get_approval_service

        data = request.get_json() or {}
        reason = data.get('reason', '')

        service = get_approval_service()
        success = service.reject_fact(
            fact_id,
            reviewed_by=g.user_email,
            reason=reason
        )

        if success:
            return jsonify({
                'status': 'success',
                'message': f'Fact {fact_id} rejected'
            })
        else:
            return jsonify({'error': 'Failed to reject fact'}), 400

    except Exception as e:
        logger.error(f"Error rejecting fact: {e}")
        return jsonify({'error': str(e)}), 500


@approval_bp.route('/<int:fact_id>/edit', methods=['POST'])
@require_auth
def edit_fact(fact_id: int):
    """Edit a fact and approve it."""
    try:
        from src.memory.fact_approval import get_approval_service

        data = request.get_json()
        if not data or 'fact_text' not in data:
            return jsonify({'error': 'fact_text is required'}), 400

        service = get_approval_service()
        success = service.edit_and_approve_fact(
            fact_id,
            new_fact_text=data['fact_text'],
            reviewed_by=g.user_email
        )

        if success:
            return jsonify({
                'status': 'success',
                'message': f'Fact {fact_id} edited and approved'
            })
        else:
            return jsonify({'error': 'Failed to edit fact'}), 400

    except Exception as e:
        logger.error(f"Error editing fact: {e}")
        return jsonify({'error': str(e)}), 500


@approval_bp.route('/<int:fact_id>/review', methods=['GET'])
@require_auth
def get_llm_review(fact_id: int):
    """Get LLM second opinion on a pending fact."""
    try:
        from src.memory.fact_approval import get_approval_service
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import os

        service = get_approval_service()

        # Get the pending fact
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'postgres'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            dbname=os.environ.get('POSTGRES_DB', 'companion'),
            user=os.environ.get('POSTGRES_USER', 'companion'),
            password=os.environ.get('POSTGRES_PASSWORD', '')
        )

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                f"SELECT * FROM {T.PENDING_FACTS} WHERE id = %s",
                (fact_id,)
            )
            pending = cursor.fetchone()

        if not pending:
            return jsonify({'error': 'Fact not found'}), 404

        # If already reviewed, return cached review
        if pending.get('llm_review'):
            import json
            return jsonify({
                'status': 'success',
                'review': json.loads(pending['llm_review']),
                'cached': True
            })

        # Get fresh LLM review
        fact = {
            'subject': pending['subject'],
            'fact': pending['fact_text'],
            'category': pending['category'],
            'confidence': pending['confidence']
        }

        review = service.get_llm_review(fact, context=pending.get('source_message'))

        # Store the review
        service.store_llm_review(fact_id, review)

        return jsonify({
            'status': 'success',
            'review': review,
            'cached': False
        })

    except Exception as e:
        logger.error(f"Error getting LLM review: {e}")
        return jsonify({'error': str(e)}), 500


def register_approval_routes(app):
    """Register approval routes with the Flask app."""
    app.register_blueprint(approval_bp)
    logger.info("Registered approval routes")
