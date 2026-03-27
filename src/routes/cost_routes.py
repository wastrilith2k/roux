"""
Cost Tracking Routes -- API endpoints for API-spend analytics.

WHAT: Three endpoints: cost summary (today/month/projected), historical daily
      breakdown for charting, and CSV export.

WHY:  The companion calls multiple paid APIs (Fireworks, OpenAI, RunComfy, etc.).
      These endpoints feed the admin dashboard so the owner can monitor spend
      against per-service budgets and spot anomalies.

HOW:  Delegates to `CostTracker` (SQLite-backed) which aggregates per-service
      daily totals. All endpoints require legacy session-token auth.
"""

import os
import sys

# Legacy path manipulation -- some imports still rely on these being on sys.path.
# TODO: Migrate remaining bare imports to fully-qualified src.* paths.
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(src_dir, 'services'))
sys.path.insert(0, os.path.join(src_dir, 'user'))
sys.path.insert(0, src_dir)

from flask import Blueprint, request, jsonify, session, Response
from src.database.simple_auth import token_required
from src.services.cost_tracker import get_cost_tracker

# Create blueprint
cost_bp = Blueprint('costs', __name__, url_prefix='/api/costs')


@cost_bp.route('/summary', methods=['GET'])
@token_required
def get_cost_summary():
    """Get complete cost summary across all services"""
    try:
        username = session.get('username', os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin'))
        cost_tracker = get_cost_tracker()
        summary = cost_tracker.get_cost_summary(username)

        # Convert dataclass to dict for JSON serialization
        return jsonify({
            'total_today': summary.total_today,
            'total_month': summary.total_month,
            'total_budget': summary.total_budget,
            'projected_month_end': summary.projected_month_end,
            'percentage_used': summary.percentage_used,
            'days_in_month': summary.days_in_month,
            'day_of_month': summary.day_of_month,
            'services': [
                {
                    'service_name': s.service_name,
                    'display_name': s.display_name,
                    'icon': s.icon,
                    'cost_today': s.cost_today,
                    'cost_month': s.cost_month,
                    'budget_month': s.budget_month,
                    'percentage_used': s.percentage_used,
                    'remaining_budget': s.remaining_budget,
                    'usage_details': s.usage_details,
                    'links': s.links
                }
                for s in summary.services
            ],
            'alerts': summary.alerts
        })
    except Exception as e:
        print(f"Error fetching cost summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@cost_bp.route('/history', methods=['GET'])
@token_required
def get_cost_history():
    """Get historical cost data for charting"""
    try:
        username = session.get('username', os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin'))
        days = int(request.args.get('days', 30))
        cost_tracker = get_cost_tracker()
        history = cost_tracker.get_historical_data(username, days)

        return jsonify({
            'history': history,
            'days': days
        })
    except Exception as e:
        print(f"Error fetching cost history: {e}")
        return jsonify({'error': str(e)}), 500


@cost_bp.route('/export', methods=['GET'])
@token_required
def export_costs():
    """Export cost data to CSV"""
    try:
        username = session.get('username', os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin'))
        days = int(request.args.get('days', 30))
        cost_tracker = get_cost_tracker()
        history = cost_tracker.get_historical_data(username, days)

        # Generate CSV
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(['Date', 'Fireworks', 'OpenAI', 'Hedra', 'RunComfy', 'Twilio', 'Google', 'Total'])

        # Data rows
        for row in history:
            writer.writerow([
                row['date'],
                f"${row['fireworks_cost_usd']:.2f}",
                f"${row['openai_cost_usd']:.2f}",
                f"${row['hedra_cost_usd']:.2f}",
                f"${row['runcomfy_cost_usd']:.2f}",
                f"${row['twilio_cost_usd']:.2f}",
                f"${row['google_cost_usd']:.2f}",
                f"${row['total_cost_usd']:.2f}"
            ])

        csv_data = output.getvalue()
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=companion_costs_{days}days.csv'}
        )
    except Exception as e:
        print(f"Error exporting costs: {e}")
        return jsonify({'error': str(e)}), 500
