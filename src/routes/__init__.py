"""
Routes Module

Registers all Flask blueprints for the application.
"""

from flask import Flask


def register_blueprints(app: Flask):
    """
    Register all route blueprints with the Flask app.

    Args:
        app: Flask application instance
    """
    # Import blueprints
    from routes.auth_routes import auth_bp
    from routes.settings_routes import settings_bp
    from routes.cost_routes import cost_bp
    from routes.feed_routes import feed_bp

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(cost_bp)
    app.register_blueprint(feed_bp)

    print(" All route blueprints registered")
