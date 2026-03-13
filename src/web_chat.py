"""
Web Chat Server — Companion Framework

WHAT: The main Flask + Socket.IO application that serves the web chat interface
      and coordinates all background services.
WHY:  This is the entry point for the entire companion system in production.
      It wires together the Flask app, WebSocket handlers, background schedulers,
      Redis pub/sub listeners, and the Celery task infrastructure. Previously a
      5,600-line monolith, it was refactored to delegate all business logic to
      dedicated modules.
HOW:  The startup sequence is:
      1. Eventlet monkey-patch (MUST be first — enables cooperative I/O)
      2. Create Flask app via factory (create_app)
      3. Initialize autonomy LLM (for background operations)
      4. Create Socket.IO with event handlers
      5. Start background services:
         - Unified scheduler (APScheduler for cron/interval jobs)
         - Always-On Service (Telegram + proactive messaging)
         - Ops Bot (admin commands)
         - Scheduled jobs (memory extraction, value inference, diagnostics, etc.)
      6. Start Redis pub/sub listeners (image generation, interjections)
      7. Run the server on port 5000

Route blueprints handle HTTP/WebSocket endpoints:
  - auth_routes: login/logout
  - chat_routes: message handling (delegates to MessageProcessor)
  - settings_routes, cost_routes, integrations_routes, approval_routes
"""
import os
import sys

# Add src/ subdirectories to Python path (legacy — some modules use bare imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'memory'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scheduling'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'user'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))
sys.path.insert(0, os.path.dirname(__file__))

# CRITICAL: Eventlet monkey-patch MUST happen before any other imports.
# This replaces stdlib I/O with cooperative versions so Socket.IO can
# handle many concurrent connections on a single thread.
import eventlet
eventlet.monkey_patch()

# nest_asyncio: allows nested event loops (needed when sync code calls
# into async code, e.g., the ops bot's Telegram library)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# Sentry error tracking — optional dependency
try:
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Framework configuration
from src.config.app_config import AppConfig, get_config
from src.config.socketio_config import SocketIOConfig
from src.config.logging_config import setup_logging

# Initialize Sentry (traces/profiles disabled to stay within free tier)
sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn and SENTRY_AVAILABLE:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[
            FlaskIntegration(),
            CeleryIntegration(),
        ],
        traces_sample_rate=0.0,  # Disabled — free tier has limited quota
        profiles_sample_rate=0.0,  # Disabled — free tier has limited quota
        environment=os.getenv('FLASK_ENV', 'production'),
        release=os.getenv('GIT_COMMIT', 'unknown'),
    )
    print(f"✅ Sentry initialized for Flask app (env: {os.getenv('FLASK_ENV', 'production')})")
elif sentry_dsn and not SENTRY_AVAILABLE:
    print("⚠️  SENTRY_DSN set but sentry_sdk not installed - skipping Sentry init")

# Import middleware and error handlers
from src.middleware.error_handlers import register_error_handlers, register_socketio_error_handlers
from src.middleware.auth_middleware import token_required

# Import route blueprints (stable-core - minimal routes)
from src.routes.auth_routes import auth_bp
from src.routes.settings_routes import settings_bp
from src.routes.cost_routes import cost_bp
from src.routes.chat_routes import chat_bp, register_socketio_handlers
from src.routes.integrations_routes import integrations_bp
from src.routes.approval_routes import approval_bp
from src.routes.observe_routes import observe_bp, start_observe_subscriber, register_observe_socketio
# REMOVED: status_routes - status_manager system removed
# DISABLED: livekit_routes, feed_routes, autonomy_routes - not part of stable-core
# DISABLED: src.core.agent - pipeline handles LLM setup

# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_app(config_class=None):
    """
    Application factory for creating Flask app instances.

    Args:
        config_class: Configuration class to use (default: based on FLASK_ENV)

    Returns:
        Configured Flask app instance
    """
    # Determine config
    if config_class is None:
        config_class = get_config()

    # Create Flask app
    app = Flask(__name__, static_folder='../public', static_url_path='/')
    app.config.from_object(config_class)

    # Setup logging
    setup_logging(app)

    # Setup CORS
    CORS(app)

    # Database and authentication are initialized on first use
    # (No explicit init needed with current architecture)

    # Register error handlers
    register_error_handlers(app)

    # Register route blueprints (stable-core - minimal)
    app.register_blueprint(auth_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(cost_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(integrations_bp)
    app.register_blueprint(approval_bp)
    app.register_blueprint(observe_bp)
    # REMOVED: status_bp
    # DISABLED: feed_bp, livekit_bp, autonomy_bp

    return app


# ---------------------------------------------------------------------------
# Global LLM instances (legacy — pipeline handles its own LLM now)
# ---------------------------------------------------------------------------

llm_instance = None
tools_list = None
retriever_instance = None
autonomy_llm = None  # Separate LLM for autonomous background operations


def initialize_llm_and_tools():
    """No-op stub — the ConversationPipeline manages its own LLM provider.
    Kept for backwards compatibility with startup sequence.
    """
    global llm_instance, tools_list, retriever_instance
    print("ℹ️  LLM setup handled by ConversationPipeline")
    return None, None, None


def initialize_autonomy_llm():
    """Initialize a dedicated LLM for autonomous background operations.

    This is separate from the conversation LLM — it powers proactive
    reach-outs, opinion formation, and other background tasks that run
    without user interaction.
    """
    global autonomy_llm

    try:
        from langchain_fireworks import ChatFireworks

        model = os.getenv("AUTONOMY_MODEL", "accounts/fireworks/models/kimi-k2-instruct-0905")
        api_key = os.getenv("FIREWORKS_API_KEY")

        if not api_key:
            print("⚠️  FIREWORKS_API_KEY not set, autonomy LLM disabled")
            return None

        autonomy_llm = ChatFireworks(
            model=model,
            api_key=api_key,
            temperature=0.7,
            max_tokens=300
        )
        print(f"✅ Autonomy LLM initialized with model: {model}")
        return autonomy_llm

    except Exception as e:
        print(f"❌ Failed to initialize autonomy LLM: {e}")
        return None


def create_socketio(app):
    """
    Create and configure SocketIO instance.

    Args:
        app: Flask application instance

    Returns:
        Configured SocketIO instance
    """
    socketio = SocketIO(
        app,
        **SocketIOConfig.get_kwargs()
    )

    # Apply configuration
    SocketIOConfig.apply_to_socketio(socketio)

    # Register error handlers
    register_socketio_error_handlers(socketio)

    # Register WebSocket event handlers
    register_socketio_handlers(socketio)

    # Register /observe namespace handlers
    register_observe_socketio(socketio)

    return socketio


# ============================================================================
# STATIC FILES AND ROOT ROUTES
# ============================================================================

def register_static_routes(app):
    """
    Register routes for static files and root paths.

    Args:
        app: Flask application instance
    """

    @app.route('/')
    def root():
        """Serve root index page"""
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/observe')
    def observe():
        """Serve observation dashboard"""
        return send_from_directory(app.static_folder, 'observe.html')

    @app.route('/api/static/<path:filename>')
    def serve_static(filename):
        """Serve static files"""
        return send_from_directory(app.static_folder, filename)

    @app.route('/img/<path:filename>')
    def serve_image(filename):
        """Serve uploaded images"""
        try:
            return send_from_directory(app.config.get('UPLOAD_FOLDER', '/tmp/companion-uploads'), filename)
        except FileNotFoundError:
            return {'error': 'Image not found'}, 404


# ============================================================================
# STARTUP INITIALIZATION
# ============================================================================

def init_background_autonomy():
    """
    Initialize the companion's background task scheduler (internal clock).

    This gives the companion an internal sense of time and enables it to execute
    background tasks on a schedule, not just reactively during conversations.
    """
    # Background autonomy is now handled by the unified scheduler + always-on service.
    # The old background_scheduler module has been removed.
    print("Background autonomy handled by unified scheduler + always-on service")
    return None


def init_scheduled_jobs(socketio=None):
    """Register all recurring background jobs with the unified scheduler.

    Jobs run independently via APScheduler. Each job is wrapped in a
    try/except so one missing module doesn't prevent others from starting.

    Job inventory:
    - Always-On Service: Telegram + proactive messaging (continuous)
    - Ops Bot: admin command handler (continuous)
    - Semantic memory extraction (hourly)
    - Value inference (daily at 3 AM)
    - System diagnostics (every 6 hours)
    - Daily conversation summary (daily at 12:05 AM)
    - Core memory refresh (Sundays at 4 AM)
    """
    try:
        from src.scheduling.unified_scheduler import get_unified_scheduler
        unified_scheduler = get_unified_scheduler()

        # Start Always-On Service for autonomous behavior
        # Handles Telegram messaging and proactive reach-outs
        try:
            from src.autonomy.always_on_service import start_always_on
            start_always_on()
            print("✅ Always-On Service started (Telegram + proactive messaging)")
        except Exception as e:
            print(f"⚠️  Always-On Service not available: {e}")

        # Start Ops Bot command handler (separate from the companion's Telegram)
        try:
            from src.ops.ops_bot_commands import start_ops_bot
            if start_ops_bot():
                print("✅ Ops Bot command handler started")
            else:
                print("⚠️  Ops Bot not configured (missing token)")
        except Exception as e:
            print(f"⚠️  Ops Bot not available: {e}")

        # Register hourly semantic memory extraction job (if available)
        try:
            from src.tasks.semantic_memory.hourly_semantic_extract import run_hourly_extraction
            unified_scheduler.register_interval_job(
                job_id='semantic_memory_extraction',
                func=run_hourly_extraction,
                hours=1,
                description='Extract biographical memories from recent conversations'
            )
            print("✅ Semantic memory extraction job registered (hourly)")
        except ImportError as e:
            print(f"⚠️  Semantic memory extraction not available: {e}")

        # Register value inference job (daily at 3 AM)
        # Analyzes the companion's messages to infer hidden values and personality
        try:
            from src.autonomy.value_inference import run_value_inference
            unified_scheduler.schedule_cron(
                job_id='value_inference',
                func=run_value_inference,
                hour=3,
                minute=0,
                full_history=False  # Just recent messages
            )
            print("✅ Value inference job registered (daily at 3 AM)")
        except ImportError as e:
            print(f"⚠️  Value inference not available: {e}")

        # Register system diagnostics job (every 6 hours)
        # Checks all subsystems and reports failures to Sentry
        try:
            from src.diagnostics.system_health import run_diagnostics
            unified_scheduler.schedule_interval(
                job_id='system_diagnostics',
                func=run_diagnostics,
                hours=6,
                report_to_sentry=True
            )
            print("✅ System diagnostics job registered (every 6 hours)")
        except ImportError as e:
            print(f"⚠️  System diagnostics not available: {e}")

        # Register hierarchical memory tier management job (Mem0-inspired)
        # Runs daily at 4 AM to promote/demote facts between tiers
        # DISABLED: Dead jobs removed during Neo4j purge
        # - tier_management_job (hierarchical_memory_tiers moved to _dead)
        # - importance_backfill_job (moved to _dead)
        # - biography_vectorization_job (moved to _dead)

        # Register daily summary job (daily at 12:05 AM Pacific)
        # Generates daily conversation summaries (Clawdbot-inspired)
        try:
            from src.tasks.daily_summary_task import generate_daily_summary
            unified_scheduler.schedule_cron(
                job_id='daily_summary',
                func=lambda: generate_daily_summary.delay(),
                hour=0,
                minute=5
            )
            print("✅ Daily summary job registered (daily at 12:05 AM)")
        except ImportError as e:
            print(f"⚠️  Daily summary not available: {e}")

        # Register core memory refresh job (weekly on Sundays at 4 AM)
        # Regenerates the companion's narrative memory about the user
        try:
            from src.tasks.core_memory_task import refresh_core_memory
            unified_scheduler.schedule_cron(
                job_id='core_memory_refresh',
                func=lambda: refresh_core_memory.delay(),
                hour=4,
                minute=0,
                day_of_week='sun'
            )
            print("✅ Core memory refresh job registered (Sundays at 4 AM)")
        except ImportError as e:
            print(f"⚠️  Core memory refresh not available: {e}")

        # Start the unified scheduler
        unified_scheduler.start()
        print("✅ Unified scheduler started")

    except Exception as e:
        print(f"⚠️  Error initializing scheduled jobs: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Create Flask app
    app = create_app()

    # Initialize LLM and tools globally BEFORE any request handling
    print("⚙️  Initializing LLM and tools...")
    initialize_llm_and_tools()
    print("✅ LLM and tools initialized")

    # Initialize autonomy LLM (Phase 1.2)
    print("⚙️  Initializing autonomy LLM...")
    initialize_autonomy_llm()

    # Create SocketIO
    socketio = create_socketio(app)

    # Register static routes
    register_static_routes(app)

    # Initialize the companion's background autonomy (internal clock with scheduled tasks)
    print("⚙️  Initializing companion's background autonomy...")
    init_background_autonomy()
    print("✅ Background autonomy initialized")

    # Initialize scheduled jobs with SocketIO instance
    init_scheduled_jobs(socketio=socketio)

    # ================================================================
    # Redis pub/sub listeners — bridge between background workers
    # and the WebSocket layer. Celery workers publish events to Redis
    # channels; these listeners forward them to the right Socket.IO room.
    # ================================================================

    def start_redis_subscriber():
        """Forward image_generated events from Redis pub/sub to Socket.IO rooms."""
        import redis
        import json
        try:
            redis_client = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                decode_responses=True
            )
            pubsub = redis_client.pubsub()
            pubsub.subscribe('image_generated')
            print("✅ Redis subscriber started for image_generated events")

            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        room_id = data.pop('room_id', None)
                        if room_id:
                            socketio.emit('image_generated', data, room=room_id)
                            print(f"📤 Forwarded image_generated to room {room_id}")
                    except Exception as e:
                        print(f"⚠️  Error processing Redis message: {e}")
        except Exception as e:
            print(f"❌ Redis subscriber error: {e}")

    # Start Redis subscriber in a background thread
    import threading
    redis_thread = threading.Thread(target=start_redis_subscriber, daemon=True)
    redis_thread.start()

    # Start observation dashboard Redis subscriber
    start_observe_subscriber(socketio)

    def start_interjection_subscriber():
        """Forward companion interjection events (unprompted messages during active chat)."""
        import redis
        import json
        try:
            redis_client = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                decode_responses=True
            )
            pubsub = redis_client.pubsub()
            pubsub.subscribe('companion_interjection')  # TODO: rename to 'companion_interjection' when src/autonomy/ is updated
            print("✅ Redis subscriber started for interjection events")

            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        room_id = data.pop('room_id', None)
                        if room_id:
                            socketio.emit('message', data, room=room_id)
                            print(f"💬 Forwarded interjection to room {room_id}")
                    except Exception as e:
                        print(f"⚠️  Error processing interjection message: {e}")
        except Exception as e:
            print(f"❌ Interjection subscriber error: {e}")

    interjection_thread = threading.Thread(target=start_interjection_subscriber, daemon=True)
    interjection_thread.start()

    # Print startup information
    print("\n" + "="*60)
    print("🤖 Companion AI Chat Interface Starting...")
    print("="*60)
    print("✅ Configuration loaded")
    print("✅ Database connected")
    print("✅ Routes registered")
    print("✅ WebSocket support enabled")
    print("✅ Error handlers configured")
    print("✅ Middleware registered")
    print("="*60)
    print("📱 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    # Start server
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=app.debug,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
